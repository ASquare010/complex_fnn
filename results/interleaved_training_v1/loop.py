"""Common phase-instrumented continuation for all native AdamW modes."""

import sympy  # noqa: F401
import torch
import torch._dynamo  # noqa: F401

assert not torch.cuda.is_initialized()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import copy  # noqa: E402
import math  # noqa: E402
import statistics as st  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

from results.fp32_classifier_profile_v1.source.common import (  # noqa: E402
    MemoryLedger,
    clear_boundary,
    cpu_tree,
    set_blocks,
    tensor_hash,
    tree_hash,
)
from results.fp32_decoder_resource_v1.source.common import (  # noqa: E402
    attention_context,
    training_loss,
)
from results.optimizer_memory_v1.source.prepare import ROOT, hashes, read, sha  # noqa: E402
from results.streamed_evaluation_v1.source.evaluation import evaluate  # noqa: E402
from src.core.config import ModelConfig, TrainConfig  # noqa: E402
from src.core.data import TokenData  # noqa: E402
from src.core.optimization import parameter_groups  # noqa: E402
from src.core.reproducibility import environment, write_json  # noqa: E402
from src.core.transformer import Transformer  # noqa: E402


def inventory(model, optimizer, data, extra):
    families = dict(
        parameters=list(model.parameters()),
        buffers=list(model.buffers()),
        gradients=[p.grad for p in model.parameters() if p.grad is not None],
        moments=[s[k] for s in optimizer.state.values() for k in ("exp_avg", "exp_avg_sq")],
        counters=[s["step"] for s in optimizer.state.values()],
        dataset=[data.train, data.valid],
        batch_and_loss=extra,
    )
    seen, sizes = set(), {}
    for name, tensors in families.items():
        size = 0
        for tensor in tensors:
            if tensor is None or not tensor.is_cuda:
                continue
            storage = tensor.untyped_storage()
            key = (tensor.device.index, storage.data_ptr())
            if key not in seen:
                seen.add(key)
                size += storage.nbytes()
        sizes[name] = size
    assert sizes["moments"] == 2 * sizes["parameters"]
    allocated = torch.cuda.memory_allocated()
    known = sum(sizes.values())
    assert allocated >= known
    return dict(
        storage_bytes=sizes, known_storage_bytes=known, unattributed_live_bytes=allocated - known
    )


def describe(values):
    return dict(
        n=len(values),
        mean=st.mean(values),
        median=st.median(values),
        sample_variance=st.variance(values),
        min=min(values),
        max=max(values),
    )


def save_state(path, model, optimizer, data, cfg, step, seed):
    state = dict(
        model=cpu_tree(model.state_dict()),
        optimizer=cpu_tree(optimizer.state_dict()),
        sampler_state=data.generator.get_state().cpu(),
        model_config=asdict(cfg),
        step=step,
        seed=seed,
    )
    torch.save(state, path)
    return dict(
        path=path.as_posix(),
        sha256=sha(path),
        model_hash=tree_hash(state["model"]),
        optimizer_hash=tree_hash(state["optimizer"]),
        sampler_hash=tensor_hash(state["sampler_state"]),
    )


def run_case(fixture, mode, protocol):
    setup_started = time.perf_counter()
    label = fixture["label"] + "__" + mode
    folder = ROOT / "runs" / label
    folder.mkdir(exist_ok=False)
    ledger = MemoryLedger()
    source = torch.load(fixture["checkpoint"], map_location="cpu", weights_only=True)
    assert source["step"] == 800
    cfg = ModelConfig(**source["model_config"])
    tc = TrainConfig(
        steps=30, batch_size=8, learning_rate=0.0006, seed=fixture["seed"], precision="fp32"
    )
    model = Transformer(cfg, fixture["seed"]).cuda()
    model.load_state_dict(source["model"])
    set_blocks(model)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, tc),
        lr=0.0006,
        betas=(0.9, 0.95),
        eps=1e-8,
        **protocol["modes"][mode],
    )
    state = copy.deepcopy(source["optimizer"])
    for group in state["param_groups"]:
        group.update(protocol["modes"][mode])
    optimizer.load_state_dict(state)
    assert tree_hash(optimizer.state_dict()["state"]) == tree_hash(source["optimizer"]["state"])
    data = TokenData(
        Path(protocol["datasets"][fixture["dataset"]]["path"]), "cuda", fixture["seed"] + 10000
    )
    data.generator.set_state(source["sampler_state"])
    names = {id(p): name for name, p in model.named_parameters()}
    groups = [
        dict(
            options={k: v for k, v in g.items() if k != "params"},
            names=[names[id(p)] for p in g["params"]],
            ids=s["params"],
        )
        for g, s in zip(optimizer.param_groups, optimizer.state_dict()["param_groups"], strict=True)
    ]
    for group in groups:
        assert group["options"]["lr"] == 0.0006 and group["options"]["eps"] == 1e-8
        assert tuple(group["options"]["betas"]) == (0.9, 0.95)
    initial_model_hash = tree_hash(model.state_dict())
    assert initial_model_hash == tree_hash(source["model"])
    initial_moment_hash = tree_hash(source["optimizer"]["state"])
    initial_sampler_hash = tensor_hash(data.generator.get_state())
    del source, state, group
    setup_wall_ms = 1000 * (time.perf_counter() - setup_started)

    def mark(name, extra=()):
        row = ledger.mark(name)
        row.update(inventory(model, optimizer, data, extra))
        return row

    mark("construction")
    before_score = evaluate(model, data, 8, 512, 10**9, "classifier_chunks")
    mark("initial_validation")
    records = []
    first_state = first_gradients = None
    for step in range(1, protocol["steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        x, y = data.batch(8, 512)
        tokens_hash, targets_hash = tensor_hash(x), tensor_hash(y)
        mark(f"batch_{step}", [x, y])
        events = [torch.cuda.Event(enable_timing=True) for _ in range(8)]
        started_ns = time.time_ns()
        started = time.perf_counter()
        events[0].record()
        loss = training_loss(model, x, y, fixture["loss_policy"])
        events[1].record()
        events[2].record()
        with attention_context(fixture["loss_policy"]):
            loss.backward()
        events[3].record()
        if step == 1:
            raw = {k: p.grad.detach().cpu() for k, p in model.named_parameters()}
            mark("first_raw_gradient_copy", [x, y, loss])
        events[4].record()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        events[5].record()
        if step == 1:
            clipped = {k: p.grad.detach().cpu() for k, p in model.named_parameters()}
        events[6].record()
        optimizer.step()
        events[7].record()
        events[7].synchronize()
        wall_ms = 1000 * (time.perf_counter() - started)
        until_ns = time.time_ns()
        event_complete_ms = events[0].elapsed_time(events[7])
        mark(f"optimizer_{step}", [x, y, loss, norm])
        event_ms = {
            name: events[2 * i].elapsed_time(events[2 * i + 1])
            for i, name in enumerate(("forward", "backward", "clipping", "optimizer"))
        }
        row = dict(
            step=step,
            absolute_step=800 + step,
            tokens_hash=tokens_hash,
            targets_hash=targets_hash,
            loss=loss.item(),
            norm=norm.item(),
            wall_ms=wall_ms,
            event_complete_ms=event_complete_ms,
            at_ns=started_ns,
            until_ns=until_ns,
            event_ms=event_ms,
            event_sum_ms=sum(event_ms.values()),
        )
        assert math.isfinite(row["loss"]) and math.isfinite(row["norm"])
        records.append(row)
        if step == 1:
            path = folder / "first_gradients.pt"
            torch.save(dict(raw=raw, clipped=clipped), path)
            first_gradients = dict(
                path=path.as_posix(),
                sha256=sha(path),
                raw_hash=tree_hash(raw),
                clipped_hash=tree_hash(clipped),
            )
            first_state = save_state(
                folder / "first.pt", model, optimizer, data, cfg, 801, fixture["seed"]
            )
            del raw, clipped
            mark("first_artifact_serialization", [x, y, loss, norm])
    after_score = evaluate(model, data, 8, 512, 10**9, "classifier_chunks")
    mark("final_validation", [x, y, loss, norm])
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    assert all(
        float(s["step"]) == 830
        and all(bool(torch.isfinite(s[k]).all()) for k in ("exp_avg", "exp_avg_sq"))
        for s in optimizer.state.values()
    )
    mark("final_finite_checks", [x, y, loss, norm])
    final_state = save_state(folder / "final.pt", model, optimizer, data, cfg, 830, fixture["seed"])
    mark("final_artifact_serialization", [x, y, loss, norm])
    warm = records[protocol["warmup"] :]
    timing = {
        k: describe([r[k] for r in warm]) for k in ("wall_ms", "event_sum_ms", "event_complete_ms")
    }
    stability = {}
    for k in timing:
        halves = [st.median(r[k] for r in part) for part in (warm[:10], warm[10:])]
        stability[k] = max(halves) / min(halves)
    output = dict(
        label=label,
        setup_wall_ms=setup_wall_ms,
        fixture=fixture,
        mode=mode,
        groups=groups,
        initial_model_hash=initial_model_hash,
        initial_moment_hash=initial_moment_hash,
        initial_sampler_hash=initial_sampler_hash,
        before_score=before_score,
        after_score=after_score,
        first_gradients=first_gradients,
        first_state=first_state,
        final_state=final_state,
        updates=records,
        phases=ledger.records,
        timing=timing,
        timing_stability=stability,
        peak_allocated_bytes=max(m["peak_allocated_bytes"] for m in ledger.records),
        peak_reserved_bytes=max(m["peak_reserved_bytes"] for m in ledger.records),
        parameters=cfg.total_parameters,
        training_updates=30,
        training_targets=30 * 4096,
        training_backwards=30,
        full_validation_scores=2,
        finite=True,
    )
    write_json(folder / "result.json", output)
    print(label, "30 updates complete", flush=True)
    return output


def run():
    assert not (ROOT / "result.json").exists()
    p = read(ROOT / "protocol.json")
    for field in ("sources", "maintained_files", "input_hashes", "library_hashes"):
        hashes(p[field])
    write_json(ROOT / "environment.json", environment())
    started = time.perf_counter()
    cases, boundaries = [], [clear_boundary()]
    for fixture in p["fixtures"]:
        for mode in fixture["mode_order"]:
            cases.append(run_case(fixture, mode, p))
            boundaries.append(clear_boundary())
    write_json(
        ROOT / "result.json",
        dict(
            status="COMPLETE",
            cases=cases,
            boundaries=boundaries,
            wall_seconds=time.perf_counter() - started,
            training_updates=360,
            training_targets=1474560,
            training_backwards=360,
            full_validation_scores=24,
            broad_goal_achieved=False,
        ),
    )


if __name__ == "__main__":
    try:
        run()
    except Exception:
        write_json(ROOT / "failure.json", dict(traceback=traceback.format_exc()))
        raise
