"""H117 fresh three-seed training; one common loop for all policies."""

import sympy  # noqa: F401 -- established import preloading, before CUDA initialization
import torch
import torch._dynamo  # noqa: F401

assert not torch.cuda.is_initialized()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import hashlib  # noqa: E402
import json  # noqa: E402
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
    POLICIES,
    attention_context,
    qualify,
    training_loss,
)  # noqa: E402
from results.fp32_training_replication_v1.source.initialize import generate_initials  # noqa: E402
from results.streamed_evaluation_v1.source.evaluation import evaluate  # noqa: E402
from src.core.config import ModelConfig, TrainConfig  # noqa: E402
from src.core.data import TokenData  # noqa: E402
from src.core.diagnostics import gradient_stats, inspect_layers  # noqa: E402
from src.core.optimization import group_summary, parameter_groups  # noqa: E402
from src.core.reproducibility import environment, sha256, write_json  # noqa: E402
from src.core.transformer import Transformer  # noqa: E402

ROOT = Path("results/fp32_training_replication_recovery_v1")


def describe(values):
    return dict(
        mean=st.mean(values),
        median=st.median(values),
        sample_variance=st.variance(values),
        min=min(values),
        max=max(values),
        n=len(values),
    )


def run_case(fixture, policy, protocol):
    beginning = time.perf_counter()
    label = fixture["label"] + "__" + policy
    folder = ROOT / "runs" / label
    folder.mkdir(exist_ok=False)
    ledger = MemoryLedger()
    state = torch.load(fixture["checkpoint"], map_location="cpu", weights_only=True)
    assert state["model_config"] == fixture["model_config"]
    assert state["step"] == fixture["source_step"]
    cfg = ModelConfig(**state["model_config"])
    tc = TrainConfig(
        steps=800,
        batch_size=fixture["batch"],
        learning_rate=0.0006,
        seed=fixture["seed"],
        precision=POLICIES[policy][0],
    )
    model = Transformer(cfg, fixture["seed"]).cuda()
    model.load_state_dict(state["model"])
    set_blocks(model)
    assert tree_hash(model.state_dict()) == tree_hash(state["model"])
    initial_model_hash = tree_hash(state["model"])
    optimizer = torch.optim.AdamW(
        parameter_groups(model, tc), lr=tc.learning_rate, betas=(0.9, 0.95), eps=1e-8
    )
    optimizer.load_state_dict(state["optimizer"])
    assert tree_hash(optimizer.state_dict()["state"]) == tree_hash(state["optimizer"]["state"])
    source_optimizer_hash = tree_hash(state["optimizer"])
    source_lrs = [g["lr"] for g in optimizer.param_groups]
    for group in optimizer.param_groups:
        assert tuple(group["betas"]) == (0.9, 0.95) and group["eps"] == 1e-8
        assert group.get("lr_scale", 1) == 1
        group["lr"] = tc.learning_rate
    assert all(float(s["step"]) == fixture["source_step"] for s in optimizer.state.values())
    initial_optimizer_hash = tree_hash(optimizer.state_dict())
    del state, group
    data = TokenData(
        Path(protocol["datasets"][fixture["dataset"]]["path"]), "cuda", fixture["seed"] + 10000
    )
    sampler_state = data.generator.get_state()
    initial_sampler_hash = tensor_hash(sampler_state)
    ledger.mark("construction_data_and_optimizer")
    initial_eval = evaluate(model, data, fixture["batch"], cfg.context, 10**9, "classifier_chunks")
    ledger.mark("initial_full_validation")
    x, y = data.batch(fixture["batch"], cfg.context)
    model.train()
    loss = training_loss(model, x, y, policy)
    with attention_context(policy):
        loss.backward()
    initial_probe = dict(
        loss=loss.item(),
        tokens_hash=tensor_hash(x),
        targets_hash=tensor_hash(y),
        gradient_statistics=gradient_stats(model),
    )
    gradients = {k: p.grad.detach().cpu() for k, p in model.named_parameters()}
    assert all(bool(torch.isfinite(g).all()) for g in gradients.values())
    gradient_path = folder / "initial_gradients.pt"
    torch.save(gradients, gradient_path)
    initial_probe.update(
        path=gradient_path.as_posix(),
        sha256=sha256(gradient_path),
        gradients_hash=tree_hash(gradients),
    )
    del gradients, loss, x, y
    data.generator.set_state(sampler_state)
    del sampler_state
    ledger.mark("initial_gradient_probe_and_serialization")
    records, order, checkpoints = [], hashlib.sha256(), []
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    loop_start = time.perf_counter()
    for step in range(1, 801):
        x, y = data.batch(fixture["batch"], cfg.context)
        batch_hash, targets_hash = tensor_hash(x), tensor_hash(y)
        order.update(batch_hash.encode())
        optimizer.zero_grad(set_to_none=True)
        ledger.mark(f"before_update_{step}")
        at_ns = time.time_ns()
        start = time.perf_counter()
        events[0].record()
        loss = training_loss(model, x, y, policy)
        events[1].record()
        with attention_context(policy):
            loss.backward()
        events[2].record()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        events[3].record()
        events[3].synchronize()
        end = time.perf_counter()
        until_ns = time.time_ns()
        memory = ledger.mark(f"update_{step}")
        record = dict(
            step=step,
            absolute_step=fixture["source_step"] + step,
            warmup=step <= 20,
            at_ns=at_ns,
            until_ns=until_ns,
            loss=loss.item(),
            preclip_norm=norm.item(),
            tokens_hash=batch_hash,
            targets_hash=targets_hash,
            wall_update_ms=1000 * (end - start),
            event_forward_ms=events[0].elapsed_time(events[1]),
            event_backward_ms=events[1].elapsed_time(events[2]),
            event_optimizer_ms=events[2].elapsed_time(events[3]),
            event_update_ms=events[0].elapsed_time(events[3]),
            **memory,
        )
        assert math.isfinite(record["loss"]) and math.isfinite(record["preclip_norm"])
        if step in (1, 20, 200, 400, 800):
            record["gradient_statistics_after_clip"] = gradient_stats(model)
        records.append(record)
        with (folder / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if step % 100 == 0:
            print(json.dumps(dict(active=label, step=step, loss=record["loss"])), flush=True)
        if step in (200, 400):
            score = evaluate(model, data, fixture["batch"], cfg.context, 10**9, "classifier_chunks")
            ledger.mark(f"validation_{step}")
            saved = dict(
                model=cpu_tree(model.state_dict()),
                optimizer=cpu_tree(optimizer.state_dict()),
                model_config=asdict(cfg),
                training_config=asdict(tc),
                seed=fixture["seed"],
                step=step,
                sampler_state=data.generator.get_state().cpu(),
            )
            path = folder / f"step{step}.pt"
            torch.save(saved, path)
            checkpoints.append(
                dict(
                    step=step,
                    path=path.as_posix(),
                    sha256=sha256(path),
                    model_hash=tree_hash(saved["model"]),
                    optimizer_hash=tree_hash(saved["optimizer"]),
                    sampler_hash=tensor_hash(saved["sampler_state"]),
                    validation=score,
                )
            )
            del saved, score
            ledger.mark(f"checkpoint_{step}")
    loop_seconds = time.perf_counter() - loop_start
    del loss, norm, x, y, events
    ledger.mark("after_update_logging")
    final_eval = evaluate(model, data, fixture["batch"], cfg.context, 10**9, "classifier_chunks")
    ledger.mark("final_full_validation")
    fixed_x, _ = next(data.validation(fixture["batch"], cfg.context, 1))
    model.eval()
    diagnostics = inspect_layers(model, fixed_x, "cuda", "bf16")
    finite = all(
        bool(torch.isfinite(p).all()) and bool(torch.isfinite(p.grad).all())
        for p in model.parameters()
    )
    finite = finite and all(
        bool(torch.isfinite(v).all())
        for s in optimizer.state.values()
        for v in s.values()
        if isinstance(v, torch.Tensor)
    )
    assert finite and all(row["finite"] for row in diagnostics.values())
    assert all(float(s["step"]) == fixture["source_step"] + 800 for s in optimizer.state.values())
    ledger.mark("final_layer_and_finite_diagnostics")
    checkpoint = dict(
        model=cpu_tree(model.state_dict()),
        optimizer=cpu_tree(optimizer.state_dict()),
        model_config=asdict(cfg),
        training_config=asdict(tc),
        seed=fixture["seed"],
        step=fixture["source_step"] + 800,
        continuation_steps=800,
        sampler_state=data.generator.get_state().cpu(),
    )
    path = folder / "final.pt"
    torch.save(checkpoint, path)
    final_model_hash = tree_hash(checkpoint["model"])
    final_optimizer_hash = tree_hash(checkpoint["optimizer"])
    final_sampler_hash = tensor_hash(checkpoint["sampler_state"])
    del checkpoint, fixed_x
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    gradient_bytes = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters())
    optimizer_bytes = sum(
        v.numel() * v.element_size()
        for s in optimizer.state.values()
        for v in s.values()
        if isinstance(v, torch.Tensor)
    )
    ledger.mark("final_checkpoint_and_accounting")
    timed = records[20:]
    timing = {
        k: describe([r[k] for r in timed])
        for k in (
            "wall_update_ms",
            "event_update_ms",
            "event_forward_ms",
            "event_backward_ms",
            "event_optimizer_ms",
        )
    }
    block_means = [
        st.mean(r["wall_update_ms"] for r in timed[start : start + 260]) for start in (0, 260, 520)
    ]
    output = dict(
        status="COMPLETE",
        label=label,
        fixture=fixture,
        policy=policy,
        execution=dict(
            decoder_precision=POLICIES[policy][0],
            attention_backend=POLICIES[policy][1],
            classifier_policy=POLICIES[policy][2],
        ),
        source_checkpoint_sha256=protocol["checkpoint_hashes"][fixture["checkpoint"]],
        initial_model_hash=initial_model_hash,
        source_optimizer_hash=source_optimizer_hash,
        initial_optimizer_hash=initial_optimizer_hash,
        source_learning_rates=source_lrs,
        training_config=asdict(tc),
        optimizer_groups=group_summary(optimizer.param_groups),
        initial_sampler_hash=initial_sampler_hash,
        final_sampler_hash=final_sampler_hash,
        data_order_hash=order.hexdigest(),
        initial_probe=initial_probe,
        initial_validation=initial_eval,
        final_validation=final_eval,
        records=records,
        memory_phases=ledger.records,
        timing=timing,
        timing_block_means=block_means,
        timing_stability_ratio=max(block_means) / min(block_means),
        final_model_hash=final_model_hash,
        final_optimizer_hash=final_optimizer_hash,
        checkpoint=dict(path=path.as_posix(), sha256=sha256(path)),
        intermediate_checkpoints=checkpoints,
        parameter_count=cfg.total_parameters,
        ffn_parameters=cfg.unique_ffn_parameters,
        parameter_bytes=parameter_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_bytes=optimizer_bytes,
        peak_job_allocated_bytes=max(r["peak_allocated_bytes"] for r in ledger.records),
        peak_job_reserved_bytes=max(r["peak_reserved_bytes"] for r in ledger.records),
        peak_training_allocated_bytes=max(r["peak_allocated_bytes"] for r in records),
        weights_gradients_moments_finite=finite,
        diagnostics=diagnostics,
        clip_fraction=sum(r["preclip_norm"] > 1 for r in records) / 800,
        optimizer_updates=800,
        profile_backward_passes=801,
        training_targets=800 * fixture["batch"] * cfg.context,
        training_loop_wall_seconds=loop_seconds,
        case_wall_seconds=time.perf_counter() - beginning,
    )
    write_json(folder / "metrics.json", output)
    print(
        json.dumps(
            dict(
                completed=label,
                nll=final_eval["nll"],
                peak_mib=output["peak_job_allocated_bytes"] / 2**20,
                wall_ms=timing["wall_update_ms"]["median"],
                stability=output["timing_stability_ratio"],
            )
        ),
        flush=True,
    )
    return output


def run():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    for name, digest in protocol["sources"].items():
        assert sha256(Path(name)) == digest, name
    for name, digest in protocol["checkpoint_hashes"].items():
        assert sha256(Path(name)) == digest, name
    initials = generate_initials(protocol)
    write_json(ROOT / "initializations.json", initials)
    protocol["checkpoint_hashes"] = {r["path"]: r["sha256"] for r in initials["initializations"]}
    env = environment()
    env.update(
        tf32=False,
        threads=torch.get_num_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    )
    write_json(ROOT / "environment.json", env)
    write_json(ROOT / "qualification.json", qualify())
    print("20 full-model double comparisons / 24 qualification backwards passed", flush=True)
    boundaries, cases = [clear_boundary()], []
    beginning = time.perf_counter()
    for fixture in protocol["fixtures"]:
        for policy in fixture["policy_order"]:
            cases.append(run_case(fixture, policy, protocol))
            boundaries.append(clear_boundary())
            write_json(
                ROOT / "progress.json",
                dict(
                    completed=len(cases),
                    optimizer_updates=800 * len(cases),
                    last=cases[-1]["label"],
                ),
            )
    assert len(cases) == 18
    write_json(
        ROOT / "result.json",
        dict(
            status="COMPLETE",
            cases=cases,
            boundaries=boundaries,
            optimizer_updates=sum(c["optimizer_updates"] for c in cases),
            profile_backward_passes=sum(c["profile_backward_passes"] for c in cases),
            training_targets=sum(c["training_targets"] for c in cases),
            elapsed_seconds=time.perf_counter() - beginning,
            all_phase_peaks_recorded=True,
            qualification_backward_passes=24,
            fresh_training_runs=18,
            broad_goal_achieved=False,
        ),
    )


if __name__ == "__main__":
    try:
        run()
    except Exception:
        write_json(ROOT / "failure.json", dict(traceback=traceback.format_exc()))
        raise
