"""Twelve disposable native first steps, using frozen probe gradients only."""

import sympy  # noqa: F401
import torch
import torch._dynamo  # noqa: F401

assert not torch.cuda.is_initialized()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import time  # noqa: E402
import traceback  # noqa: E402

from results.adam_update_sensitivity_v1.source.prepare import ROOT, hashes, read, sha  # noqa: E402
from results.fp32_classifier_profile_v1.source.common import (  # noqa: E402
    MemoryLedger,
    clear_boundary,
    cpu_tree,
    tree_hash,
)
from src.core.config import ModelConfig, TrainConfig  # noqa: E402
from src.core.optimization import parameter_groups  # noqa: E402
from src.core.reproducibility import environment, write_json  # noqa: E402
from src.core.transformer import Transformer  # noqa: E402


def run_case(case, device, protocol):
    label = device + "__" + case["label"]
    folder = ROOT / "runs" / label
    folder.mkdir(exist_ok=False)
    ledger = MemoryLedger() if device == "cuda" else None
    state = torch.load(protocol["initial_path"], map_location="cpu", weights_only=True)
    assert state["step"] == 0 and not state["optimizer"]["state"]
    model = Transformer(ModelConfig(**state["model_config"]), protocol["seed"]).to(device)
    model.load_state_dict(state["model"])
    assert tree_hash(model.state_dict()) == protocol["initial_model_hash"]
    tc = TrainConfig(steps=800, batch_size=8, learning_rate=0.0006, seed=101, precision="fp32")
    optimizer = torch.optim.AdamW(
        parameter_groups(model, tc), lr=0.0006, betas=(0.9, 0.95), eps=1e-8
    )
    optimizer.load_state_dict(state["optimizer"])
    assert not optimizer.state
    raw = torch.load(case["probe"]["path"], map_location="cpu", weights_only=True)
    assert tree_hash(raw) == case["probe"]["gradients_hash"]
    assert raw.keys() == dict(model.named_parameters()).keys()
    for name, parameter in model.named_parameters():
        assert parameter.shape == raw[name].shape and raw[name].dtype == torch.float32
        parameter.grad = raw[name].to(device).clone()
    names = {id(p): name for name, p in model.named_parameters()}
    groups = []
    for group, saved in zip(
        optimizer.param_groups, optimizer.state_dict()["param_groups"], strict=True
    ):
        assert tuple(group["betas"]) == (0.9, 0.95) and group["eps"] == 1e-8
        assert group["lr"] == 0.0006 and group["lr_scale"] == 1
        assert group["foreach"] is None and group["fused"] is None
        groups.append(
            dict(
                options={k: v for k, v in group.items() if k != "params"},
                names=[names[id(p)] for p in group["params"]],
                ids=saved["params"],
            )
        )
    if ledger:
        ledger.mark("construction_and_raw_gradient")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    clipped = {k: p.grad.detach().cpu().clone() for k, p in model.named_parameters()}
    clip_path = folder / "clipped.pt"
    torch.save(clipped, clip_path)
    if ledger:
        ledger.mark("clipping_and_serialization")
    start = time.perf_counter()
    optimizer.step()
    if ledger:
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - start) * 1000
    if ledger:
        ledger.mark("single_optimizer_step")
    updated = cpu_tree(dict(model=model.state_dict(), optimizer=optimizer.state_dict()))
    assert all(bool(torch.isfinite(t).all()) for t in updated["model"].values())
    for item in updated["optimizer"]["state"].values():
        assert item["step"].item() == 1
        assert all(bool(torch.isfinite(item[k]).all()) for k in ("exp_avg", "exp_avg_sq"))
    updated_path = folder / "updated.pt"
    torch.save(updated, updated_path)
    if ledger:
        ledger.mark("updated_state_serialization")
    row = dict(
        label=label,
        input_label=case["label"],
        policy=case["policy"],
        device=device,
        source_gradient_sha256=case["probe"]["sha256"],
        initial_model_hash=protocol["initial_model_hash"],
        norm_before_clip=norm.item(),
        groups=groups,
        parameters=sum(p.numel() for p in model.parameters()),
        clipped=dict(
            path=clip_path.as_posix(), sha256=sha(clip_path), tree_hash=tree_hash(clipped)
        ),
        updated=dict(
            path=updated_path.as_posix(), sha256=sha(updated_path), tree_hash=tree_hash(updated)
        ),
        single_step_wall_ms=wall_ms,
        memory_phases=ledger.records if ledger else None,
        disposable_optimizer_steps=1,
        language_training_updates=0,
        forwards=0,
        backwards=0,
        finite=True,
        state_steps_all_one=True,
    )
    write_json(folder / "result.json", row)
    print(label, "one disposable update complete", flush=True)
    return row


def run():
    assert not (ROOT / "result.json").exists()
    protocol = read(ROOT / "protocol.json")
    for field in ("sources", "maintained_files", "input_hashes", "library_hashes"):
        hashes(protocol[field])
    beginning, cases, boundaries = time.perf_counter(), [], []
    for device in protocol["devices"]:
        if device == "cuda":
            assert not torch.cuda.is_initialized()
            write_json(ROOT / "environment.json", environment())
            boundaries.append(clear_boundary())
        for case in protocol["cases"]:
            cases.append(run_case(case, device, protocol))
            if device == "cuda":
                boundaries.append(clear_boundary())
    assert len(cases) == 12 and len(boundaries) == 7
    write_json(
        ROOT / "result.json",
        dict(
            status="COMPLETE",
            cases=cases,
            boundaries=boundaries,
            wall_seconds=time.perf_counter() - beginning,
            disposable_optimizer_steps=12,
            language_training_updates=0,
            training_targets=0,
            forwards=0,
            backwards=0,
            validation_scores=0,
            broad_goal_achieved=False,
        ),
    )


if __name__ == "__main__":
    try:
        run()
    except Exception:
        write_json(ROOT / "study_failure.json", dict(traceback=traceback.format_exc()))
        raise
