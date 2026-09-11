"""Bounded H101 qualification/resource study; run with UV from repository root."""

import argparse
import gc
import hashlib
import json
import statistics
import time
import traceback
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from results.token_memory_v1.source.execution import POLICIES, common_loss, install
from src.core.benchmark import autocast
from src.core.config import ModelConfig, TrainConfig
from src.core.data import TokenData
from src.core.optimization import parameter_groups
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.transformer import Transformer

ROOT = Path("results/token_memory_v1")
PLAN = Path("research/token_memory_plan.md")
WIDTHS = {
    "gelu": 1536,
    "swiglu": 1024,
    "gelu_narrow": 456,
    "swiglu_narrow": 304,
    "blockshuffle_swiglu": 2048,
}


def tensor_hash(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def relative_error(left, right):
    return float((left.double() - right.double()).norm() / right.double().norm().clamp_min(1e-8))


def qualify():
    torch.set_num_threads(4)
    records = []
    for device in ("cpu", "cuda"):
        for variant in WIDTHS:
            cfg = ModelConfig(
                variant=variant,
                width=24,
                hidden=48,
                layers=2,
                heads=3,
                groups=8,
                vocab_size=32,
                context=7,
            )
            x = torch.arange(14, device=device).reshape(2, 7)
            reference = None
            for policy in ("native", "ffn_chunks", "loss_chunks", "both_chunks"):
                model = Transformer(cfg, 17).to(device)
                if device == "cpu":
                    model.double()
                identities = {n: id(p) for n, p in model.named_parameters()}
                keys = tuple(model.state_dict())
                install(model, policy, 5)
                assert identities == {n: id(p) for n, p in model.named_parameters()}
                assert keys == tuple(model.state_dict())
                with autocast(device, "bf16" if device == "cuda" else "fp32"):
                    # Avoid the legacy .loss FP32 cast for the CPU double witness.
                    if device == "cpu" and policy in ("native", "ffn_chunks"):
                        loss = torch.nn.functional.cross_entropy(
                            model(x).flatten(0, 1), (x + 1).flatten()
                        )
                    else:
                        loss = model.loss(x, x + 1)
                loss.backward()
                grads = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()}
                if reference is None:
                    reference = (loss.item(), grads)
                errors = {n: relative_error(g, reference[1][n]) for n, g in grads.items()}
                if device == "cpu":
                    for n, g in grads.items():
                        torch.testing.assert_close(g, reference[1][n], rtol=1e-9, atol=1e-11)
                limit = 0.02 if device == "cuda" else 1e-9
                loss_error = abs(loss.item() / reference[0] - 1)
                passed = max(errors.values()) <= limit and loss_error <= (
                    0.001 if device == "cuda" else 1e-9
                )
                model.eval()
                with torch.no_grad(), autocast(device, "bf16" if device == "cuda" else "fp32"):
                    predicted = model(x)
                records.append(
                    dict(
                        device=device,
                        variant=variant,
                        policy=policy,
                        loss_relative_error=loss_error,
                        gradient_errors=errors,
                        eval_finite=bool(torch.isfinite(predicted).all()),
                        passed=passed,
                    )
                )
                del model, loss, grads, predicted
    write_json(
        ROOT / "qualification.json",
        {"records": records, "passed": all(r["passed"] for r in records)},
    )
    assert all(r["passed"] for r in records), (
        "Frozen fidelity gate failed; inspect qualification.json"
    )
    print(f"Qualification: {len(records)} full-model cases pass", flush=True)


def score(model, batches):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad(), autocast("cuda", "bf16"):
        for x, y in batches:
            total += common_loss(model, x, y).item() * y.numel()
            count += y.numel()
    model.train()
    return total / count


def run_case(variant, batch, context, seed, policy, data):
    gc.collect()
    torch.cuda.empty_cache()
    cfg = ModelConfig(
        variant=variant,
        width=384,
        hidden=WIDTHS[variant],
        layers=8,
        heads=6,
        groups=8,
        vocab_size=4096,
        context=context,
    )
    model = Transformer(cfg, seed).cuda()
    assert sum(p.numel() for p in model.parameters()) == cfg.total_parameters
    install(model, policy, 512)
    tc = TrainConfig(learning_rate=0.0006, weight_decay=0.1, seed=seed)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, tc), lr=tc.learning_rate, betas=(0.9, 0.95), eps=1e-8
    )
    data.generator.manual_seed(seed + 10000)
    records = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    input_hash = hashlib.sha256()
    for step in range(12):
        x, y = data.batch(batch, context)
        input_hash.update(tensor_hash(x).encode())
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with autocast("cuda", "bf16"):
            loss = model.loss(x, y)
        torch.cuda.synchronize()
        forward_end = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward_end = time.perf_counter()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        end = time.perf_counter()
        records.append(
            dict(
                step=step + 1,
                loss=loss.item(),
                grad_norm=norm.item(),
                forward_ms=1000 * (forward_end - start),
                backward_ms=1000 * (backward_end - forward_end),
                optimizer_ms=1000 * (end - backward_end),
                update_ms=1000 * (end - start),
            )
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    gradient_bytes = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters())
    optimizer_bytes = sum(
        v.numel() * v.element_size()
        for s in optimizer.state.values()
        for v in s.values()
        if isinstance(v, torch.Tensor)
    )
    # Capture memory before validation and CPU checkpoint export.
    valid = list(data.validation(batch, context, 8))
    nll = score(model, valid)
    repeat_nll = score(model, valid)
    assert nll == repeat_nll
    states = {n: tensor_hash(p) for n, p in model.state_dict().items()}
    finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    key = f"{variant}_b{batch}_t{context}_s{seed}_{policy}"
    checkpoint = None
    if policy == "native":
        checkpoint = ROOT / "checkpoints" / f"{key}.pt"
        checkpoint.parent.mkdir(exist_ok=True)
        torch.save(
            {
                "model_config": asdict(cfg),
                "seed": seed,
                "model": {n: p.detach().cpu() for n, p in model.state_dict().items()},
            },
            checkpoint,
        )
    timed = records[4:]
    result = dict(
        key=key,
        variant=variant,
        batch=batch,
        context=context,
        seed=seed,
        policy=policy,
        config=asdict(cfg),
        training=asdict(tc),
        training_steps=12,
        training_tokens=12 * batch * context,
        data_order_sha256=input_hash.hexdigest(),
        parameter_count=cfg.total_parameters,
        ffn_parameter_count=cfg.unique_ffn_parameters,
        parameter_bytes=parameter_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_bytes=optimizer_bytes,
        peak_allocated_bytes=peak,
        peak_reserved_bytes=reserved,
        validation_nll=nll,
        validation_targets=sum(y.numel() for _, y in valid),
        independent_repeat_nll=repeat_nll,
        weights_finite=finite,
        clip_fraction=sum(r["grad_norm"] > 1 for r in records) / len(records),
        state_hashes=states,
        checkpoint=str(checkpoint) if checkpoint else None,
        records=records,
        timing={
            name: {
                "mean": statistics.mean(r[name] for r in timed),
                "median": statistics.median(r[name] for r in timed),
                "sample_variance": statistics.variance(r[name] for r in timed),
            }
            for name in ("forward_ms", "backward_ms", "optimizer_ms", "update_ms")
        },
    )
    write_json(ROOT / "cases" / key / "metrics.json", result)
    print(
        f"{key}: {peak / 2**20:.2f} MiB, {result['timing']['update_ms']['median']:.2f} ms, NLL {nll:.6f}",
        flush=True,
    )
    del model, optimizer, loss, x, y, valid, norm
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run():
    assert not (ROOT / "protocol.json").exists(), "Use a new experiment directory to rerun"
    assert json.loads((ROOT / "qualification.json").read_text())["passed"]
    data = TokenData(Path("data/wikitext2_v1"), "cuda", 10017)
    prov = provenance()
    sources = {
        **prov["source_files"],
        **{p.as_posix(): sha256(p) for p in (ROOT / "source").glob("*.py")},
        PLAN.as_posix(): sha256(PLAN),
    }
    with zipfile.ZipFile(ROOT / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, path)
    protocol = dict(
        started_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=sha256(PLAN),
        sources=sources,
        provenance=prov,
        environment=environment(),
        data=data.manifest,
        seeds=[17, 29, 43],
        shapes=[[16, 128], [8, 512]],
        policies=POLICIES,
        widths=WIDTHS,
        chunk_size=512,
        warmup_steps=4,
        timed_steps=8,
    )
    write_json(ROOT / "protocol.json", protocol)
    results = []
    start = time.perf_counter()
    for batch, context in ((16, 128), (8, 512)):
        for seed_index, seed in enumerate((17, 29, 43)):
            policies = POLICIES[seed_index:] + POLICIES[:seed_index]
            for variant in WIDTHS:
                for policy in policies:
                    results.append(run_case(variant, batch, context, seed, policy, data))
    write_json(
        ROOT / "result.json",
        dict(
            status="COMPLETE",
            cases=results,
            elapsed_seconds=time.perf_counter() - start,
            protocol_sha256=sha256(ROOT / "protocol.json"),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("qualify", "run"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        qualify() if args.mode == "qualify" else run()
    except Exception:
        write_json(ROOT / f"{args.mode}_failure.json", {"traceback": traceback.format_exc()})
        raise
