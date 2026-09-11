"""Use H120's complete-update loop with scoped storage adapters."""

# ruff: noqa: I001
from results.optimizer_memory_v1.source import profile as loop
from results.optimizer_memory_v1.source import audit as audit_helpers
from results.checkpoint_input_offload_v1.source.common import boundary, host_stats
from results.compact_training_v1.adapter import Adapter
from results.checkpoint_input_offload_v1.source.prepare import hashes, read
from src.core.config import ModelConfig
from pathlib import Path
from unittest.mock import patch
import math

ROOT = Path("results/compact_training_v1")
torch = loop.torch


class HostLedger(loop.MemoryLedger):
    def mark(self, phase):
        result = super().mark(phase)
        result["host"] = host_stats()
        torch.cuda.memory.reset_peak_host_memory_stats()
        return result


def qualify_one(combined, expected):
    cfg = ModelConfig(
        variant="gelu", vocab_size=32, width=24, hidden=48, layers=2, heads=3, context=16
    )
    with Adapter(loop, combined, loss_scale=100) as adapter:
        model = loop.Transformer(cfg, 125).cuda()
        loop.set_blocks(model)
        x = torch.arange(32, device="cuda").reshape(2, 16) % 31
        y = x + 1
        loss = loop.training_loss(model, x, y, "fp32_default_chunks")
        loss.backward()
        torch.cuda.synchronize()
        raw = {k: p.grad.detach().cpu() for k, p in model.named_parameters()}
        if combined:
            assert all(
                loop.tensor_hash(v) == loop.tensor_hash(adapter.store.buffers[k])
                for k, v in raw.items()
            )
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True).item()
        assert norm > 1
        clipped = {k: p.grad.detach().cpu() for k, p in model.named_parameters()}
        arrays = audit_helpers.arrays(raw)
        double_norm = math.sqrt(sum(audit_helpers.norm2(v) for v in arrays.values()))
        clip_error = audit_helpers.error(
            audit_helpers.arrays(clipped), {k: v / (double_norm + 1e-6) for k, v in arrays.items()}
        )["distance"]
        assert clip_error <= 1e-6
        model.eval()
        logits = []
        for enabled in (False, True):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
                logits.append(model(x).cpu())
        if expected is not None:
            reference = audit_helpers.arrays(expected["raw"])
            ge = audit_helpers.error(arrays, reference)["distance"]
            te = max(
                audit_helpers.error({k: arrays[k]}, {k: reference[k]})["distance"] for k in arrays
            )
            assert ge <= 1e-5 and te <= 1e-4
            assert all(torch.equal(a, b) for a, b in zip(logits, expected["logits"], strict=True))
        else:
            ge = te = 0.0
        result = dict(
            combined=combined,
            active_norm=norm,
            clip_error=clip_error,
            gradient_global=ge,
            gradient_tensor=te,
            passed=True,
        )
    return result, dict(raw=raw, logits=logits)


def run():
    assert not (ROOT / "result.json").exists()
    p = read(ROOT / "protocol.json")
    for field in ("sources", "input_hashes", "maintained_files"):
        hashes(p[field])
    boundaries = [boundary()]
    q0, expected = qualify_one(False, None)
    boundaries.append(boundary())
    q1, _ = qualify_one(True, expected)
    boundaries.append(boundary())
    loop.write_json(
        ROOT / "qualification.json", dict(rows=[q0, q1], backwards=2, eval_forwards=4, passed=True)
    )
    cases = []
    for i, f in enumerate(p["fixtures"]):
        for arm in ("baseline", "combined") if i == 0 else ("combined", "baseline"):
            with (
                patch.object(loop, "ROOT", ROOT / arm),
                patch.object(loop, "MemoryLedger", HostLedger),
                Adapter(loop, arm == "combined") as adapter,
            ):
                c = loop.run_case(f, "default", p)
                restoration = list(adapter.restorations)
            if arm == "combined":
                assert len(restoration) == 30 and all(
                    v == dict(hooks=50, restored_bytes=36398592) for v in restoration
                )
            cases.append(dict(arm=arm, measurement=c, restorations=restoration))
            boundaries.append(boundary())
    loop.write_json(
        ROOT / "result.json",
        dict(
            cases=cases,
            boundaries=boundaries,
            training_updates=120,
            training_targets=491520,
            backwards=122,
            full_scores=8,
            broad_goal_achieved=False,
        ),
    )


if __name__ == "__main__":
    run()
