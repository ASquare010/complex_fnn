"""CPU-only seed-17 duration diagnosis; no training, scoring or architecture change."""
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from src.blockshuffle_ffn import BlockShuffleLinear
from src.core.config import ModelConfig, TrainConfig
from src.core.optimization import group_summary, initialize_dense_width, parameter_groups
from src.core.trainer import learning_rate
from src.core.transformer import Transformer

torch.set_num_threads(4)
root = Path("results/duration_optimizer_geometry_v1")
root.mkdir(exist_ok=False)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

@torch.no_grad()
def map_norm(module):
    if isinstance(module, BlockShuffleLinear):
        # Offline CPU materialization only; no dense matrix enters any model run.
        matrix = module.double()(torch.eye(module.input_width, dtype=torch.float64))
        return matrix.norm().item()
    return module.weight.double().norm().item()

old = json.loads(Path("results/long_duration_v1/preflight.json").read_text())
long = json.loads(Path("results/long_duration_v1/result.json").read_text())
records = []
for trial in long["trials"]:
    recipe = trial["recipe"]
    for steps, record in ((800, old["references"][recipe]), (3200, trial)):
        path = Path("results/runs") / record["run"]
        assert sha(path / "metrics.json") == record["metrics_sha256"]
        assert sha(path / "checkpoint.pt") == record["checkpoint_sha256"]
        m = json.loads((path / "metrics.json").read_text())
        mc, tc = ModelConfig(**m["model"]), TrainConfig(**m["training"])
        assert tc.steps == steps and tc.seed == 17 and tc.learning_rate == .0012
        model = Transformer(mc, 17)
        initialize_dense_width(model, tc)
        groups = parameter_groups(model, tc)
        assert group_summary(groups) == m["optimizer_parameter_groups"]
        group_by_id = {id(p): (g["lr_scale"], g["weight_decay"]) for g in groups for p in g["params"]}
        rates = [learning_rate(i, tc) for i in range(steps)]
        projections = []
        for layer in (0, 7):
            for name in ("up", "gate", "down"):
                module = getattr(model.blocks[layer].ffn, name, None)
                if module is None:
                    continue
                treatment = [group_by_id[id(p)] for p in module.parameters()]
                log_multiplier = sum(math.log1p(-eta * scale * decay) for eta in rates for scale, decay in treatment)
                projections.append({"layer": layer, "projection": name, "factor_treatments": treatment,
                    "decay_only_map_multiplier": math.exp(log_multiplier),
                    "initial_map_frobenius_norm": map_norm(module)})
        checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        for entry in projections:
            module = getattr(model.blocks[entry["layer"]].ffn, entry["projection"])
            entry["trained_map_frobenius_norm"] = map_norm(module)
            entry["trained_to_initial_norm_ratio"] = entry["trained_map_frobenius_norm"] / entry["initial_map_frobenius_norm"]
        diag = json.loads((path / "final_diagnostics.json").read_text())
        records.append({"recipe": recipe, "steps": steps, "metrics_sha256": sha(path / "metrics.json"),
            "checkpoint_sha256": sha(path / "checkpoint.pt"), "sum_global_lr": sum(rates),
            "clipped_step_fraction": m["clipped_step_fraction"], "validation_loss": m["validation_loss"],
            "layers": {str(l): diag[f"layer_{l}.ffn"] for l in (0, 7)}, "projections": projections})
        del model, checkpoint
record = {"status": "PASS", "seed": 17, "layers": [0, 7], "records": records,
    "optimizer_updates": 0, "validation_targets_scored": 0, "compute_device": "cpu",
    "scope": "Decay-only scalar products assume zero gradients and Adam moments. Trained Frobenius norms are separate observed checkpoint properties, not spectral norms or causal explanations. Initializers and actual optimizer metadata checked. Dense map materialization used only offline on CPU.",
    "script_sha256": sha(Path(__file__))}
(root / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
for r in records:
    down = next(p for p in r["projections"] if p["layer"] == 0 and p["projection"] == "down")
    print(json.dumps({"recipe": r["recipe"], "steps": r["steps"], "sum_lr": r["sum_global_lr"], "down_decay_only": down["decay_only_map_multiplier"], "down_actual_norm_ratio": down["trained_to_initial_norm_ratio"]}), flush=True)
