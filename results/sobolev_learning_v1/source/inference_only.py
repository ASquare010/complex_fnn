"""Posthoc fresh-process inference profiles; original training-process gates stay fixed."""

import json
from pathlib import Path

import torch

from results.sobolev_learning_v1.source.model import from_checkpoint
from results.sobolev_selection_v1.source.study import profile
from src.core.reproducibility import sha256, write_json
from src.dense_ffn import DenseFFN

ROOT = Path("results/sobolev_learning_v1")


def run():
    result = json.loads((ROOT / "result.json").read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    students, teachers = [], []
    for kind, info in protocol["teachers"].items():
        assert sha256(Path(info["path"])) == info["sha256"]
        weights = torch.load(info["path"], map_location="cpu", weights_only=True)["model"]
        for case in (c for c in result["cases"] if c["teacher"] == kind):
            layer, seed = case["layer"], case["seed"]
            folder = ROOT / "pairs" / f"{kind}_l{layer}_s{seed}"
            data = torch.load(folder / "data.pt", map_location="cpu", weights_only=True)
            x = data["x"][36864:37120].clone()
            del data
            prefix = f"blocks.{layer}.ffn."
            state = {k.removeprefix(prefix): v for k, v in weights.items() if k.startswith(prefix)}
            teacher = DenseFFN(384, state["up.weight"].shape[0], kind)
            teacher.load_state_dict(state)
            teachers.append({"teacher": kind, "layer": layer, "seed": seed, **profile(teacher, x)})
            del teacher
            for row in (
                r
                for r in result["rows"]
                if (r["teacher"], r["layer"], r["seed"]) == (kind, layer, seed)
            ):
                record = row["endpoints"][row["selected_index"]]["checkpoint"]
                assert sha256(Path(record["path"])) == record["sha256"]
                model = from_checkpoint(
                    torch.load(record["path"], map_location="cpu", weights_only=True)
                )
                students.append(
                    {
                        "teacher": kind,
                        "layer": layer,
                        "seed": seed,
                        "method": row["method"],
                        "checkpoint": record,
                        **profile(model, x),
                    }
                )
                del model
            print(f"Profiled {kind} layer {layer} seed {seed}", flush=True)
    write_json(
        ROOT / "inference_only.json",
        {
            "posthoc": True,
            "original_gates_unchanged": True,
            "gradients_or_optimizers_in_process": False,
            "neural_updates": 0,
            "students": students,
            "full_teachers": teachers,
        },
    )


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run()
