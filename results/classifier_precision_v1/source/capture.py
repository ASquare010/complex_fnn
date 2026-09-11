"""Capture actual final-normalized rows from all fixed H112 endpoint states."""

import json
from pathlib import Path

import torch

from results.classifier_precision_v1.source.common import clear, tensor_hash
from src.core.config import ModelConfig
from src.core.data import TokenData
from src.core.reproducibility import sha256
from src.core.transformer import Transformer

ROOT = Path("results/classifier_precision_v1")


def capture_all(protocol):
    old = json.loads(Path("results/whole_job_memory_workspace_v1/result.json").read_text())
    captures = []
    for row in old["cases"]:
        cache = Path(protocol["datasets"][row["dataset"]]["path"])
        for step in (100, 800):
            source = next(c for c in row["checkpoints"] if c["step"] == step)
            assert sha256(Path(source["path"])) == source["sha256"]
            checkpoint = torch.load(source["path"], map_location="cpu", weights_only=True)
            model = Transformer(ModelConfig(**row["model_config"]), row["seed"]).cuda().eval()
            model.load_state_dict(checkpoint["model"], strict=True)
            data = TokenData(cache, "cpu", row["seed"] + 30000)
            tokens, targets = data.batch(8, 512)
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                value = model.embedding(tokens.cuda())
                for block in model.blocks:
                    value = block(value)
                hidden = model.norm(value).cpu()
            assert hidden.dtype == torch.float32
            peak = torch.cuda.max_memory_allocated()
            label = f"{row['dataset']}_s{row['seed']}_{row['policy']}_{step}"
            path = ROOT / "captures" / (label + ".pt")
            assert not path.exists()
            payload = {
                "hidden": hidden,
                "weight": model.embedding.weight.detach().cpu(),
                "tokens": tokens,
                "targets": targets,
            }
            torch.save(payload, path)
            captures.append(
                {
                    "label": label,
                    "dataset": row["dataset"],
                    "seed": row["seed"],
                    "training_policy": row["policy"],
                    "step": step,
                    "source_checkpoint": source,
                    "model_config": row["model_config"],
                    "path": path.as_posix(),
                    "sha256": sha256(path),
                    "tensor_hashes": {k: tensor_hash(v) for k, v in payload.items()},
                    "capture_peak_allocated_bytes": peak,
                }
            )
            del checkpoint, model, block, data, tokens, targets, value, hidden, payload
            clear()
            print("Captured " + label, flush=True)
    assert len(captures) == 24
    return captures
