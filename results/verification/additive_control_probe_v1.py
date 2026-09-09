"""Full-size retained-control probe; run with cwd selecting the source snapshot."""

import hashlib
import json
import sys
from pathlib import Path

# An explicit cwd isolates historical src imports from this helper's location.
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import torch

from src.core import trainer
from src.core.config import ModelConfig, TrainConfig
from src.core.diagnostics import gradient_stats, inspect_layers
from src.core.optimization import group_summary, initialize_dense_width, parameter_groups
from src.core.transformer import Transformer


def tensor_sha(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def probe(request_path, output):
    assert not output.exists()
    request = json.loads(request_path.read_text())
    torch.set_num_threads(4)
    tokens = torch.from_numpy(np.load(request["train_path"], mmap_mode="r")[:33].astype(np.int64))
    x, y = tokens[:32].reshape(2, 16), tokens[1:33].reshape(2, 16)
    rows = {}
    for cell in request["controls"]:
        config, training = ModelConfig(**cell["model"]), TrainConfig(**cell["training"])
        model = Transformer(config, training.seed)
        initialize_dense_width(model, training)
        if hasattr(trainer, "configure_gate_recomputation"):
            trainer.configure_gate_recomputation(model, training)
        elif training.recompute_gate:
            assert config.variant == "blockshuffle_swiglu"
            for block in model.blocks:
                block.ffn.recompute_gate = True
                block.ffn.gate_recompute_method = training.gate_recompute_method
        groups = parameter_groups(model, training)
        row = {
            "initial_weights": {n: tensor_sha(p) for n, p in model.named_parameters()},
            "optimizer_groups": group_summary(groups),
            "ffn_parameters": config.unique_ffn_parameters,
            "total_parameters": config.total_parameters,
        }
        optimizer = torch.optim.AdamW(
            groups, lr=training.learning_rate, betas=(0.9, 0.95), eps=1e-8
        )
        logits, loss = model(x), model.loss(x, y)
        loss.backward()
        row.update(
            logits=tensor_sha(logits),
            loss=loss.item(),
            gradients={n: tensor_sha(p.grad) for n, p in model.named_parameters()},
            gradient_stats=gradient_stats(model),
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        row["updated_weights"] = {n: tensor_sha(p) for n, p in model.named_parameters()}
        row["diagnostics"] = inspect_layers(model, x, "cpu", "fp32")
        rows[cell["run"]] = row
        del model, optimizer, groups, logits, loss
    output.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({"source_root": str(Path.cwd()), "controls": len(rows), "status": "PASS"}),
        flush=True,
    )


if __name__ == "__main__":
    probe(Path(sys.argv[1]), Path(sys.argv[2]))
