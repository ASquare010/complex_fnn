"""Independent checkpoint/score and batch-order audit for H102 and H101 diagnosis."""

import gc
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from src.core.config import ModelConfig
from src.core.data import TokenData
from src.core.reproducibility import sha256, write_json
from src.core.transformer import Transformer


def tensor_hash(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


if __name__ == "__main__":
    root = Path("results/token_memory_replication_v1")
    assert json.loads((root/"completion.json").read_text())["status"] == "COMPLETE"
    original = json.loads(Path("results/token_memory_v1/protocol.json").read_text())
    for name, digest in original["sources"].items():
        assert sha256(Path(name)) == digest, name
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = TokenData(Path("data/wikitext2_v1"), "cuda", 10017)
    paths = sorted(root.glob("*/cases/*/metrics.json"))
    paths += sorted(Path("results/token_memory_diagnosis_v1").glob("*/cases/*/metrics.json"))
    records = []
    for path in paths:
        metrics = json.loads(path.read_text())
        cfg = ModelConfig(**metrics["config"])
        endpoint = path.parents[2]/"endpoint.pt"
        state = torch.load(endpoint, map_location="cpu", weights_only=True)["model"]
        assert {n:tensor_hash(p) for n,p in state.items()} == metrics["state_hashes"]
        assert all(bool(torch.isfinite(p).all()) for p in state.values())
        assert sum(p.numel() for p in state.values()) == metrics["parameter_count"]
        data.generator.manual_seed(metrics["seed"]+10000)
        order = hashlib.sha256()
        for _ in range(12):
            x, _ = data.batch(metrics["batch"], metrics["context"])
            order.update(tensor_hash(x.cpu()).encode())
        assert order.hexdigest() == metrics["data_order_sha256"]
        model = Transformer(cfg, metrics["seed"]).cuda()
        model.load_state_dict(state, strict=True)
        model.eval()
        weighted, targets = 0.0, 0
        with torch.inference_mode():
            for x, y in data.validation(metrics["batch"], metrics["context"], 8):
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    logits = model(x)
                loss = F.cross_entropy(logits.float().reshape(-1,cfg.vocab_size), y.reshape(-1))
                weighted += loss.item()*y.numel()
                targets += y.numel()
        nll = weighted/targets
        assert targets == metrics["validation_targets"]
        assert nll == metrics["validation_nll"], (metrics["key"],nll,metrics["validation_nll"])
        assert len(metrics["records"]) == 12 and metrics["training_tokens"] == 12*metrics["batch"]*metrics["context"]
        records.append({"key":metrics["key"],"checkpoint_sha256":sha256(endpoint),
                        "metrics_sha256":sha256(path),"nll":nll,"exact_score_match":True,
                        "parameter_hashes_match":True,"data_order_match":True})
        print(f"Audited {metrics['key']}",flush=True)
        del model, state, logits, loss, x, y
        gc.collect()
        torch.cuda.empty_cache()
    write_json(root/"audit.json",{"passed":True,"cases":records,"source_hashes_match":True})
