"""Full-scale matched-state gradient diagnosis, separate from training trajectories."""

import json
from pathlib import Path

import torch

from results.token_memory_v1.source.execution import install
from results.token_memory_v1.source.study import relative_error
from src.core.config import ModelConfig
from src.core.data import TokenData
from src.core.reproducibility import write_json
from src.core.transformer import Transformer

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
root = Path("results/token_memory_diagnosis_v1")
data = TokenData(Path("data/wikitext2_v1"), "cuda", 10017)
x, y = data.batch(16, 128)
cfg = ModelConfig(width=384, hidden=1536, layers=8, heads=6, context=128)
records = []
for state_name in ("initial", "gelu_native", "gelu_loss"):
    reference = None
    for policy in ("native", "block", "loss_chunks", "both_chunks"):
        model = Transformer(cfg, 17).cuda()
        if state_name != "initial":
            model.load_state_dict(torch.load(root/state_name/"endpoint.pt", weights_only=True)["model"])
        install(model, policy, 512)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(x, y)
        loss.backward()
        grads = {n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
        if reference is None:
            reference = (loss.item(), grads)
        errors = {n:relative_error(g, reference[1][n]) for n,g in grads.items()}
        row = {"state":state_name,"policy":policy,"loss":loss.item(),
               "max_gradient_relative_l2":max(errors.values()),"gradient_errors":errors}
        print(json.dumps({k:v for k,v in row.items() if k != "gradient_errors"}),flush=True)
        records.append(row)
        del model, loss, grads
write_json(root/"matched_state_fidelity.json", records)
