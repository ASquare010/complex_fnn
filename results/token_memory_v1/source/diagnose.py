"""Fresh-process replay of the exact H101 case loop; retain anomalous endpoints."""

import argparse
from pathlib import Path

import torch

from results.token_memory_v1.source import study
from src.core.data import TokenData
from src.core.reproducibility import write_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("policy")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--variant", default="gelu")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    study.ROOT = Path(args.root)
    study.ROOT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = TokenData(Path("data/wikitext2_v1"), "cuda", 10017)
    captured = []
    factory = study.Transformer

    def capture(*args, **kwargs):
        model = factory(*args, **kwargs)
        captured.append(model)
        return model

    study.Transformer = capture
    result = study.run_case(args.variant, 16 if args.context == 128 else 8,
                            args.context, args.seed, args.policy, data)
    model = captured[0]
    torch.save({"model": {n:p.detach().cpu() for n,p in model.state_dict().items()}},
               study.ROOT/"endpoint.pt")
    write_json(study.ROOT/"endpoint_stats.json", {n:{"norm":p.norm().item(),
               "max_abs":p.abs().max().item(), "grad_norm":p.grad.norm().item()}
               for n,p in model.named_parameters()})
