"""H112 explicitly adapts only H110's corpus, output root and evaluation callable."""

import json
import sys
from pathlib import Path

import sympy
import torch
import torch._dynamo

from results.token_memory_duration_v1.source import worker as original
from results.token_memory_duration_v1.source.common import qualify
from results.whole_job_memory_v1.source.adapters import EvaluationAdapter
from src.core.data import TokenData
from src.core.reproducibility import write_json

ROOT = Path("results/whole_job_memory_v1")
assert not torch.cuda.is_initialized()
print(
    f"CPU preload: Python={sys.version.split()[0]} torch={torch.__version__} sympy={sympy.__version__}",
    flush=True,
)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if sys.argv[1] == "qualify":
    checks = qualify()
    assert len(checks) == 8
    write_json(ROOT / "qualification.json", {"passed": True, "checks": checks})
else:
    dataset, seed, policy = sys.argv[1:]
    protocol = json.loads((ROOT / "protocol.json").read_text())
    cache = Path(protocol["datasets"][dataset]["path"])
    evaluation_policy = protocol["evaluation_policy"]

    def corpus_adapter(path, device, sample_seed):
        assert path == Path("data/wikitext2_v1"), "Unexpected upstream corpus reference"
        return TokenData(cache, device, sample_seed)

    original.ROOT = ROOT / dataset
    original.TokenData = corpus_adapter
    directory = original.ROOT / "runs" / f"b8_t512_s{seed}_{policy}"
    evaluation_adapter = EvaluationAdapter(evaluation_policy, int(seed), directory)
    original.evaluate = evaluation_adapter
    original.run(8, 512, int(seed), policy)
    assert evaluation_adapter.calls == 6 and evaluation_adapter.extra_checkpoint is not None
    write_json(
        directory / "adapter.json",
        {
            "evaluation_policy": evaluation_policy,
            "dataset": dataset,
            "extra_checkpoints": [evaluation_adapter.extra_checkpoint],
            "evaluation_calls": 6,
        },
    )
