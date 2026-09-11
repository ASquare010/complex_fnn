"""Describe stored native/candidate gradient differences; no new GPU work."""

import json
from pathlib import Path

import torch

from results.checkpoint_input_offload_v1.source.prepare import hashes, read

ROOT = Path("results/native_buffer_layout_v1")
torch.set_num_threads(4)
assert not torch.cuda.is_initialized()
p = read(ROOT / "pair_protocol.json")
hashes(p["files"])
r = read(ROOT / "result.json")
rows = []
for dataset in ("wikitext2", "tinystories"):
    cases = {
        x["arm"]: x["measurement"] for x in r["cases"] if x["measurement"]["dataset"] == dataset
    }
    tensors = {
        arm: torch.load(c["gradient"]["path"], map_location="cpu", weights_only=True)
        for arm, c in cases.items()
    }
    layers = {}
    sqdiff = sqref = 0.0
    for key, base in tensors["native"].items():
        v = tensors["reuse"][key]
        delta = v.double() - base.double()
        dn = delta.square().sum().item()
        rn = base.double().square().sum().item()
        sqdiff += dn
        sqref += rn
        layers[key] = dict(
            bitwise_equal=torch.equal(v, base),
            relative_l2=(dn / max(rn, 1e-60)) ** 0.5,
            max_abs=delta.abs().max().item(),
            finite=bool(torch.isfinite(v).all()),
        )
    rows.append(
        dict(
            dataset=dataset,
            global_relative_l2=(sqdiff / sqref) ** 0.5,
            exact_tensors=sum(x["bitwise_equal"] for x in layers.values()),
            total_tensors=len(layers),
            layers=layers,
        )
    )
assert not torch.cuda.is_initialized()
output = dict(
    pairs=rows,
    backwards=0,
    training_updates=0,
    scope="post-hoc stored-pair description; does not alter H128 gate",
)
(ROOT / "pair_analysis.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
print(json.dumps([{k: v for k, v in row.items() if k != "layers"} for row in rows]))
