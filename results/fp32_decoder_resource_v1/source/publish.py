"""Publish H116 navigation from checked, preserved pre-study documents."""

import hashlib
import json
from pathlib import Path

ROOT = Path("results/fp32_decoder_resource_v1")
protocol = json.loads((ROOT / "protocol.json").read_text())
documents = {}
for name, record in protocol["before_documents"].items():
    value = Path(name).read_bytes()
    assert hashlib.sha256(value).hexdigest() == record["sha256"], name
    assert value == Path(record["preserved_path"]).read_bytes(), name
    documents[name] = value.decode("utf-8")


def prepend(content, addition):
    title, rest = content.split("\n", 1)
    return title + "\n\n" + addition + rest.lstrip("\n")


documents["README.md"] = prepend(
    documents["README.md"].replace(
        "The latest [decoder diagnosis]", "The previous [decoder diagnosis]", 1
    ),
    """The latest [full-job FP32 resource screen](research/fp32_decoder_resource_results.md)
finds a qualified narrow-model execution recipe: **26.97-27.23% lower allocation**,
with update time approximately unchanged versus BF16 native training. Default
attention with FP32 decoder/classifier chunking passes numerical, memory and
short-continuation quality gates on both narrow corpus fixtures. Thirty matched
continuations / 1,500 updates and independent gradient/state/scoring checks
complete. These are correlated single-seed fixtures and 50-update screens;
broader replication and duration are still required. Full GELU/SwiGLU and math
attention fail the memory target. There is no new layer or parameter reduction.

""",
)
documents["research/CURRENT_STATE.md"] = prepend(
    documents["research/CURRENT_STATE.md"].replace("## Latest:", "## Previous:", 1),
    """## Latest: default FP32 chunks pass the narrow-model full-job resource gates

[H116](fp32_decoder_resource_results.md) completes 30 matched continuations /
1,500 updates in 291.67 seconds. Default-attention FP32 decoder/classifier
chunking reduces job allocation from 407.25 to 297.43 MiB on WikiText and
400.22 to 291.25 MiB on TinyStories: **26.97% / 27.23% saved**. Median warmed
update time changes -0.12% / +0.33% versus BF16 native. Same-backend native
FP32 controls show 26.81% / 27.23% saving with 16.03% / 17.70% slower updates.

All original gates pass for these two narrow scopes. Full gradient error is
below 9.354e-7. The independent audit passes all 24 FP32 gradient replays,
36 native scores and 1,500 batch/state checks. Default-chunk NLL changes stay
within -0.00159% to +0.00170% versus BF16 after 50 updates. This does not prove
long-run quality: the four narrow states are correlated seed61 pairs, with
single-seed full controls. No architectural or parameter-count change occurs.

Forced math attention fails memory and BF16-relative time on both narrow
corpora. Full GELU/SwiGLU fail the 15% job-memory gate under either backend.
Retain only default FP32 chunks on narrow context512 for broader replication
and duration testing. No fresh training grid or default change occurs here.

CPU analysis initially hits a Windows access violation. Its failure is retained;
a fresh process runs the unchanged analyzer successfully, with no scientific
retry. The 61 maintained files remain unchanged from the earlier 116-test pass.
The broad goal stays open. [Plan](fp32_decoder_resource_plan.md),
[receipt](../results/verification/fp32_decoder_resource_final_v1.json).

""",
)
documents["research/PROGRESS_OVERVIEW.md"] = prepend(
    documents["research/PROGRESS_OVERVIEW.md"].replace(
        "## Newest result:", "## Previous result:", 1
    ),
    """## Newest result: approximately 27% lower allocation at comparable update time

[H116](fp32_decoder_resource_results.md) turns the prior gradient diagnosis into
a passing resource screen for narrow context-512 models. Default FP32 chunks
save 26.97% / 27.23% job allocation on WikiText/TinyStories, with -0.12% / +0.33%
median update-time changes versus BF16 native. Same-precision native controls
also pass the memory/time comparison. Every numerical and short-NLL gate passes.

The 30 cases contain 1,500 updates. Independent verification covers 24 FP32
backward replays, 36 native scores, all batches and endpoint model/Adam states.
Math attention and full GELU/SwiGLU fail the memory target and are eliminated
from this screen. Four narrow fixtures are correlated seed61 pairs, so the
next earned work is broader replication and duration, not a general quality
claim. This establishes a promising execution component with unchanged
parameter count; the full research objective remains unmet.

""",
)
documents["research/idea_bank.md"] += """

| H116 | Can FP32 decoder fidelity coexist with lower whole-job VRAM and practical runtime? | PROMISING COMPONENT: default FP32 chunks pass both narrow T512 scopes with 26.97-27.23% lower allocation and comparable BF16-relative time; 24 gradient replays/36 native scores/all 1,500 batches verify. Math and full T128 settings are ELIMINATED from this resource gate. Correlated single-seed, 50-update scope; broader replication/duration remains required. |
"""
documents["research/ARTIFACTS.md"] += """

## H116 FP32 decoder resource screen

[Plan](fp32_decoder_resource_plan.md), [report](fp32_decoder_resource_results.md),
[source guide](../results/fp32_decoder_resource_v1/source/README.md) and
[receipt](../results/verification/fp32_decoder_resource_final_v1.json) cover
30 continuations / 1,500 updates and independent gradient/scoring/state checks.
Compact result/audit JSON, logs and tables retain every case/update. The
original CPU analyzer access violation and unchanged-code recovery are preserved.

Sixty tensor artifacts, per-case metrics/histories, raw aggregate JSON/logs and
pre-study document snapshots stay local and ignored. All 115 prospective sources
and 61 maintained files are preserved. The receipt records 1,578 backwards
including qualification/audit, 36 native scores and 3,210 memory intervals.
Neither a new training grid nor a broad research success is claimed.
"""
documents["research/literature.md"] += """

## H116: precision and classifier memory as a measured joint recipe

[PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html)
supports synchronized timing and distinct allocated/reserved accounting.
[SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
documents backend choices and numerical differences. [Cut Your Losses](https://arxiv.org/abs/2411.09009)
is established classifier-memory precedent; chunking and FP32 are not novel here.

[H116](fp32_decoder_resource_results.md) measures default FP32 chunks at about
27% lower full-job allocation and comparable BF16-relative update time on narrow
T512 fixtures. Same-precision native controls isolate the chunking cost. Forced
math attention fails the practical memory/time gate, and full T128 controls
fail memory. Twenty-four independent FP32 backward replays pass; this does not
establish global determinism. The short, correlated fixture scope earns broader
replication/duration only, with no language-quality or architectural novelty claim.
"""
documents[".gitignore"] += """

# H116: complete compact resource evidence; tensors and raw histories stay local.
/results/fp32_decoder_resource_v1/runs/
/results/fp32_decoder_resource_v1/before_documents/
/results/fp32_decoder_resource_v1/result.json
/results/fp32_decoder_resource_v1/audit.json
/results/fp32_decoder_resource_v1/*.log
!/results/fp32_decoder_resource_v1/*.json.gz
!/results/fp32_decoder_resource_v1/*.csv.gz
!/results/fp32_decoder_resource_v1/*.log.gz
!/results/fp32_decoder_resource_v1/*_exit.txt
!/results/fp32_decoder_resource_v1/*_protocol.json
!/results/fp32_decoder_resource_v1/*failure*.json
!/results/fp32_decoder_resource_v1/qualification.json
!/results/fp32_decoder_resource_v1/environment.json
"""
for name, content in documents.items():
    Path(name).write_text(content, encoding="utf-8", newline="\n")
print("Published H116 navigation; all prior document bytes remain preserved")
