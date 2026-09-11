"""Publish H115 navigation only after checking the frozen pre-study documents."""

import hashlib
import json
from pathlib import Path

ROOT = Path("results/decoder_gradient_transport_v1")
protocol = json.loads((ROOT / "protocol.json").read_text())
documents = {}
for name, record in protocol["before_documents"].items():
    data = Path(name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == record["sha256"], name
    assert data == Path(record["preserved_path"]).read_bytes(), name
    documents[name] = data.decode("utf-8")

readme = """The latest [decoder diagnosis](research/decoder_gradient_transport_results.md)
separates repeat variability from gradient transport error. Thirty fixed-state
conditions and 320 backward passes, with **zero optimizer updates**, show that
BF16 can turn classifier-gradient differences around 1e-6 into decoder-gradient
differences around 0.003. Removing checkpointing does not solve this. Both
FP32 decoder modes pass the unchanged numerical gates and earn a separate
whole-model memory/runtime screen. No new language training is earned yet.

"""
state = """## Latest: FP32 decoder modes pass the fixed numerical diagnosis

[H115](decoder_gradient_transport_results.md) completes 30 conditions on six
saved fixtures: correlated WikiText/TinyStories seed61 pairs and full GELU/SwiGLU
seed17 controls. This is 300 main plus 20 qualification backwards, zero optimizer
updates, with 189.85 seconds for the main study. No new model is introduced.

BF16/default's median paired total-gradient error is 0.003181. Disabling
checkpointing does not help. BF16/math removes the observed repeat variability
but still gives median paired error 0.003030. Both FP32 modes reduce paired
error below 9.354e-7 in every fixture/repetition and pass the original transport
and numerical replay gates. FP32/default still has tiny bitwise replay changes;
neither mode establishes global determinism. An exact CPU/CUDA scalar witness
demonstrates 32,768-fold perturbation amplification at a BF16 rounding boundary.

All same-mode forward boundaries match. Precision changes forward values too:
FP32 normalized hidden states differ from BF16 by 0.249-0.384% relative L2.
The independent audit verifies 30 native forward captures and saved gradient
arithmetic, with zero backward replays. H114's failed audits remain failures.

Retain FP32/default/block and FP32/math/block for an **uninstrumented whole-model
resource screen** only. Diagnostic memory excludes optimizer state and hooks
affect timing. No practical VRAM win, fresh LM allocation or default change is
earned here. The broad goal remains open. All 61 maintained files remain
unchanged from the earlier 116-test pass; that suite was not rerun.
[Plan](decoder_gradient_transport_plan.md),
[scoped receipt](../results/verification/decoder_gradient_transport_final_v1.json).

"""
progress = """## Newest result: decoder precision passes diagnosis; resource test is next

[H115](decoder_gradient_transport_results.md) separates two numerical effects.
On fixed incoming gradients, default BF16 decoder repeats sometimes vary.
Forced math attention removes the observed repeat variability but leaves
roughly 0.003 paired gradient error from roughly 1e-6 classifier differences.
Removing checkpointing does not solve the problem. Both FP32 decoder modes
pass unchanged transport/replay thresholds across all 30 diagnostic conditions.

An elementary rounding witness produces exactly 32,768-fold perturbation
amplification in CPU/CUDA BF16 linear backward. This explains a possible
mechanism, without claiming a new theorem or isolating every H114 kernel cause.
The study uses 320 backwards including qualification, **zero optimizer updates**,
and earns only a separate uninstrumented whole-job memory/runtime screen.
All 30 native hidden-state recaptures and stored first-gradient arithmetic verify.
Actual lower VRAM with preserved quality remains the primary unmet objective.

"""


def prepend_after_title(content, block):
    title, rest = content.split("\n", 1)
    return title + "\n\n" + block + rest.lstrip("\n")


documents["README.md"] = prepend_after_title(
    documents["README.md"].replace("The latest [complete-model", "The previous [complete-model", 1),
    readme,
)
documents["research/CURRENT_STATE.md"] = prepend_after_title(
    documents["research/CURRENT_STATE.md"].replace("## Latest:", "## Previous:", 1), state
)
documents["research/PROGRESS_OVERVIEW.md"] = prepend_after_title(
    documents["research/PROGRESS_OVERVIEW.md"]
    .replace("## Newest result:", "## Previous result:", 1)
    .replace("## Current priority and latest result", "## Previous classifier diagnosis", 1),
    progress,
)
documents["research/idea_bank.md"] += """

| H115 | Do checkpointing, attention backend or decoder precision explain transport/replay failures? | DIAGNOSIS COMPLETE: no-checkpoint BF16 fails; math BF16 repeats exactly but transport fails; both FP32 modes pass numerical gates. Exact scalar rounding witness and independent forward/stored-arithmetic audit pass. Only an uninstrumented resource screen is earned; 320 backwards, zero updates, no broad goal or novelty claim. |
"""
documents["research/ARTIFACTS.md"] += """

## H115 decoder gradient transport

[Report](decoder_gradient_transport_results.md),
[plan](decoder_gradient_transport_plan.md) and
[source guide](../results/decoder_gradient_transport_v1/source/README.md) explain
the 30-condition fixed-state diagnosis. Compact result/audit JSON and logs are
losslessly compressed. Four tables retain 30 conditions, 120 pairs, 180
nontrivial replays and 1,200 gradient boundary comparisons. Protocols,
qualification, environment, summary, figure and exit records are public.

Sixty tensor artifacts and per-condition raw metrics in `conditions/`, raw
aggregate JSON/logs, and pre-study document snapshots stay local and ignored.
The [receipt](../results/verification/decoder_gradient_transport_final_v1.json)
verifies 109 frozen sources, 61 maintained files, 18 source inputs, the tensor
hashes, all arithmetic exports and the original gates. The independent audit
checks 30 native forwards and saved gradient arithmetic with zero backwards;
it does not independently replay the entire gradient experiment. Prior H114
failures and evidence remain preserved. No optimizer updates or new language
training occur in H115.
"""
documents["research/literature.md"] += """

## H115: finite-precision decoder transport and SDPA controls

[PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
documents backend-dependent numerical behavior and FP32 intermediates in math
attention for BF16 inputs. [Numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)
and [reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)
describe floating-point and reproducibility limits. These are general facts,
not evidence identifying a particular operator in the current failed replays.

[H115](decoder_gradient_transport_results.md) distinguishes repeat variability
from paired perturbation transport by holding classifier dH fixed across decoder
replays. Default attention exposes EfficientAttentionBackward, not a measured
FlashAttention node. Math BF16 removes observed repeat variability but still
amplifies paired differences; both FP32 modes pass the numerical gates. Forward
values change under precision/backend interventions. The exact 32,768-fold
scalar rounding witness is elementary arithmetic, not a new theorem or a
condition-number measurement. Chunked classifier memory has established prior
work, including [Cut Your Losses](https://arxiv.org/abs/2411.09009). H115 makes
no novelty or training-quality claim and earns only resource screening.
"""
documents[".gitignore"] += """

# H115: compact numerical diagnosis; raw tensors and snapshots remain local.
/results/decoder_gradient_transport_v1/conditions/
/results/decoder_gradient_transport_v1/before_documents/
/results/decoder_gradient_transport_v1/result.json
/results/decoder_gradient_transport_v1/audit.json
/results/decoder_gradient_transport_v1/*.log
!/results/decoder_gradient_transport_v1/*.json.gz
!/results/decoder_gradient_transport_v1/*.csv.gz
!/results/decoder_gradient_transport_v1/*.log.gz
!/results/decoder_gradient_transport_v1/*_exit.txt
!/results/decoder_gradient_transport_v1/*_protocol.json
!/results/decoder_gradient_transport_v1/qualification.json
!/results/decoder_gradient_transport_v1/environment.json
"""
for name, content in documents.items():
    Path(name).write_text(content, encoding="utf-8", newline="\n")
print("Updated six navigation documents and .gitignore; preserved all before-documents")
