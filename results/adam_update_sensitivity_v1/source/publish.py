"""Publish concise navigation while retaining every pre-study document byte."""

from pathlib import Path

from results.adam_update_sensitivity_v1.source.prepare import ROOT, read, sha

p = read(ROOT / "protocol.json")
for path, info in p["before_documents"].items():
    assert sha(path) == info["sha256"] == sha(info["preserved_path"])

intro = """The latest [Adam first-update diagnostic](research/adam_update_sensitivity_results.md)
measures **61–64x amplification** of tiny saved gradient differences, while the
resulting update-direction difference remains about 0.0017%. Larger epsilon
reduces it but changes the direction by 10.8–32.8%, failing the fixed distortion
criterion. All GPU numerical checks pass; the combined audit fails CPU clipping.
Twelve disposable steps use **zero new language training**. This scoped mechanism
does not repair the old quality failure or establish new VRAM savings.

"""
section = """## Latest: first-step sensitivity supports a mechanism, not a training cure

[H119](adam_update_sensitivity_results.md) uses six frozen initial probe gradients
from the selected WikiText seed and 12 disposable native AdamW steps. Across all
nine cross-policy pairs, ideal first-update direction differences are amplified
**61.21–63.51x**; at least **99.888%** of squared difference comes from gradients
within 10 epsilon of zero. The direction difference is still only about 0.0017%.
This does not prove the cause of the later 800-step quality gap.

The independent NumPy calculation verifies all 45 direction pairs, 12 distortions,
15 native pairs and six device comparisons. All six CUDA cases pass the fixed
numerical checks. **The overall numerical audit fails**: CPU clipping differs
from FP64 by up to 6.872e-6, above the fixed 1e-6 bound. All parameter/moment
checks pass. Larger epsilons reduce discrepancy but change directions by
10.79%/32.76%, so both fixed low-distortion remedies are rejected.

The original process completed six CPU steps and then failed its CUDA-init
guard. Its outputs and exit are preserved; a frozen continuation runs only the
remaining six GPU cases through the unchanged function. Total is 12 disposable
optimizer steps, zero language-training updates/forwards/backwards/scores,
24 tensor artifacts and seven zero CUDA allocator boundaries. No tolerance or
case is repeated. All 145 original frozen sources and 61 maintained files remain
intact; recovery sources have their own manifest. The prior 116-test pass is
historical, with no new maintained-suite run.

H117's failed two-corpus gate stays failed, H118 stays inconclusive, and no
optimizer or model default is promoted. Return the memory investigation to
measuring optimizer temporaries versus persistent moments at the complete-update
peak under a separate plan; no new long training sweep is allocated. The broad
VRAM/quality and architectural goals remain open.
[Plan](adam_update_sensitivity_plan.md),
[recovery](adam_update_sensitivity_recovery_plan.md),
[evidence receipt](../results/verification/adam_update_sensitivity_final_v1.json).

"""


def original(path):
    return Path(path).read_bytes().decode("utf-8").replace("\r\n", "\n")


def save(path, value):
    Path(path).write_text(value, encoding="utf-8", newline="\n")


text = original("README.md")
head, rest = text.split("\n\n", 1)
save(
    "README.md",
    head
    + "\n\n"
    + intro
    + rest.replace("The latest [repeat-variability", "The preceding [repeat-variability", 1),
)
for name in ("research/CURRENT_STATE.md", "research/PROGRESS_OVERVIEW.md"):
    head, rest = original(name).split("\n\n", 1)
    save(name, head + "\n\n" + section + rest.replace("## Latest:", "## Previous:", 1))
save(
    "research/idea_bank.md",
    original("research/idea_bank.md")
    + """
| H119 | Does Adam magnify saved initial-gradient differences, and can epsilon reduce that without materially changing the direction? | LOCAL MECHANISM SUPPORTED: all nine cross-policy pairs amplify 61–64x with >99.88% near-zero contribution. GPU checks pass; combined numerical audit FAILS CPU clipping. Both larger-epsilon low-distortion criteria FAIL (10.8–32.8% direction change). Twelve disposable steps, zero LM updates; six completed CPU steps preserved across a guard-failure continuation. No new quality, VRAM or architecture result. |
""",
)
save(
    "research/ARTIFACTS.md",
    original("research/ARTIFACTS.md")
    + """
## H119: saved-gradient first-update diagnosis

[Report](adam_update_sensitivity_results.md), [plan](adam_update_sensitivity_plan.md)
and [recovery plan](adam_update_sensitivity_recovery_plan.md) document 12 disposable
steps with no new language training. Keep the combined CPU-clipping audit failure
separate from the passing evidence-accounting receipt. Source, protocols, compact
45-pair table, JSON/log archives, summary and figure belong in Git. The 24 raw
clipped-gradient/model/optimizer tensor files and navigation snapshots stay local.
The original six CPU results and guard failure remain; recovery adds only six
unexecuted CUDA cases. Frozen hashes are preserved, not changed to accept recovery.
""",
)
save(
    "research/literature.md",
    original("research/literature.md")
    + """
## H119: first-step Adam sensitivity is a known equation, not a new optimizer

[H119](adam_update_sensitivity_results.md) derives g/(|g|+epsilon) at the first
Adam step and its derivative epsilon/(|g|+epsilon)^2 directly from
[Adam](https://arxiv.org/abs/1412.6980) and
[PyTorch AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html).
Those primary sources were checked again on 2026-09-10; installed Adam/AdamW,
clipping and optimizer sources are pinned separately. The measured 61–64x
amplification and failed epsilon-distortion criteria are local observations,
not priority claims or a proof of long-training causality. CPU clipping fails
the preset FP64 tolerance; the six CUDA checks pass. Existing
[numerical-accuracy guidance](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)
does not guarantee cross-device equality. AdamW's documented foreach temporary
storage motivates a separate memory profile, not an assumed saving.
""",
)
save(
    ".gitignore",
    original(".gitignore")
    + """
# H119: keep compact diagnostic evidence, raw tensor/state artifacts stay local.
/results/adam_update_sensitivity_v1/runs/
/results/adam_update_sensitivity_v1/before_documents/
/results/adam_update_sensitivity_v1/result.json
/results/adam_update_sensitivity_v1/analysis.json
/results/adam_update_sensitivity_v1/audit.json
/results/adam_update_sensitivity_v1/*.log
!/results/adam_update_sensitivity_v1/*.json.gz
!/results/adam_update_sensitivity_v1/*.csv.gz
!/results/adam_update_sensitivity_v1/*.log.gz
!/results/adam_update_sensitivity_v1/*_exit.txt
!/results/adam_update_sensitivity_v1/*_protocol.json
!/results/adam_update_sensitivity_v1/environment.json
!/results/adam_update_sensitivity_v1/study_failure.json
""",
)
print("Updated seven navigation documents; original bytes preserved.")
