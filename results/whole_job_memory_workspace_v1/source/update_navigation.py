"""Publish concise navigation after completed native audit; preserve prior documents."""

import hashlib
import json
from pathlib import Path

ROOT = Path("results/whole_job_memory_workspace_v1")


def run():
    assert json.loads((ROOT / "summary.json").read_text())["completed_trials"] == 12
    assert json.loads((ROOT / "audit.json").read_text())["passed"]
    before = ROOT / "before_documents"
    before.mkdir(exist_ok=True)
    names = [
        "README.md",
        "research/CURRENT_STATE.md",
        "research/PROGRESS_OVERVIEW.md",
        "research/idea_bank.md",
        "research/ARTIFACTS.md",
        "research/literature.md",
    ]
    hashes = {}
    for name in names:
        path = Path(name)
        saved = before / name.replace("/", "__")
        if saved.exists():
            expected = json.loads((ROOT / "before_documents.json").read_text())[name]
            assert hashlib.sha256(saved.read_bytes()).hexdigest() == expected
            if name == "README.md" and path.stat().st_size == 0:
                # Recover the one interrupted cp1252 write from its verified original.
                path.write_bytes(saved.read_bytes())
            assert path.read_bytes() == saved.read_bytes(), name
        data = path.read_bytes()
        if not saved.exists():
            saved.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    (ROOT / "before_documents.json").write_text(json.dumps(hashes, indent=2) + "\n")

    path = Path("README.md")
    old = path.read_text(encoding="utf-8")
    start, end = old.index("The latest [800-update memory study]"), old.index("## Start here")
    replacement = """The latest [two-corpus memory study](research/whole_job_memory_results.md)
completes **12 fresh trials / 9,600 updates**. Chunked training with the same
streamed evaluator for both arms reduces measured job tensor allocation:

| Corpus, three seeds | Reference → candidate peak | Quality and runtime | Decision |
|---|---:|---|---|
| TinyStories | 402.8 → 269.6 MiB (**33.06% saved**) | NLL changes -0.51% to +0.32%; updates 13.8-14.5% slower | Scoped component qualifies |
| WikiText-2 | 408.6 → 276.1 MiB (**32.43% saved**) | Two seeds exceed the 1% NLL allowance | Fixed recipe fails |

Independent audit verifies **72 native scores, 48 checkpoints and every sampled
batch**. These are PyTorch tensor allocations including training, diagnostics,
validation and corpus caches; driver/context memory is excluded. All trials use
the same narrow GELU architecture and parameter count. The two-corpus claim is
rejected, while the TinyStories memory component is retained. This is established
execution technology, with no new activation or general superiority claim.

The preceding [evaluation study](research/streamed_evaluation_results.md) qualifies
classifier streaming on all 24 saved states: 16-31% less evaluation allocation,
approximately 1% median timing overhead and negligible score drift. Processing
fewer complete sequences costs 3.7-7.1 times as much and is rejected at these shapes.
Native runtime failures and bounded recoveries are preserved; their root cause
remains unresolved. Maintained code/defaults stay unchanged from the prior
116-test pass. Both studies use the existing GPU and UV-managed environment.

Other recent results remain fully documented:

- [800-update execution study](research/token_memory_duration_results.md):
  training savings pass at short context, but native evaluation limits job savings.
- [Derivative initialization](research/sobolev_learning_results.md): its initial
  advantage largely disappears under value-only learning; fixed recipes closed.
- [Derivative-aware calibration](research/sobolev_selection_results.md): useful
  local component, while complete compressed-model quality gates fail.
- [Affine residuals](research/affine_residual_fit_results.md),
  [projection/rank limits](research/real_subspace_results.md) and
  [feature discovery](research/spectral_fitting_results.md): audited negative
  results that constrain the next architecture.

"""
    path.write_text(old[:start] + replacement + old[end:], encoding="utf-8")

    path = Path("research/CURRENT_STATE.md")
    old = path.read_text(encoding="utf-8").replace(
        "## Latest: training memory improves; whole-job gains remain limited or unreplicated",
        "## Previous: training memory improves; whole-job gains remain limited or unreplicated",
        1,
    )
    heading, rest = old.split("\n", 1)
    new = """
## Latest: whole-job saving qualifies on TinyStories; WikiText quality fails

H112 completes **12 fresh trials / 9,600 updates** across seeds 61/73/89 on two
corpora. Both training arms get H111's qualified classifier-streaming evaluator.
Measured job allocation falls **32.43% on WikiText-2 and 33.06% on TinyStories**,
with 13.65-14.52% slower median updates. These actual per-trial tensor peaks
include validation, diagnostics and CUDA corpus caches; they exclude driver/
context memory. All training, model and Adam quantities remain finite.

TinyStories passes every seed's <=1% NLL degradation, >=15% allocation saving
and <=25% timing-cost gates. WikiText-2 loses 1.465% / 0.042% / 1.954% NLL and
fails two seeds. **Retain the scoped TinyStories component; reject the fixed
two-corpus recipe.** H110's favorable one-seed long-context observation does
not replicate on these fresh WikiText seeds. Its old fidelity stop stays intact.

Independent audit verifies 72 native scores, 48 checkpoints, twelve exact
initializations and all 9,600 training batches. Maximum streamed/native score
drift is 3.40e-8, so evaluation drift does not explain the quality changes.
The grid runs once in 922.36 seconds, using fresh models in one process with
25 verified zero-allocation boundaries. cuBLAS workspaces are cleared only
between trials and fully charged within trials. Original startup failures,
the workspace diagnosis and postprocessing recoveries remain preserved. Native
failure root cause remains unknown; process-isolation changes are explicit.

H111 separately qualifies classifier streaming at T128/T512 on 24 old states
with zero updates: 16-31% lower evaluation allocation and roughly 1% median
time overhead. Sequence microbatching fails cost at 3.71x / 7.07x. H111 uses
persistent-state fixtures; H112 supplies the separate actual-job measurements.

The next discriminator is a bounded matched-state gradient/precision comparison
before allocating any changed training policy. No causal mechanism has been
proved and no further training grid is allocated by these results. Models,
parameters and defaults remain unchanged: two model folders, five variants,
eight recipes, with the prior 116-test pass kept separate. The broader goal
remains unmet. [H112 report](whole_job_memory_results.md),
[H111 report](streamed_evaluation_results.md),
[final audit](../results/verification/whole_job_memory_final_v1.json).

"""
    path.write_text(heading + "\n" + new + rest.lstrip("\n"), encoding="utf-8")

    path = Path("research/PROGRESS_OVERVIEW.md")
    old = path.read_text(encoding="utf-8")
    start = old.index("## Current priority and latest result")
    end = old.index("## Previous derivative-initializer learning result")
    new = """## Current priority and latest result

**Measured VRAM remains primary; the broader goal is unmet.** H112 completes
12 fresh 800-update trials on WikiText-2 and TinyStories with three seeds each.
Classifier/loss chunking reduces actual measured job tensor allocation by
32.43% / 33.06%, with 13.65-14.52% slower updates. Both arms use the same
memory-aware evaluator and unchanged narrow GELU model.

TinyStories passes every quality/memory/time gate. Its NLL changes range from
-0.514% to +0.316%. WikiText misses the 1% quality allowance in two seeds,
losing 1.465% and 1.954%. The fixed two-corpus recipe is rejected. The useful
result is a scoped TinyStories execution component, with no new architecture,
parameter saving or general superiority claim.

All 72 native scores, 48 saved checkpoints, twelve initializations and 9,600
sampled batches pass independent audit. Training takes 15.37 minutes and is
not repeated. Native startup failures and explicit recoveries are retained;
their root cause remains unknown. Tensor allocations exclude driver/context
memory. Trials use fresh models in one process with 25 verified empty tensor
storage boundaries; library workspaces are fully charged during measurement.
[Complete report and every seed](whole_job_memory_results.md).

H111 first verified the evaluator on 24 existing states with no learning.
Classifier streaming saves 16-31% evaluation allocation at roughly 1% median
time overhead. Sequence microbatching saves slightly more but costs 3.7-7.1x,
so it fails. This resolves H110's evaluation bottleneck without erasing its
earlier fidelity failure. [Evaluation report](streamed_evaluation_results.md),
[earlier duration result](token_memory_duration_results.md).

The maintained code, two model folders, five variants and eight recipes remain
unchanged from the prior 116-test pass. A numerical-gradient diagnosis is the
next justified question, before further language training is allocated.

"""
    path.write_text(old[:start] + new + old[end:], encoding="utf-8")
    print("Updated concise README, current state and progress; originals preserved")


if __name__ == "__main__":
    run()
