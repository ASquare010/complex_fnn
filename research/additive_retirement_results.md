# H062: finalize the shortlist after the additive screen

**Cleanup complete.** The active repository contains three model folders and
six registered variants: full/narrow GELU and SwiGLU controls, plain BlockShuffle,
and native rational BlockShuffle. Nine recipes and the shared core remain.
The research performance target is still unmet.

The [H061 additive screen](additive_block_lowrank_screen_results.md) fails every
quality gate, despite passing memory and parameter limits. Its model folder,
two experiment drivers, two test files and one configuration have therefore moved
to [the readable archive](archive/h061_retired/). Seven shared integration files
are restored to their exact, previously verified H059 bytes. No retained model
computation or recipe changes.

Before retirement, [H061's source snapshot](archive/h061_source.zip) preserved
230 verified source, configuration, documentation and verification-script files.
Its [manifest](archive/h061_manifest.json) records each hash and metadata for
3,212 protected result/data files. All 171 language-model/profile runs remain,
including the three new rejected trials. Checkpoints, cached data, histories,
diagnostics, negative outcomes and the scoped construction are preserved.
Readable archived Markdown links are updated; the source ZIP preserves the
original bytes. Historical source assertions must be run against their snapshots.

## Verification

- **89 active tests pass** in 19.09 seconds, with sources unchanged during testing.
- **12 CPU FP32/CUDA BF16 cases match exactly** across all six retained variants:
  initialization, nonzero rational parameters, predictions/losses, gradients,
  optimizer groups, two AdamW updates and layer diagnostics.
- All retained source/configuration hashes match the verified H059 test state.
  The archive contents and protected evidence metadata match their manifests.
- All **49 frozen plans** retain their original bytes. Live documentation links
  resolve; the three previously recorded historical links resolve in H057's
  complete snapshot, as explained by the archive guide.
- The final audit checks active imports, lint, CLI counts, entry-point help and
  strict loading plus a finite synthetic forward from six real saved checkpoints.
  These checks add no corpus training, validation ranking or test scoring.

The signature wrapper had two setup failures: first its output filename collided
with its process-status file; then the isolated script lacked the repository
import path. Both stopped before signature computation, and both logs remain.
The corrected v3 wrapper uses distinct filenames and an explicit source path;
all twelve comparisons then pass. Model code and the already passed test suite
were unchanged. These wrapper failures are separate from H061's pre-training
preflight failure and do not represent repeated language-training cells.

[Test record](../results/verification/additive_retirement_tests_v1.json),
[successful signature process](../results/verification/additive_retirement_process_signatures_v3.json),
[signature values](../results/verification/additive_retirement_signature_values_v3.json),
[move manifest](../results/verification/additive_retirement_moves_v1.json), and
[final audit](../results/verification/additive_retirement_final_v1.json) preserve
this verification. The final audit is the authority for the stated checks.

Plain BlockShuffle remains the best supported compressed reference, with its
longer-training failure disclosed. Native rational BlockShuffle remains only a
memory/fidelity investigation candidate. No automatic new training allocation,
quality promotion or breakthrough claim follows from this cleanup.
