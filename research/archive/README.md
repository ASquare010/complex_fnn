# Archived research

The active shortlist is in [CURRENT_STATE](../CURRENT_STATE.md). Discarded model
code has been removed from `src`, and obsolete drivers, tests and recipes have
been removed from their active directories. Historical results remain available;
archiving a model does not erase its evidence or invalidate a scoped proof.

## Recoverable snapshots

- [Complete H057 source snapshot](h057_source.zip): all pre-cleanup source, tests,
  configurations, research documents, project files and verification scripts.
- [Verified manifest](h057_manifest.json): SHA256 for all 526 archived files and
  metadata for 2,178 protected result/data files.
- [Retired files](retired/): readable copies of removed folders, drivers, tests and
  configurations. This is a partial historical tree; use the ZIP for execution.

The ZIP is 1,715,596 bytes with SHA256
`575d29dd0882cddd96e01626bca752457982a5d14bf6cc9ec243f477e2fa1f75`.
Every entry was read back and hash-checked before retirement. Existing checkpoints
and datasets were not copied or deleted. Each original run also retains its own
`source.zip`, which is the appropriate source for reproducing that particular run.

## What was retired

| Branch | Reason to leave the active shortlist |
|---|---|
| Bezier / quadratic FFNs and shifted activation banks | No earned language-quality promotion; repeated-coordinate curves also collapse algebraically |
| Grouped-only, fixed/adaptive coupling | Failed controlled quality screens; the reusable block operator moved to `src/core/structured_linear.py` |
| Shared and affine-shared FFNs | No qualifying result supporting continued allocation over the retained reference |
| Paired-feature variants | Failed to establish a useful quality advantage; duplicate algebra retained as history |
| Parallel, square-headwise and overcomplete FFNs | Quality gates failed; some also failed memory; H056 rules out further tuning of the tested overcomplete recipe |
| Narrow learned activations and affine BlockShuffle | Failed promotion; affine's replicated matched-rate gain was only 0.101%, with 2/3 wins |
| Two rational compiler backends | Full training repeats failed fidelity despite local numerical/memory checks |
| ReLU/SiLU-only registry entries | Redundant legacy controls; full/narrow GELU and SwiGLU remain active |

At H059, native rational BlockShuffle remained a secondary memory-limited candidate.
It is now retired by H087, below. Its historical implementation moved to `src/rational_blockshuffle_ffn`; all other learned
activation implementations are historical. Plain BlockShuffle remains the main
compressed reference, with its longer-training failure clearly disclosed.

## Reproduce historical work

Run from the repository root and extract into a fresh destination:

```powershell
uv run --extra compile --extra data python -m zipfile -e research/archive/h057_source.zip results/reproductions/h057
```

Use that extracted tree as the working directory when reading the old code/tests.
Its dependencies are specified in its `pyproject.toml` and `uv.lock`. Run-specific
source archives may predate H057 and should be used for exact historical trials.
Pass the original data location to commands that accept `--cache`. Drivers with
fixed relative data/results paths need the corresponding retained artifacts in
those locations inside a separate reproduction workspace; they are not bundled
in this small source ZIP. Do not overwrite completed result directories.

Frozen plans retain their exact original bytes, variant names, source paths and
commands. In the reduced live tree, three old Markdown links inside frozen plans
refer to removed paths: the retired rational compiler configuration and the old
learnable-activation model notes (referenced by two plans). They resolve in the
complete snapshot. Readable live counterparts are
[the old compiler config](retired/configs/wikitext2_blockshuffle_rational_compiled.json)
and [the old activation notes](retired/src/learnable_activation_ffn/model.md).
Source-hash assertions in old workers deliberately fail against today's changed
registry; use the recorded source instead of weakening those assertions.

The archive is excluded from active Ruff discovery and pytest uses `tests/`.
Historical ledger labels such as PROMISING describe the evidence at that time;
[current selection](../CURRENT_STATE.md) controls future allocation.

## H061 additive block/low-rank retirement

The [H061 screen](../additive_block_lowrank_screen_results.md) rejects the tested
additive recipe: selected NLL 6.032030 is 2.105% above full SwiGLU and 2.023% above
narrow, despite passing parameter and memory limits. The active registry returns
to six variants; no automatic repair or longer training is earned.

- [H061 source snapshot](h061_source.zip): 230 verified source, configuration,
  documentation and verification-script files, including the qualified candidate.
- [H061 manifest](h061_manifest.json): per-file SHA256 and protected evidence metadata.
- [Readable retired candidate](h061_retired/src/additive_block_lowrank_ffn/README.md),
  [mathematical notes](h061_retired/src/additive_block_lowrank_ffn/model.md),
  and [drivers/tests/recipe](h061_retired/).
- [Executed screen source](../../results/additive_screen_v2/source.zip) and
  [qualification source](../../results/additive_qualification_v1/source.zip)
  preserve the exact states for those runs. All checkpoints remain in results.

H061 source ZIP SHA256:
`8c4dd6bd2a5e3cde80d26c3e6966687505a3c43a93878fb2792ef9eb10e08401`.
Readable retired Markdown links are adjusted for their new location; the ZIP
retains pre-retirement bytes. Extract the complete snapshot into a fresh workspace
for execution, following the reproduction guidance above. Frozen plan references
to original source paths remain meaningful in the snapshot.

## H087 rational retirement and compact source release

Native rational BlockShuffle is no longer active. Its initial one-seed loss signal
did not establish a reliable learned-shape benefit, and the original, native
recompute and staged memory variants failed resource qualification.

- [Pre-retirement source snapshot](h086_source.zip):72 source/config/document files,
  178,731 bytes, independently read back before retirement.
- [Snapshot manifest](h086_manifest.json): source hashes and the H086 partial-audit anchor.
- [Readable retired model](h087_retired/src/rational_blockshuffle_ffn/README.md),
  [dedicated tests](h087_retired/tests/test_learnable_activation.py),
  [memory driver](h087_retired/src/core/rational_memory_audit.py) and
  [recipe](h087_retired/configs/wikitext2_blockshuffle_rational_screen.json).

The archive SHA256 is4a25939854f5d21fd65b93da0670cab9cdb0733b1b525da49892f6a0cca30b55.
Use that snapshot for the former active source paths. The readable retired tree
is intentionally historical and does not participate in active tests or imports.
No data, checkpoint or completed result was deleted. Large raw artifacts remain
local under the [artifact policy](../ARTIFACTS.md).
