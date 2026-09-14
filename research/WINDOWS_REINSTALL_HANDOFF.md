# Windows reinstall handoff — 2026-09-14

Research is paused for the requested code checkpoint and Windows reinstall.
No new training was launched for this handoff.

## Preserve before reinstalling

A local Git commit is on the same drive as Windows. It does not protect against
formatting that drive. Copy this repository to an external drive before wiping
Windows, or push the committed code and verify the remote contains the commit.
The configured remote is https://github.com/ASquare010/complex_fnn.git.
This handoff does not claim a remote push or external backup has happened.

Git includes source, configurations, tests, `pyproject.toml`, `uv.lock`, research
reports, figures, checkpoint manifests and compact evidence. Large raw evidence
has lossless gzip copies indexed in [the archive manifest](reinstall_evidence_manifest.json).
Original raw bytes were preserved locally. Git ignores model tensors, datasets,
caches and the Python environment. The small historical source ZIP remains
tracked because it contains source, not model weights.

For a complete recoverable research backup, copy `data/`, `results/` and
`src/experimental/checkpoints/` to an external drive as well as the committed code.
These folders contain ignored data and state that a Git clone cannot restore.
The previous cleanup deliberately removed old unneeded checkpoints; its record is
[here](STORAGE_CLEANUP_2026-09-13.md). Their historical hashes do not imply the
deleted tensors are still available.

If backup space is limited, prioritize:

- `src/experimental/checkpoints/exact_offload_wikitext2_s101_step800.pt`:
  the selected H156 exact-helper checkpoint, indexed by `manifest.json` nearby.
- H163 `final.pt` and `resume.pt` files listed in
  [h163.json](../src/experimental/checkpoints/h163.json). This manifest references
  files under `results/exact_offload_long_scale_v1/`; it does not contain weights.
- Dataset/tokenizer files under `data/` and the retained H156/H160–H164 experiment
  folders if endpoint audits or exact continuation are needed.

The two FP16 checkpoints under `src/experimental/checkpoints/` are historical
diagnostics. H160 failed qualification; they are not recommended models.

## Restore the working environment

Install Git, uv and the NVIDIA driver on the new Windows installation. Clone the
verified remote or restore the externally saved repository. From its root run:

```powershell
uv python install 3.12.9
uv sync --frozen --python 3.12.9
uv run --no-sync python scripts/restore_evidence.py --apply
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
```

The lockfile selects the configured CUDA PyTorch wheels. Optional `compile` and
`data` extras can be restored with `uv sync --frozen --python 3.12.9 --extra compile
--extra data` when needed. Do not copy `.venv` as a portable environment.
Restore the ignored files separately. Confirm checkpoint SHA-256 values against
the manifests before using them. Some frozen research launchers reference the
old absolute Python/workspace paths: inspect them before running on a new machine.
The evidence restoration command verifies both compressed and original hashes,
restores only missing files and refuses to overwrite changed evidence.

## Resume from these findings

- **H163 passed its scoped qualification:** exact helpers reduced allocated GPU
  memory by 18.1% in six 800-update runs of a 22.2M-parameter WikiText2 model.
  Complete-update CUDA time increased by approximately 1.2–5.9%. Maintained helper
  usage is in [training_memory_usage.md](training_memory_usage.md).
- **H160 failed:** FP16 storage missed one quality gate and one timing-stability
  gate. Do not promote the approximate codec based on its earlier short tests.
- **H164 was rejected:** balanced BTT factors improved fitting over greedy factors
  but missed wide-model quality and used 17.7% more local training VRAM.
- **H165 is an unexecuted draft:** `results/btt_execution_v1/model.py`, `check.py`
  and [the plan](btt_execution_plan.md) exist. Lint passed previously; CPU numerical
  checks, the study runner and GPU experiments are unfinished. No H165 result or
  stability claim exists. The previous write/run command was rejected by automatic
  approval review because of the usage limit; it did not execute.

The architectural goal remains open. Parameter savings have repeatedly failed
to preserve quality or reduce actual activation/temporary memory. Resume with a
bounded, falsifiable experiment and use the qualified exact helper as the current
practical memory improvement. See [CURRENT_STATE.md](CURRENT_STATE.md) for reports.
