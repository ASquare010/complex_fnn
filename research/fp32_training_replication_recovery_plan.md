# H117 startup-order recovery — no training trial has run

The original H117 attempt stopped at `generate_initials`' no-CUDA guard.
`run()` collected `environment()` before generating CPU initial states; that
metadata helper queries device name/properties and initializes CUDA. This is
a startup-order implementation error, not a scientific gate failure or a
recurrence of the earlier native Windows access violation.

Preserve the original protocol, 121 frozen sources, environment, qualification,
failure traceback, log and exit1. The original qualification completed 24
CPU backwards. No initial checkpoint, run directory, training batch, optimizer
update, candidate score or partial training run was produced.

Use a new root, `results/fp32_training_replication_recovery_v1`. Move generation
of the six CPU initial states ahead of environment collection and qualification.
Keep the same frozen initializer and its before/after no-CUDA assertions.
Only then query hardware, qualify the unchanged loss adapters, and start CUDA
training. Sources are frozen again before this recovery executes. The original
source is not edited. Regenerate all initial states independently during audit.

All scientific choices in [the H117 plan](fp32_training_replication_plan.md)
remain unchanged: two corpora, seeds101/113/127, three policies, 18 fresh trials,
800 updates each, identical evaluation schedule, numerical/resource/quality
gates and no best-checkpoint selection. This is the first training allocation
actually executed, not a repetition of completed training.

The failed qualification adds24 CPU backwards to accounting. Planned recovery
uses14,418 main backwards,24 qualification backwards and12 audit backwards:
**14,478 total backwards across both attempts**, with14,400 optimizer updates.
No retry loop, additional seeds, tuning or threshold changes are introduced.
The original seven pre-study document snapshots remain authoritative for
publication. The broader research goal remains open.
