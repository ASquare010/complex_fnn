# H134 import recovery (prospective)

Initial case00 exited 3221225477 (Windows access violation) during NumPy import.
Its case directory has no scientific files; no GPU process remained. The driver
stopped before later cases. No training update or evaluation completed.

Permit one fresh launch per scheduled case, retaining every original log.
Use PYTHONMALLOC=pymalloc for ALL training and independent audit processes.
Keep frozen worker/loop, GPU policy, order, fixtures, budgets and gates unchanged.
This allocator change is an operational hypothesis, not a diagnosed crash cause.
Recovery logs/exits have a recovery_ prefix. Stop at the next process failure;
never replay a completed case. No automatic unbounded retries. Any later recovery
requires a new prospective record and inspection of completed work.
