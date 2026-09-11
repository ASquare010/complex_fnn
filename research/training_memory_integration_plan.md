# H141: opt-in maintained memory helper

Previous turn: progress. H140 passed all six paired short-segment comparisons.
H138 remains failed; no sustained-throughput or new-FFN claim is established.

Add src/core/training_memory.py and targeted tests without editing existing
maintained sources/defaults. Public APIs: buffer_cross_entropy, buffer_model_loss,
and a restoring offload_checkpoint_inputs context. Keep the H128 classifier
arithmetic exactly, including native input-gradient layout. Add CPU-only argument
checks. Support first-order eager FP32/FP64 hard-label mean CE, ignore_index=-100;
reject autocast and unsupported shapes. Offload is explicit, requires native block
checkpointing, preserves parameter paths/objects, preflights all selected blocks,
and restores instance methods on exceptions. CPU/eval/no-grad bypass transfer.

Before GPU work: targeted tests and AST equivalence against H127-forward/H128-
backward; freeze source, tests, plan and six H140 source checkpoints. Run full
repository tests. For each of all six saved states, execute one frozen and one
maintained FP32 complete Adam step on identical data. Alternate arm order by
fixture. Require matching provenance/batches, finite states, parameter identities
and state paths, selected block indices and clean restoration; raw/clipped gradient
global relative L2<=1e-5 and maximum per-tensor<=1e-4; parameter/moment relative
L2<=1e-6; loss relative error<=1e-6. Memory peak may not exceed frozen reference by
more than1MiB for this integration replay. Ordinary attention remains nondeterministic.
Save both outputs and report bitwise equality descriptively, not as a gate.

Budget12 optimizer updates/backwards,49,152 targets,13 zero CUDA boundaries;
no validation scores (H140's independently verified scores are prior evidence).
This is source integration verification at the same scale, not a performance or
new-quality experiment. No runtime claim from single steps. Every old maintained
file hash must remain unchanged. Stop on failure; preserve logs/artifacts and
never retry a completed GPU step merely for a more favorable result. A pass earns
an opt-in documented helper; next qualify another workload scale while pursuing
the broader parameter-efficient FFN objective. No promotion into training defaults.
