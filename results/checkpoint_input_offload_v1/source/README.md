# H121: checkpoint-input offload diagnostic

Read [the result](../../../research/checkpoint_input_offload_results.md) first.
The native-classifier path saves 48 MiB; the chunked path saves only about
15 MiB and fails the fixed 10% gate. No training run or default change follows.

`offload.py` contains the only execution change: temporary wrappers around
the maintained `Block.forward`, using PyTorch's `save_on_cpu(pin_memory=True)`.
Its diagnostic hooks never retain original GPU inputs. `common.py` reconstructs
the existing H120 model/optimizer/data fixture and reuses earlier loss,
accounting and statistics helpers. `profile.py` has one shared no-update
probe loop for both storage modes; no candidate trainer is copied.

Execution order used:

1. `prepare`: verify H120 evidence, snapshot navigation, freeze the plan and
   eight scientific Python files plus inherited sources.
2. `qualify`: six FP64 GPU backwards and twelve finite-difference forwards.
3. `profile`: eight cases, 30 untraced repetitions and one trace each; zero updates.
4. `prepare_audit`: seal all results and 16 gradient files.
5. `audit`: **failed during SymPy import**, before any replay began.
6. `prepare_recovery`, `audit_recovery`: preserve the failure and run the
   unchanged audit function for the eight outstanding replay backwards.
7. `analyze`, `report`, `plot`: apply fixed gates and produce compact evidence.
8. `publish`: update navigation after checking its pre-study snapshots.
9. `results/verification/checkpoint_input_offload_final_v1.py`: check hashes,
   budgets, phase accounting, trace lifetimes, numerical gates and tables.

`launch.py` selects the UV-managed Python 3.12.9 interpreter and installed
`.venv` dependencies. Scientific helpers set four CPU threads and FP32/TF32-off
settings. Each stage uses a fresh process and refuses log overwrite. Run one
GPU process at a time. Do not rerun a sealed stage in place; a reproduction
needs a new study namespace and prospective manifest, reusing these helpers.

Evidence includes protocol/audit/recovery manifests, summary, compressed
qualification/result/audit JSON, metrics/repetition/phase CSV and logs. The
24 raw gradient tensors, per-case JSON and navigation snapshots remain local;
their hashes are in the final receipt. Neither historical tests nor fixed-state
gradient checks establish endpoint quality. The broader goal remains open.
