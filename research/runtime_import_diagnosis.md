# H110 runtime import investigation

The source of the intermittent native Windows failures is **unresolved**. This
diagnostic does not reject a neural architecture or establish a hardware fault.
It records why H110 needed an explicit continuation after its first control.

The original candidate process failed with exit3221225477 (`0xC0000005`) while
AdamW initialization imported Torch compiler dependencies and SymPy. It made
zero updates. The first recovery preloaded dependencies on CPU: two probes
passed, but its candidate process then failed during SymPy import, before Torch
import or CUDA initialization. Thus that failure cannot be attributed to a
candidate CUDA forward or backward operation.

## Bounded CPU experiment

The [frozen probe](../results/runtime_import_diagnosis_v1/source/probe.py) imports
SymPy, checks a simple symbolic derivative, imports Torch and `torch._dynamo`,
and asserts CUDA is uninitialized. Twelve independent subprocesses compare
UV Python 3.12.9 and 3.12.12, with or without existing bytecode caches. All use the
same installed CPython3.12 packages, `PYTHONMALLOC=malloc` and hash seed 107.
Three replications rotate the condition order. No model training or GPU
calculation is part of this diagnostic.

| Interpreter | Bytecode policy | Successful probes | Observed failures |
|---|---|---:|---|
| UV 3.12.9 | Existing caches | 1/3 | Two native access violations |
| UV 3.12.9 | Bypass caches | 3/3 | None in these three probes |
| UV 3.12.12 | Existing caches | 2/3 | Generated Python SyntaxError during Torch import |
| UV 3.12.12 | Bypass caches | 2/3 | One native access violation |

Bypass means `-B -X pycache_prefix=<absent directory>`: old cache files are not
read, and no new bytecode is written. Existing caches and installations are
untouched; the flags are documented in the
[Python command-line reference](https://docs.python.org/3.12/using/cmdline.html).
The retained RECORD audit checks **1,572 hashed SymPy files**, with
zero missing files and zero hash mismatches. This verifies those source bytes,
not all Python/Torch binaries, RAM or runtime behavior.

Eight of twelve probes pass. Both interpreter versions encounter failures, and
one fails even with caches bypassed. These results do not prove cache corruption,
source corruption, a particular interpreter defect or a GPU/driver cause.
The three successful UV 3.12.9 bypass probes justify only a bounded, recorded
continuation attempt, as specified in the [recovery plan](token_memory_duration_recovery_plan.md).
All original exceptions and native traces remain preserved.

Before that continuation, an independent audit using UV 3.12.9 with cache bypass
reproduces the completed original control's initialization and three saved
checkpoints: five NLL checks match exactly, all 800 training batches replay
exactly, and the final optimizer is finite at step800. The audit performs zero
optimizer updates. This supports use of the existing control but does not
establish general runtime reliability.

After the recorded continuation finishes training, a separate outer `uv`
invocation panics before launching the checkpoint audit. The retained failure
reports an installation-version parsing assertion. Directly invoking the same
absolute UV-managed Python 3.12.9 interpreter starts the audit successfully;
all 40 subsequent NLL checks match exactly. This bypasses that dispatch attempt,
but does not establish a fix or connect the UV panic causally to the native
import failures. No scientific training is repeated.

See the [complete diagnostic record](../results/runtime_import_diagnosis_v1/result.json),
[frozen protocol](../results/runtime_import_diagnosis_v1/protocol.json), and
[H110 duration report](token_memory_duration_results.md) for the scientific
outcome and any subsequent runtime events. Small sample counts are descriptive;
no statistical superiority or root-cause claim is made.
