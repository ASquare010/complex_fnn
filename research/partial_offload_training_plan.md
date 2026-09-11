# H137: checkpoint-input offload amount

Previous turn: progress. H136 completed 18 fresh runs and independent audits.
Quality and memory passed for every seed, but seed101 WikiText-2 exceeded the
15% runtime limit. That long-training gate remains failed.

Hypothesis: fewer CPU transfers retain a useful portion of the memory saving
with lower runtime cost. Test ordinary native loss with no offload, then H128
buffer loss with0,4,8 offloaded blocks. Four means the last four of eight blocks,
chosen prospectively for simplicity, not claimed optimal. The existing native
save_on_cpu adapter wraps only selected blocks; parameter identities and model
registration remain unchanged. No new activation/algorithm novelty is claimed.

Use H136 ordinary seed101 final step800 checkpoints for both corpora. Reuse
H134's complete30-update continuation loop, H135 ordinary policy and numerical
noise calibration. No cuBLAS env/setter, default8.125MiB, FP32, TF32off, four
threads, PYTHONMALLOC=pymalloc, UV-managed Python. Original clipping/Adam, learning
rate, batch8/context512, native RMSNorm, resident gradients and validation remain.
All arms start from identical model/optimizer/sampler states and see identical
batches. Input offload is the only change among the three buffer arms.

Per corpus run ordinary,buffer0,buffer4,buffer8,buffer8,buffer4,buffer0,ordinary.
Compare mirrored repeats against their same-repeat ordinary control.16 runs,
480 updates/backwards,1,966,080 training targets,32 full study scores plus16
independent native scores,48 tensor artifacts,32 worker +17 audit zero boundaries.
No extra training or hidden retries. Stop driver on failure; any recovery must
first inspect evidence and record a bounded prospective protocol.

Apply EACH corpus/repeat: memory ratio<=0.90, complete CUDA/wall<=1.15, NLL<=1.01,
candidate pinned-host peak<=128MiB, halves timing ratio<=1.15, passive telemetry
coverage, unchanged sources and numerical audit. Report all candidate results.
The8-block arm is a diagnostic control and cannot be promoted by this short test
because H136 already failed its long-run gate. Among NEW0/4-block variants that
pass every corpus/repeat, prefer the one with greater memory savings, then time.
A short pass only earns stronger validation; no maintained defaults are changed.

CPU audit reuses independent NumPy clipping/first-Adam checks and native scoring.
H135's bounded native-repeat calibration applies to raw/clipped first gradients:
global min(1e-5,max(1e-6,10*noise)); tensor min(1e-4,max(1e-5,10*noise)). Native
noise must itself be within the caps. Record final weight drift descriptively;
ordinary attention is nondeterministic. Exact H128 operator proof remains.

Same H134 timing scope: forward/backward/clipping/Adam and transfers, excluding
batch generation, grad clearing, validation and inventory. Whole-job CUDA peak
includes all phases, excluding driver/context/other processes. Record host peak
separately. Passive GPU telemetry every200ms, stopped in finally. No hardware
clock/power changes. All results and failed gates remain available.
