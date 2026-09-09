# CUDA graph serving check - frozen execution experiment

The locked three-seed candidate improves on the narrow control but fails the
1% NLL target. Rotating-order eager serving is also below the 80% throughput
floor: factorized median 0.657 times full SwiGLU, packed factors 0.780 times.
Packing reduces launches but has extra tensor movement. Neither is an accepted
runtime result. The dense merged cache reaches about 1.00 times, but uses twice
the full reference's FFN matrix FLOPs and additional cache buffers.

Test CUDA graph replay on the same final seed-17 checkpoints. Apply replay to
all four modes: factorized, packed factors, BF16 dense cache, and full reference.
No weight updates, custom kernels or compiler changes. Hold batch 16, context
128, BF16 and full-sequence output fixed. Warm up on a side stream, retain
static input/output storage, copy each new input into the static buffer, and
include that device-to-device copy in replay timing. Verify fresh-input replay
against eager output and evaluate all 32,768 validation targets for every mode.

Measure graph construction time and single-model peak allocated GPU memory
including graph storage separately. Then rotate mode order for eight timing
rounds with four warmups and 30 synchronized timed replays per mode. Retain
all timings; simultaneous model residency during paired timing is not used as
a single-model memory measurement. Report median and range of paired ratios.
Compare each candidate's graph execution against the dense graph reference.
This is a fixed-shape execution result; it does not establish arbitrary-shape
serving, token-by-token generation speed, training speed or architecture novelty.

Implementation follows the primary
[PyTorch CUDA graph documentation](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs).