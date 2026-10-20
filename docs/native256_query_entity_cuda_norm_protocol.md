# Query reader actual-objective numerical gate

18 October2026, no optimizer updates. Preserve three failed zero-update input/
numeric exports. The first selected two short source spillovers without query
labels; the second confirmed no <=128-token segment contains both. The two
shortest genuine combined segments are438/439 tokens. This changes gate inputs,
not the model, data, objective, sampler or training recipe.

The third gate's absolute2e-5 norm threshold failed at norm20.90377. Diagnosis
records CE difference4.77e-7, query loss2.38e-7, source loss0, combined4.65e-7,
all clipped gradients5.14e-7, but FP32 norm difference1.7929e-4 (8.577e-6 relative).
An independent explicit-denominator/repeated-GPU audit summed every unclipped
parameter gradient's square in FP64. CPU norm20.903774869/GPU20.903771265 differ
only1.7244e-7 relatively. Raw gradients differ<=5.961e-7, GPU repeats<=1.193e-7.
CPU FP32 reduction has8.789e-6 relative error versus FP64, GPU below4.93e-8.
This establishes a norm-reduction issue, not different objective semantics.

New actual-objective gate retains absolute2e-5 checks on CE, query/source loss,
combined objective and all clipped parameter gradients, plus GPU global/micro
gradients. Change ONLY norm comparison to relative2e-5, bound to the independent
FP64 and repeated-GPU receipts. The numerical change does not certify quality or
justify continuing a trained pilot. All weights/empty Adam must remain exact;
run all-source-bearing raw512/memory39/eight-row capacity with two moment-sized
buffers before export. Timing excludes Adam step/scratch/encoder/IO/eval/save.
