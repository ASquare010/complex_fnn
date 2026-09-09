# H095: remove a redundant readout without retraining

H094's three richer bases fail their fixed gates. The duplicate-even control is
the strongest selected aggregate recipe: reporting MSE ratio0.870784574 versus
narrow GELU. This is an observed control result, not a prespecified candidate win.
Before refining or assigning more training, remove its algebraic redundancy.

The model is A*x + V*E(U*x) + W*E(U*x), E(z)=z^2/(1+abs(z)). Replace V,W by
B=V+W, giving A*x+B*E(U*x). This retains the exact real-valued function and input
Jacobian. It reduces weights350,208 ->282,624:76.0416667% below1,179,648 full
weights. Shared projection U is unchanged. It does not preserve ordinary AdamW
training coordinates, states or future trajectories, so do not resume training
with the folded checkpoint and claim an equivalent optimization experiment.

Freeze this plan and source before executing. No optimizer update, rate selection,
precision recovery or training extension. Use the12 H094 duplicate checkpoints
selected already on the selection split, plus their independently selected narrow
GELU, full GELU and full SwiGLU timing references. Reporting data is already used;
this is a post-selection algebra/resource check, not an untouched validation.

One CPU FP64 output/input-Jacobian equivalence check on seed17 smooth weights
before float32 folding, tolerance1e-11. For every selected duplicate checkpoint,
compare256 FP32 outputs and input gradients with atol/rtol1e-5. Save paired tensors.
Rescore all4,096 reporting rows after FP32 folding: require absolute AND relative
MSE change <=1e-5. Record every change; the real-arithmetic theorem does not imply
bitwise equality after float32 addition and different GEMM widths. Stop on failure.

Measure native FP32 inference at batch256 for folded, original duplicate, narrow
GELU, full GELU and full SwiGLU. Use inference_mode,20 warmups and seven timing
blocks of20 forwards with CUDA events and synchronization. Rotate model order
across task/seed. Record every block, median GPU-event forward time, synchronized
wall time and peak allocated bytes with only the probe resident. This is neither
compiled inference, autoregressive decoding nor native training throughput.

One RTX4070 Laptop GPU, four CPU threads, TF32off, same hardware/runtime as H094.
Independently reload saved pairs, compare them again and check counts and relative
MSE changes. Preserve the frozen H094 result and checkpoint hashes. No new active
model folder and no gold-goal success follow from this scoped experiment.
