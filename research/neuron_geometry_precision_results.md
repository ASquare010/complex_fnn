# Pair geometry: qualification failure and explicit recovery

The corrected execution recipe passes all 27 qualification checks. This is
evidence about formulas, derivatives and GPU execution; it is not evidence that
the activation learns better. The fitting result is documented separately in
[the geometry study](neuron_geometry_results.md).

## What failed

H088 completed 21 CPU checks, then failed its first whole-FFN BF16 comparison.
For the learned pair twist, 55 of 175,104 up-projection gradient entries exceeded
the original absolute/relative tolerance of 0.02. The largest absolute mismatch
was 0.0625. Training had not started: there was no dataset, checkpoint or
optimizer update. The failed run remains intact in
[its terminal record](../results/neuron_geometry_v1/coordinator_status.json).

The rational implementation cast the residual correction to BF16 before adding
it to the input. The independently derived trigonometric reference cast the
complete activation once. Those placements describe the same real function but
different floating-point computations.

## Diagnostic, then a different execution recipe

H089 froze a separate diagnostic: four candidates, FP32 and BF16, and four
evaluation modes per case. It performed 32 forward/backward evaluations and
zero optimizer updates. Computing each complete activation in FP32 and casting
its result once removed every reference tolerance violation. All 20 FP32
output/gradient pairs were bit-for-bit unchanged by that precision region.
The diagnostic preserved all 41 files from the failed H088 run, including
their hashes, sizes and modification times.

[Diagnostic evidence](../results/neuron_precision_v1/result.json) and
[frozen diagnostic plan](neuron_precision_plan.md) retain the measurements.
No tolerance was relaxed, and BF16 equality to the original implementation is
not claimed.

H090 therefore made one explicit qualification recovery. It wraps the original
pair-twist and local-curve functions with FP32 input conversion and one final
output cast, while retaining FP64 inputs for numerical gradient checks. The
original state dictionaries, parameter/buffer roles, formulas and initialization
remain unchanged. In the planned FP32 fitting grid these casts are identity
operations.

All 27 checks passed in the recovery. The 21 original CPU observations matched,
including exact equality of six saved tensor payloads. The original failed run
was preserved. The unchanged FP32 training grid was allocated only after these
checks passed. See the [recovery plan](neuron_geometry_recovery_plan.md) and
[qualification process record](../results/neuron_geometry_recovery_v1/qualification_process.json).

## Scope of the mathematical evidence

The centered pair twist preserves distance to its fixed center. Its real-valued
inverse negates the learned amplitude, its Jacobian determinant is one, and
its activation Jacobian singular values lie between sqrt(2)-1 and sqrt(2)+1.
The proof and its assumptions are in the
[direction note](research_direction_2026_09_10.md).

These are properties of the activation. Dense weights can still amplify or
attenuate gradients, and compositions need not retain those bounds. None of
the checks proves favorable optimization, convergence, full-network gradient
stability, novel mathematics, or parameter-efficient generalization.
