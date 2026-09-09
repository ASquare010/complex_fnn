# Rational correction-only compiler isolation

**Status: complete; earns a training repeat: True.** This diagnostic retains native SiLU and gate multiplication, compiling only the rational residual. It does not modify the previously failed full-product backend or its training result.

The [frozen protocol](activation_correction_execution_plan.md) requires all-parameter gradient relative L2<=0.0001, stricter than the original local audit. Native checkpoints, full-validation targets, fidelity batch, optimizer state and transient profiling-loop AST are held fixed. Source archives, checkpoint hashes, environment and reused memory controls are verified before execution.

| Local quantity | Correction-only | Previous full product |
|---|---:|---:|
| logits_max_abs | 0.0 | 0.0 |
| logits_relative_l2 | 0.0 | 0.0 |
| all_gradients_relative_l2 | 3.952930266539674e-08 | 0.000784290506311299 |
| activation_gradients_relative_l2 | 1.1478558797925649e-07 | 0.00016879276423865594 |
| finite | True | True |

Allocated training peak: 717.313 MiB. Full-validation NLL: 5.898002606, native 5.898002606. Frozen gates: {'memory_within_ten_percent_full_gelu': True, 'memory_within_ten_percent_full_swiglu': True, 'strict_gradient_fidelity': True}.

This isolates a different compilation boundary. A smaller derivative error suggests numerical improvement only on the measured batch; it does not prove the cause of the earlier 200-step drift or guarantee a matching training trajectory. Transient optimizer updates, if reached, are discarded and are not a trained model result.

[Raw audit](../results/activation_correction_execution_v1/result.json). All execution artifacts and the original checkpoint remain retained.
