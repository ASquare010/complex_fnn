# Query prototype V2: temporary context registration repair

18 October2026. No corpus run/optimizer update has started. V1 zero-update capacity,
ordinary-CE CPU/GPU agreement and unit receipts remain immutable and are not a
certificate for its later auxiliary training path. The first actual auxiliary
test exposed a PyTorch module-registration error: a temporary reference to the
embedding Parameter was registered as `reader._embedding`. Clearing it left a
registered None entry, so the subsequent detached Tensor context was rejected.

New namespace `token_memory_64_query_entity_v2` changes ONLY transient context
assignment to bypass Module parameter/buffer registration. The mathematical
reader, initialization, budget, KV layout and prospective coefficients remain
the same. The replacement reader is constructed inside a forked RNG scope.
Require exact V1/V2 initial tensors/RNG, unchanged parameter/buffer/state keys
during/after both attached and detached contexts and repeated auxiliary calls,
exception cleanup, all original invariant checks, actual auxiliary gradient
routing/microbatch/serialized Adam tests, and a new full-size actual-objective
GPU/independent CPU preflight. Preserve the failed V1 training-path observation;
no optimization is permitted merely on the earlier ordinary-only receipt.
