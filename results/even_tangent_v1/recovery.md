# Bounded H153 preflight recovery

The original check process exited1 at the scalar inverse assertion, before any
feature/ridge/replay cases. Its protocol, source and log remain unchanged.
Python scalar operands in torch.where produced FP32 coefficient tensors; the
inverse's1+a arithmetic therefore rounded in FP32 despite FP64 y. This is not
the intended FP64 inverse test. The recovery explicitly constructs theta and
eta as FP64 tensors. No formula, tolerance, seed or diagnostic budget changes.
Run exactly one new check_dtype64 stage with protocol_dtype64.json. Preserve the
original failure. No training or GPU work occurred in the failed stage.
