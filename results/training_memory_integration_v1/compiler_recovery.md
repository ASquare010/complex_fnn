# Bounded optional-test environment recovery

The full suite returned126 passes and six failures before Triton kernel execution:
its compiler search uses the UV interpreter's sysconfig directory while packages
are loaded from this repo's .venv. The installed compiler exists at
.venv/Lib/site-packages/triton/runtime/tcc/tcc.exe. Preserve suite.log/exit unchanged.
Run only the six failed fused-inference tests once with process-local CC pointing
to that bundled compiler. No source/default/tool installation changes, no optimizer
updates, and no research timing repeats. Record compiler hash and final outcome.
