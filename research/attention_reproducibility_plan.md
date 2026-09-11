# H129: establish a reproducible whole-model comparison

H128 is progress: operator exactness/resource gates pass, whole-model bitwise
replay fails even for native controls. Investigate the reference before any
further training. No rescue of H128 or claim that small drift is harmless.

Three fresh sequential processes on the same UV runtime/GPU:
1. original_default: deterministic algorithms off; cuBLAS workspace env unset;
   default attention dispatch, cuDNN deterministic off, benchmark off.
2. deterministic_default: deterministic algorithms on (errors, not warnings),
   cuBLAS workspace :4096:8, cuDNN deterministic on, default attention dispatch.
3. deterministic_math: same deterministic settings, explicit SDPA MATH context
   around both forward and backward, including checkpoint recomputation.

The deterministic policy changes a bundle of settings. Comparisons do not
isolate one flag's causal effect. CPU profiler events record actual dispatched
SDPA forward/backward names in every probe; these are operator names, not a
claim about the internal GPU reduction algorithm. No throughput claims from
instrumented probes.

Within each policy, use both H121 seed-101 native FP32 step-800 checkpoint
fixtures and the identical sampled batch. Construct each probe afresh with
unaltered model/Adam states. Three arms: native classifier/resident checkpoint
inputs; native classifier/CPU-offloaded checkpoint inputs; H128 layout-aware
classifier/CPU-offloaded inputs. Ordinary RMSNorm, resident parameter gradients,
no optimizer update. Two fresh constructions per arm. Interleave arms in
opposite corpus order; compare each result with the resident native anchor and
with its own first repetition. All parameters, moments, input bytes and sampler
states must match. Save every probe's gradient hashes and comparison metrics;
save first-repetition full gradients for every arm (18 artifacts maximum).

Budget: 3 policies x2 corpora x3 arms x2 repetitions =36 backwards maximum,
147,456 diagnostic targets, zero training updates. No extra audit replay.
Offline CPU audit recomputes all comparisons whose two tensor artifacts are
saved, and checks every repeat hash comparison, budgets and provenance. Policy
errors are recorded and stop that policy; other predeclared policies continue.
Do not rerun completed probes or silently fall back after unsupported kernels.

Acceptance, separately per policy/corpus: all six losses and 50 parameter
gradients bitwise identical; same-arm repeats bitwise identical; all gradients
finite; expected provenance and zero allocator boundaries. Numerical L2/max
errors and matching-tensor counts are descriptive only. A deterministic policy
passing both corpora earns a separate complete-update test using that exact
policy. It does not establish long-run quality, cross-policy bitwise equality,
new-FFN success or resource savings under the new execution policy.

[PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html),
[SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention),
[deterministic algorithms](https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html).
No novelty claim. All maintained defaults remain unchanged; broad goal open.
