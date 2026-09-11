# H129: whole-model reproducibility under explicit attention policies

**PASS scoped reproducibility: deterministic_default, deterministic_math.** This test addresses the unreliable native reference in H128.
It measures equality at two fixed model states, not training quality or speed.
All recorded scientific outcomes pass independent CPU evidence verification.

| Execution policy | Corpus | Exact versus native anchor | Exact same-arm repeats | Maximum relative L2 versus anchor | Gate |
|---|---|---:|---:|---:|---|
| original_default | wikitext2 | 2/6 | 0/3 | 6.807e-08 | fail |
| original_default | tinystories | 1/6 | 0/3 | 1.112e-07 | fail |
| deterministic_default | wikitext2 | 6/6 | 3/3 | 0.000e+00 | pass |
| deterministic_default | tinystories | 6/6 | 3/3 | 0.000e+00 | pass |
| deterministic_math | wikitext2 | 6/6 | 3/3 | 0.000e+00 | pass |
| deterministic_math | tinystories | 6/6 | 3/3 | 0.000e+00 | pass |

Each corpus/policy has three arms: native/resident checkpoint inputs,
native/CPU-offloaded checkpoint inputs, and H128 classifier/CPU-offloaded inputs.
Each arm is constructed twice from the same step-800 state and sampled batch.
The six anchor comparisons include the native anchor's trivial self-comparison;
the three repeat comparisons each compare two independently constructed probes.
Loss and all 50 parameter gradients must match bitwise. L2 differences are
reported for context and never substitute for the exactness gate.

Ordinary execution matches none of the three same-arm repeat pairs on either
corpus. Both deterministic policies match every comparison, with zero gradient
difference. The default deterministic policy retains the efficient-attention
operator family; forcing the math backend is unnecessary for these fixed-state
checks. H128 memory savings cannot yet be transferred to this changed policy
without a fresh resource measurement.

## What the policy changes

`original_default` disables deterministic algorithms and unsets the cuBLAS
workspace environment variable. `deterministic_default` enables strict
deterministic algorithms and cuDNN deterministic execution, with
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, while retaining default SDPA selection.
`deterministic_math` uses that same deterministic configuration and forces
SDPA MATH around both forward and backward, including recomputation.
Four CPU threads, FP32 and disabled TF32 apply to all policies. Optimizer and
model states are unchanged; no compact RMSNorm or gradient staging is used.

The policy changes a bundle of settings. The result does not isolate one
flag as the cause. These are fresh sequential processes on the same RTX 4070
Laptop GPU and UV-managed Python/PyTorch installation. CPU profiler events in
every probe identify the dispatched attention operators:

- `original_default`: `aten::_scaled_dot_product_efficient_attention`, `aten::_scaled_dot_product_efficient_attention_backward`, `aten::scaled_dot_product_attention`
- `deterministic_default`: `aten::_scaled_dot_product_efficient_attention`, `aten::_scaled_dot_product_efficient_attention_backward`, `aten::scaled_dot_product_attention`
- `deterministic_math`: `aten::_scaled_dot_product_attention_math`, `aten::scaled_dot_product_attention`

Math attention is decomposed and need not expose a fused SDPA backward entry.
The recorded operators identify dispatch, not every internal CUDA reduction
choice. Profiling may affect scheduling; no throughput or memory-saving claim
is drawn from this instrumented experiment. A passing policy demonstrates
repeatability for these probes, not a guarantee across hardware or versions.
See [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
and [SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).

## Evidence and limits

Actual execution: **36 successful backwards**, 36 backward
attempts, 147,456 diagnostic targets, **zero optimizer updates**.
There are 18 first-repetition gradient artifacts and 39
zero GPU allocator boundaries. Every probe records gradient hashes, exact
loss, model/optimizer/input/sampler provenance, finiteness and actual attention
operators. All 61 maintained file hashes remain unchanged.

An independent CPU verifier rehashes the saved gradients, recomputes their
per-tensor and whole-gradient distances using NumPy, checks all repeat-hash
comparisons and verifies budgets/provenance. Second-repetition tensors are
represented by recorded hashes and comparison metrics rather than extra
full tensor files; numeric distances for those repetitions cannot be
independently recomputed from retained tensors. Their bitwise comparisons
are checked against first-repetition hashes. No additional GPU audit was run.

Policy failures: `[]`. No completed probe was retried.
H127's operator failure and H128's full-model replay failure remain recorded;
this separate policy comparison does not retroactively change either gate.
Parameter count remains 9,099,648. No activation, FFN or algorithmic novelty
is claimed. No maintained default changes.

## Next decision

Run a separately frozen complete-update and resource comparison under `deterministic_default`, including all transfers, original clipping/Adam and native validation. Match the policy in both arms. Compare its cost with ordinary execution separately before claiming practical efficiency.
The broad VRAM, quality and parameter-efficient FFN research goal remains open.

[Prospective plan](attention_reproducibility_plan.md),
[independent verification](../results/attention_reproducibility_v1/audit.json),
[evidence receipt](../results/attention_reproducibility_v1/receipt.json).
