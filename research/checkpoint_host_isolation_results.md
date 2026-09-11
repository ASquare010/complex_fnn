# H158: Isolated host-allocation qualification

**PASS fresh-process resource/numerical qualification.** Eighteen fresh-process GPU backwards, six source fixtures,
zero optimizer updates. This resolves a resource-measurement question; it does
not establish training quality, throughput, parameter savings or novelty.

## Why this follows H157

H157 passed numerical and GPU-allocation subchecks but failed its absolute-zero
pinned-host gate. Its single process retained about 64 MiB from an earlier offload
probe. H158 preserves that failed result and preregisters separate fresh processes
for native0, offload4 and FP16 on every fixture. Each process exits before the next
starts; arm order alternates by fixture. The maintained API and H157 codec/probe
are reused unchanged. No cached bytes are retrospectively subtracted.

The new claim is **no additional pinned allocator footprint relative to native**,
not zero pinned bytes. FP16 must have allocated and active peaks no greater than
native, and allocated peak at most 1% of offload4. All three processes must start
with zero allocated host bytes. This is a new H158 gate, not a relaxed H157 pass.
PyTorch allocated-byte statistics include active and cached allocations; active
bytes are reported separately. See the sources and discussion in
[H157](checkpoint_fp16_results.md). The five-byte incidental allocation's source
has not been traced, so no stronger claim about host transfers is made.

## Results: every seed

| Corpus | Seed | Global gradient error | Max tensor error | GPU allocation saved vs native | GPU / offload4 | Pinned allocated peak bytes: native / offload4 / FP16 | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| wikitext2 | 101 | 0.01672% | 0.02976% | 22.83% | 1.0000 | 5 / 67,108,869 / 5 | True |
| wikitext2 | 113 | 0.01575% | 0.02810% | 22.83% | 1.0000 | 5 / 67,108,869 / 5 | True |
| wikitext2 | 127 | 0.01522% | 0.02806% | 22.83% | 1.0000 | 5 / 67,108,869 / 5 | True |
| tinystories | 101 | 0.01906% | 0.02961% | 23.04% | 1.0000 | 5 / 67,108,869 / 5 | True |
| tinystories | 113 | 0.01948% | 0.02864% | 23.04% | 1.0000 | 5 / 67,108,869 / 5 | True |
| tinystories | 127 | 0.01961% | 0.03118% | 23.04% | 1.0000 | 5 / 67,108,869 / 5 | True |

Absolute allocator peaks across all six fixtures (MiB):

| Arm | Minimum | Maximum | Mean |
|---|---:|---:|---:|
| native0 | 650.141 | 656.320 | 653.231 |
| offload4 | 500.329 | 506.508 | 503.419 |
| fp16 | 500.329 | 506.508 | 503.419 |

Across the three seeds, gradient relative-L2 statistics (dimensionless; sample variance):

| Corpus | Mean | Median | Variance |
|---|---:|---:|---:|
| wikitext2 | 0.0001589378 | 0.0001574859 | 5.795859e-11 |
| tinystories | 0.0001938118 | 0.0001947546 | 8.327076e-12 |

The GPU gates are FP16/native <=0.90 and FP16/offload4 <=1.02. FP16 numerical
limits remain global relative L2 <=0.002, maximum tensor <=0.02, cosine >=0.99999,
relative loss error <=1e-6. Offload retains exact-path limits 1e-5 global,
1e-4 maximum tensor and 1e-6 loss. Every fixture must pass; no seed is discarded.
Full tensor errors, cosines, active and allocated host peaks remain in audit.json.

## Scope and verification

Two corpora, seeds 101/113/127, ordinary H156 step800 checkpoints, batch16,
context512, 9,099,648 parameters, eight blocks, FP32, TF32 off, four CPU threads,
ordinary attention and default cuBLAS workspace. Each arm reloads identical model,
Adam moments, dataset and sampler state, then uses the identical next batch.
The candidate stores eight block-checkpoint inputs as FP16 on GPU and recomputes
with FP32 restored inputs. Original forward evaluation remains FP32; the backward
is approximate, as established by H157's local semantic check and rounding audit.

Loaded model, moments and data remain resident. GPU peaks retain the larger of
construction and later forward/backward/gradient-copy/check peaks. There is no
Adam step, and these are not whole-update or isolated-inference memory measurements.
Driver/context, unrelated processes and total CPU RSS are outside these counters.
No cold-process timing is reported as throughput. No extra diagnostic GPU runs.

Independent NumPy FP64 calculations recompute every saved gradient error and cosine;
all 18 gradients and 36 GPU boundaries are checked. Batch, model and optimizer
hashes match both new controls and original H157. The FP16 save/unpack count and
stored bytes are checked for all eight inputs. Original sources, data and checkpoint
hashes verify. Each child has its own exclusive log and exit record. No completed case was rerun.

## Preserved startup failure

The original worker1 crashed during PyTorch import with Windows access violation
3221225477, before model construction or result files. Case00 completed and was
retained. A prospectively documented single retry added the prior H157 launcher's
unused bytecode-cache prefix; it then continued cases2-17. The original failed
logs and exit files remain. The cause of the import crash is not established.
The recovery audit differs only in which exit record it accepts for worker1.
No numerical recipe, model code or gate changed; successful backward count is 18.
[Recovery record](checkpoint_host_isolation_recovery.md).

## Decision

Proceed to a warmed, temporally paired complete-update timing comparison of FP16, offload4 and native. Only a timing survivor earns fresh multi-seed quality training. This result establishes neither of those outcomes.
The broader goal remains open. Activation-compressed training is established
prior art; no novel activation or breakthrough is claimed.

[Prospective plan](checkpoint_host_isolation_plan.md),
[full audit](../results/checkpoint_host_isolation_v1/audit.json),
[original failed H157 audit](../results/checkpoint_fp16_v1/audit.json).
