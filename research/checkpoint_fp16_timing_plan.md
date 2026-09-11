# H159: Complete-update timing of FP16 checkpoint storage

Previous turn: progress. H158 independently qualified isolated resource and
approximate-gradient gates on all six H156 ordinary step800 states. Preserve
H157's failed zero-host gate and H158's recorded pre-model startup crash/recovery.
This is the next elimination test, not activation novelty or a quality claim.

Reuse H142's maintained 30-update loop unchanged. Six H158 fixtures, three arms:
ordinary native loss/GPU checkpoints; buffered loss/four-block CPU offload; buffered
loss/eight FP16 GPU checkpoint inputs. Each segment reloads the same step800 model,
Adam moments and sampler. Batch16/context512, validation batch8, FP32/TF32 off,
four CPU threads, ordinary attention, default 8.125 MiB cuBLAS workspace, AdamW
LR0.0006/betas0.9,0.95/eps1e-8, decay0.1 for matrices and0 otherwise, clipping1.

For each fixture run ABC CBA (reverse to CBA ABC on alternating fixtures), with
repeat0 the first occurrence and repeat1 the second. 36 segments x30 updates:
1080 backwards/updates,8,847,360 targets; 72 study validation scores. First10
updates warm up; remaining20 measured. Timers include forward/backward, casts,
checkpoint recomputation, clipping and Adam, with synchronized CUDA and wall
measurements. Sampling, gradient clearing, logging, evaluation and serialization
are excluded from update timing but included in broader segment/memory reporting.
First-gradient/first-state copies occur only during warmup. No timing correction.
One GPU model resident; zero allocated/reserved boundaries; passive200ms telemetry.
Expected artifacts ~11GiB; require16GiB free. No hardware changes or completed reruns.

Every fixture must pass these prospective gates:
- FP16 complete-update combined40-sample median CUDA and wall ratios <=1.15 versus
  ordinary, <=1.05 versus offload4. Report both neighboring repeat ratios separately.
- FP16 maximum whole-segment allocated peak <=0.90 ordinary and <=1.02 offload4.
- Each FP16 repeat's final validation NLL <=1.01 corresponding ordinary AND offload4.
- Every segment's two10-sample half-median ratio <=1.15 (CUDA and wall); every arm's
  between-repeat median ratio <=1.15. All segments have telemetry during timed work.
- Allocator-owned pinned peak <=128MiB. Shared process may retain offload cache;
  this study makes no new isolated-host claim. H158 already provides that comparison.
- All states, moments, losses and gradients finite; counters801/830; identical
  batches/source hashes and exactly240 saved/unpacked FP16 inputs per segment.

Independent NumPy verifies clipping and the first Adam update/moments using each
arm's actual saved gradients, with unchanged H142 tolerances. Independent native
scoring and regenerated batches verify all36 final states/1080 sampled batches;
this adds no backwards or updates. Full first-gradient errors compare each arm
with same-repeat native: exact-path global<=1e-5,max tensor<=1e-4; FP16 global<=.002,
max tensor<=.02,cosine>=.99999. Loss relative error<=1e-6 for all. Also compare the
two native first gradients. These numerical caps do not imply equal trajectories.
Report mean, median, sample variance, all repeats and failed gates. Freeze scripts,
plan, reused loops, helpers, inputs and H158 receipt before GPU work. No winner
selection or relaxed thresholds after observing results.

A full pass earns fresh multi-seed long training. Failure rejects promotion under
this recipe; diagnose timing instability separately from consistent compute cost.
Neither this study nor H158 establishes preserved long training quality, fewer
parameters, isolated inference savings, a new activation or a breakthrough.
