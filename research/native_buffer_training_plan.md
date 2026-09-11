# H130: complete-update validation of native-buffer reuse

H129 made progress by qualifying a reproducible whole-model reference. Test
whether H128 buffer reuse plus checkpoint-input offload preserves optimization
and saves complete-job VRAM under that exact deterministic-default policy.
Retain all prior failures and keep the broader research goal open.

Reuse H120's unchanged 30-update continuation, original clipping/default AdamW,
BF16 evaluation, step-800 seed-101 states, and batch order on both corpora.
Two sequential fresh processes: ordinary native control (determinism off,
workspace unset), then deterministic-default with :4096:8 workspace, strict
deterministic algorithms, cuDNN deterministic on/benchmark off, default SDPA.
All use FP32 training, TF32 off, four CPU threads. In the deterministic process,
three arms: native/resident checkpoint inputs; native/CPU-offloaded inputs;
H128 buffer classifier/CPU-offloaded inputs. Ordinary RMSNorm/resident gradients.
Reverse deterministic arm order across corpora. No profiler in timed regions.

Eight cases x30 =240 updates/backwards, 983,040 training targets. Each scores
initial/final validation (16 full scores), saves first raw/clipped gradients,
step-801 model/optimizer and step-830 model/optimizer (24 tensor artifacts).
Independent inherited NumPy audit checks clipping and first Adam update/moments;
eight native final scores reconstruct all 240 batches and sampler endpoints.
No extra qualification or gradient replay: exact operator/path already passed
H128/H129, and original optimizer is unchanged.

Gates per corpus:
- Three deterministic arms: bitwise identical first raw/clipped gradients,
  saved model/optimizer/sampler states at 801/830, all 30 losses/gradient norms,
  and validation NLL. These checkpoints do not expose every intermediate state.
- Candidate versus deterministic native AND offloaded-native AND ordinary
  native: complete-job allocated peak <=0.9x; median event and synchronized wall
  update time <=1.15x; each arm's timing-half stability <=1.15; host peak <=128 MiB.
- Candidate versus ordinary native final NLL <=1.01x; deterministic peers exact.
- All finite/provenance/inherited numerical-score tolerances, zero boundaries.

Timing uses updates 11–30, retaining all steps, mean/median/variance. Peak memory
includes construction, initial/final evaluation, all warmup/training phases,
transfers, diagnostics, clipping/Adam and artifact serialization. Native
validation replay is separate verification, not folded into training-job peak.
A pass earns longer fresh-initialization training, multiple seeds and scale.
A failure blocks that expansion until its cause is tested. Short continuations
cannot establish long-run quality or a breakthrough. No parameter or novelty
claim; maintained defaults remain unchanged.
