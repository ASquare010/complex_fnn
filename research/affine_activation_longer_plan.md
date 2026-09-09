# H039: frozen longer WikiText comparison

Written after H037 selection and before any 800-step WikiText run. Affine
BlockShuffle selected NLL5.894765 passes all short-budget quality, parameter and
memory gates. The rational variant had NLL5.898003 but failed native memory.
This earns a longer comparison, not a claim of convergence or a breakthrough.

## Fixed cohort and budget

Five recipes: full GELU, full SwiGLU, calibrated narrow SwiGLU, unmodified
BlockShuffle and affine BlockShuffle. Use each recipe's independently selected
200-step rate (respectively0.0012,0.0012,0.0012,0.0006,0.0012).
All other architecture, optimizer, precision and initialization settings remain
as in the archived 200-step trials. The affine model has eight groups, amplitude
0.25, zero theta initialization, no shape decay or special shape rate. Eager
execution only; no compiler backend in this training cohort.

Train from initialization, seed17,800steps,B16,context128:1,638,400 sampled tokens
per trial, five trials8,192,000 tokens. d384/L8/H6,vocab4096, the unchanged pinned
WikiText train/validation cache. Evaluate322,688 validation targets at
initialization and steps1,200,400,600,800. Each recipe uses the same10% warmup and
cosine schedule stretched to800steps. All runs record gradients, clipping,
initial/final layer statistics, weights, source archives and hardware/software.
Official test remains unfetched/unscored. This is eight hundred steps, not
established convergence.

One fresh GPU worker at a time, order full SwiGLU, calibrated narrow,
unmodified BlockShuffle, full GELU, affine BlockShuffle. Deadline2400seconds
per worker. Never overwrite any completed or partial run. Stop on infrastructure
or numerical failure, retain all logs and outputs, and require an explicit
recorded continuation before retry. Do not extend rates, alter configurations,
or pick a best intermediate checkpoint after seeing results. Previous selections
at the upper rate boundary remain an unresolved tuning limitation.

## Verification and decisions

Before reuse, verify all five selected source archives, checkpoint hashes,
configurations and data hashes. The training-loop AST is unchanged from the
200-step archives. Each new run must reproduce its own archived initial NLL
within1e-7. All five final sampling-RNG states must agree, proving identical
sampling consumption in this deterministic common data implementation.

Evaluate FINAL800-step NLL only. For affine promotion to independent seeds,
require >=70% fewer FFN weights, <=1% relative NLL degradation versus BOTH
full controls, beat calibrated narrow, >=0.2% improvement versus unmodified
BlockShuffle and allocated training peak <=1.1timesBOTHfull references.
Every recorded gradient/activation/slope must be finite. Report all five trials
and every failed gate, regardless of the outcome. Serial wall time and inference
measurements do not establish paired throughput superiority.

A pass earns a separately scheduled replication at seeds29and43 using these
same settings, then TinyStories transfer and an activation-path ablation.
A failure stops that promotion; do not redefine success from the 200-step win.
Independent seeds, actual convergence, equally tuned scale/corpus controls,
throughput and credible novelty remain parts of the original research goal.
