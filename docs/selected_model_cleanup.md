# Selected-model cleanup

Curve-Wide and the dense baseline are the only active model families. The retained
curve implementation accepts only `self_curve_wide`; the old model identifier
`channel_curve_transformer` and weight names remain compatible with saved weights.
Unused curve interventions, older curve modes, repeated dense passes and
BlockShuffle helpers were removed from active code. Existing kernels are unchanged.

The [archive](../ffn_experiments/README.md) contains Signed Linear, Self-Curve and
Basis Readout, their family controls, supporting definitions, representative
configurations and result reports. It is excluded from the active trainer.
Other implementations were removed; their [reports](retired_models/README.md),
[historical study configurations](study_configs) and compact records remain.
Local pre-cleanup source snapshots are in `dump/cleanup-selected-v1`.

The active presets now pair Curve-Wide with the full dense baseline on TinyStories
and WikiText at the completed 8000-update budget. Historical configuration fields
remain readable for leaderboard metadata; obsolete model configurations are
rejected for new training. Old runs retain strict source checks and require their
original source snapshots for exact resumption or recorded evaluation.

Verification compares the simplified Curve-Wide with the archived pre-cleanup
implementation: initialization, parameter counts, CPU FP64 and CUDA FP32/BF16
outputs and gradients, and logits from the winning checkpoint. The baseline
checkpoint loads; all three archived families pass forward/backward checks.
[Verification record](../records/selected-model-cleanup-checks.json).

The main research charter is unchanged. Historical evidence records remain
unchanged, including their original source hashes. Earlier audit records describe
the source and documentation at the time of each experiment; they are not claims
that the refactored working tree has those historical hashes. Git preserves exact
file bytes to avoid breaking those evidence links through newline conversion.
No new language-training claim is made by this cleanup.
