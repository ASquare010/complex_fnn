# H063: native recomputation for the retained rational activation

Frozen before implementation and resource measurement. H062 is completed progress:
the rejected additive recipe is archived. This tests an execution change to the
retained models, not a new activation or architecture.

## Mechanism and prior art

Current rational training checkpoints only the activation product. Test two wider
native checkpoint boundaries: the whole FFN after its input norm (ffn), and the
whole decoder block including attention and both residual additions (block).
Keep each recipe's existing inner gate recomputation unchanged. Use PyTorch
non-reentrant checkpointing, preserve_rng_state=True, default early stopping;
no compilation, offload, altered precision, chunking, or arithmetic rewrite.
Evaluation and no-grad calls bypass the new recomputation. The default is none.

Checkpointing trades saved activations for repeated forward work; this is known
practice, not novelty. [PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/checkpoint.html)
describes non-reentrant recomputation and its constraints. See also
[Korthikanti et al.](https://arxiv.org/abs/2205.05198) for the memory/compute tradeoff
in Transformer training. Their distributed results are not predictions for this
single-GPU setup. Real arithmetic is unchanged, but numerical equivalence must
be measured; the earlier compiler repeats demonstrate why local checks are weak.

## Staged allocation

First preserve current source and verify the H062 completion record. Add one
shared opt-in recomputation setting; retain six variants and all parameter names.
Test all six variants on CPU FP32 and CUDA BF16 at seeds 17/29/43 with nonzero
rational coefficients: logits/loss, every gradient, groups and two clipped AdamW
updates must match none exactly for both scopes. Test evaluation bypass, invalid
scope rejection and parameter/checkpoint identity. Recheck all twelve H059 default
CPU/GPU signatures. No resource dispatch before the local/full test suite passes.

Then run exactly 15 fresh isolated GPU workers: five selected seed-17 200-step
checkpoints (full SwiGLU, full GELU, calibrated narrow, plain BlockShuffle and native
rational) times scopes none/ffn/block. Pin every input checkpoint, metrics and source
archive hash before dispatch. The controls are H061's selected references; rational
is wikitext2_blockshuffle_rational_lr1200_s17_200. Model width 384, layers 8,
context 128, vocabulary 4096, batch 16. Use each original optimizer policy and
loaded moments, with its selected peak LR held constant for this synthetic test.

Each worker first records one full-size synthetic forward/backward without an
optimizer update, then clears gradients. Record all parameter-gradient hashes,
loss and logits, plus original weight and optimizer hashes. It then performs
20 synthetic updates; first ten warm up, final ten are synchronized and timed.
All workers use the identical CPU-generated token tensor of shape [20,16,129],
seed 60017, and transfer the corresponding next-token batch to CUDA per step.
All updates use BF16 autocast, FP32 parameters, clip 1, AdamW and four CPU threads.
Record every loss/norm, allocated/reserved peak, time samples, final weights and
moments, configuration, source, environment, RNG and token hashes. Save the
synthetic continuation checkpoint at update 220, clearly outside the LM leaderboard.

This allocates 300 synthetic updates / 614,400 target exposures plus 15 no-update
2,048-target compatibility probes. It scores no corpus validation or test targets,
fetches no data and does not rerun any completed language cell. Rotate scope order
by recipe index; run recipes in the order listed above, one worker at a time.
All five recipes receive equivalent checkpoint options, as required by the user.

## Gates and decision

All allocated workers must finish, remain finite, and preserve original checkpoints.
For every recipe, both new scopes must match none exactly for initial logits/loss,
every initial gradient, all 20 losses/norms, and final weights plus optimizer moments.
RNG and synthetic token hashes must match. No relaxed tolerances after observing data.

A rational scope earns a separate full native-training repeat only if it also:

- preserves at least 70% FFN-weight reduction;
- reduces its own native allocated peak by at least 10%;
- stays within 110% of BOTH full controls using the SAME scope.

Comparisons to uncheckpointed dense controls are diagnostics, not promotion gates.
Report narrow and plain controls under the same policy too. If both scopes pass,
choose ffn as the smaller recomputation boundary. Report timing overhead and all
scope results; no speedup claim from fewer stored tensors or these short windows.
A pass is execution qualification, not LM quality, convergence or novelty.

If both fail, close this particular native-boundary repair. Preserve evidence;
no automatic offload, smaller batch, new activation or further checkpoint search.
A future full training repeat requires its own frozen allocation and must compare
the complete saved native trajectory and final weights/moments before promotion.
