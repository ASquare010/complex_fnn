# Additional unscored validation-tail check

Frozen before any model NLL is measured on this tail. Earlier training,
selection and serving used only the first256 nonoverlapping context128 windows
of the200803-token TinyStories validation cache:32768 targets. The same cache
contains1568 complete windows. The remaining1312 windows provide167936 targets
that have not been scored by the retained LM protocols. Data preparation has
already deduplicated the full validation set against training; this is not a
new corpus and does not satisfy the broader-corpus requirement.

After the locked seed29/43 cohort finishes, evaluate all twelve width384/layer8
800-step checkpoints, with no further training or weight edits. Use native
CUDA BF16 and the existing forward loss evaluator. Windows start at index256
and end before1568, giving82 batches of16 windows at context128. Their target
positions are32769 through200704 inclusive; the input at32768 is the prior
prefix's final target, but tail targets do not overlap scored prefix targets.
The last98 validation tokens have insufficient context-window length and are
omitted, as under the existing fixed-window convention.

First reproduce the prefix NLL for each checkpoint within .0002 absolute.
Then report prefix and tail NLL separately, plus the target-weighted aggregate.
Keep all four recipes and three seeds. Report whether the original <=1% NLL
limits versus both full references and the advantage over calibrated narrow
hold on the tail; never choose different windows or thresholds after seeing it.
No speed claim follows from this evaluation. Preserve checkpoint/source/data
hashes, every per-window loss and the exact target ranges so disjointness and
weighted aggregation can be verified. Compare paired seeds, report means and
sample SD, and retain any reversal of the prefix ranking.

This is an additional check on repeatedly used selection data, not an
independent dataset, a convergence experiment, a new training result or a
complete statistical remedy for earlier model selection. Once measured, this
tail is no longer unscored and must not be presented as fresh evidence for
future choices. Broader-corpus and equal-tuning/convergence studies remain.
