# H105: projected real FFNs

**All three energy projection recipes are rejected.** No optimizer updates or
language-model promotion follow. [Results](../../../research/real_subspace_results.md)
include all 378 comparisons and a post-screen output-rank obstruction.

`model.py` estimates seven training-only input subspaces and folds input/output
projection into small affine factors. It retains the original GELU/SwiGLU hidden
width. `study.py` fixes two pretrained teachers, three layers, three calibration
seeds and ranks 32/64/128. `qualification.py` checks exact folding, input gradients,
rank-full identity, the linear-response optimum and real capture parity.
These files, the plan and shared capture helper remain frozen to protocol hashes.

The independent `audit.py` recaptures all tokens through the complete decoder,
recomputes statistics in chunks and evaluates explicit factors with a different
batch partition. It does not call the candidate's forward or estimator.
`diagnose.py` computes an empirical output-rank floor after the screen;
`audit_rank.py` checks its covariance calculation against direct rectangular SVD.
`analyze.py` writes all gates/tables, while `plot.py` runs separately without Torch.

The screen/audits use the existing UV-managed Python 3.12.9 runtime and project
packages. Its environment is recorded. All source, compact results and logs are
retained; tensor pairs, covariance/eigenvector statistics and teacher checkpoints
remain local. Restore `result.json` from its lossless gzip before historical
readers in a fresh clone, and rerun in an empty output directory to reconstruct
raw artifacts. Existing completed or partial protocols are never overwritten.

The rank-64 output class has error floors above the frozen 5% threshold on all
18 captured datasets. This explains why refining its input initializer cannot
make the fixed projection recipe pass. The energy candidates also lose to
simpler methods independently of that absolute gate. Structured full-rank maps
and other memory-reduction mechanisms remain separate research questions.
