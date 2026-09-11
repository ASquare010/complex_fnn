# H124: compact analytical RMSNorm backward

Previous goal turn: progress. H123's gradient staging saved 9.46–9.98%, below
its fixed 10% gate. Test reducing normalization intermediates, independently
and combined with that staging. Prior failures remain failed.

For r=(mean(x^2)+epsilon)^(-1/2), y=w*x*r and incoming gradient g:
  dw=sum_over_tokens(g*x*r)
  dx=r*(g*w) - x*r^3*mean(g*w*x).
Save only x, w and r. Compute dw first, then reuse an owned dx scratch buffer
with addcmul_ to avoid materializing the final correction. Never modify saved
inputs or upstream gradients. FP32 execution only; FP64 mathematical checks.
No higher-order differentiation. In exact arithmetic this is the existing
RMSNorm, not a new normalization function. Its Jacobian norm is bounded by
max(abs(w))/sqrt(epsilon): the unweighted normalization Jacobian has tangent
eigenvalue r and radial eigenvalue epsilon*r^3, both <=1/sqrt(epsilon).
This layerwise bound does not guarantee stable gradients across a network.

Reuse H123's run_case unchanged through a temporary constructor adapter and
separate result roots. Both corpora, chunked FP32 loss, checkpoint-input CPU
offload and four arms: ordinary/compact norm x resident/staged gradients.
Ten backwards per case, three warmup; all gradient restoration costs included.
Reverse arm order across corpora. No new trainer or maintained-code changes.

Budget: three FP64 shapes x reference/custom =6 backwards; two joint finite
 differences per shape =12 extra loss forwards. Then eight cases x10=80
backwards. Total86 backwards, zero updates, 327,680 diagnostic targets, eight
gradient artifacts. One GPU process using the existing UV runtime.

Require toy gradient atol/rtol1e-10 and finite-difference error <=1e-7 times
max(1,abs(analytical)). Include zero inputs. Stop full cases if qualification
fails. Full-model gradients: symmetric global L2 <=1e-5, max tensor <=1e-4,
loss relative error <=1e-6 against H121 audited references; unchanged model,
optimizer moments and sampler; exact gradient restoration for staged arms;
finite saved gradients and zero GPU boundaries. Source hashes remain fixed.

For each candidate, compare BOTH corpora against ordinary norm/resident
 gradients in this same study. Require full diagnostic-case allocated peak
<=90% baseline, median event-sum and synchronized wall <=1.15x baseline, and
pinned-host allocated peak <=128 MiB. Report all four arms, including whether
compact norm adds value within each gradient-storage policy. Include every
construction, warmup, backward, restore, serialization and diagnostic phase.
Report mean/median/variance; no sustained-throughput or language-quality claim.
Only passing both corpora earns a separate complete-training study. No old
failure is overwritten; no completed case is repeated.

Prior art: RMSNorm https://arxiv.org/abs/1910.07467 and standard analytical
normalization backward. Novelty is not claimed for this storage implementation.
