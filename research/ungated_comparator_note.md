# Comparator gaps alongside H077

Checked 2026-09-08 while the frozen H077 trials run. This read-only prior-art
review changes no recipe, rate, gate or training allocation. Paper results below
are author-reported and have not been reproduced in this repository.

| Primary source | Relevant evidence | Consequence for this project |
|---|---|---|
| [StructuredFFN, official project](https://claire-labo.github.io/StructuredFFN/) | Identifies BlockShuffle with two block-diagonal factors and a shuffle; also studies LowRank, BlockDense and dense-to-structured self-guided training | Our GELU/operator combination does not establish novelty. Self-guidance adds a temporary dense branch, requiring separate training-parameter and resource accounting. Published wide models also change attention via GQA. |
| [BLAST, v1, sections 2 and 4.1](https://arxiv.org/html/2410.21262v1) | Shares row/column block bases with block-specific diagonal couplings. Includes GPT-2 training from scratch on WikiText-103 for 100 epochs, separately from pretrained-model compression | BLAST is a relevant from-scratch comparator, not only an inference compressor. Its reported perplexity/FLOP tradeoff does not prove our fixed-attention, 70%-FFN target. |
| [Continuous structured-matrix search, v1, sections 5-6](https://arxiv.org/html/2410.02117v1) | Studies structured operators, structure-dependent initialization/LR and BTT-MoE routing within individual projections, including attention | Total expert weights and active compute must be counted separately. Its reported compute savings cannot be treated as 70% fewer learned FFN weights or as an FFN-only reproduction. |

## A count-compatible conventional control, not an allocation

For a rectangular BLAST projection with output m, input n, b row/column groups
and rank parameter r, count the factor shapes directly:

- Row bases: b * (m/b) * r = mr.
- Column bases: b * (n/b) * r = nr.
- Diagonal couplings: b * b * r.

Thus P = r(m + n + b^2). This is our rectangular count derived from the published
factorization, without biases. A two-projection GELU FFN has 2r(d + h + b^2)
weights. At d384, h3200, b8 and r48, each FFN has **350,208** weights; eight layers
have **2,801,664**, exactly matching H077's h3264 BlockShuffle GELU and both narrow
controls. The wider h3264 choice with r48 would instead use 2,850,816 weights and
miss the 70% reduction threshold. [BLAST factor definition](https://arxiv.org/html/2410.21262v1#S2)

This arithmetic qualifies only a possible budget. It does not qualify an
implementation, initialization, optimizer, gradient behavior, runtime or learning
advantage. No BLAST prototype or worker is created. A future comparison would
need its own frozen qualification and equal tuning budget. The paper's
matrix-factorization descent result is not an AdamW Transformer-training theorem.

A passing H077 replication would still leave duration/convergence, scale, broader
data and strong published comparators unresolved. It would not support a claim
of superiority over current state of the art.
