# Group-product v2: a smaller bank before language selection

Registered after v1's terminal hardware screen and before v2 profiling or quality.
V1 at hidden 512 had a 433.0 MiB peak: 19.73% below full GELU, narrowly missing
the unchanged 20% gate. Its throughput stayed within the allowed deterioration.
Retire that exact size; its language quality is unknown. Do not round 19.73% to
a pass. Raw evidence is `records/group-product-v1-profiles.json`.

Use the same [group-product equation and hypothesis](group_product_hypothesis.md)
with hidden 480 and 30 groups, still 16 features per group. Width 512, four layers,
16 heads, context 256, vocabulary 4096, batch eight and BF16 stay fixed. Feature
capacity is smaller and is not assumed to preserve v1's unknown quality.

FFN parameters: 1,968,000, 76.72% fewer than full SwiGLU's 8,454,144. Projection
FLOPs: 3,932,160/token, 76.74% fewer, excluding group and pointwise operations.
Compact SwiGLU hidden 320 and compact GELU hidden 480 both have 1,966,080 FFN
weights, 0.10% fewer than the candidate. Plain SiLU hidden 480 has the same count
as these compact references. Square control hidden 480 has exactly the candidate's
count. Full SwiGLU/GELU hidden widths remain 1376/2064.

Repeat the numerical checks at the new grouping and hardware screen on both
corpora, three alternating-order rounds, 20 warmup plus 100 measured updates.
Use median throughput, largest allocated peak, the unchanged >=20% memory OR
>=1.2x speed gate against both full controls, <=5% deterioration in the other
resource and <6 GiB reserved. If it fails, retire before language. There is no
further size selection under this study ID.

If it passes, integrate an equivalent full Transformer in production and repeat
Trainer exact resume. Train group products, square, plain SiLU, compact fused
SwiGLU, compact GELU, full fused SwiGLU and full GELU on both corpora: seed 101,
2,000 updates, learning rate 0.0006, warmup 200, AdamW decay 0.1. Apply the same
full 1% loss-cost and compact 1% gain thresholds, controls, inference removals and
confirmation requirements as v1. Test loss is not used. This is hardware-driven
size selection before quality, not a relaxed acceptance contract. Originality
remains unverified and the existing prior-art caveats still apply.
