# BlockShuffle SwiGLU

**Main compressed reference; the final quality target remains unmet.** Each dense
projection is replaced by two block-diagonal factors and fixed channel shuffles.
The FFN remains `down(SiLU(up(x)) * gate(x))`.

At width 384, hidden width 2048, eight groups and eight layers:

| Quantity | BlockShuffle | Full SwiGLU |
|---|---:|---:|
| FFN weights | 2,801,664 | 9,437,184 |
| Total model weights | 9,099,648 | 15,735,168 |
| FFN weight reduction | 70.3125% | 0% |

The three-seed WikiText mean NLL is 4.759029 at 800 steps, but 4.147443 at 3,200
steps is 1.256% above full SwiGLU. The longer result fails the target. Read the
[model notes](model.md) and [current state](../../research/CURRENT_STATE.md).

Use [the retained WikiText recipe](../../configs/wikitext2_blockshuffle_stronger_800.json).
Factor-specific learning rates and native gate recomputation are explicit parts
of this recipe. Fewer parameters do not guarantee lower wall time or memory.

The optional [Triton inference implementation](triton_inference.py) retains the
measured TinyStories four-kernel path. Its existing 1.206x compiled/full speed
result applies to one fixed full-sequence inference setup. It supports plain
BlockShuffle only; rational activation needs its own faithful execution path.
