# Reading Curve-Wide

Start in [transformer.py](../src/models/channel_curve_transformer/transformer.py).
Read `CurveFFN.forward`, then `Block` and `Model`, all in that one file.
Like the dense baseline, it reuses attention and normalization from components.
There is one selected FFN architecture.

```text
token IDs -> embedding -> four Transformer blocks -> next-token scores
                         each block:
                         normalize -> attention -> add to input
                         normalize -> CurveFFN  -> add to input

CurveFFN: 512 features -> learned linear mix -> learned curves -> linear mix
```

A tensor is an array of numbers. Its shape [batch, tokens, width] says how many
sequences, token positions and feature coordinates it holds. A feature is a
current signal; a parameter is a learned setting. `nn.Linear` learns weighted
combinations. `nn.Parameter` registers numbers for training. `forward` computes
outputs. Autograd computes gradients; an optimizer uses them to update parameters.

For each feature z, three branches compute `SiLU(a*z+b) * (c*z+e)` and add their
responses. SiLU is `u*sigmoid(u)`. The arrays a/b change the activation's input;
c/e change the multiplying factor. These are not probabilities or named concepts.
The feature mixing happens in the surrounding matrices. The initial mean/scale
calibration is fixed, not learned; retain it when comparing against saved weights.

The forward method directly expresses the whole FFN in ordinary PyTorch:

```python
z = self.up(x)              # Learn combinations of features.
# Three learned responses are added in the loop.
# Apply the fixed initial mean and scale.
return self.down(value)     # Mix the transformed features.
```

Both CPU and GPU use this forward method and automatic gradients. There is no
custom CUDA compilation or handwritten backward function in the active code.
The dtype lines keep curve arithmetic in FP32 during mixed-precision training,
then return to the projection's dtype. Parameter names, initialization and
checkpoint structure remain unchanged.

`calibration()` at the bottom computes a fixed starting mean and scale once
when the model is constructed. It averages the initial response over weighted
points from a normal distribution. This is initialization mathematics, not a
GPU optimization. Keeping it preserves the selected recipe and saved weights.

Equal equations preserve the intended model; execution can change numerical
rounding, memory and speed. Output, gradient and saved-checkpoint checks passed.
Normal autograd retains more intermediate tensors during training. The short
GPU check measured 553 MiB for plain Curve-Wide, 401 MiB for archived optimized
Curve-Wide and 561 MiB for dense. Plain Curve-Wide also ran slower. The parameter
saving remains; the historical 28% training-memory saving does not apply to this
version. See the [execution report](../src/models/channel_curve_transformer/result.md).
