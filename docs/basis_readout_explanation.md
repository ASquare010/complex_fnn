# What the basis-readout experiment actually changes

The input projection learns 512 combinations of the residual features. For each
combination, compute six different scalar responses: three shifted smooth
activations, and those same responses multiplied by their input coordinate.
The group readout learns how much each response contributes to each of 32 group
outputs. The final projection mixes those outputs globally.

Some artifact keys call the unmultiplied SiLU responses the "linear" branch.
That is bookkeeping terminology: SiLU is nonlinear, so this is not an affine
path through the FFN. The later activation-map diagnostic measures a different
object: the best affine approximation of a trained full FFN's entire map.

The preceding self-curve recipe sums three responses before its output matrix.
That makes their output vectors proportional for a given input coordinate.
Here the group readout can assign different directions to the six responses.
The tied control learns their coefficients but still sends them through one
direction per coordinate. Comparing those two asks whether independent response
directions help, with equally sized dense controls checking whether the extra
readout weights are simply better spent in an ordinary FFN.

There is no extra nonlinear depth between B and Q: both are linear, and their
composition is a structured map from the expanded response bank to the residual
output. In a group, all 192 responses still use at most 32 intermediate output
directions. Different responses do not equal independent information, and a
larger bank does not recover input information that its learned projections
failed to preserve. The six scalar basis functions are closely related.

The economical allocation is therefore the hypothesis, not a proven new neuron
type. Fixed shapes do not mean the FFN cannot learn its transformation: P, B
and Q are learned. Letting slopes/offsets train was separately tested for
resources and failed this implementation's slowdown limit; its language quality
is unknown. Fixed-template language quality is still being measured.

Only FFNs change. Each model remains a complete Transformer with embeddings,
attention, normalization, residual connections and a tied output head. Those
components stay identical across comparisons. A loss improvement would mean
better predictions under the recorded protocol, not a direct measurement of
semantic information or proof that all inputs are retained.
