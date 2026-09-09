# Structured wide FFN controls

## Hypothesis and inspiration
Keep four times as many GELU features (or the full SwiGLU hidden width) as a
narrow FFN while spending the same number of matrix weights. Based on grouped
linear maps and channel shuffling. Related to
[structured FFNs](https://arxiv.org/abs/2406.16450) and
[Monarch](https://arxiv.org/abs/2204.00595); no novelty claim.

## Architecture
F(x)=D P GELU(Ux), or D P [SiLU(Ux) * Vx].
U,V,D are grouped block-diagonal maps; P interleaves hidden channels.
Every model uses the same external decoder, attention and training loop.
Our one-block-per-projection sandwich differs from replacing each linear map
with two shuffled block factors as in the cited BlockShuffle implementation.

## Parameters and compute
GELU: 2dh/G. SwiGLU: 3dh/G. With d=192,G=4,h=768 or 512:
73728 weights/block, 75% FFN reduction, 1672896 total model weights.
The matrix FLOPs equal the parameter-matched narrow controls; elementwise
activation work and memory do not. Fixed shuffles add no learned parameters.

## Initialization and risks
Group weights use .02*sqrt(G) to compensate for reduced fan-in, then the common
residual-output scale. This initialization choice must be ablated if gains arise.
Grouped nonlinear interactions are restricted; batched small matrices and
shuffle copies may be slow. Count the full live activation footprint.

## Results and verdict
REJECTED for promotion in the seed-17 200-step screen: grouped GELU NLL 4.22408,
grouped SwiGLU 4.64820; narrow GELU 4.24112 and narrow SwiGLU 4.41907.
Neither meets the frozen promotion gate. This does not reject more expressive
structured matrix products such as Monarch. See the
[protocol](../../../../second_screen_plan.md) and
[later report](../../../../second_screen_report.md).
