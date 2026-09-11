# H148: real autograd reconstruction resource screen

Previous turn: progress. H147 passed inverse/VJP checks but measured no optimized
training. Implement its reverse behind a custom first-order autograd boundary.
Use eight scalar/orthogonal layers, d384, rho2/8; per-channel shape and bias.
Fixed dense orthogonal matrices are registered buffers and fully counted.
Reconstruction saves final output, theta, bias and Q; no interior activations.
No higher derivatives, compile, autocast or stochastic-module support is claimed.

Arms: full GELU and SwiGLU residual stacks, each eager and checkpointed;
budget GELU checkpointed; reversible learned shape eager/checkpoint/reconstruct;
fixed-shape reconstruction; affine reconstruction. All use depth8/d384.
The reversible learned and affine models have6144 parameters; fixed shape3072.
Shape controls theta are uniform[-1.5,1.5], biases normal sd.1; CPU FP64 QR
matrices cast to FP32. The fixed and learned scalar families start identically;
affine replaces phi(z) by (1+rho*tanh(theta))*z. Matrices are fixed throughout.
This mechanism has prior art (RevNets/flows); no novelty or quality claim.

Preflight: CPU FP64 gradcheck input/theta/bias for nonlinear and affine custom
backward at nonzero shape; eager/checkpoint/reconstruct gradients must agree at
rtol1e-10/atol1e-12. Trace unique saved storage on CPU at batch2048/d384,
classifying model parameters, fixed buffers, original input and other tensors.
Hooks are outside GPU runs. Count source dependencies and preserve H147 evidence.

Frozen screen: batches128/2048; fresh seeds173/191/211; 10 arms;12 updates each,
60 runs,720 backward/optimizer updates,61 clean GPU boundaries. Reuse H146's
measurement loop verbatim except model import, output root and diagnostic flag.
Same Gaussian/orthogonal target construction, LR.003, betas(.9,.95), decay0,
clip1, FP32/TF32off, ordinary default workspace, four CPU threads, upstream input
gradients. Four warmups, eight timed complete updates; gradient reset outside.
Rotate ten-arm order. Save states, raw timing and first/final gradient diagnostics.

Primary candidate learned:reconstruct must meet <=.8 parameters, <=.9 allocated
peak, <=1.15 CUDA/wall median against BOTH checkpointed full controls in EVERY
fixture. Candidate and both control half-window stability must also be<=1.15.
Show eager full controls, budget GELU, learned eager/checkpoint, fixed-shape and
affine controls. This gate earns only a separate quality study. No retries/tuning.
No loss-based selection: twelve updates on one fixed batch are diagnostic only.

Independent audit: saved FP32 states explicitly scored in CPU FP64 with batch257,
relative tolerance1e-5, without model.forward/custom backward. Regenerate all six
datasets, validate counts INCLUDING fixed buffers, optimizer counters12, finite
states, timing summaries, gradients and cleanup. Additionally compare learned
execution modes' final states and fixed-batch losses descriptively; floating
reconstruction is not promised to reproduce the same training trajectory.
Audit performs no optimizer/backward/GPU work. No maintained defaults change.
