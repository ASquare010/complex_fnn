# CE-only latent fitting diagnostic, 2026-10-09

## Latest follow-up: encoder-target MSE only (completed)

User requested removing the decoder from training and trying MSE only. One
1000-update predictor run completed in `dump/latent-mse-only-v1`. Same fresh
initial checkpoint, same16training pairs,4.33M architecture, AdamW/weight decay,
and LR schedule as the earlier CE-only predictor diagnostic. Objective is mean
squared vector error divided per dimension by the saved training-set standard
deviation squared. Target/input vectors verified with the encoder-only export;
decoder loaded only after all1000updates for no-gradient evaluation. No CE term.

| Measurement | Result |
|---|---:|
| Fitting normalized MSE at100updates | 0.005187 |
| Fitting normalized MSE at200updates | 5.692e-8 |
| Fitting normalized MSE at1000updates | 2.481e-9 |
| Fitting decoded accuracy/exact | 100%,16/16chunks |
| Separate16development examples normalized MSE | 1.51529 |
| Separate16development decoded accuracy/exact | 0.0977%,0/16chunks |

This shows the existing predictor can memorize these mappings with direct
vector supervision, unlike the tested CE-only recipe. It supports an objective/
optimization issue rather than insufficient capacity for these16examples. It
does not establish generalization, chatbot quality, or superiority to a baseline.
The tiny development check was not optimized and is not an untouched final test;
encoder exposure to the underlying corpus remains possible. The development
error increased during fitting, consistent with overfitting this tiny sample.

Saved checkpoint replay reproduces fitting/development vector metrics exactly;
independent audit confirms1000 consecutive updates and checkpoint hash. Encoder
file unchanged; no encoder gradients. Training targets are supervision, not
prediction inputs. Existing CE trainer/defaults are unchanged. No larger-data
MSE run or combined-loss experiment was launched.

Earlier diagnostic evidence follows.

## Follow-up: free latent optimization (completed)

User authorized the proposed direct-vector check. One 1000-update run in
`dump/latent-direct-ce-v1` optimized 16 independent vectors (4096 scalar
parameters), initialized from the preceding fixed-16 predictor's outputs.
The transformer was removed from optimization; the codec stayed frozen.
Parameterization: z = initial + training_std * delta, delta initially zero.
Adam, no weight decay, 25-update warmup to0.03 then cosine to0.003; CE only.
This is a coordinate-specific diagnostic recipe, not a matched LR comparison
against transformer weight training or a general search over optimizers.

| Measurement | Before | After1000 |
|---|---:|---:|
| Decoded CE | 8.71375 | 8.48699 |
| Token accuracy | 23.1445% | 25.0977% |
| Exact chunks | 0/16 | 0/16 |
| Normalized MSE to encoder targets (metric only) | 1.64025 | 1.68450 |

Vector coordinates moved by0.1772 RMS in training-normalized units. Gradients
were finite and nonzero, codec tensors remained exactly unchanged, and saved
vector reload reproduced all final metrics exactly. Independent log audit
verified1000 consecutive updates and the checkpoint hash. Correct encoder
vectors still supply the known successful solution; they were not initialization
or loss targets. Only original token IDs supplied the CE supervision.

The fitted initialization did not converge to the correct text even when every
example had free, independent vector coordinates. This strengthens evidence
for difficulty optimizing CE through this frozen decoder, and weakens the case
that predictor parameter count alone explains the failure. It does not prove
all CE-only initializations/optimizers fail or establish a unique root cause.
This is per-example optimization with known answers, not a generative model or
held-out quality result. No larger model, MSE, codec unfreezing or extra updates
were run. A later explicitly authorized target-vector-supervised comparison
could test whether direct guidance toward known decodable vectors helps.

The original predictor diagnostic follows.

User authorized running/checking the proposed 16-example fitting test. Completed
one 1000-update run of the same 4,327,680-parameter predictor, restored from the
batch32 run's initial model (fresh weights and its training-only normalization).
Frozen encoder/decoder, CE only, original 50-step warmup/cosine LR and AdamW.
Each update uses the same 16 distinct training paragraphs' first chunk to predict
the second independent 64-token chunk. This is a memorization diagnostic with
one input vector per example, not a validation or long-context experiment.

## Results

| Input to decoder | CE | Token accuracy | Exact chunks |
|---|---:|---:|---:|
| Correct target encoder vectors | 0.00003175 | 100% | 16/16 |
| Fresh predictor | 50.0154 | 0% | 0/16 |
| Existing batch32 predictor on these pairs | 16.1978 | 0.0977% | 0/16 |
| Predictor fitted for 1000 updates on these pairs | 8.7138 | 23.1445% | 0/16 |
| Fitted predictor with shuffled input chunks | 19.1993 | 0.3906% | 0/16 |
| Fitted predictor with zero input vectors | 49.4962 | 0.1953% | 0/16 |

Gradients at inspected steps1/25/250/1000 are finite, with nonzero gradients to
predicted vectors and trainable parameters. The frozen codec had no parameter
gradients and all its state tensors remained exactly unchanged. A single-position
input does not exercise attention selection across multiple past vectors; zero
query/key gradients in that case do not imply a broken attention implementation.

The real target-vector mean norm is132.07; fitted predictions reach133.20, so the
final failure is not explained by vector magnitude alone. Normalized vector MSE
increases from1.1452 to1.6402 despite CE improving. MSE is a diagnostic metric only;
it was never optimized. Low CE need not require the unique original encoder code.

Straight-line interpolation from fitted predictions toward exact target vectors
reveals a non-monotonic decoder objective. At interpolation fractions0/0.1/0.5/1,
CE is8.71/12.47/25.94/0.00003175. This is direct evidence of a loss barrier along
that particular path, not proof of a local minimum or absence of another path.
The larger-corpus predictor also gives very similar outputs across these inputs
(mean per-dimension across-example std0.342 versus7.901 for target vectors),
consistent with the repetitive samples. The tiny fitting run becomes more
input-dependent, but still does not solve even the fixed training examples.

## Interpretation and next decision

Verified: the codec can decode the target representations, target ordering is
correct, gradients pass through the frozen decoder, and the predictor can learn
some dependence on context. Not established: a specific implementation bug,
insufficient parameter count, or that a larger model/longer training would solve
the problem. The CE-only frozen-decoder optimization interface needs investigation
before treating this as a model-capacity limit. No MSE was added, no codec was
unfrozen, no larger model or further training was launched.

If further diagnostics are authorized, optimizing a small set of free latent
variables with the same CE (without a predictor) could isolate decoder optimization
from predictor capacity. This is a proposed next test, not completed evidence.

## Preservation and verification

Evidence: dump/latent-diagnostic-v2/{receipt,baselines,history,gradients,result,
verification}.json, last.pt and source.zip. Independent reload reproduced CE and
accuracy exactly; checkpoint/prepared hashes and optimizer step1000 verified.
No original checkpoints or training sources were changed.

The first attempt stopped before optimization on an overly strict bitwise check
after changing encoder batch shape. The measured discrepancy was7.629e-6; all16
original paragraph batches re-encoded bitwise exactly. Original attempt retained
in dump/latent-diagnostic-v1, including batch_check.json. V2 checks exact identity
using the original batching. A read-only diagnostic command also had a quoting
syntax error before execution, then was replaced by a saved script. No native
failure, dataset replacement or environment change occurred.
