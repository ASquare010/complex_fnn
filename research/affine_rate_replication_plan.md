# H044: replicate the stronger same-rate plain control

Frozen after both H043 cells finish, before either new seed is trained.
At seed17 and LR0.0012, plain NLL4.750900 nearly matches affine4.749982
(0.0193% difference). At LR0.0006 affine helps0.337%, but that pair is worse
than either higher-rate recipe. The original three-seed selected-rate result
is therefore insufficient evidence of a material activation-specific gain.

## Fixed comparison and cost

Train only plain BlockShuffle at global peak LR0.0012, seeds29 then43, 800steps.
Reuse H043 seed17 and all three original H041 affine LR0.0012 checkpoints.
Each new configuration equals that seed's original plain H039/H041 configuration
except learning_rate. Native SiLU/product recomputation stays native; no shape
parameters are added. Do not retune, extend the rate grid or change execution.

Keep width384, layers8, heads6, context128, batch16, frozen WikiText and tokenizer,
BF16, optimizer scales/decay/clipping, initial weights, warmup/cosine schedule,
full322,688-target validation at initialization and steps1/200/400/600/800.
Each new trial uses1,638,400 sampled tokens. Total new budget3,276,800 tokens.
Two sequential fresh GPU workers, 2400-second deadline each. Complete both
unless infrastructure/numerical failure; retain every outcome and failure.

Verify the original full/narrow/plain/affine references at each seed and their
source archives. Verify new critical computation hashes, model/config/counts,
initial NLL, optimizer metadata, histories and finite diagnostics. Require
sampling RNG equality with that seed's original affine and plain checkpoints.
No official test download or scoring, no source checkpoint overwrites.

## Report and interpretation

Report every seed and mean +/- sample SD for plain and affine at LR0.0012,
relative mean NLL change and paired differences. Reuse the exploratory t interval
and exact one-sided sign test with their H041 limitations; seed17 selected the
new rate control. Report the original parameter/quality/memory gates for the
stronger plain recipe against each seed's full SwiGLU, full GELU and calibrated
narrow controls. These full controls already use LR0.0012.

A material activation benefit in this comparison requires >=0.2% lower mean
NLL AND three strict paired wins. Failure retains affine as an implemented
expressive alternative, but does not promote its TinyStories transfer or new
gain/bias variants on the basis of the now-confounded selected-rate advantage.
Three plain gate passes establish the stronger local control, not a new primitive.
Serial training speed is descriptive; no paired speed or convergence claim.

The two recomputation/backward implementations still differ, so even a surviving
same-rate difference is not an isolated activation-learning effect. H042 removal
and any future inference audit remain valid checkpoint interventions without
proving the training mechanism. This corrective comparison takes precedence
over further activation search; preserve all earlier frozen decisions as historical.
