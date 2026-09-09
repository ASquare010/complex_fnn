# H043: complete the two-by-two learning-rate comparison

Frozen after H041/H042, before either missing 800-step rate cell is trained.
The selected WikiText recipes use different global peak rates: plain BlockShuffle
0.0006, affine BlockShuffle 0.0012. Their three-seed result compares selected
recipes; it cannot isolate the activation from learning rate. Short-screen
selection can reverse with training duration. Test this explanation before
additional mechanism variants or corpus transfer.

## Fixed experiment

Reuse the two original seed-17 800-step checkpoints at their selected rates.
Train precisely two missing cells from initialization, in this order:
plain BlockShuffle at 0.0012, affine BlockShuffle at 0.0006. Use each recipe's
original H039 configuration with ONLY learning_rate changed. Keep width384,
eight layers, batch16, context128, seed17, native BF16, 800 optimizer steps,
1,638,400 sampled tokens per new trial, frozen WikiText/tokenizer and complete
322,688-target validation at initialization and steps1/200/400/600/800.
Gate recomputation remains native for plain, checkpoint for affine. This existing
execution difference remains disclosed; H036/H040 showed that tiny numerical
differences can affect optimization. No causal claim from this two-by-two alone.

Fresh sequential GPU workers, 2400-second deadline each. No changed architecture,
optimizer group scales, decay, initialization, clipping, attention or sampling.
Verify all critical computation hashes against H039/H041, initial NLL and all
common optimizer group metadata against each original recipe. Require final
sampling RNG equality across all four cells. Archive source/config/data hashes,
checkpoints, histories, finite diagnostics, memory and all results. Reuse all
three full/narrow seed-17 controls without retuning. Official test stays unscored.
New budget: two trials, 3,276,800 sampled tokens; no adaptive rate extension.

## Decision

Report affine minus plain NLL separately at EACH rate, both in NLL units and
relative percent. A material same-rate benefit requires >=0.2% lower NLL.
Only if both comparisons pass may we describe the benefit as robust across
these two tested rates. Report the original quality/parameter/memory gates for
every cell against full SwiGLU, full GELU and calibrated narrow. Any inference
about the newly compared rate cells remains one-seed exploratory evidence.

A failure at either rate qualifies H041's recipe-level conclusion; it does not
erase that result. If affine's gain disappears at the higher rate, investigate
rate/execution controls before attributing the gain to activation shape or
expanding gain/bias variants. TinyStories transfer and deployment execution
remain separate outstanding experiments; neither is answered here.
