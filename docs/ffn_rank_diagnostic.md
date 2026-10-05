# Exploratory diagnostic: where the full FFN stores its weight energy

Registered on 2026-10-04 before reading spectra. No training or new validation
measurement is involved. The signed-context language campaign continues unchanged.

Use the completed full native SwiGLU controls from context-bank v1 on TinyStories
and WikiText, seed 101, 2,000 updates. Verify their archived dense implementation
and initialization components match current code. Compare trained matrices to
the common named initial weights. Run on CPU with one thread; do not use the GPU
or modify any frozen training source.

For each layer, form `W_up.T @ W_up` for the fused detector/gate input matrix
and `W_down @ W_down.T` for its readout. Compute FP64 eigenvalues and the fraction
of squared singular-value energy retained by ranks 16, 32, 64, 128 and 256. Also
report the corresponding optimal relative Frobenius approximation error.
Record checkpoint/config/source provenance and every layer, not only favorable ones.

This asks whether the learned input/readout matrices concentrate more energy
than their initial random matrices. It can motivate a future allocation of
linear versus nonlinear capacity. It does not measure activation covariance,
retained semantic information, language loss under compression, or the ability
of a model trained from scratch to recover quality. Low-energy directions may
matter. No rank is qualified or selected solely from this diagnostic, and no
trained weights may be reused without accounting for their training budget.

The size, quality, resource and confirmation gates are unchanged. A factorized
FFN would need a separately registered controlled experiment before training.
