# Research rules

1. Evidence wins. Never fabricate measurements, assert unverified novelty, or call
   one seed a breakthrough. Distinguish structural proofs from empirical claims.
2. Keep shared data, tokenizer, attention, training, evaluation and timing in
   src/core. Candidate folders contain architecture and readable model notes only.
3. Hold data order, token budget, model dimensions outside the FFN, optimizer,
   precision and tuning effort fixed. Compare full, narrow, parameter-matched and
   compute-matched controls as appropriate. Report every changed variable.
4. Use uv, PyTorch and CUDA. Prefer BF16 when supported. Establish eager correctness
   before compile or custom kernels. One bounded GPU experiment at a time.
5. Record source snapshot/hash, git state, configuration, seed, data/tokenizer hashes,
   software/hardware, checkpoint, metrics and failure information. Never hide runs.
6. Monitor pre-clip gradient norms, clipping frequency, per-layer activations and
   gradients, saturation, finite values, throughput and allocated peak VRAM.
7. Time after warmup with CUDA synchronization. Label full-sequence forward timing
   separately from autoregressive generation. Fewer parameters do not imply faster.
8. Promote via cheap screens; useful ideas must earn larger budgets and >=3 seeds.
   Check scale and a broader corpus before declaring success.
9. Explain hypotheses mathematically, test simpler mechanisms and perform ablations.
   Simplify or eliminate weak branches. Do not build architecture soup.
10. Update the candidate ledger and current state after each round. Preserve useful
    components of failures. Keep documentation clear and the original structure
    wherever practical. No automatic publication or external messaging is implied.

States: IDEA, HYPOTHESIS, IMPLEMENTING, EXPERIMENTING, PROMISING,
NEEDS_INVESTIGATION, ELIMINATED, VALIDATED, BREAKTHROUGH.
Model verdicts: ACCEPTED, PROMISING, INCONCLUSIVE, REJECTED.
