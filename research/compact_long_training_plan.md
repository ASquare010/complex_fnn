# H126: long training from original initialization

Previous turn: progress. H125 passed short complete-update validation. Test
whether the result survives early optimization and the historical WikiText
native-versus-chunked quality problem before allocating more seeds.

Reuse H117's unchanged 800-step run_case and H125's qualified adapter. Two
original seed101 step0 checkpoints, WikiText/TinyStories, three arms each:
- native: ordinary RMSNorm, ordinary whole-block checkpointing, FP32 native loss;
- chunks: ordinary RMSNorm, checkpoint-input CPU offload, FP32 chunked loss;
- combined: compact RMSNorm plus gradient staging/restoration, checkpoint-input
  offload, FP32 chunked loss; ordinary RMSNorm for BF16 evaluation.

All model weights, initial optimizer state, initialization, batches, learning
rate, clipping and AdamW settings are matched. Native control has no host
storage adapter. Reverse arm order across corpora. Same UV runtime, FP32,
TF32 disabled, four CPU threads, one GPU process. No new candidate trainer.

Budget: six runs x800 =4800 updates, 19,660,800 training targets; six initial
probe backwards plus six independent replay backwards =4812 backwards total.
24 full study validation scores; 20 independent native audit scores (two
initializations plus three checkpoints per run). Regenerate two initial states
exactly and reconstruct all4800 training batches. Save24 tensor artifacts:
six initial gradients and18 trained model/optimizer states at200/400/800.
No extra qualification, since the exact adapter/normalization was qualified
in H124/H125. Freeze all reused source and evidence before execution.

Candidate gates must pass separately on both corpora against BOTH controls:
- final NLL <=1.01x each control;
- complete-job allocated peak <=0.9x each control;
- median event and synchronized wall update time <=1.15x each control;
- timing block-mean stability <=1.15 in every arm;
- pinned-host allocated peak <=128 MiB;
- all states/gradients finite, correct optimizer steps and data provenance;
- initial gradient replay <=1e-5 global /1e-4 max tensor, loss <=1e-6;
- all native audit scores agree within1e-6 and GPU boundaries return to zero.

Record all phases including warmup, initial probes, validation, optimizer,
transfers, diagnostics and serialization before resetting peaks. Timings use
steps21–800. Preserve mean, median, variance and every step. Do not substitute
training-only memory for complete-job memory. First-gradient checks do not
prove long-run quality; endpoint scores decide that gate.

This is one intentionally difficult seed per corpus, not multiseed proof.
Passing earns a separate seed/scale replication; failure blocks that expansion.
No completed case repeats, no threshold changes and no old failed gate rescue.
Parameter count is unchanged; the broader architectural/VRAM goal remains open.
