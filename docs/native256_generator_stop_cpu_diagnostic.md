# Generator-controlled stopping: CPU diagnostic

18 October2026. A process-local read-only ablation on the preserved query V2
step1000 checkpoint. No source model, checkpoint, cache or current job changed.
The classifier uses no expected answer or answer length at inference.

Keep the old nonEOS output distribution conditional on continuing. Set EOS
logit to the maximum valid content log probability plus the generator's EOS
logit minus its largest valid nonEOS logit, then normalize. Thus greedy EOS
wins exactly when the generator prefers EOS, independently of copy dominance.
The CPU wrapper repeats the generator matmul for diagnosis; this is not a speed
claim or production integration. Reader23,209,216 parameters and KV schema stay
unchanged. All1,058 public-forward calls satisfy finite normalization (maximum
logsumexp error4.77e-7) and the EOS-decision identity. Checkpoint tensors/hash
remain unchanged; no optimizer is constructed.

Preselected development cases: first four-way unit from each of seven access
groups, both original and value-free-prefix forms, plus first four positive and
four missing notebook cases. Selection never depends on predictions.

| Group | Existing exact | Generator-controlled exact |
|---|---:|---:|
| Unconstrained access |0/28|0/28|
| Prefix-assisted access |0/28|15/28|
| Notebook positive+missing subset |4/8|3/8|

The lost notebook answer is truncated after Remy, despite the original copy
model producing the complete correct answer. Therefore generator EOS can be
wrong before an answer is finished, even when it is correct at supplied gold
endpoints. Do not adopt this override. It supports the measured stopping
mechanism but does not solve format selection or preserve existing quality.

Full outputs/report and an independent aggregate/input/score audit are saved
under dump/native256-generator-stop-cpu-20261018. The aggregate audit is not a
second model implementation or full-dev/GPU/cache/quality proof. The archived
math script is probe_cpu.txt; the launch added an artifact-copy statement before
loading weights, without changing its model math. No reserved TEST was used.
