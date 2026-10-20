# Conditional EOS development diagnostic: verified trade-off

18 October2026. Same immutable V2 step1000 checkpoint, zero optimizer updates.
The rule retains the old output when copying assigns positive probability to
EOS; otherwise it lets the generator's EOS-versus-content margin control stopping.
No target text or gold length enters inference. Numerically underflowed EOS copy
probability is treated as absence. The encoder,64:1 global attention positions,
23,209,216 reader parameters, literal-ID metadata and KV layout are unchanged.
The diagnostic wrapper repeats the generator matmul and is not a speed result.

The full diagnostic and independent audit completed successfully. The audit
checks all1,740 case outputs/tokens/text/EOS/strict scores, original case files,
source/input hashes, exact shuffled-donor replay and six previously executed CPU
public/cached replays against the new GPU outputs. Full-dev document counts and
loss totals are independently reaggregated. It does not cover a fresh full-budget
training comparison, reserved TEST, historical transfer or the reasoning probe.

| Development measure | Preserved V2 | Conditional EOS |
|---|---:|---:|
| Full language accuracy |44.213472%|43.741220%|
| Positive fact answers |225/285|225/285|
| Missing answers |30/63|30/63|
| Changed-value paired correct |49/64|49/64|
| Changed-query paired correct |39/64|39/64|
| Unconstrained access complete |0/224|0/224|
| Prefix-assisted access complete |3/224|95/224|
| Blocked prefix-assisted access |0/56|0/56|
| Notebook full replies |197/256|197/256|
| Notebook full all-four sets |37/64|37/64|
| Notebook proper completions |256/256|256/256|
| Notebook completion all-four sets |64/64|64/64|

Language loss is−.472252 percentage points, paired fixed-document95% interval
[−.502633,−.442032],4,852 clusters/5,000 draws. These are one fixed checkpoint's
paired document intervals, not training-seed uncertainty. The95 access answers
require a supplied value-free answer prefix; unconstrained requests remain0.
Thus this is a measured partial stopping repair with a language cost, not a
quality-preserving compression solution. Do not adopt it or start training from
this read-only diagnostic as though the target were achieved.

Artifacts: dump/native256-generator-stop-availability-dev-20261018/report.json
and independent-verification.json. Existing runs/checkpoints/data/cache/source
files are unchanged. Full two-original-epoch parity and a strong matched raw
reference remain required by the active goal.
