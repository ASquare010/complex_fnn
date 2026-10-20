# Raw64 gated cross-attention: final comparison (2026-10-16)

The completed cross-attention model is slightly worse than plain64 on overall
language prediction and substantially behind original512. These factual probes
show no reliable recovery improvement. The new growing query reader is a separate,
untrained experiment; equal parameter budgets do not imply equal quality or speed.

## Matched training

All three final/best checkpoints are at 48,762 updates, with every Adam counter
matching. All 48,762 logged batches have identical original sample indices, label
counts, learning rates and sampler cursors: two epochs and 361,357,104 assistant
label exposures. Each model has 23,209,216 trainable parameters, fresh seed47,
FP32, effective batch32, the same frozen encoder and original unified-v4 data.
Microbatch sizes differ. Hybrid reader/cross-attention parameters are funded by
smaller FFNs, so this is not a single-variable attention ablation.

Both trained cross-attention gates are nonzero: tanh-gate norms are 1.0083 and
1.0772 at layers4/8. The result is not explained by gates remaining exactly shut.
Nonzero gates do not establish useful information extraction.

## Language prediction

| Metric | Original512 | Plain64 | Gated cross-attention64 |
|---|---:|---:|---:|
| Development CE, lower is better | 1.804710 | 2.214284 | 2.247860 |
| Development token accuracy | 61.0132% | 53.5046% | 53.0322% |
| Test CE, lower is better | 1.806954 | 2.216948 | 2.249793 |
| Test token accuracy | 60.9990% | 53.5087% | 53.0127% |
| Test perplexity | 6.0919 | 9.1793 | 9.4858 |

All 7,646 test segments and 1,757,421 supervised labels were reevaluated using
the same frozen codes. Hybrid test accuracy is 0.4961 percentage points below
plain64 and 7.9864 points below original512. A 5,000-draw paired document bootstrap
gives hybrid-minus-plain accuracy [-0.5410, -0.4510] points and CE
[0.03130, 0.03432]; versus original, accuracy [-8.1510, -7.8283] points and CE
[0.43489, 0.45093]. These intervals describe document sampling, not seed variation.

Hybrid CE is worse than original on all17 sources, and worse than plain64 on16/17,
on both splits. The memory-source CE improves 0.382394 to0.360782 versus plain64,
but its test token accuracy decreases 90.1136% to89.9919%. Predictable template
tokens and teacher-forced prefixes mean this source's CE is not factual recall.

## Free factual recovery

All312 previously used prompts were replayed:128 notebook cases with correct and
blocked fact memory,32 changed facts, and8 color cases in recent/compressed/blocked
conditions. Greedy decoding,64-token cap, PAD/BOS suppression and EOS rule match
across models. Completed generated blocks are encoded live at each model's own
rollover boundary. Gold answers are never fed back.

| Probe | Original512 | Plain64 | Cross-attention64 |
|---|---:|---:|---:|
| Old notebook facts, exact answer | 0/103 | 0/103 | 0/103 |
| Changed notebook fact, exact updated answer | 0/32 | 0/32 | 0/32 |
| Both original and changed fact correct | 0/32 | 0/32 | 0/32 |
| Missing fact, correct missing response | 13/25 | 17/25 | 18/25 |
| False exact missing response when answer exists | 49/103 | 65/103 | 78/103 |
| Recent color probe, correct color | 8/8 | 0/8 | 0/8 |
| Compressed color probe, correct color | 0/8 | 1/8 | 0/8 |
| Blocked color probe, correct color | 1/8 | 1/8 | 2/8 |

The hybrid's other25 answerable notebook responses were inspected: all are wrong;
none contain the complete expected location. For example, a notebook in a yellow
cabinet in the hallway was answered as being in the bedroom. The better missing
score comes with more false missing responses, not demonstrated factual reading.
The false-missing count uses exact templates, not a general abstention classifier.

Changing an old fact changes only5/32 hybrid outputs, without a correct fact pair.
With correct notebook memory,128/128 outputs stop; blocking fact codes reduces
this to8/128. Blocking preserves memory positions and the full later question,
and masks only fact-containing codes. The blocked color hits are insufficient
evidence of retrieval: those facts are inaccessible and the outputs do not stop.
Recent color prompts initially fit raw64 but cross its boundary after five new
tokens; this is a live-rollover probe, not permanently raw-only decoding.

## Verification and evidence

The evaluator uses the hybrid's public forward/packed path. An independent
unfolded raw-attention plus explicit cross-attention reference agrees within
2.2888e-5 FP32. Altering blocked code contents leaves logits exactly unchanged.
Four serial generations per model replay exactly through the independent path.
All312 old outputs and every old test metric reproduce the historical evaluation.
All checkpoint/source hashes and frozen encoder tensors remain unchanged; zero
optimizer updates. Saved per-segment/source metrics and notebook exact/stop
scores were independently recalculated from artifacts.

- Evaluator: `tools/compare_token_memory_crossattn.py`.
- Report and full outputs: `dump/token-memory-crossattn-comparison-20261016/`.
- Detailed checks: `report.json`, `verification.json`, `abstention_analysis.json`.
- Trained hybrid checkpoint SHA256:
  `9f96ae069dc5d7afd71d39f3024e57888e82b960a3f2230aaa35a07a6f94a1e6`.

One initialization and a small synthetic factual suite limit generalization.
This test split was already used in research. No architecture change, further
training, checkpoint deletion or current-run interruption was performed.

## New growing reader: full two-epoch command

The query-conditioned reader keeps one position per code and unchanged K/V
shapes. It remains untrained. With the completed hybrid no longer using the GPU,
microbatch4 passed a real-data benchmark and a synthetic maximum39-code/512-query
capacity check with Adam-sized buffers: 1.340GiB allocated,1.547GiB reserved in
the latter. Effective batch32, seed47 and the48,762-update schedule stay unchanged.

From `D:\Git\complex_fnn`:

```powershell
& .\.venv-cuda64\Scripts\python.exe .\src\models\growing_token_memory\training.py `
  --name growing-token-query-reader-v1 `
  --data .\dump\data\unified-chat-v4 `
  --cache-dir I:\growing-token-cache-v4 `
  --epochs 2 --batch-size 32 --microbatch 4 --encoder-batch 256
```

There is no `--stop-after 4000`, so this runs the full two epochs. It starts fresh;
do not resume the old growing-model checkpoint with the new architecture. The
existing frozen prefix cache is reused after hash verification. No training was
launched during this comparison.

At microbatch4, three alternating benchmark rounds measured median cached
forward/backward0.7267s per effective32 for the query reader versus0.5057s for the
old growing reader: about44% more time in this measured workload. Cached-query
latency was28.62ms versus18.36ms. These exclude disk loading, Adam, validation,
saving and live encoder work; they are not end-to-end training/generation rates.
Both use exactly1,277,952 KV bytes for39 completed codes. Quality remains unknown.
See `docs/growing_query_reader_setup.md` for details and benchmark artifacts.
