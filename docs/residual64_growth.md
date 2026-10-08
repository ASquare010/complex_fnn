# Residual64 modest-growth comparison

Authorized: three plain-residual candidates, 4,000 additional updates each,
max three parallel workers. Same 64 tokens/vector, width256, max context512,
125 packing features. No margin loss. No further experiments authorized.

| Arm | Encoder/decoder layers | Total parameters |
|---|---:|---:|
| continue_plain |4/1|10,456,576|
| encoder_plain |6/1|12,554,752|
| balanced_plain |5/2|12,554,752|

The two larger models add20.07% parameters. All warm-start the904/1000 short,
103/206 packed plain endpoint from residual512-screen-4000-v5/s64_plain/last.pt,
SHA2568c1aa819efb11fc05a5e2a68c037d4d69f5e4d30d55359d728800a5ce3d4d503.
30,000 inherited updates; fresh AdamW lr3e-5,weight decay.01,clip1,micro2xaccum4.
Sampler seed43 generates a new shared schedule; not independent model seeds.

Growth preserves every old tensor, the four-layer learned depth routing and
compressed-vector width. New encoder blocks are ordinary residual refinement
blocks AFTER the existing depth mixture, before encoder normalization. New decoder
blocks follow the existing decoder. Both output projections in each added block
start at zero, preserving the original function; random internal weights can learn
as output weights open. This does NOT reroute all six encoder layers jointly.
Each worker verifies exact initial CPU FP32 logits and vectors on a padded sample,
without optimizer updates. This is a transfer check, not a full validation replay.

Reuse existing immutable packed512 data. Same25%short/25%packed/25%uniform/
25%pattern schedule for every arm. No new context curriculum: isolate growth from
additional training. Complete independent decoded train/validation identities,
source/parent hashes, finite gradients and1300/1500MiB allocated/reserved guards
remain. Prior separate probes/initial validation waivers retained, not passed.

Logical CPUs16/17/18 pin workers; controller16. Use uv run --no-sync with existing
CUDA environment. Prior native failures, including the unpinned post-training CLR
audit failure, remain unexplained; no automatic retry after any new native/data
failure. Old weights/evidence retained. No inference checkpoint promotion.

Frozen at launch: dump/residual64-growth-v1 (training .py files).
Output: D:/Git/latent-text-compressor/artifacts/runs/residual64-growth-4000-v1.
Progress: powershell -NoProfile -File D:\Git\complex_fnn\dump\residual64-growth-v1\progress.ps1

Full206 packed and1000 short validation every500; best chosen by packed exact,
then token recovery/NLL, as before. Report short-best separately if different.
Final/best full replay, encoder export parity, source/checkpoint/record hashes and
matched target schedules required for completion. Test receives identity checks
only; no model evaluation yet. One inherited seed and reused validation limit
claims. The goal is100% exact recovery, not rounded token accuracy.

Status: prepared for direct launch; not yet a trained result.

## Diagnosed controller repair

V1 controller failed on Windows PermissionError/WinError5 replacing status.json
while it was read. Both larger workers passed identities and transfer checks but
exited before update1 after seeing the failed-controller status. The unchanged
continuation passed the barrier earlier and is still running; it was not restarted.
This was not a model, native-runtime or data-integrity failure.

V2 launches ONLY encoder_plain and balanced_plain, each4000 updates, with identical
computational sources, parent and sampler43. Metadata writes retry only a transient
PermissionError on rename for up to2seconds; no optimizer-step retries. V1 sources
and failure evidence remain intact. Total budget remains12000 updates, max3 live
workers. Compare v1/continue_plain against v2/encoder_plain and v2/balanced_plain.
A combined completion audit must verify all three target schedules, per-arm audits,
source provenance and final/best replay before calling this comparison complete.
See dump/residual64-growth-v2/repair.md. V1 status remains historically failed even
while its orphaned continuation worker trains; inspect individual worker status.

## Launch verified

All three are now training: unchanged continuation500/4000 (validating), encoder
50/4000, balanced25/4000 at observation. All passed complete decoded identities,
exact parent tensor preservation and padded-sample initial vector/logit parity.
Total GPU2045/8188MiB. Larger early training peaks494.42/546 and494.90/538MiB
allocated/reserved; these are not final resource results. No failures in repaired
v2. Sources frozen. Progress: dump/residual64-growth-v2/progress.ps1.

## Completed comparison

| Arm | Final short /1000 | Final packed /206 | Packed-selected best short / packed | Training allocated / reserved MiB |
|---|---:|---:|---:|---:|
| Same size continuation |957|151|958 /155 at3000|424.58 /474|
| Encoder growth |942|153|963 /154 at3000|496.89 /560|
| Balanced growth |959|169|959 /169 at4000|497.22 /558|

All completed4000 additional updates; endpoints34000 cumulative updates. Parent
short904 at30000; earlier parent781 at26000. Thus unchanged model has8000 updates
across the last TWO rounds, but only4000 in this growth comparison.

Pinned read-only saved-evidence verification passed all three final/best/encoder,
result,persistent-record and frozen-source archive hashes;4000 update sequences and
identical target schedules; all final/best replay,export parity and decoded-identity
receipts true. This check did not rerun model validation or retry the prior failed
independent serialization script. V1 failed controller evidence remains retained.

Training-update loop totals20.50/23.73/23.25minutes respectively exclude validation
and preparation. GPU shared with staggered launches: not isolated throughput.
Balanced model wins final long reconstruction; same-size model essentially matches
its short score with fewer parameters and lower memory. Still not100% exact;
one inherited seed,reused validation,no untouched test evaluation.
No further continuation launched. Product checkpoint unchanged.
