# Query V2 matched-budget timing scope

18 October2026, zero optimizer updates. Compare default ordinary Span and repaired
query V2, each exactly23,209,216 reader weights. This corrects the preliminary
shared-core timing's ordinary control: it used31,368 FFN units/23,206,144 weights,
versus the new reader's23,209,216. The preliminary timing remains an isolation
measurement, not the final equal-budget benchmark.

Use identical genuine original packet[112342]+[217414]*7, raw512/maxmemory39,
with per-row valid/visible masks unchanged. Benchmark ordinary assistant CE
forward/backward/clip in alternating order, three warmed rounds with two
moment-sized buffers. Then compare actual cached one-token inference in alternating
order for batch1 and8 on a genuine query-bearing prompt. Derive its old count and
raw prefix from the original first-answer query label, not reconstructed gold
answer text. Both caches retain the existing raw64 arrays, one global KV/code,
same capacity and byte count. Reset only raw_count to the saved prompt boundary
for repeated identical one-token feeds; the test overwrites that same next slot
each time. Token IDs are fixed ordinary non-special input, no output scoring.

Include all Python dispatch, selector/scope/key/matching/read and core/head work
inside timed bodies, with GPU synchronization at boundaries. Model initialization,
code appends/prompt prefill, frozen E/IO, optimizer step, evaluation/save are
outside timing. State explicitly that these timings are not full training or
whole-generation throughput and establish no quality improvement. Record all
initial weights unchanged, zero optimizer updates, model/KV/buffer byte counts,
input/index provenance and source hashes. Previous completed runs remain immutable.
