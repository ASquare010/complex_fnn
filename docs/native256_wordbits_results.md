# Exact lexical metadata within existing FP32 code slots

Storage/numerical diagnostic, 18 October 2026. **No trained word-bit language
reader exists.** This does not establish QA quality, language preservation,
reasoning, speed or completion of the overall goal.

The separate component preserves all 256 semantic coordinates while replacing
their four lowest mantissa bits with source-token metadata. Four FP32 values carry
one 16-bit payload, so the existing 256-number code holds all 64 payloads. Each
payload allocates 12 bits to a vocabulary ID, two to input-known speaker role,
one to input-known message start and one reserved bit. Future message ends, EOS,
answers and gold entity positions are not permitted metadata.

This deliberately adds exact lexical information to the representation. It does
not show the old encoder learned exact token recovery. Original float values
change by at most 15 ULP; normal-value relative changes were at most 1.7881e-6 in
the tested inputs. Sign and exponent bits remain unchanged. It is attention/KV
position compression, **not byte compression relative to raw token IDs**.

## Completed component checks

Four CPU tests pass. The zero-update CPU and CUDA gates both check:

- Every 16-bit payload combination, including the reserved bit.
- Every vocabulary ID at every one of the 64 source positions (262,144 checks).
- Exact role/start recovery, selected-position extraction, signed zero and
  subnormal payloads, preserved sign/exponent, finite outputs and numerical bound.
- Copy into the existing 21-layer KV allocation: 8,860,878 bytes in the fixture,
  with **zero added cache bytes**.
- Explicit identity straight-through gradients to semantic values. The forward
  is bitwise; this is an approximation to the derivative of quantized storage.

The unmodified frozen decoder was evaluated at every position in 512 genuine
held-out source blocks, without success filtering. Before and after adding word
bits, it reconstructed 32,735/32,736 content tokens and 32,735/32,768 total tokens.
No argmax prediction changed. Maximum logit change was .000253677. Existing
special-token and one content-token error remain; this decoder is not lossless.
The metadata itself recovers all 32,768 source IDs exactly.

Using the **same immutable native256 step8750 checkpoint**, four genuine original
development segments supplied 573 assistant labels and 1–10 memory positions.
No predicted label changed. Maximum hidden change was 1.11461e-5, maximum logit
change 1.39475e-5. This is a small numerical check, not a full language evaluation.
The original checkpoint SHA is
`57ccdacbf4ae79be637ace6d104848a5543bda33a8fe7486a39b24c7e8144052`.

## Limits before integration

Bit payloads survive FP32 load/store only. Normalization, projections, casts and
ordinary arithmetic can destroy them. A future reader must extract metadata
before arithmetic and reinsert it after learned V projections into the existing
value allocations. It must not retain a second old-token bank or add memory
positions. These integration requirements are **not implemented** by this
storage component. The current tests do not establish a production cache path.

No LM, encoder or decoder was trained or changed. There are zero additional
component parameters. A future actual model must count all query/readout weights
inside the user-confirmed 23M reader budget; the encoder is separate. It still
needs causal query/span/field selection, general language use of memory, full
model speed and KV measurements before training, and broad matched fresh
two-original-epoch evaluations. Exact stored IDs alone solve none of those tests.

Protocol: [prospective storage gate](native256_wordbits_protocol.md). Artifacts:
`dump/native256-wordbits-storage-gate-20261018`. The probe session12191 completed
successfully. Independent verification27237 also finished successfully with a
saved `passed: true`. The verifier independently
uses NumPy bit arithmetic, reaggregates all saved decoder/native predictions,
replays sampled CPU decoder positions and a native segment, and checks protected
old files. It recovered all32,768 saved IDs, replayed128 CPU decoder positions and
native segment2047, and verified all1,901 protected files. No old source,
checkpoint, cache or running model is modified. All workers have finished.
