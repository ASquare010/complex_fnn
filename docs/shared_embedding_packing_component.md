# Shared-embedding packing component check

18 October 2026. CPU component experiment only; no trained production model.
The reader budget is 23,209,216 parameters; the encoder is separate, matching
baseline accounting. Existing native256 and all pilot checkpoints are unchanged.

The temporary component summarizes 64 shared token embeddings into one width512
vector. Learned within-block rotations supply position information before the
sum; RMS normalization, a learned gate and memory-type vector follow. The
existing exact token-ID payload is retained in the first256 float low bits.
This payload is added literal metadata, not proof of learned lossless compression.

Component checks passed: exact metadata round trip for64:1 and1:1 modes;
changing later blocks leaves earlier codes unchanged; all packing parameters
and the shared embedding receive finite nonzero gradients. The component has
1,792 parameters. Actual constructed eight-layer width512 core inventory has
21,884,416 parameters (FFN hidden total11,107); the prior local-reader inventory
adds1,323,008, totaling23,209,216. The complete transformer/cache integration
has not been implemented or tested. The proposed KV-only ratio .761905 to the
21-layer native256 model is an inventory calculation, not measured performance.

Report: `dump/shared-embedding-packing-cpu-v2-20261018/report.json`.
The first temporary constructor collided with nn.Module.type; its failure is
preserved separately. No production forward, optimizer update, GPU preflight,
quality or latency result exists for this candidate. It is a hypothesis for
aligning memory summaries with the reader's embedding basis, not an adopted fix.
