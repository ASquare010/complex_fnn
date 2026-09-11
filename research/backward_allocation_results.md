# H122: backward allocation diagnostic

The **requested-payload peak moves from block 7 backward to block 0 backward**
when checkpoint inputs are offloaded, in both corpora. Blocks are indexed 0–7.
This is evidence of a changing memory bottleneck, not proof of the exact
allocated-VRAM peak's location. The original exact-allocation gate **failed**.

| Corpus | Input storage | Requested peak MiB | Allocated peak MiB | Requested-peak interval |
|---|---|---:|---:|---|
| wikitext2 | offload | 278.699 | 280.337 | after block 0 recomputation |
| wikitext2 | native | 295.577 | 296.553 | after block 7 recomputation |
| tinystories | offload | 272.993 | 275.089 | after block 0 recomputation |
| tinystories | native | 289.871 | 290.374 | after block 7 recomputation |

Native peaks occur after block 7 recomputation and before block 6 begins.
Offloaded peaks occur after block 0 recomputation and before backward finishes.
The trigger's Python stack does not resolve the native backward operator.
Do not assign it to a specific attention, FFN or gradient tensor.

The original analyzer incorrectly treated event requested bytes as allocated
block sizes. Rounded blocks and unsplit remainders make them differ. Snapshot
marker indices also needed +1: a returned snapshot excludes its own history
event. The frozen analyzer and failure log remain intact. Separate offline
reanalysis exactly reconciles requested bytes with final active requested
sizes, validates all 76 markers and records site groups at the requested peak.
It does **not** rescue the original allocated-peak attribution gate. Some
grouped allocation sites remain unidentified; that category includes persistent
storage and must not be called wholly temporary backward memory.

Four cases used the same two trained H121 checkpoints, chunked FP32 loss,
resident Adam moments/data and native versus offloaded checkpoint inputs.
Two warmup backwards plus one traced backward each: **12 backwards, zero
optimizer updates, 49,152 diagnostic targets**. No training-quality or timing
claim. All four gradients match H121's independently audited references:
maximum global relative L2 7.778e-08; maximum tensor relative L2
1.986e-07. Model/moment states are unchanged and all five GPU boundaries
are zero. H121's original four-fixture gate remains failed.

A CPU inspection encountered a transient JSON integer-decoding error; a fresh
read of all four traces succeeded. Both logs remain. No completed GPU case
was repeated. This does not diagnose the recurrent runtime failures. The first report pass had a path-type error before writing output; it was corrected without rerunning any scientific stage. Its failure log remains.

**Next hypothesis:** completed parameter gradients may contribute to the late
backward peak after input offloading. A separate storage inventory can test
that before attempting gradient offload or altered scheduling. Do not infer
their contribution from the unidentified stack bucket. No maintained code or
default changes; the broader VRAM, parameter-efficiency and quality goal is open.

[Plan](backward_allocation_plan.md), [offline reanalysis](backward_allocation_reanalysis.md),
[evidence](../results/backward_allocation_v1/receipt.json).
The trace uses the existing [PyTorch memory-history API](https://docs.pytorch.org/docs/2.14/torch_cuda_memory.html);
this is diagnostic work, not algorithmic novelty.
