# Paired readout execution v2: reuse temporary adjoint storage

Registered after all 48 v1 profiles completed and the terminal audit passed,
before v2 implementation or measurement. Preserve v1's frozen source and results.
V1 passed resources against native full SwiGLU on both corpora, but its allocated
memory reduction was 19.80% versus GELU and 19.96% versus kernel SwiGLU. TinyStories
also failed the 5% slowdown ceiling versus GELU (throughput .9469x). Under the
preregistered all-control gate, retire v1 before language without changing gates.

Keep the original paired-readout mathematics, width 1152, signs, pairing,
calibration, weights, saved projected z/readout bank, counts and controls.
The change is buffer lifetime in its explicit backward function:

1. Gather incoming feature adjoints into a fresh local tensor. Multiply signs,
   optional removal mask and response scale into that tensor in the original order.
2. Allocate native SiLU(z) as the direct branch buffer and multiply it in place
   by the scaled incoming adjoint.
3. Multiply the incoming-adjoint buffer by z, then use the installed native
   `aten.silu_backward.grad_input` output overload to reuse that same buffer for
   the indirect branch. Verify this supported alias arrangement against allocating
   native execution before relying on it.
4. Add indirect into direct in place, cast with the same native BF16 operation,
   and retain both shared-P master-gradient contributions exactly as in v1.

Only newly created local buffers may be mutated. Never mutate caller input,
incoming gradient, model parameters, saved x/z/u, signs or partner buffers.
In FP32, z.float() can alias saved z; it remains read-only. No reassociation,
fused scalar backward, custom BF16 casts, detached branch, precision change or
approximate gradient. No hardware improvement is assumed from fewer allocations.

Before profiling, repeat the complete original CPU suite and all 120 GPU FFN
fixtures/20 whole-model comparisons with the original tolerances. Additionally
compare v1 and v2 forwards/all gradients for all modes at production BF16 shape,
and assert every saved/caller tensor and parameter retains its value and version
counter after backward. Check a repeated forward/backward with retain_graph to
detect accidental saved-state mutation. Preserve failure fixtures if any check
fails; numerical failure retires v2 without resource or language work.

Only a complete numerical pass permits the same frozen 48-profile screen:
eight variants, both corpora, three alternating rounds, 20 warmup/100 measured
updates and unchanged training/data/backbone. Include all real buffers/dispatches.
All-control resource thresholds remain >=1.2x speed with memory <=1.05x OR
>=20% allocated-memory reduction with update time <=1.05x. No concurrent GPU
experiments or live source edits. No replacement of the primary by a control.

Language, removals, equal tuning, repeated seeds, independent confirmation and
sustained resource gates remain exactly those in the original
[paired-readout hypothesis](paired_readout_hypothesis.md). This is an execution
revision, not new architecture, evidence of language quality or novelty.
