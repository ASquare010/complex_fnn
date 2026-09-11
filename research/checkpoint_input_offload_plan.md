# H121 — selective host storage of checkpoint inputs

## Hypothesis, precedent and scope

H120 eliminated optimizer temporaries as the complete-job memory solution:
backward set every peak. Test whether moving only checkpoint-boundary inputs
to pinned host memory can lower a complete forward/backward probe peak by
at least 10%, with at most 15% time overhead and preserved gradients.

This uses existing PyTorch APIs, not a novel offloading algorithm. Primary
references: [save_on_cpu](https://docs.pytorch.org/docs/2.14/autograd.html#torch.autograd.graph.save_on_cpu),
[checkpoint](https://docs.pytorch.org/docs/2.14/checkpoint.html), and PyTorch's
[checkpoint/offload wrapper](https://github.com/pytorch/pytorch/blob/main/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py).
Inspect and pin installed source: in this runtime checkpoint saves its inputs
before entering its internal saved-tensor hook, so a surrounding CPU hook can
capture those boundary inputs. Verify that behavior empirically.

Eight B8/T512/d384 FP32 inputs contain 8×8×512×384×4 = 50,331,648 bytes (48 MiB).
This is payload, not a promised VRAM saving. Lifetimes, other references and
transient recomputation/transfer buffers can limit peak reduction. Host pinned
allocation is additional system memory, and must be reported separately.

## Minimal implementation

Temporarily wrap each existing block's normal `forward` in
`torch.autograd.graph.save_on_cpu(pin_memory=True)` only when training with
gradients and whole-block checkpointing. Call the existing `Block.forward`
unchanged; do not replace attention, FFN, normalization, optimizer, checkpoint
implementation or numerical precision. Parameter identities/names and state
dict keys must stay unchanged. Restore instance overrides after the probe.
Native mode uses the ordinary block path.

Trace passes wrap the same API's pack/unpack callbacks to record shape/dtype,
bytes, CPU-pinned status, pack/unpack/release order and copy timing. Trace uses
weak references/finalizers, never a retained reference to the original GPU
input. Verify copied values by synchronized hashes; count the additional
diagnostic device-to-host copies separately. Native trace uses a detached
passthrough packed tensor. Trace timing is not the runtime comparison.

Logical saved-payload lifetime is measured by packed-tensor object release,
not claimed as physical allocator lifetime. Record PyTorch host active/cached
allocation statistics too. Host stats are rounded, per-bucket peak estimates;
do not equate them with payload or demand that the host cache become zero.

## Fixed staged budget

1. GPU FP64 qualification: two small tanh matrix chains, three paths each
   (uncheckpointed, native checkpoint, CPU-offloaded checkpoint). Six backward
   passes. Three joint directional finite-difference checks per shape add
   twelve loss forwards, no backwards. Require loss relative/absolute agreement
   1e-12, gradients relative1e-10/absolute1e-12 and directional error <=1e-7
   times max(1, absolute analytical derivative). Stop before full probes if any
   qualification fails.
2. Four H120 corpus/loss fixtures: WikiText/TinyStories seed101 at the same
   saved update800 states, native/chunked FP32 classifier. Compare native versus
   pinned checkpoint-input storage; alternate mode order. Each case performs
   30 repeats of the SAME first continuation batch, first10 warmup/last20 timed.
   Eight cases =240 forward/backward probes. No optimizer step is called.
3. One separately marked trace forward/backward per case adds eight backwards.
   Save untraced and traced gradient maps (16 tensor files).
4. Independent fresh-process replay adds eight backwards and eight saved
   replay-gradient maps. Verify model/moment/sampler provenance and gradient
   agreement, with zero optimizer steps or validation scores.

Total budget: **262 backwards** (6 tiny qualifications +240 probes +8 traces
+8 replays), **zero optimizer/training updates**, no new training recipe.
Full-model diagnostic target evaluations total 256×4096 =1,048,576; these are
repeated fixed-state gradient probes, not 1,048,576 training targets.
Total new gradient tensor artifacts:24. Failed stages remain recorded; no
silent retry or adaptive additional allocation.

Keep all H120 model/loss/precision/hardware settings and its default AdamW
state resident, including both moments and the dataset cache. Do not execute
AdamW. Use the existing UV-managed Python/PyTorch environment, four CPU threads,
TF32 off and one GPU process. Source states/moments must be unchanged at exit.
This measures forward/backward resource viability, not complete training-job
VRAM, endpoint language quality or a new architecture.

## Measurements and gates fixed before execution

Include construction, warmup, every forward/backward, gradient serialization,
trace, graph release and diagnostics in allocation ledgers before resetting
peaks. Record allocated/reserved CUDA bytes and pinned-host stats separately.
Keep all GPU allocator boundaries at zero before/between/after cases.
Save phase-event times and synchronized wall times, mean/median/sample variance
and split-half timing stability. Hash first-batch tokens/targets and source
model/moments; all repetitions use the same batch.

Require all of the following, separately in every fixture:

- qualification and independent replay pass;
- exactly eight nonempty boundary inputs are packed/unpacked per trace, each
  of shape [8,512,384], FP32, with total payload48 MiB;
- offloaded tensors are pinned CPU tensors; native trace tensors stay on CUDA;
- every unpacked value hashes exactly to its source value; all logical packed
  payloads release after loss/graph deletion and garbage collection;
- native/offloaded, trace/untraced and replay/stored losses differ by <=1e-6
  relative; parameter-gradient symmetric global relative L2 <=1e-5 and maximum
  tensor relative L2 <=1e-4; all values/gradients finite;
- measured complete diagnostic-case allocated peak <=90% of native (including
  all warmup/untimed trace/serialization phases); record normal-probe peaks too;
- offloaded median summed forward/backward event time and synchronized wall
  time <=1.15×native; split-half ratios <=1.15 in both arms;
- logical offloaded saved-payload peak <=48 MiB and host allocator reported
  allocated/cached peak <=128 MiB. Record any delayed host-active reclamation.

Only an all-four-fixture pass earns a separate short complete-training screen.
A failed scope remains failed; a passing mean or selected trace cannot replace
it. Keep memory benefits with an explicit failed-runtime label when appropriate.
No maintained default, parameter-count or breakthrough claim is authorized by
this diagnostic alone. H117/H119/H120 decisions remain unchanged.
