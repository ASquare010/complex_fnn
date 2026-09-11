# H111: retain training savings during evaluation

Status before execution: HYPOTHESIS. The preceding H110 turn made progress:
eight completed trials, exact independent rescoring, and evidence that native
evaluation becomes the candidate's allocation peak. H110's stopped long-context
fidelity comparison remains stopped. Its positive one-seed quality observation
is not a replicated result or a reason to relax its thresholds.

## Question and controls

Can evaluation avoid the N-by-V classifier workspace while preserving scores,
so that a subsequent complete training job can retain its training-memory
savings? This is an established execution mechanism, not a novel architecture.
Compare three fixed evaluation policies on every H110 saved checkpoint:

1. Native full-batch forward and mean CE.
2. Classifier chunks: unchanged decoder batch, classifier/CE in 512-token chunks.
3. Sequence chunks: native forward/CE on at most 512 token rows of complete
   sequences per sub-batch (B4 at T128, B1 at T512). Never split a sequence's
   attention context. Weight partial-batch means by their valid-target counts.

Use all 24 checkpoints at updates100/400/800 from the eight completed H110
trials. Score all complete contexts of the same WikiText-2 valid split, preserving
the original outer batch order, tokenizer and target count. All checkpoints are
development evidence, not new independent training seeds. There are zero SGD
updates and no altered model/optimizer state. No test split is accessed.

For each state, native evaluation must exactly reproduce H110's full score at
step800. Score all three policies at every endpoint, requiring absolute relative
NLL difference <=0.0001 (0.01%) from native, identical target/input order, finite
scores, unchanged model tensors, training/eval flag and random generators.
Tiny CPU-double qualifications additionally cover uneven batch/chunk boundaries,
masked targets including an entirely masked sub-batch, and an all-masked case.
Require native vs either policy agreement at absolute/relative tolerance1e-10.
No gradient qualification is claimed for a no-grad evaluation-only change.

## Memory and time before training allocation

For final checkpoints, use the same real saved Adam moment tensors on CUDA,
CPU scalar steps, and explicit zero-gradient buffers with the original shapes
and dtypes. These buffers reproduce persistent allocation, not gradient values.
There is no optimizer construction or update. The CUDA train/valid cache remains
present and counts. Earlier checkpoints receive the same final persistent-state
fixture; their purpose is score comparison, not optimizer reconstruction.

Report allocated/reserved peak for full evaluation and warmed timing on the
first16 original validation batches (two warmups, five repeats, cyclic policy
order by state). Reset allocation peaks only after preparation and retain all
persistent fixtures. Also record the allocation peak of the entire preparation
and evaluation process separately. These are local tensor allocations, not
driver-visible VRAM. The fixture is explicitly NOT a measured training job.

For each shape, a streaming policy qualifies if all endpoint scores pass,
every endpoint uses >=10% less evaluation allocation, and its median paired
evaluation-time ratio across states is <=1.5. Keep all failed policies/metrics.
For a future training recipe, rank eligible policies by the median estimated
job peak max(H110 training peak, new evaluation fixture peak), then median
evaluation time; exact ties prefer sequence chunks. This composed estimate
guides allocation only. An actual whole-job saving requires a fresh integrated
training measurement with the same optimized evaluation policy for BOTH arms.

If no policy qualifies at a shape, allocate no learning at that shape. Complete
the fixed 24-state evaluation grid despite a quality/cost gate failure; stop
immediately on native-score mismatch, nonfinite data, source mismatch or runtime
failure and preserve the interruption. Never retry silently or choose favorable
checkpoints, batch sizes, chunk sizes or tolerances.

## Runtime, provenance and audit

Use the original UV-managed Python3.12.9 directly, with existing installed
packages, PYTHONMALLOC=malloc, hash seed107, four CPU threads, TF32 off and BF16
autocast/FP32 model weights. Bypass bytecode using -B and an unused pycache prefix.
This carries forward an observed workaround, not a claim that Windows import
failures are fixed. No installed package or cache is modified. One GPU job runs
at a time on the RTX4070 Laptop. Freeze this plan, source, maintained files and
all endpoint hashes before qualification or scoring.

Independently audit score/count/order consistency, timing/memory arithmetic,
policy selection and state preservation. Preserve raw scores, all timing repeats,
fixture bytes, summaries, sources and failures. The maintained model structure,
five variants and eight recipes stay unchanged. Do not rerun the maintained
suite merely to re-count its prior116-test pass.

The subsequent quality-oriented replication, if earned, needs a separate plan
with fresh training seeds, two corpora, equal training tokens and a one-sided
quality criterion. It must not rewrite H110's fidelity decision. Neither H111
alone nor a composed memory estimate completes the broad VRAM/architecture goal.

Primary precedents: [PyTorch CE](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cross_entropy.html)
defines reduction/ignore semantics; [Cut Your Losses](https://arxiv.org/abs/2411.09009)
establishes classifier-memory optimization. We reuse native PyTorch chunks and
do not implement or claim improvement over CCE.
