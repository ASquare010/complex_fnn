# Larger-model replication after the serving gate passes

Frozen before any seed29/43 larger-model training. At width384/layer8, seed17,
800 steps, native BlockShuffle passes local quality/memory gates, and the new
four-kernel execution passes serving at1.206x equally compiled full SwiGLU.
This earns replication of the existing locked recipes, not a new architecture
or optimizer search. The full research target and prior negative results stay.

## Fixed cohort and budget

Train exactly eight fresh 800-step runs: seeds29 and43, each with full SwiGLU,
full GELU, width-calibrated narrow SwiGLU and BlockShuffle. Use the existing
configs/scale384_*_800.json configurations, overriding only seed. Preserve
width384, layers8, heads6, context128, batch16, BF16, optimizer/init recipes,
frozen TinyStories data/tokenizer, and 32768 fixed validation targets. Token
budget is1638400 per run. Compare both full controls and calibrated narrow.
All initial non-FFN tensors must match within each seed, as in the first cohort.

Run one GPU training process at a time, with no concurrent GPU tests/profiles.
Retain every run, including ordinary quality misses. The cohort is not selected
by which seeds pass. Stop for nonfinite training, OOM or execution failure;
preserve the failure before deciding whether an unchanged retry is justified.
Use a900-second process deadline per run as an execution guard, not a quality
early-stop rule. The expected training duration is minutes per run on this GPU;
no unbounded width, LR or activation sweep is authorized by this plan.

Only native training is used. The fused path is inference-only and the
post-training condition floor is not applied during these training runs.
Source archives distinguish the later tooling additions from the unchanged
trainer/model/optimizer/data implementations; verify their hashes against the
seed17 training sources. The optional compiler package does not change PyTorch
or CUDA versions. Record actual environment and all shared protocol fields.

## Acceptance and reporting

Report per-seed NLL and paired differences plus mean/sample SD across17,29,43.
Require >=70% FFN weight reduction, <=1% relative NLL degradation against BOTH
full references, beating calibrated narrow, finite training and allocated
training peak<=1.1x full SwiGLU. Report both per-seed gates and the aggregate;
a mean passing cannot hide a failed seed. Report total model weights and compute.

After each completed structured checkpoint, a fixed inference-only numerical
check may compare native and four-kernel eager/compiled outputs on three fresh
inputs and all fixed validation targets, using the existing .001 NLL/.01 RMS/
.15 max-logit tolerances. Any later serving timing is a separate paired audit;
same shapes do not justify inventing speed measurements for unmeasured seeds.

The800-step schedule still does not prove convergence, and this remains a
small fixed corpus. Even all gates passing in three seeds would not establish
broad superiority, equal historical tuning effort, architectural novelty or
a full-network gradient lower bound. Convergence and broader-corpus experiments
must be specified separately after observing this locked cohort.
