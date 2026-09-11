from pathlib import Path

R = Path("results/batch_scale_training_v1")
text = Path("results/interleaved_training_v1/finish.py").read_text()
text = (
    text.replace("interleaved_training_v1", "batch_scale_training_v1")
    .replace("interleaved_training_plan.md", "batch_scale_training_plan.md")
    .replace("interleaved_training_results.md", "batch_scale_training_results.md")
)
text = text.replace(
    "results/timing_drift_audit_v1/receipt.json",
    "results/training_memory_integration_v1/receipt.json",
)
text = (
    text.replace('"H140"', '"H142"')
    .replace("[H140]", "[H142]")
    .replace(
        "H140: interleaved complete-training comparison",
        "H142: doubled-batch maintained-helper qualification",
    )
)
text = text.replace("scoped interleaved timing gate", "scoped batch-16 qualification gate")
text = text.replace(
    "720 updates/backwards,2,949,120 targets", "720 updates/backwards,5,898,240 targets"
)
text = text.replace(
    "## Latest: interleaved complete-training timing",
    "## Latest: doubled-batch memory-helper qualification",
)
text = text.replace("interleaved timing]", "batch-16 qualification]")
text = text.replace(
    "All six ordinary H138 step800 model/Adam/sampler states are used.",
    "All six ordinary H138 step800 model/Adam/sampler states are used. Training batch\nsize is16 instead of8; context512 and validation batch8 remain unchanged.\nThe candidate uses the maintained buffer_model_loss and offload_checkpoint_inputs\nAPIs without editing their source. This tests a larger activation workload, not\na larger model or a different domain.",
)
text = text.replace(
    "An AST audit proves the reused H134 loop differs only in fixture seed\nmetadata and setup-wall instrumentation.",
    "AST audits prove the reused H140 loop differs only in training batch size and\ntarget accounting; the independent H120 audit differs only in sampled batch size.\nNative validation and numerical Adam/clipping checks remain unchanged.",
)
text = text.replace(
    "Design an explicit opt-in maintained implementation with source-equivalence tests, preserving defaults, then qualify another workload scale. This pass concerns only balanced short segments at saved states; H138 remains failed and sustained deployment throughput remains unproven.",
    "The maintained helper now has bounded evidence at batch8 and batch16. Preserve ordinary defaults and H138's failed long-run result. Return to the unresolved structural FFN/parameter-efficiency question using the repository's prior candidate eliminations; do not infer novel neuron geometry or sustained throughput from these memory-only results.",
)
(R / "finish.py").write_text(text)
