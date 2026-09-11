from pathlib import Path

R = Path("results/batch_scale_training_v1")
old = Path("results/interleaved_training_v1/loop.py").read_text()
new = (
    old.replace("batch_size=8", "batch_size=16")
    .replace("data.batch(8, 512)", "data.batch(16, 512)")
    .replace("training_targets=30 * 4096", "training_targets=30 * 8192")
)
(R / "loop.py").write_text(new)
old = Path("results/optimizer_memory_v1/source/audit.py").read_text()
(R / "native_audit.py").write_text(old.replace("data.batch(8, 512)", "data.batch(16, 512)"))
for name in ("launch.py", "prepare_audit.py", "analyze.py", "audit.py"):
    text = (
        Path("results/interleaved_training_v1", name)
        .read_text()
        .replace("interleaved_training_v1", "batch_scale_training_v1")
        .replace('"H140"', '"H142"')
    )
    if name == "audit.py":
        text = text.replace(
            "import old, gradient_check, final_distance",
            "import gradient_check, final_distance\nfrom results.batch_scale_training_v1 import native_audit as old",
        )
    if name == "analyze.py":
        text = text.replace(
            '        and c["parameters"] == 9099648', '        and c["parameters"] == 9099648'
        )
    (R / name).write_text(text)
w = (
    Path("results/interleaved_training_v1/worker.py")
    .read_text()
    .replace("interleaved_training_v1", "batch_scale_training_v1")
)
w = w.replace("from results.partial_offload_training_v1.offload import install_subset\n", "")
w = w.replace(
    "from results.native_buffer_layout_v1.operator import model_loss",
    "from src.core.training_memory import buffer_model_loss, offload_checkpoint_inputs\nfrom contextlib import ExitStack",
)
start = w.index("def one(")
end = w.index("\ndef run():", start)
w = (
    w[:start]
    + """def one(row, p, folder):
    stack = ExitStack()
    installed = False
    selected = []
    original = loop.training_loss

    def loss(model, tokens, targets, policy, chunk=512):
        nonlocal installed
        if row["arm"] == "ordinary":
            return original(model, tokens, targets, policy, chunk)
        assert policy == "fp32_default_native"
        if not installed:
            stack.enter_context(offload_checkpoint_inputs(model, 4))
            selected.extend(i for i, b in enumerate(model.blocks) if "forward" in b.__dict__)
            installed = True
        return buffer_model_loss(model, tokens, targets)

    try:
        with (patch.object(loop, "ROOT", folder), patch.object(loop, "MemoryLedger", HostLedger),
              patch.object(loop, "training_loss", loss), patch.object(loop, "write_json", write_json)):
            result = loop.run_case(row["fixture"], "default", p)
            assert selected == ([4,5,6,7] if row["arm"] == "buffer4" else [])
            return dict(**result, offloaded_blocks=selected)
    finally:
        stack.close()

"""
    + w[end:]
)
(R / "worker.py").write_text(w)
