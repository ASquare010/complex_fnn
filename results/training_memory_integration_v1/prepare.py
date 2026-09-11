"""Freeze integration scope and prove unchanged classifier arithmetic."""

import ast
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/training_memory_integration_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/interleaved_training_v1/receipt.json")["files"])
assert (ROOT / "test_exit.txt").read_text().strip() == "0"


def method(path, name, method_name):
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name
    ).body


native = method("results/native_buffer_loss_v1/operator.py", "NativeBufferLoss", "forward")
# Native prototype starts with dtype/autocast guards; maintained API checks earlier.
start = next(
    i
    for i, n in enumerate(native)
    if isinstance(n, ast.Assign)
    and isinstance(n.targets[0], ast.Name)
    and n.targets[0].id == "logp"
)
new = "src/core/training_memory.py"
assert ast.dump(ast.Module(body=native[start:], type_ignores=[])) == ast.dump(
    ast.Module(body=method(new, "_BufferCrossEntropy", "forward"), type_ignores=[])
)
assert ast.dump(
    ast.Module(
        body=method("results/native_buffer_layout_v1/operator.py", "LayoutBufferLoss", "backward"),
        type_ignores=[],
    )
) == ast.dump(ast.Module(body=method(new, "_BufferCrossEntropy", "backward"), type_ignores=[]))
base = read("results/interleaved_training_v1/protocol.json")
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(base[field])
for values in base["base_verified"].values():
    hashes(values)
p = dict(
    study="H141",
    fixtures=base["fixtures"],
    datasets=base["datasets"],
    maintained_files=base["maintained_files"],
    input_hashes={
        **base["input_hashes"],
        "results/interleaved_training_v1/receipt.json": sha(
            "results/interleaved_training_v1/receipt.json"
        ),
    },
    sources={
        f.as_posix(): sha(f)
        for f in [
            *ROOT.glob("*.py"),
            Path(new),
            Path("tests/test_training_memory.py"),
            Path("research/training_memory_integration_plan.md"),
            Path("research/training_memory_usage.md"),
        ]
    },
    training_updates=12,
    backwards=12,
    classifier_ast_equal=True,
    current_state_before=sha("research/CURRENT_STATE.md"),
    readme_before=sha("README.md"),
)
for name, path in [("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")]:
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
write_json(ROOT / "protocol.json", p)
print("Classifier forward/backward AST equality verified;12-step integration replay frozen.")
