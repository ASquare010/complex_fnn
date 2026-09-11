import hashlib
import json
import os
from pathlib import Path

compiler = Path(".venv/Lib/site-packages/triton/runtime/tcc/tcc.exe").resolve()
assert compiler.is_file()
os.environ["CC"] = str(compiler)
Path("results/training_memory_integration_v1/compiler.json").write_text(
    json.dumps(dict(path=str(compiler), sha256=hashlib.sha256(compiler.read_bytes()).hexdigest()))
)
import pytest  # noqa: E402
import sympy  # noqa: E402, F401
import torch  # noqa: E402
import torch._dynamo  # noqa: E402, F401

torch.set_num_threads(4)
raise SystemExit(
    pytest.main(
        [
            "-q",
            "tests/test_triton_inference.py",
            "-k",
            "test_fused_matches_native_on_fresh_non_tile_aligned_inputs",
        ]
    )
)
