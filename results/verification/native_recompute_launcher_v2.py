"""Run the frozen H063 phases once, preserving every process result."""

from results.verification.native_recompute_process_v1 import run
from src.core.native_recompute_audit import RECIPES, SCOPES

run("preflight_runtime_retry", ["-m", "src.core.native_recompute_audit", "preflight"])
for index, recipe in enumerate(RECIPES):
    order = SCOPES[index % 3 :] + SCOPES[: index % 3]
    for scope in order:
        cell = f"{recipe}_{scope}"
        run(f"worker_{cell}", ["-m", "src.core.native_recompute_audit", "worker", "--cell", cell])
run("finish", ["-m", "src.core.native_recompute_audit", "finish"])
