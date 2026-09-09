from results.verification.rational_memory_process_v1 import run
from src.core.rational_memory_audit import CELLS

run("full_tests", ["-m", "pytest", "-p", "no:anyio", "-q"])
run("preflight", ["-m", "src.core.rational_memory_audit", "preflight"])
for cell in CELLS:
    run(f"worker_{cell}", ["-m", "src.core.rational_memory_audit", "worker", "--cell", cell])
run("finish", ["-m", "src.core.rational_memory_audit", "finish"])
