"""Apply unchanged H110 descriptive statistics and gates to completed evidence."""

from pathlib import Path

from results.token_memory_duration_v1.source import analyze

analyze.ROOT = Path("results/token_memory_duration_fresh_v1")
analyze.run()
