"""H083 binding-only repair; all model and fitting bodies remain in frozen H082."""

import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

from results.latent_activation_v1.source import study

study.ROOT = Path("results/latent_activation_recovery_v1")
study.PLAN = Path("research/latent_activation_recovery_plan.md")


@contextmanager
def bound():
    replacements = {
        "ROOT": study.ROOT, "PLAN": study.PLAN, "TASKS": study.TASKS, "FORMS": study.FORMS,
        "SEEDS": study.SEEDS, "RATES": study.RATES, "STEPS": study.STEPS,
        "COUNTS": study.COUNTS, "HIDDEN": study.HIDDEN, "make_model": study.make_model,
        "make_data": study.make_data, "target": study.target, "diagnostics": study.diagnostics,
        "heldout": study.heldout, "summarize": study.summarize, "write_json": study.durable_json,
    }
    with ExitStack() as stack:
        for name, value in replacements.items():
            stack.enter_context(patch.object(study.base, name, value))
        stack.enter_context(patch.object(torch, "save", study.save_tensor))
        yield


study.bound = bound

if __name__ == "__main__":
    try:
        with bound():
            study.base.run()
    except BaseException as exc:
        study.durable_json(study.ROOT / "failure.json", {
            "error": repr(exc), "traceback": traceback.format_exc(),
            "explicit_preflight_repetitions": 1, "fitting_repetitions": 0,
        })
        raise
