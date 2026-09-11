"""Explicit evaluator dependency and extra checkpoint for complete rescoring."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from results.streamed_evaluation_v1.source.evaluation import evaluate
from src.core.reproducibility import sha256


@dataclass
class EvaluationAdapter:
    policy: str
    seed: int
    directory: Path
    calls: int = 0
    extra_checkpoint: dict | None = None

    def __call__(self, model, data, batch, context, batches):
        expected = (0, 100, 200, 400, 800, 800)
        assert self.calls < len(expected)
        step = expected[self.calls]
        self.calls += 1
        assert batches == (10**9 if self.calls == 6 else 16)
        result = evaluate(model, data, batch, context, batches, self.policy)
        if step == 200:
            path = self.directory / "step200.pt"
            assert not path.exists()
            torch.save(
                {
                    "model_config": asdict(model.config),
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "seed": self.seed,
                    "step": step,
                },
                path,
            )
            self.extra_checkpoint = {"step": step, "path": path.as_posix(), "sha256": sha256(path)}
        return result
