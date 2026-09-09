"""H090 changes only activation precision regions and isolated output roots."""

from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from torch import float64, nn

from results.neuron_geometry_v1.source import study as original_study
from results.neuron_geometry_v1.source import test_qualification as original_tests
from results.neuron_geometry_v1.source.model import GeometryFFN, LocalCurve, PairTwist

ROOT = Path("results/neuron_geometry_recovery_v1")
original_reference = original_tests.reference


class FloatPairTwist(PairTwist):
    def forward(self, z):
        return super().forward(z if z.dtype == float64 else z.float()).to(z.dtype)


class FloatLocalCurve(LocalCurve):
    def forward(self, z):
        return super().forward(z if z.dtype == float64 else z.float()).to(z.dtype)


class PrecisionFFN(GeometryFFN):
    def __init__(self, form, seed=17):
        super().__init__(form, seed)
        original = self.curve
        if isinstance(original, PairTwist):
            replacement = FloatPairTwist(
                original.theta.numel(),
                learned=isinstance(original.theta, nn.Parameter),
                center=original.center,
            )
        elif isinstance(original, LocalCurve):
            replacement = FloatLocalCurve(original.family, original.groups)
        else:
            return
        replacement.load_state_dict(original.state_dict(), strict=True)
        self.curve = replacement


def reference(z, curve):
    return original_reference(z if z.dtype == float64 else z.float(), curve).to(z.dtype)


@contextmanager
def bindings():
    with ExitStack() as stack:
        for module, name, value in (
            (original_study, "ROOT", ROOT),
            (original_study, "GeometryFFN", PrecisionFFN),
            (original_tests, "ROOT", ROOT / "qualification"),
            (original_tests, "GeometryFFN", PrecisionFFN),
            (original_tests, "reference", reference),
        ):
            stack.enter_context(patch.object(module, name, value))
        yield


if __name__ == "__main__":
    with bindings():
        original_study.main()
