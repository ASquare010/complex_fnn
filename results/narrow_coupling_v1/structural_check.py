"""Post-hoc exact finite witness; no trained data, GPU, or empirical gate changes."""

from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import sha
from results.ordinary_long_training_v1.io import write_json

root = Path("results/narrow_coupling_v1")
assert not (root / "structural_check.json").exists()
d = 32
x = [[int(j == i or j == (i + 1) % d) for j in range(d)] for i in range(d)]
y = [[row[j] * row[(j + 1) % d] for j in range(d)] for row in x]
assert y == [[int(i == j) for j in range(d)] for i in range(d)]
assert all([(-v[j]) * (-v[(j + 1) % d]) for j in range(d)] == y[i] for i, v in enumerate(x))
# Zero plus all basis outputs is affinely independent. A collision requires
# a singular square readout after any injective core; singular readout cannot
# contain these d+1 outputs in its affine range. This is exact representability
# only, not a positive infimum for approximation error.
write_json(
    root / "structural_check.json",
    dict(
        passed=True,
        scope="Exact finite representability only; no approximation-error bound",
        input_dimension=d,
        pairs=x,
        outputs=y,
        integer_basis_witness=True,
        zero_output=[0] * d,
        source_sha256=sha(__file__),
        protocol_sha256=sha(root / "protocol.json"),
        post_hoc=True,
        training_updates=0,
        gpu_used=False,
    ),
)
print("Exact integer collision/full-affine-span witness verified.")
