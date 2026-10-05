# Readout reuse execution v3: forward repaired, gradient gate fails

Retire v3 before resource or language work. The actual v3 implementation passes
CPU equations/gradients, initialization, counts/quadrature and exact recovery.
CUDA completed 192 independent FFN fixtures and 18 whole-model comparisons.
The actual full BF16 grouped model's output is bit-identical to ordinary
execution, but its embedding gradient fails the unchanged .0001/.05 tolerance.
The repeat rejects the same three of 2,097,152 entries; maximum disallowed
absolute difference is .0001691557. Overall maximum embedding-gradient difference
is .0003688149, including entries allowed by relative tolerance. No complete
CUDA pass record exists, and later raw/cast/regression checks were not reached.

The first v3-labelled preparation suites accidentally imported v2. Those
records remain preserved and explicitly marked as a setup error, not actual
v3 evidence. Corrected module/fingerprint paths then produced the genuine
v3 result above. All setup, patch and numerical errors are recorded.

Saved-fixture diagnosis uses identical native forward inputs and incoming
gradients for each FFN. Forward outputs match exactly in all four layers.
Fused grouped/scalar backward changes 63/79/64/69 BF16 projected adjoints;
using native grouped adjoints with the same raw scalar derivative still changes
16/25/25/29. Differences are small (max projected-adjoint differences up to
4.8e-7), but they cross BF16 rounding boundaries. Same-input local input-gradient
differences reach 2.4e-7 and projection-weight differences 3.1e-5. The whole-model
embedding comparison remains decisive; local tolerance passes do not substitute
for it. This identifies both group accumulation and scalar derivative ordering
as relevant on this fixture, not a proven performance bottleneck or universal
precision bound.

Weights and mathematical counts remain 2,506,752 (70.35% fewer FFN weights),
9,732,096 forward projection FLOPs/token. VRAM, throughput and language quality
are unknown for v3. Gates, widths, controls and initialization were not changed.

Evidence: records/readout-reuse-v3-cpu-checks.json,
records/readout-reuse-v3-gradient-failure.json,
records/readout-reuse-v3-gradient-diagnostic.json,
records/readout-reuse-v3-development-errors.json. Actual fixture:
dump/readout-reuse-v3-gradient-failure.pt. Earlier preparation records carry
their v2 source hashes and are not accepted as v3 validation.
