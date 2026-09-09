"""Alias all eight unchanged H082 checks after correcting only adapter binding."""

from results.latent_activation_recovery_v1.source.adapter import study
from results.latent_activation_v1.source.test_harness import (
    test_counts_shared_initialization_and_optimizer as test_counts_shared_initialization_and_optimizer,
)
from results.latent_activation_v1.source.test_harness import (
    test_finite_differences as test_finite_differences,
)
from results.latent_activation_v1.source.test_harness import (
    test_identity_and_checkpoint_fidelity as test_identity_and_checkpoint_fidelity,
)
from results.latent_activation_v1.source.test_harness import (
    test_promotion_and_failure_rules as test_promotion_and_failure_rules,
)
from results.latent_activation_v1.source.test_harness import (
    test_scalar_bounds_and_bezier_identity as test_scalar_bounds_and_bezier_identity,
)
from results.latent_activation_v1.source.test_harness import (
    test_scaling_sampler_and_permutation as test_scaling_sampler_and_permutation,
)

assert study.ROOT.as_posix() == "results/latent_activation_recovery_v1"
