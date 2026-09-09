"""Run the unchanged H079 test functions with only artifact storage rebound."""

from pathlib import Path

from results.blast_operator_recovery_v1.source.storage import record_observation
from results.blast_operator_v1.source import test_harness as frozen

ROOT = Path("results/blast_operator_recovery_v1/observations")


def durable_record(label, kind, data, tensors=None):
    record_observation(ROOT, label, kind, data, frozen.cpu(tensors))


frozen.ROOT = ROOT
frozen.record = durable_record
# Aliases preserve the original functions and their parametrization markers.
test_shapes_and_counts = frozen.test_shapes_and_counts
test_projection_fp64 = frozen.test_projection_fp64
test_small_finite_difference = frozen.test_small_finite_difference
test_ffn_checkpoint = frozen.test_ffn_checkpoint
test_exact_blockshuffle_embedding = frozen.test_exact_blockshuffle_embedding
test_hadamard_witness = frozen.test_hadamard_witness
