"""One import, separately initialized H112 trials, verified zero-storage boundaries."""

import gc
import json
import sys
import time
from pathlib import Path

import sympy
import torch
import torch._dynamo

from results.token_memory_duration_v1.source import worker as original
from results.token_memory_duration_v1.source.common import qualify
from results.whole_job_memory_v1.source.adapters import EvaluationAdapter
from src.core.data import TokenData
from src.core.reproducibility import sha256, write_json

ROOT = Path("results/whole_job_memory_continuation_v1")


def clear_boundary(label):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    assert allocated == 0, (label, allocated)
    return {
        "label": label,
        "allocated_bytes": allocated,
        "reserved_bytes": torch.cuda.memory_reserved(),
    }


def run():
    assert not torch.cuda.is_initialized()
    print(
        f"CPU preload passed: Python={sys.version.split()[0]} torch={torch.__version__} sympy={sympy.__version__}",
        flush=True,
    )
    protocol = json.loads((ROOT / "protocol.json").read_text())
    for path, digest in protocol["sources"].items():
        assert sha256(Path(path)) == digest, path
    checks = qualify()
    assert len(checks) == 8
    write_json(ROOT / "qualification.json", {"passed": True, "checks": checks})
    rows, boundaries = [], [clear_boundary("after_qualification")]
    start = time.perf_counter()
    for corpus_index, dataset in enumerate(("wikitext2", "tinystories")):
        cache = Path(protocol["datasets"][dataset]["path"])
        for seed_index, seed in enumerate((61, 73, 89)):
            policies = (
                ("block", "loss_chunks")
                if (seed_index + corpus_index) % 2 == 0
                else ("loss_chunks", "block")
            )
            pair = []
            for policy in policies:
                label = f"b8_t512_s{seed}_{policy}"
                full_label = f"{dataset}_{label}"
                boundaries.append(clear_boundary("before_" + full_label))
                write_json(
                    ROOT / "coordinator_status.json",
                    {"status": "RUNNING", "label": full_label, "completed_trials": len(rows)},
                )

                def corpus_adapter(path, device, sample_seed):
                    assert path == Path("data/wikitext2_v1")
                    return TokenData(cache, device, sample_seed)

                original.ROOT = ROOT / dataset
                original.TokenData = corpus_adapter
                directory = original.ROOT / "runs" / label
                adapter = EvaluationAdapter(protocol["evaluation_policy"], seed, directory)
                original.evaluate = adapter
                original.run(8, 512, seed, policy)
                assert adapter.calls == 6 and adapter.extra_checkpoint is not None
                write_json(
                    directory / "adapter.json",
                    {
                        "evaluation_policy": protocol["evaluation_policy"],
                        "dataset": dataset,
                        "extra_checkpoints": [adapter.extra_checkpoint],
                        "evaluation_calls": 6,
                    },
                )
                row = json.loads((directory / "metrics.json").read_text())
                row.update(
                    dataset=dataset,
                    evaluation_policy=protocol["evaluation_policy"],
                    source_root=original.ROOT.as_posix(),
                    extra_checkpoints=[adapter.extra_checkpoint],
                )
                boundaries.append(clear_boundary("after_" + full_label))
                rows.append(row)
                pair.append(row)
                write_json(
                    ROOT / "completed_progress.json",
                    {
                        "completed_trials": len(rows),
                        "updates": 800 * len(rows),
                        "boundaries": boundaries,
                    },
                )
                print(
                    json.dumps(
                        {
                            "completed_trials": len(rows),
                            "dataset": dataset,
                            "label": label,
                            "streamed_nll": row["full_validation"]["nll"],
                            "training_mib": row["peak_training_allocated_bytes"] / 2**20,
                            "job_mib": row["peak_job_allocated_bytes"] / 2**20,
                            "elapsed_seconds": time.perf_counter() - start,
                        }
                    ),
                    flush=True,
                )
            for key in (
                "initial_state_hashes",
                "initial_sampler_sha256",
                "final_sampler_sha256",
                "data_order_sha256",
            ):
                assert pair[0][key] == pair[1][key], key
    assert len(rows) == 12 and len(boundaries) == 25
    write_json(
        ROOT / "result.json",
        {
            "status": "COMPLETE",
            "cases": rows,
            "updates": 9600,
            "training_targets": sum(r["training_targets"] for r in rows),
            "boundaries": boundaries,
            "elapsed_seconds": time.perf_counter() - start,
            "primary_quality_requires_native_audit": True,
            "same_process_fresh_models": True,
            "repeated_scientific_updates": 0,
            "broad_goal_achieved": False,
        },
    )
    write_json(ROOT / "coordinator_status.json", {"status": "COMPLETE", "completed_trials": 12})


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        run()
    except Exception:
        import traceback

        write_json(ROOT / "failure.json", {"traceback": traceback.format_exc()})
        write_json(ROOT / "coordinator_status.json", {"status": "FAILED", "see": "failure.json"})
        raise
