"""All-condition analysis; retain every repeat and the original numerical gates."""

import faulthandler

faulthandler.dump_traceback_later(45, repeat=True)

import sympy  # noqa: E402,F401
import torch  # noqa: E402
import torch._dynamo  # noqa: E402,F401

assert not torch.cuda.is_initialized()
torch.set_num_threads(4)

import csv  # noqa: E402
import gzip  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import statistics as st  # noqa: E402
from pathlib import Path  # noqa: E402

from results.decoder_gradient_transport_v1.source.common import MODES, tensor_error  # noqa: E402
from results.fp32_classifier_profile_v1.source.common import gradient_error  # noqa: E402

ROOT = Path("results/decoder_gradient_transport_v1")


def read(path):
    return json.loads(Path(path).read_text())


def describe(values):
    return dict(
        n=len(values),
        mean=st.mean(values),
        median=st.median(values),
        sample_variance=st.variance(values) if len(values) > 1 else None,
        min=min(values),
        max=max(values),
    )


def table(path, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    Path(path).write_bytes(gzip.compress(stream.getvalue().encode(), mtime=0))


def run():
    result, audit = read(ROOT / "result.json"), read(ROOT / "audit.json")
    assert result["status"] == "COMPLETE" and audit["passed"]
    rows, pairs, replays, traces = [], [], [], []
    for case in result["conditions"]:
        incoming = case["classifier_difference"]["hidden_gradient"]["relative_l2"]
        nontrivial = [r for r in case["records"] if r["repetition"] > 0]
        row = dict(
            label=case["label"],
            fixture=case["fixture"]["label"],
            scope=case["fixture"]["scope"],
            mode=case["mode"],
            backend_nodes="|".join(case["attention_backward_nodes"]),
            incoming_hidden_gradient_error=incoming,
            classifier_weight_gradient_error=case["classifier_difference"]["weight_gradient"][
                "relative_l2"
            ],
            paired_total_median=st.median(
                p["total"]["global_relative_l2"] for p in case["paired_repetitions"]
            ),
            paired_total_max=max(
                p["total"]["global_relative_l2"] for p in case["paired_repetitions"]
            ),
            paired_tensor_max=max(
                p["total"]["max_tensor_relative_l2"] for p in case["paired_repetitions"]
            ),
            maximum_replay_global=max(r["replay_total"]["global_relative_l2"] for r in nontrivial),
            maximum_replay_tensor=max(
                r["replay_total"]["max_tensor_relative_l2"] for r in nontrivial
            ),
            bitwise_repeat_matches=sum(r["bitwise_replay_total"] for r in nontrivial),
            repeated_comparisons=len(nontrivial),
            all_forward_boundaries_equal=all(r["forward_equal"] for r in case["records"]),
            peak_diagnostic_mib=case["peak_diagnostic_allocated_bytes"] / 2**20,
            peak_reserved_mib=case["peak_diagnostic_reserved_bytes"] / 2**20,
            instrumented_pass_median_seconds=st.median(
                r["instrumented_seconds"] for r in case["records"]
            ),
        )
        row["paired_relative_amplification_median"] = row["paired_total_median"] / incoming
        row["transport_gate_passed"] = (
            row["paired_total_max"] <= 0.002
            and row["paired_tensor_max"] <= 0.02
            and row["all_forward_boundaries_equal"]
        )
        row["repeat_gate_passed"] = (
            row["maximum_replay_global"] <= 1e-5
            and row["maximum_replay_tensor"] <= 1e-4
            and row["all_forward_boundaries_equal"]
        )
        rows.append(row)
        for pair in case["paired_repetitions"]:
            pairs.append(
                dict(
                    label=case["label"],
                    mode=case["mode"],
                    repetition=pair["repetition"],
                    decoder_global_error=pair["decoder"]["global_relative_l2"],
                    total_global_error=pair["total"]["global_relative_l2"],
                    total_max_tensor_error=pair["total"]["max_tensor_relative_l2"],
                )
            )
            for boundary, values in pair["trace"].items():
                traces.append(
                    dict(
                        label=case["label"],
                        mode=case["mode"],
                        repetition=pair["repetition"],
                        boundary=boundary,
                        relative_l2=values["relative_l2"],
                        absolute_error=values["max_absolute_error"],
                        relative_amplification=values["relative_l2"] / incoming,
                    )
                )
        for record in nontrivial:
            replays.append(
                dict(
                    label=case["label"],
                    mode=case["mode"],
                    repetition=record["repetition"],
                    policy=record["policy"],
                    global_error=record["replay_total"]["global_relative_l2"],
                    max_tensor_error=record["replay_total"]["max_tensor_relative_l2"],
                    bitwise_equal=record["bitwise_replay_total"],
                )
            )
    groups, decisions = [], []
    metric_keys = (
        "incoming_hidden_gradient_error",
        "paired_total_median",
        "paired_total_max",
        "paired_tensor_max",
        "maximum_replay_global",
        "maximum_replay_tensor",
        "peak_diagnostic_mib",
        "peak_reserved_mib",
        "instrumented_pass_median_seconds",
        "paired_relative_amplification_median",
    )
    for mode in MODES:
        peers = [r for r in rows if r["mode"] == mode]
        groups.append(
            dict(
                mode=mode,
                fixtures=6,
                metrics={k: describe([p[k] for p in peers]) for k in metric_keys},
                bitwise_repeat_matches=sum(p["bitwise_repeat_matches"] for p in peers),
                repeated_comparisons=36,
            )
        )
        gates = dict(
            all_forward_boundaries_equal=all(p["all_forward_boundaries_equal"] for p in peers),
            all_transport_gates=all(p["transport_gate_passed"] for p in peers),
            all_repeat_gates=all(p["repeat_gate_passed"] for p in peers),
        )
        decisions.append(
            dict(
                mode=mode,
                gates=gates,
                qualifies_uninstrumented_resource_screen=all(gates.values()),
                fresh_language_training_earned=False,
                global_determinism_proved=False,
            )
        )
    cross_mode = []
    for fixture in dict.fromkeys(c["fixture"]["label"] for c in result["conditions"]):
        selected = [c for c in result["conditions"] if c["fixture"]["label"] == fixture]
        hidden, first = {}, {}
        for case in selected:
            paths = {Path(a["path"]).name: a["path"] for a in case["artifacts"]}
            payload = torch.load(paths["inputs.pt"], map_location="cpu", weights_only=True)
            hidden[case["mode"]] = payload["hidden"]
            if case["mode"] in ("bf16_default_block", "bf16_default_none"):
                first[case["mode"]] = torch.load(
                    paths["first_passes.pt"], map_location="cpu", weights_only=True
                )
        comparisons = {
            mode: tensor_error(h, hidden["bf16_default_block"]) for mode, h in hidden.items()
        }
        checkpoint = {
            policy: gradient_error(
                first["bf16_default_none"][policy]["total"],
                first["bf16_default_block"][policy]["total"],
            )
            for policy in ("native_fp32", "chunks_fp32")
        }
        cross_mode.append(
            dict(
                fixture=fixture,
                forward_comparisons_to_bf16_default_block=comparisons,
                checkpoint_first_gradient_comparisons=checkpoint,
            )
        )
    summary = dict(
        status="COMPLETE",
        conditions=30,
        classifier_backward_passes=60,
        decoder_backward_passes=240,
        qualification_backward_passes=20,
        audit_backward_passes=0,
        optimizer_updates=0,
        elapsed_seconds=result["elapsed_seconds"],
        groups=groups,
        decisions=decisions,
        cross_mode=cross_mode,
        resource_screen_candidates=[
            d["mode"] for d in decisions if d["qualifies_uninstrumented_resource_screen"]
        ],
        whole_job_memory_measured=False,
        fresh_language_training_earned=False,
        broad_goal_achieved=False,
        independent_audit={k: v for k, v in audit.items() if k not in ("rows", "boundaries")},
    )
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name, values in (
        ("metrics", rows),
        ("pairs", pairs),
        ("replays", replays),
        ("traces", traces),
    ):
        table(ROOT / (name + ".csv.gz"), values)
    print(
        json.dumps(dict(decisions=decisions, elapsed_seconds=result["elapsed_seconds"]), indent=2)
    )


if __name__ == "__main__":
    try:
        run()
    finally:
        faulthandler.cancel_dump_traceback_later()
