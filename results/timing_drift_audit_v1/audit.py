"""CPU-only, receipt-verified timing/telemetry analysis of every H138 run."""

import bisect
import collections
import csv
import datetime as dt
import hashlib
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path("results/timing_drift_audit_v1")
SOURCE = Path("results/partial_offload_long_v1")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def stats(values):
    return (
        dict(
            n=len(values),
            min=min(values),
            median=st.median(values),
            mean=st.mean(values),
            max=max(values),
        )
        if values
        else None
    )


def correlation(x, y):
    if len(x) < 3 or st.pvariance(x) == 0 or st.pvariance(y) == 0:
        return None
    value = st.correlation(x, y)
    assert math.isfinite(value)
    return value


def summarize(records, samples):
    out = dict(
        first_step=records[0]["step"],
        last_step=records[-1]["step"],
        updates=len(records),
        telemetry_samples=len(samples),
    )
    for key in ("event_update_ms", "wall_update_ms"):
        out[key] = stats([r[key] for r in records])
    for key in records[0]:
        if key.startswith("event_") and key.endswith("_ms") and key != "event_update_ms":
            out[key] = stats([r[key] for r in records])
    out["sensors"] = {
        key: stats([s[key] for s in samples if s[key] is not None])
        for key in ("sm_mhz", "memory_mhz", "temperature_c", "power_w", "utilization_pct")
    }
    out["pstates"] = dict(collections.Counter(s["pstate"] for s in samples))
    out["sampled_updates"] = len({s["step"] for s in samples})
    return out


def run():
    assert not (ROOT / "protocol.json").exists()
    receipt = read(SOURCE / "receipt.json")
    for path, digest in receipt["files"].items():
        assert sha(path) == digest, path
    assert receipt["gate"] == "FAIL four-block fresh-training gate"
    for src, target in [
        ("research/CURRENT_STATE.md", "CURRENT_STATE.before.md"),
        ("README.md", "README.before.md"),
    ]:
        (ROOT / target).write_bytes(Path(src).read_bytes())
    inputs = dict(receipt["files"])
    inputs[(SOURCE / "receipt.json").as_posix()] = sha(SOURCE / "receipt.json")
    protocol = dict(
        study="H139",
        training_updates=0,
        backwards=0,
        measured_updates=9360,
        windows=360,
        input_hashes=inputs,
        sources={p: sha(p) for p in [__file__, "research/timing_drift_audit_plan.md"]},
    )
    write(ROOT / "protocol.json", protocol)
    result = []
    source_summary = read(SOURCE / "summary.json")
    for index in range(12):
        folder = SOURCE / f"case{index:02d}"
        case = read(folder / "case.json")
        m = case["measurement"]
        records = m["records"][20:]
        assert len(records) == 780
        assert [r["step"] for r in records] == list(range(21, 801))
        starts = [r["at_ns"] for r in records]
        assert starts == sorted(starts)
        assert all(r["until_ns"] >= r["at_ns"] for r in records)
        assert all(a["until_ns"] <= b["at_ns"] for a, b in zip(records, records[1:]))
        samples = []
        ignored = 0
        for row in csv.reader((folder / "telemetry.csv").read_text().splitlines()):
            stamp, gpu, uuid, state, temp, sm, mem, power, limit, util = [v.strip() for v in row]
            if int(gpu) != 0:
                continue
            at = int(dt.datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f").timestamp() * 1e9)
            j = bisect.bisect_right(starts, at) - 1
            if j < 0 or at > records[j]["until_ns"]:
                ignored += 1
                continue
            sample = dict(at_ns=at, step=records[j]["step"], pstate=state, uuid=uuid)
            for key, value in [
                ("temperature_c", temp),
                ("sm_mhz", sm),
                ("memory_mhz", mem),
                ("power_w", power),
                ("utilization_pct", util),
            ]:
                sample[key] = None if value in ("[N/A]", "N/A", "[Not Supported]") else float(value)
            samples.append(sample)
        windows, blocks = [], []
        for width, target in [(26, windows), (260, blocks)]:
            for start in range(0, 780, width):
                group = records[start : start + width]
                selected = [
                    s for s in samples if group[0]["step"] <= s["step"] <= group[-1]["step"]
                ]
                target.append(summarize(group, selected))
        means = [b["wall_update_ms"]["mean"] for b in blocks]
        assert means == m["timing_block_means"]
        stability = max(means) / min(means)
        assert stability == m["timing_stability_ratio"]
        paired = [w for w in windows if w["sensors"]["sm_mhz"]]
        result.append(
            dict(
                index=index,
                dataset=case["fixture"]["dataset"],
                seed=case["fixture"]["seed"],
                arm=case["arm"],
                blocks=blocks,
                windows=windows,
                stability=stability,
                sampled_updates=len({s["step"] for s in samples}),
                active_samples=len(samples),
                excluded_samples=ignored,
                clock_wall_correlation=correlation(
                    [w["sensors"]["sm_mhz"]["mean"] for w in paired],
                    [w["wall_update_ms"]["mean"] for w in paired],
                ),
                whole=summarize(records, samples),
            )
        )
    assert sum(sum(w["updates"] for w in r["windows"]) for r in result) == 9360
    assert sum(len(r["windows"]) for r in result) == 360
    assert sum(len(r["blocks"]) for r in result) == 36
    for row, old in zip(result, source_summary["metrics"], strict=True):
        assert row["index"] == old["index"] and row["stability"] == old["stability"]
        assert row["whole"]["event_update_ms"]["median"] == old["event_ms"]
    write(
        ROOT / "analysis.json",
        dict(
            study="H139",
            runs=result,
            measured_updates=9360,
            training_updates=0,
            backwards=0,
            prior_gate_unchanged=True,
        ),
    )
    rows = []
    for r in result:
        clocks = [
            b["sensors"]["sm_mhz"]["median"] if b["sensors"]["sm_mhz"] else None
            for b in r["blocks"]
        ]
        times = [round(b["wall_update_ms"]["mean"], 2) for b in r["blocks"]]
        rows.append(
            f"| {r['index']} | {r['dataset']} | {r['seed']} | {r['arm']} | {times} | {clocks} | {r['stability']:.4f} | {r['clock_wall_correlation']:.3f} |"
        )
    report = (
        """# H139: timing drift in the failed four-block validation

**H138 remains failed; no candidate is promoted.** This CPU-only audit verifies
all prior evidence hashes and accounts for every one of the 9,360 timed updates,
in 360 windows and 36 original timing blocks. It uses zero GPU updates/backwards.

| Case | Corpus | Seed | Arm | Three block mean wall times (ms) | Active-sample block median SM clocks (MHz) | Stability ratio | Window clock/time Pearson r |
|---:|---|---:|---|---|---|---:|---:|
"""
        + "\n".join(rows)
        + """

Each block has 260 updates; each correlation uses up to 30 consecutive 26-update
windows after the original 20 warmups. Sensor samples are included only when
they fall inside a measured update, excluding validation, serialization and
sampling gaps. Sampling is sparse (200 ms), not a per-kernel measurement; missing
sensors stay missing. Window observations are autocorrelated. Correlations are
descriptive, with no causal inference or significance claim. P-states, phase
summaries, sample coverage and every window are retained in `analysis.json`.

The first WikiText ordinary/candidate pair ran at different clock distributions.
The final TinyStories ordinary control changes speed substantially within its
run. These observations make sequential whole-run timing comparisons vulnerable
to time-varying device conditions. They do not establish whether temperature,
power management, scheduling or another workload caused the changes. No measured
time is clock-normalized, removed or replaced, and no failed gate is rescued.

## Next experiment

Separate the already audited convergence evidence from a new, explicitly timed
experiment: interleave short complete-training segments for both arms at the
same saved model/optimizer states, in balanced ABBA/BAAB rounds for every corpus
and seed. Keep only one model resident, account for its complete job allocation,
record state reload overhead separately, and include forward/backward/clipping/
Adam plus offload transfers inside the measured segment. Use every round, report
paired ratios and within-run drift, retain the original 15% runtime limit, and
freeze exact counts/order/numerical checks before GPU execution. This tests
runtime under temporal pairing; it does not replace H138 or establish deployment
throughput, convergence or generality. Do not repeat 9,600 training updates merely
to obtain a more favorable sequential timing result.

The broader objective still requires robust VRAM/quality/compute savings and
independent exploration of parameter-efficient FFN geometry. Offload and loss
buffer reuse are established memory mechanisms, not novelty claims.

[Plan](timing_drift_audit_plan.md) ·
[Machine-readable analysis](../results/timing_drift_audit_v1/analysis.json)
"""
    )
    Path("research/timing_drift_audit_results.md").write_text(report, encoding="utf-8")
    print("Verified: 12 runs, 9360 measured updates, 360 windows; zero GPU work.")


if __name__ == "__main__":
    run()
