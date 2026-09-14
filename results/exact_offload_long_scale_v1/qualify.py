"""Apply prospective H163 host gates; preserve the underlying H161-style summary."""

import statistics as st

from results.exact_offload_long_scale_v1.run import ROOT
from results.exact_offload_long_scale_v1.study import read, verify, write


def main():
    protocol = verify()
    summary = read(ROOT / "summary.json")
    peaks, pairs, stats = [], [], {}
    for case in protocol["schedule"]:
        data = read(ROOT / f"case{case['index']:02d}" / "host_memory.json")
        for value in data.values():
            assert value["cb"] == 80
            assert value["PeakWorkingSetSize"] >= value["WorkingSetSize"] > 0
            assert value["PeakPagefileUsage"] >= value["PrivateUsage"] > 0
        for key in ("PeakWorkingSetSize", "PeakPagefileUsage"):
            values = [
                data[stage][key] for stage in ("before_torch", "before_worker", "after_worker")
            ]
            assert values == sorted(values)
        peaks.append(
            dict(
                **case,
                working_set=data["after_worker"]["PeakWorkingSetSize"],
                private_commit=data["after_worker"]["PeakPagefileUsage"],
            )
        )
    for seed in (401, 409, 419):
        a = next(r for r in peaks if r["seed"] == seed and r["arm"] == "ordinary")
        b = next(r for r in peaks if r["seed"] == seed and r["arm"] == "helper")
        delta = {k: b[k] - a[k] for k in ("working_set", "private_commit")}
        prior_gate = next(r for r in summary["pairs"] if r["seed"] == seed)["passed"]
        gates = dict(
            numeric_resource=prior_gate,
            working_set=delta["working_set"] <= protocol["host_increment_limit_bytes"],
            private_commit=delta["private_commit"] <= protocol["host_increment_limit_bytes"],
            absolute=all(
                r[k] <= protocol["host_peak_limit_bytes"]
                for r in (a, b)
                for k in ("working_set", "private_commit")
            ),
        )
        pairs.append(
            dict(seed=seed, host_delta_bytes=delta, gates=gates, passed=all(gates.values()))
        )
    for arm in ("ordinary", "helper"):
        stats[arm] = {}
        for key in ("working_set", "private_commit"):
            values = [r[key] for r in peaks if r["arm"] == arm]
            stats[arm][key] = dict(
                mean=st.mean(values),
                median=st.median(values),
                sample_variance=st.variance(values),
                n=3,
            )
    write(
        ROOT / "qualification.json",
        dict(
            study="H163",
            passed=all(r["passed"] for r in pairs),
            peaks=peaks,
            pairs=pairs,
            statistics=stats,
            optimizer_updates=4800,
            backwards=4806,
            terminal_convergence_qualified=False,
            goal_achieved=False,
        ),
    )
    print(pairs, flush=True)


if __name__ == "__main__":
    main()
