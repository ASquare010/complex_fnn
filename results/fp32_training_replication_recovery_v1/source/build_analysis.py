"""Derive reporting from H116, retaining stricter per-seed H117 decisions."""

from pathlib import Path

ROOT = Path("results/fp32_training_replication_recovery_v1")
text = Path("results/fp32_decoder_resource_v1/source/analyze.py").read_text()
text = text.replace(
    'ROOT = Path("results/fp32_decoder_resource_v1")',
    'ROOT = Path("results/fp32_training_replication_recovery_v1")',
)
changes = [
    ("    rows, updates = [], []", "    rows, updates, curves = [], [], []"),
    (
        "        rows.append(row)",
        "        rows.append(row)\n"
        '        scores = [(0, case["initial_validation"])]\n'
        '        scores += [(c["step"], c["validation"]) for c in case["intermediate_checkpoints"]]\n'
        '        scores += [(800, case["final_validation"])]\n'
        "        for step, score in scores:\n"
        '            curves.append(dict(label=case["label"], fixture=case["fixture"]["label"],\n'
        '                               corpus=case["fixture"]["dataset"], seed=case["fixture"]["seed"],\n'
        '                               policy=case["policy"], step=step, nll=score["nll"],\n'
        '                               targets=score["targets"], order_sha256=score["order_sha256"]))',
    ),
    (
        'wall_bf16=st.median(p["wall_ratio_bf16"] for p in peers) <= 1.25,',
        'wall_bf16=all(p["wall_ratio_bf16"] <= 1.25 for p in peers),',
    ),
    (
        'wall_native=st.median(p["wall_ratio_native"] for p in peers) <= 1.25,',
        'wall_native=all(p["wall_ratio_native"] <= 1.25 for p in peers),',
    ),
    ("short_quality_bf16=", "final_quality_bf16="),
    ("short_quality_native=", "final_quality_native="),
    ("qualifies_broader_replication=", "qualifies_scoped_800_update_component="),
    ("fresh_training_allocated=False,", "new_training_allocation=False,"),
    ('"PROMISING"', '"VALIDATED IN THIS SCOPE"'),
    ('"ELIMINATED from this resource gate"', '"ELIMINATED from this fixed recipe"'),
    ("        cases=30,", "        cases=18,"),
    (
        "        audit_backward_passes=24,",
        "        audit_backward_passes=12,\n        failed_initial_qualification_backward_passes=24,\n        total_backward_passes_across_attempts=14478,",
    ),
    (
        '        pareto_note="Raw 50-update NLL is not a statistically established quality difference.",',
        '        pareto_note="Raw 800-update NLL ordering has limited statistical power with three seeds.",',
    ),
    ('if d["qualifies_broader_replication"]', 'if d["qualifies_scoped_800_update_component"]'),
    (
        "        native_validation_scores=36,",
        "        native_validation_scores=60,\n        regenerated_initializations=6,\n        trained_checkpoints_verified=54,",
    ),
    (
        '[r["final_score_relative_error"] for r in audit["rows"]]',
        '[s["relative_error"] for r in audit["rows"] for s in r["checkpoint_scores"]]',
    ),
    ("        fresh_training_runs=0,", "        fresh_training_runs=18,"),
    (
        '    (ROOT / "summary.json").write_text(',
        '    summary["two_corpus_component_qualified"] = all(d["qualifies_scoped_800_update_component"] for d in decisions)\n'
        '    summary["convergence"] = [dict(corpus=corpus, policy=policy, step=step,\n'
        '        statistics=describe([r["nll"] for r in curves if r["corpus"] == corpus and r["policy"] == policy and r["step"] == step]))\n'
        '        for corpus in ("wikitext2", "tinystories") for policy in protocol["policies"] for step in (0, 200, 400, 800)]\n'
        '    (ROOT / "summary.json").write_text(',
    ),
    (
        '    table("updates.csv.gz", updates)',
        '    table("updates.csv.gz", updates)\n    table("curves.csv.gz", curves)',
    ),
]
for before, after in changes:
    assert text.count(before) == 1, before
    text = text.replace(before, after)
target = ROOT / "source/analyze.py"
assert not target.exists()
target.write_text(text, encoding="utf-8", newline="\n")
print("Derived all-seed endpoint and convergence analysis; no new scientific execution")
