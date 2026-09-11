"""Prospective derivation of one shared runner from the frozen H114 profile."""

from pathlib import Path

ROOT = Path("results/fp32_decoder_resource_v1")
original = Path("results/fp32_classifier_profile_v1/source/study.py").read_text()
text = original
changes = [
    (
        '"""H114 complete-model continuation profile; never overwrite an existing case."""',
        '"""H116 shared full-job precision profile; all five arms use this one runner."""',
    ),
    ("    qualify,\n", ""),
    ("    training_loss,\n", ""),
    (
        "from results.streamed_evaluation_v1.source.evaluation import evaluate",
        "from results.fp32_decoder_resource_v1.source.common import (\n"
        "    POLICIES, attention_context, qualify, training_loss,\n)  # noqa: E402\n"
        "from results.streamed_evaluation_v1.source.evaluation import evaluate",
    ),
    (
        'ROOT = Path("results/fp32_classifier_profile_v1")',
        'ROOT = Path("results/fp32_decoder_resource_v1")',
    ),
    ('precision="bf16",', "precision=POLICIES[policy][0],"),
    (
        '    with torch.autocast("cuda", dtype=torch.bfloat16):\n        loss = training_loss(model, x, y, policy)\n    loss.backward()',
        "    loss = training_loss(model, x, y, policy)\n    with attention_context(policy):\n        loss.backward()",
    ),
    (
        '        with torch.autocast("cuda", dtype=torch.bfloat16):\n            loss = training_loss(model, x, y, policy)\n        events[1].record()\n        loss.backward()',
        "        loss = training_loss(model, x, y, policy)\n        events[1].record()\n        with attention_context(policy):\n            loss.backward()",
    ),
    (
        "        policy=policy,",
        "        policy=policy,\n        execution=dict(decoder_precision=POLICIES[policy][0],\n                       attention_backend=POLICIES[policy][1], classifier_policy=POLICIES[policy][2]),",
    ),
    (
        '    write_json(ROOT / "environment.json", environment())',
        "    env = environment()\n    env.update(tf32=False, threads=torch.get_num_threads(),\n"
        "               deterministic_algorithms=torch.are_deterministic_algorithms_enabled())\n"
        '    write_json(ROOT / "environment.json", env)',
    ),
    (
        '    print("16 full-model double qualification cases passed", flush=True)',
        '    print("20 full-model double comparisons / 24 qualification backwards passed", flush=True)',
    ),
    ("    assert len(cases) == 56", "    assert len(cases) == 30"),
    (
        "            fresh_training_runs=0,",
        "            qualification_backward_passes=24,\n            fresh_training_runs=0,",
    ),
]
for before, after in changes:
    assert text.count(before) == 1, before
    text = text.replace(before, after)
target = ROOT / "source/study.py"
assert not target.exists()
target.write_text(text, encoding="utf-8", newline="\n")
launch = Path("results/decoder_gradient_transport_v1/source/launch.py").read_text()
target = ROOT / "source/launch.py"
assert not target.exists()
target.write_text(
    launch.replace("decoder_gradient_transport_v1", "fp32_decoder_resource_v1"),
    encoding="utf-8",
    newline="\n",
)
print("Derived shared study and launcher before the prospective freeze")
