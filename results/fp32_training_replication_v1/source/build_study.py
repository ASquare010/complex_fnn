"""Derive the prospective shared 800-update runner from frozen H116."""

import re
from pathlib import Path

ROOT = Path("results/fp32_training_replication_v1")
text = Path("results/fp32_decoder_resource_v1/source/study.py").read_text()
text = text.replace(
    '"""H116 shared full-job precision profile; all five arms use this one runner."""',
    '"""H117 fresh three-seed training; one common loop for all policies."""',
)
text = re.sub(r"\b50\b", "800", text)
text = re.sub(r"\b51\b", "801", text)
text = text.replace(
    'ROOT = Path("results/fp32_decoder_resource_v1")',
    'ROOT = Path("results/fp32_training_replication_v1")',
)
text = text.replace('fixture["seed"] + 40000', 'fixture["seed"] + 10000')
changes = [
    (
        "from results.streamed_evaluation_v1.source.evaluation import evaluate",
        "from results.fp32_training_replication_v1.source.initialize import generate_initials  # noqa: E402\n"
        "from results.streamed_evaluation_v1.source.evaluation import evaluate",
    ),
    (
        "    records, order = [], hashlib.sha256()",
        "    records, order, checkpoints = [], hashlib.sha256(), []",
    ),
    ("        if step in (1, 20, 800):", "        if step in (1, 20, 200, 400, 800):"),
    (
        "    loop_seconds = time.perf_counter() - loop_start",
        "        if step % 100 == 0:\n"
        '            print(json.dumps(dict(active=label, step=step, loss=record["loss"])), flush=True)\n'
        "        if step in (200, 400):\n"
        '            score = evaluate(model, data, fixture["batch"], cfg.context, 10**9, "classifier_chunks")\n'
        '            ledger.mark(f"validation_{step}")\n'
        "            saved = dict(model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),\n"
        '                         model_config=asdict(cfg), training_config=asdict(tc), seed=fixture["seed"],\n'
        "                         step=step, sampler_state=data.generator.get_state().cpu())\n"
        '            path = folder / f"step{step}.pt"\n'
        "            torch.save(saved, path)\n"
        "            checkpoints.append(dict(step=step, path=path.as_posix(), sha256=sha256(path),\n"
        '                                    model_hash=tree_hash(saved["model"]), optimizer_hash=tree_hash(saved["optimizer"]),\n'
        '                                    sampler_hash=tensor_hash(saved["sampler_state"]), validation=score))\n'
        "            del saved, score\n"
        '            ledger.mark(f"checkpoint_{step}")\n'
        "    loop_seconds = time.perf_counter() - loop_start",
    ),
    (
        '        st.mean(r["wall_update_ms"] for r in timed[start : start + 10]) for start in (0, 10, 20)',
        '        st.mean(r["wall_update_ms"] for r in timed[start : start + 260]) for start in (0, 260, 520)',
    ),
    (
        "        checkpoint=dict(path=path.as_posix(), sha256=sha256(path)),",
        "        checkpoint=dict(path=path.as_posix(), sha256=sha256(path)),\n        intermediate_checkpoints=checkpoints,",
    ),
    (
        "    boundaries, cases = [clear_boundary()], []",
        "    initials = generate_initials(protocol)\n"
        '    write_json(ROOT / "initializations.json", initials)\n'
        '    protocol["checkpoint_hashes"] = {r["path"]: r["sha256"] for r in initials["initializations"]}\n'
        "    boundaries, cases = [clear_boundary()], []",
    ),
    ("    assert len(cases) == 30", "    assert len(cases) == 18"),
    ("            fresh_training_runs=0,", "            fresh_training_runs=18,"),
]
for before, after in changes:
    assert text.count(before) == 1, before
    text = text.replace(before, after)
# The deliberate scientific-library preload precedes the remaining imports.
text = text.replace(
    "from results.fp32_decoder_resource_v1.source.common import (\n",
    "from results.fp32_decoder_resource_v1.source.common import (  # noqa: E402\n",
)
target = ROOT / "source/study.py"
assert not target.exists()
target.write_text(text, encoding="utf-8", newline="\n")
launch = Path("results/fp32_decoder_resource_v1/source/launch.py").read_text()
target = ROOT / "source/launch.py"
assert not target.exists()
target.write_text(
    launch.replace("fp32_decoder_resource_v1", "fp32_training_replication_v1"),
    encoding="utf-8",
    newline="\n",
)
print("Derived the shared fresh-training runner; freeze only after inspection and lint")
