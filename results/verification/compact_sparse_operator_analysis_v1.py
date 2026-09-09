"""Independent audit of H086's completed native portion; preserve backend failure."""

import hashlib
import json
import os
import zipfile
from pathlib import Path

import torch

ROOT = Path("results/compact_sparse_operator_v1")
torch.set_num_threads(4)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    assert Path(path).read_bytes() == data


protocol = read(ROOT / "protocol.json")
assert all(sha(p) == h for p, h in {**protocol["sources"], **protocol["plans"], **protocol["anchors"]}.items())
status = read(ROOT / "coordinator_status.json")
process = read(ROOT / "qualification_process.json")
assert status["status"] == "FAIL" and process["returncode"] == 1
assert "CUTLASS not supported" in (ROOT / "qualification.log").read_text()
assert not (ROOT / "result.json").exists()
observations = {p.stem: read(p) for p in (ROOT / "observations").glob("*.json")}
assert len(observations) == 16 and all(o["status"] == "PASS" for o in observations.values())
ordinary_pairs = checkpoint_pairs = 0
for label, observation in observations.items():
    if "tensor_file" not in observation:
        assert label == "finite_difference" and len(observation["rows"]) == 5
        for row in observation["rows"]:
            assert abs(row["numerical"] - row["analytic"]) <= 1e-7 + 1e-5 * abs(row["analytic"])
        continue
    path = ROOT / "observations" / observation["tensor_file"]
    assert sha(path) == observation["tensor_sha256"]
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "actual" in payload:
        for key, value in payload["actual"].items():
            torch.testing.assert_close(value, payload["expected"][key],
                                       atol=observation.get("atol", 1e-11),
                                       rtol=observation.get("rtol", 1e-10))
            ordinary_pairs += 1
            if "checkpoint" in payload:
                assert torch.equal(value, payload["checkpoint"][key])
                checkpoint_pairs += 1
    if observation["kind"] in ("projection", "rank_witness"):
        v, p = payload["values"], payload["positions"].long()
        m, groups, _ = v.shape
        matrix = torch.zeros(m, groups * 4, dtype=v.dtype)
        rows = torch.arange(m)[:, None]
        for slot in range(2):
            columns = 4 * torch.arange(groups)[None, :] + p[..., slot]
            matrix[rows, columns] = v[..., slot]
        assert torch.equal(matrix, payload["matrix"])
        if observation["kind"] == "projection":
            x, cotangent = payload["input"], payload["cotangent"]
            torch.testing.assert_close(cotangent @ matrix, payload["actual"]["input_gradient"],
                                       atol=1e-11, rtol=1e-10)
            dense_gradient = cotangent.T @ x
            gathered = dense_gradient.reshape(m, groups, 4).gather(-1, p)
            torch.testing.assert_close(gathered, payload["actual"]["value_gradient"],
                                       atol=1e-11, rtol=1e-10)
            torch.testing.assert_close(payload["delta"].square().sum(),
                                       payload["lifted_delta"].square().sum(), atol=1e-11, rtol=1e-10)
        else:
            assert torch.equal(matrix @ matrix.T, 8 * torch.eye(384, dtype=torch.int64))
            for i in range(8):
                for j in range(8):
                    block = matrix[i*48:(i+1)*48, j*48:(j+1)*48]
                    assert torch.equal(block @ block.T, torch.eye(48, dtype=torch.int64))
    if label == "counts":
        for state in payload.values():
            assert sum(v.numel() for n, v in state.items() if n.endswith("values")) == 350208
            assert sum(v.numel() * v.element_size() for n, v in state.items() if n.endswith("positions")) == 350208
result = {"status": "PASS_PARTIAL_EVIDENCE_AUDIT", "scientific_status": "INCOMPLETE_BACKEND_QUALIFICATION",
          "native_checks_passed": 16, "hardware_cases_attempted": 1, "hardware_cases_remaining": 11,
          "hardware_error": "sparse_semi_structured_mad_op : CUTLASS not supported",
          "ordinary_tensor_pairs": ordinary_pairs, "exact_checkpoint_pairs": checkpoint_pairs,
          "tensor_archives_verified": 15, "optimizer_updates": 0, "corpus_targets": 0,
          "research_goal_achieved": False, "source_count": len(protocol["sources"]),
          "plan_count": len(protocol["plans"]), "prior_anchors_preserved": True,
          "protocol_sha256": sha(ROOT / "protocol.json"), "source_zip_sha256": sha(ROOT / "source.zip"),
          "analysis_source_sha256": sha(__file__),
          "observations": {p.as_posix(): sha(p) for p in (ROOT / "observations").iterdir()}}
write("results/verification/compact_sparse_operator_analysis_v1.json", result)
print(json.dumps({k: v for k, v in result.items() if k != "observations"}, indent=2))
