"""Independently verify H080 saved tensors, algebra and preserved H079 damage."""

import json
import math
import zipfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

import torch

from results.blast_operator_recovery_v1.source.storage import read, sha, write_json

ROOT = Path("results/blast_operator_recovery_v1")
OLD = Path("results/blast_operator_v1")
OUT = Path("results/verification/blast_operator_recovery_analysis_v2.json")
assert not OUT.exists()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
p, r = read(ROOT / "protocol.json"), read(ROOT / "result.json")
assert r["status"] == "LOCALLY_QUALIFIED" and r["checks_passed"] == 34
assert read(ROOT / "coordinator_status.json")["status"] == "PASS"
assert len(p["sources"]) == 118 and all(sha(n) == h for n, h in p["sources"].items())
assert all(sha(n) == h for n, h in r["observation_hashes"].items())
assert len(r["observation_hashes"]) == 66
assert r["optimizer_updates"] == r["corpus_targets"] == r["full_transformer_runs"] == 0
assert r["total_h079_h080_attempts"] == 2 and r["explicit_qualification_repetitions"] == 1
for phase in ("collection", "qualification"):
    event = read(ROOT / (phase + "_process.json"))
    assert event["status"] == "PASS" and event["returncode"] == 0 and event["source_unchanged"]
    assert event["log_sha256"] == sha(ROOT / (phase + ".log"))
    assert event["protocol_sha256"] == sha(ROOT / "protocol.json")
    assert event["source_archive_sha256"] == sha(ROOT / "source.zip")
assert "34 tests collected" in (ROOT / "collection.log").read_text(encoding="utf-8")
assert "34 passed" in (ROOT / "qualification.log").read_text(encoding="utf-8")
with zipfile.ZipFile(ROOT / "source.zip") as z:
    assert z.testzip() is None and len(z.namelist()) == 121
    for n, h in p["sources"].items():
        import hashlib
        assert hashlib.sha256(z.read(n)).hexdigest() == h


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def same(a, b):
    if isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in b)
    if isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def reconstruct(f):
    u, v, s = f["row"], f["column"], f["mix"]
    b, rows, rank = u.shape
    cols = v.shape[2]
    out = torch.zeros(b * rows, b * cols, dtype=u.dtype)
    for i in range(b):
        for j in range(b):
            block = out[i*rows:(i+1)*rows, j*cols:(j+1)*cols]
            for k in range(rank):
                block.add_(torch.outer(u[i, :, k], v[j, k, :]), alpha=float(s[k, i, j]))
    return out


def unshuffle(x):
    return x.reshape(*x.shape[:-1], x.shape[-1]//8, 8).transpose(-2, -1).reshape_as(x)


counts = Counter()
checkpoint_pairs = projection_pairs = 0
max_gram = max_pair_error = max_analytic_error = 0.0
prior_payloads, prior_json_hashes = {}, {}
independent = {}
integrity = read(OLD / "integrity_observation.json")
assert all(sha(n) == v["sha256"] for n, v in integrity["files"].items())
for label, q in sorted(r["observations"].items()):
    assert q == read(ROOT / "observations" / (label + ".json"))
    assert q["status"] == "PASS" and q["optimizer_updates"] == 0 and finite(q)
    kind = q["kind"]
    counts[kind] += 1
    if "tensor_file" not in q:
        if kind == "counts":
            assert {f:v["hidden"] for f,v in q["counts"].items()} == {"gelu":3200,"swiglu":1984}
            for f, projections in (("gelu",2),("swiglu",3)):
                c = projections * 48 * (384 + q["counts"][f]["hidden"] + 64)
                assert c == q["counts"][f]["ffn_weights_per_layer"] == 350208
                assert 8*c == 2801664 and 8*c+6297984 == 9099648
        else:
            assert kind == "gradcheck" and q["finite_difference_forward_calls"] > 0
            assert (q["eps"],q["atol"],q["rtol"]) == (1e-6,1e-5,1e-3)
        continue
    path = ROOT / "observations" / q["tensor_file"]
    assert q["tensor_sha256"] == sha(path) and q["tensor_payload_readback_exact"]
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
    t = torch.load(path, map_location="cpu", weights_only=True)
    assert finite(t)
    if kind in ("projection", "ffn_checkpoint"):
        assert t["actual"].keys() == t["expected"].keys() == q["comparisons"].keys()
        for name, actual in t["actual"].items():
            expected = t["expected"][name]
            observed = q["comparisons"][name]
            assert list(actual.shape) == observed["shape"] and str(actual.dtype) == observed["dtype"]
            error = (actual.double()-expected.double()).abs().max().item()
            assert error == observed["max_absolute_error"]
            # A recorded GPU reduction must be checked on the same device.
            norm_device = q.get("device", "cpu")
            assert actual.double().to(norm_device).norm().item() == observed["norm"] > 0
            assert torch.equal(actual, expected) == observed["exact"]
            if kind == "ffn_checkpoint":
                assert torch.equal(actual, expected)
                checkpoint_pairs += 1
            else:
                torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
                max_pair_error = max(max_pair_error, error)
                projection_pairs += 1
    if kind == "projection":
        n, m = q["n"], q["m"]
        matrix = reconstruct(t["factors"])
        torch.testing.assert_close(matrix, t["matrix"], rtol=1e-10, atol=1e-11)
        assert sum(x.numel() for x in t["factors"].values()) == 48*(n+m+64)
        gram = matrix.T@matrix if n < m else matrix@matrix.T
        torch.testing.assert_close(gram, torch.eye(384,dtype=torch.float64)*q["gain"]**2,
                                   rtol=1e-10,atol=1e-11)
        max_gram = max(max_gram, (gram-torch.eye(384,dtype=torch.float64)*q["gain"]**2).abs().max().item())
        residual = 1.0 if n < m else 0.25
        assert math.isclose(matrix.square().sum().item()/m, .02**2*n*residual**2, rel_tol=1e-10)
        # Hand-derived vector-Jacobian products, independent of autograd and contract().
        f=t["factors"]
        x=t["input"].reshape(14,8,n//8)
        c=t["cotangent"].reshape(14,8,m//8)
        encoded=[x[:,j] @ f["column"][j].T for j in range(8)]
        projected=[c[:,i] @ f["row"][i] for i in range(8)]
        dx=torch.zeros_like(x)
        du,dv,ds=[torch.zeros_like(f[k]) for k in ("row","column","mix")]
        for i in range(8):
            mixed=sum(encoded[j]*f["mix"][:,i,j] for j in range(8))
            du[i]=c[:,i].T @ mixed
            for j in range(8):
                ds[:,i,j]=(projected[i]*encoded[j]).sum(0)
        for j in range(8):
            back=sum(projected[i]*f["mix"][:,i,j] for i in range(8))
            dv[j]=back.T @ x[:,j]
            dx[:,j]=back @ f["column"][j]
        derived={"output":t["input"] @ matrix.T,"input":dx.reshape_as(t["input"]),
                 "row":du,"column":dv,"mix":ds}
        errors={}
        for name,value in derived.items():
            torch.testing.assert_close(value,t["actual"][name],rtol=1e-10,atol=1e-11)
            errors[name]=(value-t["actual"][name]).abs().max().item()
            max_analytic_error=max(max_analytic_error,errors[name])
        independent[label]=errors
    elif kind == "ffn_checkpoint":
        assert sum(v.numel() for v in t["state"].values()) == q["parameter_count"] == 350208
        assert t["input"].shape == t["actual"]["output"].shape
        assert list(t["input"].shape) == q["input_shape"]
        assert all(v.dtype == torch.float32 for v in t["state"].values())
        assert t["actual"]["output"].dtype == (torch.bfloat16 if q["device"]=="cuda" else torch.float32)
        assert len(t["actual"]) == (8 if q["form"]=="gelu" else 11)
    elif kind == "embedding":
        n,m=q["n"],q["m"]
        matrix=reconstruct(t["factors"])
        expected=torch.zeros(m,n,dtype=torch.float64)
        for i in range(8):
            for j in range(8):
                block=expected[i*(m//8):(i+1)*(m//8),j*(n//8):(j+1)*(n//8)]
                for ell in range(6):
                    block.add_(torch.outer(t["second"][i,:,ell*8+j],t["first"][j,i*6+ell,:]))
        assert torch.equal(matrix,expected) and torch.equal(matrix,t["matrix"])
        assert torch.equal(expected,t["reference_matrix"])
        assert torch.equal(unshuffle(t["input"] @ expected.T),t["output"])
        assert torch.equal(t["output"],t["reference_output"])
        assert sum(v.numel() for v in t["factors"].values())-t["first"].numel()-t["second"].numel()==3072
    elif kind == "witness":
        n,m=q["n"],q["m"]
        h=torch.tensor([[(-1)**((i&j).bit_count()) for j in range(8)] for i in range(8)],dtype=torch.float64)
        assert torch.equal(h,t["hadamard"])
        matrix=reconstruct(t["factors"])
        assert torch.equal(matrix,t["matrix"])
        expected=torch.zeros(m,n,dtype=torch.float64)
        eye=torch.eye(48,dtype=torch.float64)
        for i in range(8):
            for j in range(8):
                expected[i*(m//8):i*(m//8)+48,j*(n//8):j*(n//8)+48]=h[i,j]*eye
        assert torch.equal(matrix,expected)
        gram=matrix.T@matrix if n<m else matrix@matrix.T
        assert torch.equal(gram,8*torch.eye(384,dtype=torch.float64))
        assert matrix.square().sum().item()==3072
        assert torch.equal(t["all_ones_matrix"],t["all_ones_left"]@t["all_ones_right"])
        assert t["all_ones_left"].shape[1]==48 and torch.equal(t["all_ones_matrix"][:48,:48],eye)
        assert Fraction(64*42,3072)==Fraction(*q["relative_squared_bound"])==Fraction(7,8)
    else:
        raise AssertionError(kind)
    priorpath=OLD/"observations"/q["tensor_file"]
    observed=integrity["files"][priorpath.as_posix()]
    if "archive_error" not in observed and observed.get("archive_bad_member") is None:
        old=torch.load(priorpath,map_location="cpu",weights_only=True)
        prior_payloads[label]=same(old,t)
    oldjson=OLD/"observations"/(label+".json")
    if integrity["files"][oldjson.as_posix()].get("json_valid"):
        oldq=read(oldjson)
        prior_json_hashes[label]=sha(priorpath)==oldq["tensor_sha256"]
    del t
assert counts==Counter({"projection":12,"ffn_checkpoint":12,"embedding":4,"witness":4,"counts":1,"gradcheck":1})
assert checkpoint_pairs==114 and projection_pairs==60 and len(prior_payloads)==24
assert all(sha(n)==v["sha256"] for n,v in integrity["files"].items())
record={"status":"PASS","qualified_checks":34,"saved_tensor_files_verified":32,
        "checkpoint_tensor_pairs_exact":checkpoint_pairs,"projection_tensor_pairs_within_tolerance":projection_pairs,
        "projection_max_absolute_error":max_pair_error,"analytic_vjp_max_absolute_error":max_analytic_error,
        "independent_initializer_gram_max_error":max_gram,"independent_projection_errors":independent,
        "same_width_embeddings_exact":4,"hadamard_certificates_exact":4,"squared_relative_rank6_error_bound":[7,8],
        "prior_readable_tensor_payload_equality":prior_payloads,"prior_json_tensor_hash_matches":prior_json_hashes,
        "original_damaged_and_intact_files_preserved":True,"optimizer_updates":0,"corpus_targets":0,
        "research_goal_achieved":False,"result_sha256":sha(ROOT/"result.json"),
        "observation_hashes":r["observation_hashes"],"finite_difference_forward_calls":r["observations"]["gradcheck"]["finite_difference_forward_calls"]}
write_json(OUT,record)
print(json.dumps({k:v for k,v in record.items() if k not in ("observation_hashes","independent_projection_errors")}),flush=True)
