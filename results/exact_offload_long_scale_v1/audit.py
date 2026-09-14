"""Independent FP64 gradient comparison and native final-state scoring for H163."""

import json
import math
import statistics as st
from pathlib import Path

import numpy as np

from results.exact_offload_long_scale_v1.study import (
    ROOT,
    evaluate,
    read,
    sha,
    state_hash,
    tensor_hash,
    torch_setup,
    verify,
    write,
)


def main():
    torch = torch_setup()
    from src.core.config import ModelConfig
    from src.core.data import TokenData
    from src.core.transformer import Transformer

    protocol = verify()
    runs, gradients = [], {}
    for row in protocol["schedule"]:
        folder = ROOT / f"case{row['index']:02d}"
        assert (folder / "exit.txt").read_text() == "0"
        result = read(folder / "result.json")
        assert result["boundary"] == [0, 0]
        for name, expected in result["artifacts"].items():
            assert sha(folder / name) == expected
        history = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
        assert history == result["records"] and len(history) == protocol["steps"]
        assert all(
            math.isfinite(item[k])
            for item in history
            for k in ("loss", "norm", "wall_ms", "cuda_ms")
        )
        assert (folder / "telemetry.csv").stat().st_size > 0
        assert not (folder / "telemetry.err").read_text().strip()
        model = Transformer(ModelConfig(**protocol["config"]), row["seed"]).cuda()
        assert state_hash(model) == result["initial_hash"]
        data = TokenData(Path("data/wikitext2_v1"), "cuda", row["seed"] + 10000)
        initial_native = evaluate(model, data, native=True)
        assert initial_native["targets"] == result["initial_score"]["targets"]
        assert abs(initial_native["nll"] / result["initial_score"]["nll"] - 1) <= 1e-6
        result["initial_native_audit"] = initial_native
        del data
        model.load_state_dict(
            torch.load(folder / "final.pt", weights_only=True, map_location="cpu")
        )
        assert state_hash(model) == result["final_hash"]
        data = TokenData(Path("data/wikitext2_v1"), "cuda", row["seed"] + 10000)
        score = evaluate(model, data, native=True)
        relative = abs(score["nll"] / result["final_score"]["nll"] - 1)
        assert score["targets"] == result["final_score"]["targets"] and relative <= 1e-6
        result["native_audit"] = dict(**score, relative_error=relative)
        result["gradient_path"] = str(folder / "initial_gradients.pt")
        timed = history[protocol["warmup"] :]
        means = [st.mean(item["wall_ms"] for item in timed[j : j + 260]) for j in (0, 260, 520)]
        result["stability"] = max(means) / min(means)
        result["interruption_ratio"] = max(v["wall_ms"] for v in timed) / st.median(
            v["wall_ms"] for v in timed
        )
        result["timing"] = {
            key: dict(
                mean=st.mean(v[key] for v in timed),
                median=st.median(v[key] for v in timed),
                variance=st.variance(v[key] for v in timed),
            )
            for key in ("wall_ms", "cuda_ms", "forward_ms", "backward_ms", "optimizer_ms")
        }
        assert [e["step"] for e in result["endpoints"]] == [200, 400, 800]
        native_endpoints = []
        for endpoint in result["endpoints"]:
            assert sha(folder / endpoint["path"]) == endpoint["sha256"]
            if endpoint["step"] != 800:
                model.load_state_dict(
                    torch.load(folder / endpoint["path"], map_location="cpu", weights_only=True)
                )
                assert state_hash(model) == endpoint["model_hash"]
                actual = evaluate(model, data, native=True)
            else:
                assert endpoint["model_hash"] == result["final_hash"]
                actual = score
            assert actual["targets"] == endpoint["score"]["targets"]
            assert abs(actual["nll"] / endpoint["score"]["nll"] - 1) <= 1e-6
            native_endpoints.append(dict(step=endpoint["step"], **actual))
        for item in history:
            x, y = data.batch(16, 512)
            assert tensor_hash(x) + tensor_hash(y) == item["batch_hash"]
            if item["step"] in (200, 400, 800):
                endpoint = next(e for e in result["endpoints"] if e["step"] == item["step"])
                assert tensor_hash(data.generator.get_state()) == endpoint["sampler_hash"]
            del x, y
        saved = torch.load(folder / "resume.pt", map_location="cpu", weights_only=True)
        assert saved["step"] == 800 and saved["config"] == protocol["config"]
        assert saved["seed"] == row["seed"]
        assert tensor_hash(saved["sampler"]) == tensor_hash(data.generator.get_state())
        assert tensor_hash(saved["sampler"]) == result["endpoints"][-1]["sampler_hash"]
        model.load_state_dict(saved["model"])
        assert state_hash(model) == result["final_hash"]
        for group in saved["optimizer"]["param_groups"]:
            assert group["lr"] == 0.0006 and tuple(group["betas"]) == (0.9, 0.95)
            assert group["eps"] == 1e-8 and group["weight_decay"] in (0.0, 0.1)
        assert len(saved["optimizer"]["state"]) == len(list(model.parameters()))
        for state in saved["optimizer"]["state"].values():
            assert state["step"].item() == 800
            assert all(
                torch.isfinite(value).all()
                for value in state.values()
                if isinstance(value, torch.Tensor)
            )
        del state, saved
        result["native_endpoints"] = native_endpoints
        result["regenerated_batches"] = len(history)
        result["resume_verified"] = True
        runs.append(result)
        del model, data
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        assert torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0
        print("audited", row, flush=True)
    pairs = []
    for seed in (401, 409, 419):
        native = next(r for r in runs if r["seed"] == seed and r["arm"] == "ordinary")
        helper = next(r for r in runs if r["seed"] == seed and r["arm"] == "helper")
        assert native["initial_hash"] == helper["initial_hash"]
        assert [r["batch_hash"] for r in native["records"]] == [
            r["batch_hash"] for r in helper["records"]
        ]
        a = torch.load(native["gradient_path"], weights_only=True, map_location="cpu")
        b = torch.load(helper["gradient_path"], weights_only=True, map_location="cpu")
        numerator, denominator, maximum = 0.0, 0.0, 0.0
        assert a.keys() == b.keys()
        for key in a:
            av, bv = a[key].numpy().astype("float64"), b[key].numpy().astype("float64")
            assert np.isfinite(av).all() and np.isfinite(bv).all()
            err, ref = float(np.square(av - bv).sum()), float(np.square(av).sum())
            numerator += err
            denominator += ref
            maximum = max(maximum, math.sqrt(err / max(ref, 1e-300)))
        relative = math.sqrt(numerator / max(denominator, 1e-300))
        loss_error = abs(helper["initial_loss"] / native["initial_loss"] - 1)
        gradients[seed] = dict(
            global_relative=relative, max_tensor_relative=maximum, loss_relative=loss_error
        )
        del a, b, av, bv
        ratios = dict(
            memory=helper["memory"]["allocated"] / native["memory"]["allocated"],
            cuda=helper["timing"]["cuda_ms"]["median"] / native["timing"]["cuda_ms"]["median"],
            wall=helper["timing"]["wall_ms"]["median"] / native["timing"]["wall_ms"]["median"],
            nll=helper["final_score"]["nll"] / native["final_score"]["nll"],
        )
        gates = dict(
            memory=ratios["memory"] <= 0.90,
            cuda=ratios["cuda"] <= 1.15,
            wall=ratios["wall"] <= 1.15,
            quality=ratios["nll"] <= 1.01,
            host=helper["memory"]["host"]["allocated_bytes.peak"] <= 128 * 2**20,
            stability=all(
                r["stability"] <= 1.15 and r["interruption_ratio"] <= 5 for r in (native, helper)
            ),
            gradient=relative <= 1e-5 and maximum <= 1e-4 and loss_error <= 1e-6,
        )
        pairs.append(dict(seed=seed, ratios=ratios, gates=gates, passed=all(gates.values())))
    write(
        ROOT / "summary.json",
        dict(
            study="H163",
            passed=all(r["passed"] for r in pairs),
            runs=runs,
            pairs=pairs,
            gradients=gradients,
            optimizer_updates=4800,
            backwards=4806,
            native_audit_scores=24,
            terminal_convergence_qualified=False,
        ),
    )
    print(json.dumps(pairs), flush=True)


if __name__ == "__main__":
    main()
