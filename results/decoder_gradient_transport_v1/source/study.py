"""H115 fixed incoming gradients, repeated decoder VJPs, zero optimizer steps."""

import sympy  # noqa: F401
import torch
import torch._dynamo  # noqa: F401

assert not torch.cuda.is_initialized()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

from results.decoder_gradient_transport_v1.source.common import (  # noqa: E402
    MODES,
    POLICIES,
    attention_context,
    attention_nodes,
    classifier,
    encode,
    qualify,
    tensor_error,
)
from results.fp32_classifier_profile_v1.source.common import (  # noqa: E402
    MemoryLedger,
    clear_boundary,
    gradient_error,
    tensor_hash,
    tree_hash,
)
from src.core.config import ModelConfig  # noqa: E402
from src.core.data import TokenData  # noqa: E402
from src.core.reproducibility import environment, sha256, write_json  # noqa: E402
from src.core.transformer import Transformer  # noqa: E402

ROOT = Path("results/decoder_gradient_transport_v1")


def run_condition(fixture, mode, protocol):
    label = fixture["label"] + "__" + mode
    folder = ROOT / "conditions" / label
    folder.mkdir(exist_ok=False)
    beginning = time.perf_counter()
    ledger = MemoryLedger()
    source = torch.load(fixture["checkpoint"], map_location="cpu", weights_only=True)
    model = Transformer(ModelConfig(**fixture["model_config"]), fixture["seed"]).cuda()
    model.load_state_dict(source["model"])
    model.set_recompute_scope(MODES[mode][2])
    initial_hash = tree_hash(model.state_dict())
    assert initial_hash == tree_hash(source["model"])
    del source
    data = TokenData(
        Path(protocol["datasets"][fixture["dataset"]]["path"]), "cuda", fixture["seed"] + 40000
    )
    tokens, targets = data.batch(fixture["batch"], fixture["context"])
    assert tensor_hash(tokens) == fixture["probe_tokens_sha256"]
    assert tensor_hash(targets) == fixture["probe_targets_sha256"]
    ledger.mark("construction_and_data")
    with (
        attention_context(mode),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=MODES[mode][0] == "bf16"),
    ):
        h, forward_hashes = encode(model, tokens)
        nodes = attention_nodes(h)
        captured = h.detach().cpu()
    del h
    ledger.mark("forward_capture")
    classifiers = {}
    for policy in POLICIES:
        classifiers[policy] = classifier(captured.cuda(), model.embedding.weight, targets, policy)
        ledger.mark("classifier_" + policy)
    input_path = folder / "inputs.pt"
    torch.save(
        dict(hidden=captured, tokens=tokens.cpu(), targets=targets.cpu(), classifiers=classifiers),
        input_path,
    )
    classifier_difference = dict(
        hidden_gradient=tensor_error(
            classifiers["chunks_fp32"]["hidden_gradient"],
            classifiers["native_fp32"]["hidden_gradient"],
        ),
        weight_gradient=tensor_error(
            classifiers["chunks_fp32"]["weight_gradient"],
            classifiers["native_fp32"]["weight_gradient"],
        ),
        relative_loss=abs(
            classifiers["chunks_fp32"]["loss"] / classifiers["native_fp32"]["loss"] - 1
        ),
    )
    old = {}
    if mode == "bf16_default_block":
        for policy in POLICIES:
            old_name = "block_fp32" if policy == "native_fp32" else "chunks_fp32"
            old[policy] = torch.load(
                fixture["h114_gradient_probes"][old_name], map_location="cpu", weights_only=True
            )
    first, records, pairs = {}, [], []
    for repetition in range(4):
        current = {}
        for policy in POLICIES:
            model.zero_grad(set_to_none=True)
            incoming = classifiers[policy]["hidden_gradient"].cuda()
            trace = {}
            start = time.perf_counter()
            with attention_context(mode):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=MODES[mode][0] == "bf16"):
                    h, hashes = encode(model, tokens, trace)
                h.backward(incoming)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            decoder = {k: p.grad.detach().cpu() for k, p in model.named_parameters()}
            total = dict(decoder)
            total["embedding.weight"] = (
                decoder["embedding.weight"] + classifiers[policy]["weight_gradient"]
            )
            assert len(trace) == 10
            assert all(
                bool(torch.isfinite(g).all())
                for g in (*decoder.values(), *trace.values(), *total.values())
            )
            values = dict(decoder=decoder, total=total, trace=trace)
            if repetition == 0:
                first[policy] = values
            record = dict(
                repetition=repetition,
                policy=policy,
                forward_hashes=hashes,
                forward_equal=hashes == forward_hashes,
                instrumented_seconds=seconds,
                decoder_hash=tree_hash(decoder),
                total_hash=tree_hash(total),
                trace_hash=tree_hash(trace),
                replay_decoder=gradient_error(decoder, first[policy]["decoder"]),
                replay_total=gradient_error(total, first[policy]["total"]),
                replay_trace={k: tensor_error(trace[k], first[policy]["trace"][k]) for k in trace},
                bitwise_replay_total=tree_hash(total) == tree_hash(first[policy]["total"]),
                h114_total_error=gradient_error(total, old[policy]) if old else None,
            )
            records.append(record)
            current[policy] = values
            del h, incoming, decoder, total, trace, values
            ledger.mark(f"decoder_{repetition}_{policy}")
        a, b = current["chunks_fp32"], current["native_fp32"]
        pairs.append(
            dict(
                repetition=repetition,
                decoder=gradient_error(a["decoder"], b["decoder"]),
                total=gradient_error(a["total"], b["total"]),
                trace={k: tensor_error(a["trace"][k], b["trace"][k]) for k in a["trace"]},
            )
        )
        del current, a, b
        print(
            json.dumps(
                dict(
                    condition=label,
                    repetition=repetition,
                    paired_total_error=pairs[-1]["total"]["global_relative_l2"],
                    maximum_replay_error=max(
                        r["replay_total"]["global_relative_l2"] for r in records
                    ),
                )
            ),
            flush=True,
        )
    first_path = folder / "first_passes.pt"
    torch.save(first, first_path)
    assert tree_hash(model.state_dict()) == initial_hash
    ledger.mark("serialization_and_final_integrity")
    output = dict(
        status="COMPLETE",
        label=label,
        fixture=fixture,
        mode=mode,
        attention_backward_nodes=nodes,
        initial_model_hash=initial_hash,
        forward_hashes=forward_hashes,
        classifier_difference=classifier_difference,
        records=records,
        paired_repetitions=pairs,
        memory_phases=ledger.records,
        peak_diagnostic_allocated_bytes=max(r["peak_allocated_bytes"] for r in ledger.records),
        peak_diagnostic_reserved_bytes=max(r["peak_reserved_bytes"] for r in ledger.records),
        artifacts=[dict(path=p.as_posix(), sha256=sha256(p)) for p in (input_path, first_path)],
        classifier_backward_passes=2,
        decoder_backward_passes=8,
        optimizer_updates=0,
        model_unchanged=True,
        all_finite=True,
        whole_job_memory_measured=False,
        instrumented_hooks_can_affect_scheduling=True,
        elapsed_seconds=time.perf_counter() - beginning,
    )
    write_json(folder / "metrics.json", output)
    return output


def run():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    for field in ("sources", "input_hashes"):
        for path, digest in protocol[field].items():
            assert sha256(Path(path)) == digest, path
    env = environment()
    env.update(
        tf32=False,
        threads=torch.get_num_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )
    write_json(ROOT / "environment.json", env)
    write_json(ROOT / "qualification.json", qualify())
    print("Four chain-rule reconstructions and eight rounding witness cases passed", flush=True)
    cases, boundaries = [], [clear_boundary()]
    beginning = time.perf_counter()
    for index, fixture in enumerate(protocol["fixtures"]):
        modes = list(MODES)
        modes = modes[index % 5 :] + modes[: index % 5]
        if index % 2:
            modes.reverse()
        for mode in modes:
            cases.append(run_condition(fixture, mode, protocol))
            boundaries.append(clear_boundary())
            write_json(
                ROOT / "progress.json",
                dict(
                    completed_conditions=len(cases),
                    last=cases[-1]["label"],
                    classifier_backward_passes=2 * len(cases),
                    decoder_backward_passes=8 * len(cases),
                    optimizer_updates=0,
                ),
            )
    write_json(
        ROOT / "result.json",
        dict(
            status="COMPLETE",
            conditions=cases,
            boundaries=boundaries,
            classifier_backward_passes=60,
            decoder_backward_passes=240,
            qualification_backward_passes=20,
            optimizer_updates=0,
            whole_job_memory_measured=False,
            fresh_language_training_earned=False,
            elapsed_seconds=time.perf_counter() - beginning,
            broad_goal_achieved=False,
        ),
    )


if __name__ == "__main__":
    try:
        run()
    except Exception:
        write_json(ROOT / "failure.json", dict(traceback=traceback.format_exc()))
        raise
