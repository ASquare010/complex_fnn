"""Fixed-cotangent decoder transport with no model or optimizer mutation."""

from contextlib import nullcontext

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from results.classifier_precision_v1.source.common import loss_for
from results.fp32_classifier_profile_v1.source.common import gradient_error, tensor_hash
from src.core.config import ModelConfig
from src.core.transformer import Transformer

MODES = {
    "bf16_default_block": ("bf16", "default", "block"),
    "bf16_default_none": ("bf16", "default", "none"),
    "bf16_math_block": ("bf16", "math", "block"),
    "fp32_default_block": ("fp32", "default", "block"),
    "fp32_math_block": ("fp32", "math", "block"),
}
POLICIES = ("native_fp32", "chunks_fp32")


def attention_context(mode):
    return sdpa_kernel([SDPBackend.MATH]) if MODES[mode][1] == "math" else nullcontext()


def encode(model, tokens, trace=None):
    values = {}

    def observe(name, tensor):
        values[name] = tensor_hash(tensor)
        if trace is not None:

            def hook(gradient):
                trace[name] = gradient.detach().cpu()

            tensor.register_hook(hook)
        return tensor

    hidden = observe("embedding", model.embedding(tokens))
    for index, block in enumerate(model.blocks):
        hidden = observe(f"block_{index}", block(hidden))
    return observe("normalized_hidden", model.norm(hidden)), values


def attention_nodes(tensor):
    pending, seen, names = [tensor.grad_fn], set(), set()
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        name = type(node).__name__
        if "Attention" in name or "ScaledDot" in name:
            names.add(name)
        pending.extend(n for n, _ in node.next_functions)
    return sorted(names)


def tensor_error(actual, reference):
    values = gradient_error({"value": actual}, {"value": reference})
    return dict(
        relative_l2=values["global_relative_l2"],
        max_absolute_error=(actual.double() - reference.double()).abs().max().item(),
        bitwise_equal=torch.equal(actual, reference),
    )


def classifier(hidden, weight, targets, policy):
    h = hidden.detach().requires_grad_()
    w = weight.detach().requires_grad_()
    loss = loss_for(h, w, targets, policy)
    dh, dw = torch.autograd.grad(loss, (h, w))
    return dict(
        loss=loss.item(), hidden_gradient=dh.detach().cpu(), weight_gradient=dw.detach().cpu()
    )


def qualify():
    reconstruction = []
    for variant in ("gelu", "swiglu"):
        cfg = ModelConfig(
            variant=variant, width=24, hidden=48, layers=2, heads=3, vocab_size=32, context=7
        )
        tokens = torch.arange(14).reshape(2, 7) % 31
        targets = tokens + 1
        targets.reshape(-1)[::3] = -100
        for policy in POLICIES:
            reference = Transformer(cfg, 115).double()
            h, _ = encode(reference, tokens)
            loss = loss_for(h, reference.embedding.weight, targets, policy, 5)
            loss.backward()
            expected = {k: p.grad.detach().clone() for k, p in reference.named_parameters()}
            model = Transformer(cfg, 115).double()
            model.set_recompute_scope("block")
            hidden, _ = encode(model, tokens)
            hl = hidden.detach().requires_grad_()
            wl = model.embedding.weight.detach().requires_grad_()
            split_loss = loss_for(hl, wl, targets, policy, 5)
            dh, dw = torch.autograd.grad(split_loss, (hl, wl))
            hidden.backward(dh)
            actual = {k: p.grad.detach().clone() for k, p in model.named_parameters()}
            actual["embedding.weight"] += dw
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], atol=1e-10, rtol=1e-10)
            reconstruction.append(
                dict(
                    variant=variant,
                    policy=policy,
                    passed=True,
                    error=gradient_error(actual, expected),
                )
            )
    witness = []
    midpoint, epsilon = 1 + 2**-8, 2**-23
    for device in ("cpu", "cuda"):
        for precision in ("bf16", "fp32"):
            for sign in (-1, 1):
                incoming = midpoint + sign * epsilon
                x = torch.ones(1, 1, device=device, requires_grad=True)
                w = torch.ones(1, 1, device=device, requires_grad=True)
                with torch.autocast(device, dtype=torch.bfloat16, enabled=precision == "bf16"):
                    output = F.linear(x, w)
                (output.float() * incoming).sum().backward()
                expected = (1.0 if sign < 0 else 1 + 2**-7) if precision == "bf16" else incoming
                assert x.grad.item() == w.grad.item() == expected
                witness.append(
                    dict(
                        device=device,
                        precision=precision,
                        incoming=incoming,
                        input_gradient=x.grad.item(),
                        weight_gradient=w.grad.item(),
                        expected=expected,
                    )
                )
                del x, w, output
    return dict(
        passed=True,
        reconstruction_cases=reconstruction,
        rounding_witness=witness,
        qualification_backward_passes=20,
        optimizer_updates=0,
    )
