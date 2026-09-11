"""H116 execution adapters; all arms use the same maintained Transformer and loss."""

from contextlib import nullcontext

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from results.fp32_classifier_profile_v1.source.common import set_blocks
from results.fp32_classifier_profile_v1.source.common import training_loss as classifier_loss
from src.core.config import ModelConfig
from src.core.transformer import Transformer

POLICIES = {
    "bf16_default_native": ("bf16", "default", "block_bf16"),
    "fp32_default_native": ("fp32", "default", "block_fp32"),
    "fp32_default_chunks": ("fp32", "default", "chunks_fp32"),
    "fp32_math_native": ("fp32", "math", "block_fp32"),
    "fp32_math_chunks": ("fp32", "math", "chunks_fp32"),
}


def attention_context(policy):
    return sdpa_kernel([SDPBackend.MATH]) if POLICIES[policy][1] == "math" else nullcontext()


def training_loss(model, tokens, targets, policy, chunk=512):
    precision, _, classifier_policy = POLICIES[policy]
    with (
        attention_context(policy),
        torch.autocast(
            tokens.device.type, dtype=torch.bfloat16, enabled=tokens.is_cuda and precision == "bf16"
        ),
    ):
        return classifier_loss(model, tokens, targets, classifier_policy, chunk)


def qualify():
    checks = []
    for variant in ("gelu", "swiglu"):
        for shape in ((2, 7), (1, 11)):
            cfg = ModelConfig(
                variant=variant,
                vocab_size=32,
                width=24,
                hidden=48,
                layers=2,
                heads=3,
                context=shape[1],
            )
            tokens = torch.arange(shape[0] * shape[1]).reshape(shape) % 31
            targets = tokens + 1
            targets.reshape(-1)[::3] = -100
            reference = Transformer(cfg, 116).double()
            native = F.cross_entropy(reference(tokens).flatten(0, 1), targets.flatten())
            native.backward()
            expected = {k: p.grad for k, p in reference.named_parameters()}
            for policy in POLICIES:
                model = Transformer(cfg, 116).double()
                set_blocks(model)
                identities = {k: id(p) for k, p in model.named_parameters()}
                loss = training_loss(model, tokens, targets, policy, 5)
                with attention_context(policy):
                    loss.backward()
                torch.testing.assert_close(loss, native, atol=1e-10, rtol=1e-10)
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(
                        parameter.grad, expected[name], atol=1e-10, rtol=1e-10
                    )
                assert identities == {k: id(p) for k, p in model.named_parameters()}
                checks.append(dict(variant=variant, shape=shape, policy=policy, passed=True))
    return dict(
        passed=True,
        double_model_checks=checks,
        reference_backward_passes=4,
        qualification_backward_passes=24,
        optimizer_updates=0,
    )
