"""Unqualified H065 prototype: one fixed checkpoint partition, no active registration."""

import torch
from torch.utils.checkpoint import checkpoint

from src.rational_blockshuffle_ffn import RationalResidualActivation


def denominator(w, beta, u):
    return w.square() + beta * u.square()


def weighted_zero(a, w, d):
    r0 = w.square() / d
    return a.unsqueeze(-1) * r0


def weighted_one(a, u, w, d):
    r1 = u * w / d
    return a.unsqueeze(-1) * r1


def weighted_two_three(a2, a3, z, u, d):
    r2 = u.square() / d
    r3 = z * r2
    return a2.unsqueeze(-1) * r2, a3.unsqueeze(-1) * r3


def staged_residual(z, coefficients, denominator_parameter, enabled=True):
    def run(fn, *args):
        if enabled and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False)
        return fn(*args)

    beta = (1 + 0.75 * denominator_parameter.tanh()).unsqueeze(-1)
    a = coefficients.tanh().unbind(-1)
    w = (1 + z.abs()).reciprocal()
    u = z * w
    d = run(denominator, w, beta, u)
    t0 = run(weighted_zero, a[0], w, d)
    t1 = run(weighted_one, a[1], u, w, d)
    t2, t3 = run(weighted_two_three, a[2], a[3], z, u, d)
    return 0.25 * sum((t0, t1, t2, t3))


class StagedRationalActivation(RationalResidualActivation):
    def residual(self, z):
        return staged_residual(z, self.theta_coefficients, self.theta_denominator, self.training)
