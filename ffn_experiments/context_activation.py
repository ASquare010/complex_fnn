"""Complete Transformer: independent context predicates shared by detectors."""

import torch
from torch.autograd import Function
from torch.cuda.jiterator import _create_jit_fn, _create_multi_output_jit_fn
from torch.nn import functional as F

FORWARD = """template <typename T> T atlas_context_forward(T u,T v,T a,T product) {
    T p=u/(T(1)+::exp(-u)); T b=::tanh(a); T context=::tanh(v);
    return p+b*context*(product*p+(T(1)-product));
}"""
BACKWARD = """template <typename T> void atlas_context_backward(T u,T v,T a,T grad,T product,T& du,T& dv,T& da) {
    T s=T(1)/(T(1)+::exp(-u)); T p=u*s;
    T b=::tanh(a); T context=::tanh(v); T factor=product*p+(T(1)-product);
    du=grad*s*(T(1)+u*(T(1)-s))*(T(1)+product*b*context);
    dv=grad*factor*b*(T(1)-context*context);
    da=grad*factor*context*(T(1)-b*b);
}"""
forward_kernel = _create_jit_fn(FORWARD, product=1.0)
backward_kernel = _create_multi_output_jit_fn(BACKWARD, num_outputs=3, product=1.0)


class ContextActivation(Function):
    @staticmethod
    def forward(ctx, bank, coefficient, groups, product):
        hidden = coefficient.numel()
        u = bank[..., :hidden].reshape(*bank.shape[:-1], groups, -1)
        v = bank[..., hidden:].unsqueeze(-1)
        a = coefficient.reshape(groups, -1)
        ctx.save_for_backward(bank, coefficient)
        ctx.groups, ctx.product = groups, product
        if bank.is_cuda:
            value = forward_kernel(u, v, a, product=float(product))
        else:
            dtype = torch.float64 if bank.dtype == torch.float64 else torch.float32
            p = F.silu(u.to(dtype))
            b = a.tanh()
            t = v.to(dtype).tanh()
            value = p + b * t * (p if product else 1)
        return value.flatten(-2).to(bank.dtype)

    @staticmethod
    def backward(ctx, gradient):
        bank, coefficient = ctx.saved_tensors
        hidden = coefficient.numel()
        groups = ctx.groups
        u = bank[..., :hidden].reshape(*bank.shape[:-1], groups, -1)
        v = bank[..., hidden:].unsqueeze(-1)
        a = coefficient.reshape(groups, -1)
        g = gradient.reshape_as(u)
        if bank.is_cuda:
            du, dv, da = backward_kernel(u, v, a, g, product=float(ctx.product))
        else:
            dtype = torch.float64 if bank.dtype == torch.float64 else torch.float32
            work = u.to(dtype)
            t = v.to(dtype).tanh()
            b = a.tanh()
            g = g.to(dtype)
            s = work.sigmoid()
            p = work * s
            factor = p if ctx.product else 1
            du = g * s * (1 + work * (1 - s)) * (1 + b * t if ctx.product else 1)
            dv = g * factor * b * (1 - t.square())
            da = g * factor * t * (1 - b.square())
        dbank = torch.cat((du.flatten(-2).to(bank.dtype), dv.sum(-1).to(bank.dtype)), -1)
        dcoefficient = da.flatten(-2).sum(tuple(range(bank.ndim - 1))).to(coefficient.dtype)
        return dbank, dcoefficient, None, None
