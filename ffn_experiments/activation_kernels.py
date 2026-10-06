"""Fully fused SwiGLU activation: the fair dense engineering control."""

import torch
from torch.autograd import Function
from torch.cuda.jiterator import _create_jit_fn, _create_multi_output_jit_fn
from torch.nn import functional as F

FORWARD = """template <typename T> T atlas_swiglu_forward(T x,T gate) {
    float u=float(x),g=float(gate);
    return T(u/(1.0f+::expf(-u))*g);
}"""
BACKWARD = """template <typename T> void atlas_swiglu_backward(T x,T gate,T grad,T& du,T& dg) {
    float u=float(x),g=float(gate),t=float(grad);
    float s=1.0f/(1.0f+::expf(-u));
    du=T(t*g*s*(1.0f+u*(1.0f-s)));dg=T(t*u*s);
}"""
forward_kernel = _create_jit_fn(FORWARD)
backward_kernel = _create_multi_output_jit_fn(BACKWARD, num_outputs=2)


class FusedSwiGLU(Function):
    @staticmethod
    def forward(ctx, feature, gate):
        ctx.save_for_backward(feature, gate)
        if feature.is_cuda and feature.dtype != torch.float64:
            return forward_kernel(feature, gate)
        return F.silu(feature) * gate

    @staticmethod
    def backward(ctx, gradient):
        feature, gate = ctx.saved_tensors
        if feature.is_cuda and feature.dtype != torch.float64:
            return backward_kernel(feature, gate, gradient)
        s = feature.sigmoid()
        return gradient * gate * s * (1 + feature * (1 - s)), gradient * F.silu(feature)
