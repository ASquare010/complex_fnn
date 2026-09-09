"""Fixed 2:4 values; native dense materialization and cached sparse inference."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class CompactSparseLinear(nn.Module):
    def __init__(self, input_width, output_width, seed=17):
        super().__init__()
        if input_width <= 0 or input_width % 4 or output_width <= 0:
            raise ValueError("Positive widths and input divisible by four are required")
        self.input_width, self.output_width = input_width, output_width
        pairs = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
                             dtype=torch.uint8)
        mask_rng = torch.Generator().manual_seed(seed)
        choices = torch.randint(6, (output_width, input_width // 4), generator=mask_rng)
        self.register_buffer("positions", pairs[choices])
        value_rng = torch.Generator().manual_seed(seed + 104729)
        self.values = nn.Parameter(torch.randn(output_width, input_width // 4, 2,
                                               generator=value_rng) * math.sqrt(2 / input_width))

    def materialize(self):
        return self.values.new_zeros(self.output_width, self.input_width // 4, 4).scatter(
            -1, self.positions.long(), self.values).reshape(self.output_width, self.input_width)

    def forward(self, x):
        if x.ndim == 0 or x.shape[-1] != self.input_width:
            raise ValueError("Last input dimension must equal input_width")
        return F.linear(x, self.materialize())


class CompactSparseFFN(nn.Module):
    def __init__(self, form, seed=17):
        super().__init__()
        if form not in ("gelu", "swiglu"):
            raise ValueError(form)
        self.form = form
        self.hidden = 912 if form == "gelu" else 608
        self.up = CompactSparseLinear(384, self.hidden, seed)
        self.down = CompactSparseLinear(self.hidden, 384, seed + 1)
        self.gate = CompactSparseLinear(384, self.hidden, seed + 2) if form == "swiglu" else None

    def forward(self, x):
        z = self.up(x)
        z = F.gelu(z) if self.gate is None else F.silu(z) * self.gate(x)
        return self.down(z)


class CachedSparseFFN:
    """Inference only: fixed BF16 tensors; conversion is explicitly timed outside calls."""

    def __init__(self, model, backend):
        from torch.sparse import SparseSemiStructuredTensorCUSPARSELT as Cslt
        from torch.sparse import SparseSemiStructuredTensorCUTLASS as Cutlass

        cls = {"cutlass": Cutlass, "cusparselt": Cslt}[backend]
        self.form = model.form
        self.weights = {}
        self.logical = {}
        self.padded = {}
        self.storage = {}
        for name in ("up", "down", "gate"):
            module = getattr(model, name)
            if module is None:
                continue
            dense = module.materialize().detach().to(torch.bfloat16)
            rows, columns = dense.shape
            padded = F.pad(dense, (0, (-columns) % 64, 0, (-rows) % 64)).contiguous()
            sparse = cls.from_dense(padded)
            self.weights[name] = sparse
            self.logical[name] = dense
            self.padded[name] = padded
            tensors = [getattr(sparse, key, None) for key in
                       ("packed", "meta", "packed_t", "meta_t", "compressed_swizzled_bitmask")]
            self.storage[name] = {
                "logical_shape": [rows, columns], "padded_shape": list(padded.shape),
                "logical_dense_bytes": dense.numel() * dense.element_size(),
                "padded_dense_bytes": padded.numel() * padded.element_size(),
                "packed_tensor_bytes": sum(t.numel() * t.element_size() for t in tensors
                                           if t is not None),
            }

    def projection(self, name, x, mode):
        rows, columns = self.logical[name].shape
        if mode == "dense":
            return F.linear(x, self.logical[name])
        padded = F.pad(x, (0, self.padded[name].shape[1] - columns))
        weight = self.weights[name] if mode == "sparse" else self.padded[name]
        return F.linear(padded, weight)[..., :rows]

    def __call__(self, x, mode="sparse"):
        z = self.projection("up", x, mode)
        z = F.gelu(z) if self.form == "gelu" else F.silu(z) * self.projection("gate", x, mode)
        return self.projection("down", z, mode)
