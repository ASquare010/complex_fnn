"""H107 isolated affine-residual students, fair ridge initialization and export."""

import copy

import torch
from torch import nn
from torch.nn import functional as F

SPECS = {
    "affine_gelu": 256,
    "affine_swiglu": 170,
    "gelu": 448,
    "relu": 448,
    "leaky_relu": 448,
    "prelu": 448,
    "silu": 448,
    "swiglu": 298,
}


class Student(nn.Module):
    def __init__(self, form, width=384, hidden=None):
        super().__init__()
        self.form = form
        self.width = width
        self.hidden = SPECS[form] if hidden is None else hidden
        self.kind = form.removeprefix("affine_")
        self.up = nn.Linear(width, self.hidden)
        self.gate = nn.Linear(width, self.hidden) if self.kind == "swiglu" else None
        self.affine = nn.Linear(width, width) if form.startswith("affine_") else None
        self.down = nn.Linear(self.hidden, width, bias=self.affine is None)
        self.alpha = nn.Parameter(torch.tensor([0.25])) if self.kind == "prelu" else None

    def features(self, x):
        z = self.up(x)
        if self.kind == "swiglu":
            h = F.silu(z) * self.gate(x)
        elif self.kind == "gelu":
            h = F.gelu(z)
        elif self.kind == "relu":
            h = F.relu(z)
        elif self.kind == "leaky_relu":
            h = F.leaky_relu(z, 0.01)
        elif self.kind == "prelu":
            h = F.prelu(z, self.alpha)
        elif self.kind == "silu":
            h = F.silu(z)
        else:
            raise ValueError(self.kind)
        return z, h

    def forward(self, x):
        value = self.down(self.features(x)[1])
        return value if self.affine is None else value + self.affine(x)

    def optimizer_groups(self, rate):
        return [{"params": list(self.parameters()), "lr": rate}]


def normalization(x, y):
    x, y = x.double(), y.double()
    return {
        "mx": x.mean(0),
        "sx": x.std(0, correction=0).clamp_min(1e-3),
        "my": y.mean(0),
        "sy": (y - y.mean(0)).square().mean().sqrt().clamp_min(1e-6),
    }


@torch.no_grad()
def initialize_hidden(model, teacher, norm, seed):
    """All models share prefixes of a teacher-row permutation on raw inputs."""
    rows = torch.randperm(
        teacher.up.weight.shape[0], generator=torch.Generator().manual_seed(18000 + seed)
    )[: model.hidden]
    assert len(rows) == model.hidden
    w = teacher.up.weight[rows].double()
    model.up.weight.copy_(w * norm["sx"])
    model.up.bias.copy_(w @ norm["mx"])
    if model.gate is not None:
        if teacher.gate is not None:
            w = teacher.gate.weight[rows].double()
            model.gate.weight.copy_(w * norm["sx"])
            model.gate.bias.copy_(w @ norm["mx"])
        else:
            model.gate.weight.zero_()
            model.gate.bias.fill_(1)
    return rows


@torch.no_grad()
def ridge_readout(model, x, y):
    """CPU FP64 solve with standardized columns and an unpenalized intercept."""
    model.double()
    h = torch.cat([model.features(v.double())[1] for v in x.split(1024)])
    design = h if model.affine is None else torch.cat((x.double(), h), dim=1)
    mean, scale = design.mean(0), design.std(0, correction=0).clamp_min(1e-6)
    z = (design - mean) / scale
    target_mean = y.double().mean(0)
    target = y.double() - target_mean
    gram = z.T @ z / len(z)
    system = gram + 1e-4 * torch.eye(gram.shape[0], dtype=torch.float64)
    cross = z.T @ target / len(z)
    solution = torch.linalg.solve(system, cross)
    coefficient = solution / scale[:, None]
    bias = target_mean - mean @ coefficient
    if model.affine is None:
        model.down.weight.copy_(coefficient.T)
        model.down.bias.copy_(bias)
    else:
        model.affine.weight.copy_(coefficient[: model.width].T)
        model.affine.bias.copy_(bias)
        model.down.weight.copy_(coefficient[model.width :].T)
    diagnostic = {
        "stationarity_max": (system @ solution - cross).abs().max().item(),
        "ridge": 1e-4,
        "floored_columns": int((design.std(0, correction=0) < 1e-6).sum()),
        "columns": design.shape[1],
    }
    return model.float(), diagnostic


@torch.no_grad()
def normalized_teacher(teacher, kind, norm):
    model = Student(kind, teacher.up.weight.shape[1], teacher.up.weight.shape[0]).double()
    for name in ("up", "gate"):
        target = getattr(model, name)
        if target is not None:
            w = getattr(teacher, name).weight.double()
            target.weight.copy_(w * norm["sx"])
            target.bias.copy_(w @ norm["mx"])
    model.down.weight.copy_(teacher.down.weight.double() / norm["sy"])
    model.down.bias.copy_(-norm["my"] / norm["sy"])
    return model.float()


@torch.no_grad()
def fold_raw(model, norm):
    """Fold x/y standardization into existing parameters using FP64 arithmetic."""
    dtype = next(model.parameters()).dtype
    raw = copy.deepcopy(model).cpu().double()
    for layer in (raw.up, raw.gate):
        if layer is not None:
            layer.bias.sub_(layer.weight @ (norm["mx"] / norm["sx"]))
            layer.weight.div_(norm["sx"])
    raw.down.weight.mul_(norm["sy"])
    if raw.affine is None:
        raw.down.bias.mul_(norm["sy"]).add_(norm["my"])
    else:
        raw.affine.bias.sub_(raw.affine.weight @ (norm["mx"] / norm["sx"]))
        raw.affine.weight.div_(norm["sx"]).mul_(norm["sy"])
        raw.affine.bias.mul_(norm["sy"]).add_(norm["my"])
    return raw.to(dtype=dtype)
