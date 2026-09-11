import torch

from results.readout_learning_v1.data import make_data as original

TASKS = ("teacher", "product")


def make_data(task, seed):
    d = original(task, seed)
    idx = ("linear", "teacher", "product").index(task)
    d["order"] = torch.randint(
        4096, (600, 128), generator=torch.Generator().manual_seed(90000 + seed + idx)
    )
    return d
