import torch
from storage import in_dump

def autocast(device):
    return torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda")

def save(path, state):
    path = in_dump(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
