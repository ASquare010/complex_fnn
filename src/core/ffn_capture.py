"""Capture paired FFN inputs/outputs without allocating vocabulary logits."""

import torch


@torch.no_grad()
def capture_ffn_pairs(model, windows, layers, batch_size=8, bf16=True):
    """Return CPU float tensors; preserve mode and always remove temporary hooks.

    ``windows`` are explicit CPU token windows chosen by the experiment. Nothing
    in this helper samples data or fits statistics. Execution ends after the last
    requested block, which cannot affect earlier blocks in a causal decoder.
    """
    layers = tuple(layers)
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("Choose distinct block indices")
    if min(layers) < 0 or max(layers) >= len(model.blocks) or batch_size <= 0:
        raise ValueError("Invalid block index or batch size")
    if windows.ndim != 2 or windows.device.type != "cpu":
        raise ValueError("Expected CPU token windows")
    device = next(model.parameters()).device
    pairs = {layer: {"x": [], "y": []} for layer in layers}
    handles = []

    def observe(layer):
        def hook(module, args, output):
            pairs[layer]["x"].append(args[0].detach().flatten(0, 1).float().cpu())
            pairs[layer]["y"].append(output.detach().flatten(0, 1).float().cpu())

        return hook

    training = model.training
    try:
        model.eval()
        for layer in layers:
            handles.append(model.blocks[layer].ffn.register_forward_hook(observe(layer)))
        for start in range(0, len(windows), batch_size):
            tokens = windows[start : start + batch_size].to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                x = model.embedding(tokens)
                for block in model.blocks[: max(layers) + 1]:
                    x = block(x)
        return {layer: {k: torch.cat(v) for k, v in p.items()} for layer, p in pairs.items()}
    finally:
        for handle in handles:
            handle.remove()
        model.train(training)
