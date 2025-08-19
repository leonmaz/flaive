import torch

def build_optimizer(params, cfg):
    ocfg = cfg["clients"].get("optimizer", {})
    name = str(ocfg.get("name", "adamw")).lower()
    lr = float(ocfg.get("lr", 2e-3))
    wd = float(ocfg.get("weight_decay", 0.0))

    if name == "sgd":
        momentum = float(ocfg.get("momentum", 0.0))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    elif name == "adam":
        betas = tuple(ocfg.get("betas", [0.9, 0.999]))
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=wd)
    else:  # default adamw
        betas = tuple(ocfg.get("betas", [0.9, 0.999]))
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd)
