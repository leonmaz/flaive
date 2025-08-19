import torch
from torch import nn

def load_model(cfg):
    backend = cfg["model"]["backend"]
    if backend == "toy":
        return nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 2))
    # future: elif backend == "hf": return load_hf_peft_model(cfg)
    raise ValueError(f"Unknown model backend: {backend}")
