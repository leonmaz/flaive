import torch

def fedavg(updates):
    keys = updates[0].keys()
    out = {}
    for k in keys:
        acc = None
        for upd in updates:
            v = upd[k].float()
            acc = v if acc is None else acc.add_(v)
        out[k] = (acc / len(updates)).to(updates[0][k].dtype)
    return out

def fedprox(updates):
    # Aggregation is same as FedAvg; FedProx modifies client objective only.
    return fedavg(updates)

AGGREGATORS = {
    "fedavg": fedavg,
    "fedprox": fedprox,
}

def get_aggregator(cfg):
    name = str(cfg.get("aggregation", {}).get("name", "fedavg")).lower()
    agg = AGGREGATORS.get(name)
    if agg is None:
        raise ValueError(f"Unknown aggregation '{name}'. Available: {list(AGGREGATORS)}")
    return agg
