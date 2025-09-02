import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from flaive.monitoring.metrics import MetricsPusher
from flaive.utils.optim import build_optimizer

class SimClient:
    def __init__(self, client_id: int, cfg):
        self.id = client_id
        self.cfg = cfg

    def _data(self):
        X = torch.randn(512, 100)
        y = (X[:, 0] > 0).long()
        return DataLoader(
            TensorDataset(X, y),
            batch_size=self.cfg["clients"]["batch_size"],
            shuffle=True
        )

    def train(self, model: torch.nn.Module, round_idx: int):
        model.train()
        opt = build_optimizer(model.parameters(), self.cfg)
        loss_fn = torch.nn.CrossEntropyLoss()
        dl = self._data()
        pusher = MetricsPusher(
            self.cfg["monitoring"]["pushgateway_url"],
            self.cfg["monitoring"]["job_name"],
            self.id
        )
        start = time.time()
        steps, last_loss = 0, np.nan

        # Cache initial params for FedProx local proximal objective if enabled
        init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        mu = float(self.cfg.get("aggregation", {}).get("mu", 0.0))

        for _ in range(self.cfg["clients"]["local_epochs"]):
            for xb, yb in dl:
                logits = model(xb)
                loss = loss_fn(logits, yb)

                if mu > 0.0:
                    prox = 0.0
                    for k, p in model.state_dict().items():
                        if p.dtype.is_floating_point:
                            diff = p - init_state[k]
                            prox = prox + diff.float().pow(2).sum()
                    loss = loss + (mu / 2.0) * prox

                loss.backward()
                opt.step()
                opt.zero_grad()

                steps += 1
                last_loss = float(loss.item())
                if steps % 10 == 0:
                    pusher.push(last_loss, round_idx, time.time() - start)
                if steps >= self.cfg["clients"]["steps_max"]:
                    break
            if steps >= self.cfg["clients"]["steps_max"]:
                break

        elapsed = time.time() - start
        pusher.push(last_loss, round_idx, elapsed)

        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        return state, {"loss": last_loss, "elapsed": elapsed}
