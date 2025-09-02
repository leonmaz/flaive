import random
import mlflow
from flaive.models.loader import load_model
from flaive.clients.sim_client import SimClient
from flaive.server.aggregator import get_aggregator

class SimServer:
    def __init__(self, cfg):
        self.cfg = cfg
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment"])
        self.aggregate = get_aggregator(cfg)

    def run(self):
        rounds = self.cfg["rounds"]
        total = self.cfg["clients"]["total"]
        per_round = self.cfg["clients"]["per_round"]

        global_model = load_model(self.cfg)

        with mlflow.start_run(run_name="sim"):
            mlflow.log_params({
                "rounds": rounds,
                "clients_total": total,
                "clients_per_round": per_round,
                "model": self.cfg["model"]["name"],
                "backend": self.cfg["model"]["backend"],
                "aggregation": self.cfg.get("aggregation", {}).get("name", "fedavg"),
                "optimizer": self.cfg.get("clients", {}).get("optimizer", {}).get("name", "adamw"),
            })

            for r in range(rounds):
                print(f"\n=== Round {r+1}/{rounds} ===")
                client_ids = random.sample(range(total), per_round)
                updates, metrics = [], []

                for cid in client_ids:
                    client = SimClient(cid, self.cfg)
                    local = load_model(self.cfg)
                    local.load_state_dict(global_model.state_dict(), strict=True)
                    upd, m = client.train(local, round_idx=r)
                    updates.append(upd)
                    metrics.append(m)

                new_state = self.aggregate(updates)
                global_model.load_state_dict(new_state, strict=True)

                avg_loss = sum(m["loss"] for m in metrics) / len(metrics)
                avg_time = sum(m["elapsed"] for m in metrics) / len(metrics)
                print(f"Round {r}: loss={avg_loss:.4f}, time={avg_time:.2f}s")
                mlflow.log_metrics({"round_loss": avg_loss, "round_time_s": avg_time}, step=r)
