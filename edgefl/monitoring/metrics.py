from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

class MetricsPusher:
    def __init__(self, url: str, job: str, client_id: int):
        self.url = url
        self.job = job
        self.client = str(client_id)
        self.registry = CollectorRegistry()
        self.loss = Gauge("train_loss", "Training loss", ["client"], registry=self.registry)
        self.time = Gauge("train_time_s", "Training time (s)", ["client"], registry=self.registry)
        self.round = Gauge("round_index", "Round index", ["client"], registry=self.registry)

    def push(self, loss: float, round_idx: int, elapsed_s: float):
        labels = {"client": self.client}
        self.loss.labels(**labels).set(float(loss))
        self.time.labels(**labels).set(float(elapsed_s))
        self.round.labels(**labels).set(int(round_idx))
        push_to_gateway(self.url, job=self.job, registry=self.registry)
