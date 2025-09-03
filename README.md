# flaive

**flaive — Federated Learning for AI in Versatile Environments**  
An open-source framework for **configurable, lightweight federated learning** on resource-constrained devices. Run fast simulations, track experiments with MLflow, and monitor training via Prometheus/Grafana. Built to evolve from **simulation → real edge deployments** with PEFT (LoRA) and quantization.

---

## ✨ Features

- **Config-first**: number of clients, optimizer, and aggregation set in YAML
- **Algorithms**: FedAvg, FedProx (extensible registry)
- **Optimizers**: AdamW / Adam / SGD (configurable)
- **Monitoring**: clients push metrics → **Pushgateway → Prometheus → Grafana**
- **Experiment tracking**: **MLflow** (params, metrics, artifacts)
- **Portable**: Docker Compose stack (app, mlflow, pushgateway, prometheus, grafana)
- **Roadmap**: PEFT/LoRA adapters, quantization (CPU int8 / QLoRA), edge agents

---

## 🧭 Architecture (current MVP)

```
Sim Clients  ──push metrics──▶  Pushgateway  ──scraped by──▶  Prometheus ─▶ Grafana
      │
      └── local train → return updates ──▶ Sim Server ──(FedAvg/FedProx)──▶ Global Model
                                                │
                                                └── MLflow (params, metrics, artifacts)
```

---

## 📂 Project Structure

```
flaive/
├── flaive/                 # Python package
│   ├── clients/            # Simulated clients
│   ├── server/             # Server + aggregation
│   ├── models/             # Model loader(s)
│   ├── monitoring/         # Metrics pusher
│   └── utils/              # Config, optimizer factory
├── config/
│   └── config.yaml         # Main experiment config
├── docker/
│   ├── mlflow/             # MLflow volume (artifacts, backend store)
│   └── prometheus/
│       └── prometheus.yml  # Prometheus config (scrapes Pushgateway)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── run.py                  # Entry point (loads config, runs sim)
```

---

## ⚡ Quickstart

### 1) Clone & build

```bash
git clone https://github.com/YOURUSER/flaive.git
cd flaive
docker compose up --build
```

Services after boot:

- **App**: `flaive-app` (runs the simulation)
- **MLflow**: http://localhost:5000
- **Pushgateway**: http://localhost:9091
- **Prometheus**: http://localhost:9090 (Status → Targets → pushgateway: UP)
- **Grafana**: http://localhost:3000 (default: admin / admin)

### 2) Run a simulation

```bash
docker exec -it flaive-app python -m flaive.main
```

You’ll see round logs in the container output, metrics in **MLflow**, and live client metrics (loss/time/round) pushed via **Pushgateway → Prometheus** (visualize with Grafana).

---

## 🔧 Configuration

Edit `config/config.yaml` to control the run:

---

## 🧪 What’s implemented (MVP)

- **Server:** rounds, client sampling, MLflow logging
- **Clients:** synthetic dataset, configurable optimizer, optional FedProx local objective
- **Aggregation:** FedAvg (and FedProx = FedAvg at aggregation step)
- **Monitoring:** `MetricsPusher` pushes `train_loss`, `train_time_s`, `round_index` per client to Pushgateway

---

## 🗺️ Roadmap (next)

- **PEFT/LoRA adapters** (adapter-only state aggregation)
- **Quantization** (CPU int8 by default; 4-bit path when CUDA/bitsandbytes available)
- **Real edge mode** (FastAPI coordinator + edge agent)
- **IID/Non-IID partitioners** (Dirichlet α, label skew)
- **Prebuilt Grafana dashboard JSON** (provisioned)

---

## 🤝 Contributing

PRs welcome! Please open an issue first to discuss major changes.  
Code style: Black / Ruff / Isort (pre-commit optional).  
Tests: coming with data partitioning and LoRA utilities.

---

## 📜 License

MIT — see [LICENSE](LICENSE).

---

## 🧾 Citation

If you use **flaive** in your work:

```
@software{flaive2025,
  title = {flaive: Federated Learning for AI in Versatile Environments},
  year  = {2025},
  url   = {https://github.com/leonmaz/flaive}
}
```
