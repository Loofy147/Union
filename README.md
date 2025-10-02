# Unified AI Platform Orchestrator

This repository contains the complete, end-to-end implementation of the Unified AI Platform Orchestrator, a system designed for building and deploying high-performance semantic search models. It includes components for training, indexing, serving, and monitoring, providing a robust foundation for advanced retrieval systems.

## ✨ Features

- **Advanced Training:** PyTorch-based dual-encoder trainer using a Momentum Contrast (MoCo) framework with InfoNCE loss.
- **Efficient Search:** FAISS-powered index builder for creating highly optimized HNSW indexes for fast and accurate similarity search.
- **High-Performance API:** A FastAPI application serves the search API, featuring Redis caching for low latency and Prometheus instrumentation for observability.
- **Containerized Deployment:** Docker and Docker Compose are used for easy local setup and consistent development environments.
- **Production-Ready Assets:** Includes Kubernetes manifests for canary deployments, Prometheus and Grafana configuration for monitoring, and utility scripts for operational tasks.
- **Robust and Testable:** Includes unit tests for key logic (e.g., recall metrics) and a checkpointing system to ensure reliable training resumption.

## 📂 Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── cache_metrics.py    # Atomic Redis counters for cache statistics.
│   ├── main.py             # FastAPI application with search and health endpoints.
│   └── requirements.txt    # Python dependencies for the application.
├── checkpoints/            # (Git-ignored) Stores trainer checkpoint files.
├── k8s/
│   └── deployment-canary.yaml # Example Kubernetes manifest for canary deployments.
├── metrics/
│   └── recompute_recall.py # Script for calculating recall@k metrics.
├── models/
│   └── index.faiss         # (Git-ignored) The generated FAISS index file.
├── tests/
│   └── test_metrics.py     # Pytest unit tests for the recall calculation logic.
├── .gitignore              # Specifies intentionally untracked files to ignore.
├── build_index.py          # Script to generate the FAISS index from embeddings.
├── docker-compose.yml      # Defines and orchestrates the local services (API, Redis).
├── Dockerfile              # Instructions to build the Docker image for the API.
├── grafana_dashboard.json  # A skeleton Grafana dashboard for key metrics.
├── hpo_config.yaml         # Configuration for Hyperparameter Optimization.
├── prometheus.yml          # Configuration for the Prometheus monitoring server.
├── README.md               # This documentation file.
├── trainer.py              # The PyTorch script for training the dual-encoder model.
└── warmup.py               # Script to pre-warm the API cache.
```

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- [Python 3.9+](https://www.python.org/downloads/)
- [Docker](https://www.docker.com/products/docker-desktop/) and [Docker Compose](https://docs.docker.com/compose/install/)
- `sudo` access may be required for Docker commands depending on your system configuration.

### 1. Installation

Clone the repository and install the required Python dependencies:

```bash
git clone <repository-url>
cd <repository-name>
pip install -r app/requirements.txt
pip install pytest
```

### 2. Build the Search Index

Before running the API, you need to generate a search index. The `build_index.py` script creates a dummy index for demonstration purposes.

```bash
python build_index.py
```
This will create a `models/` directory and save `index.faiss` inside it. This file is ignored by Git.

### 3. Run the Local Environment

Use Docker Compose to build the container and start the API and Redis services in the background.

```bash
# Note: Use 'sudo' if you encounter Docker daemon permission errors
sudo docker compose up --build -d
```

To check the status of the running containers:
```bash
sudo docker compose ps
```

## 🛠️ Usage

### Running the API

The API is automatically started via Docker Compose. It will be available at `http://localhost:8000`. You can check its health at `http://localhost:8000/health`.

### Training the Model

The `trainer.py` script contains the logic for training the dual-encoder model. It includes a mock training loop and a robust checkpointing system.

To run the trainer:
```bash
python trainer.py
```
Checkpoints will be saved in the `checkpoints/` directory. If you run the script again, it will automatically resume from the last saved checkpoint.

### Running Tests

Unit tests are managed with `pytest`. To run the tests for the recall metrics:

```bash
pytest tests/test_metrics.py
```

### Warming Up the Cache

After a deployment or index rebuild, you can pre-warm the API's Redis cache to ensure fast initial query responses.

```bash
python warmup.py
```
This script sends a predefined list of queries to the `/search` endpoint.

## 🔌 API Endpoints

The following endpoints are available when the API is running:

| Method | Path                           | Description                                                     |
|--------|--------------------------------|-----------------------------------------------------------------|
| `GET`  | `/health`                      | Checks the health of the API, Redis, and FAISS index.           |
| `GET`  | `/search`                      | Performs semantic search. **Params**: `q` (query), `k` (top-k). |
| `GET`  | `/metrics`                     | Exposes Prometheus metrics for scraping.                        |
| `GET`  | `/metrics/cache_accounting`    | Provides a snapshot of cache statistics and checks consistency. |

## ⚙️ Configuration

- **Hyperparameters (`hpo_config.yaml`):** Defines the search space for model training and index tuning. This file is intended for use with an HPO framework like Optuna or Ray Tune.
- **Monitoring (`prometheus.yml`, `grafana_dashboard.json`):** These files provide the configuration for setting up a local monitoring stack to observe the application's performance.
```