# Product Recommendation System

Full-stack product recommendation system built in phases using MovieLens 100K, PostgreSQL, PyTorch (NCF), Qdrant, Django, and Tailwind CSS.

## Phase 1 — Data Layer (PostgreSQL)

### Schema

| Table | Purpose |
|-------|---------|
| `users` | MovieLens users (`external_id`, demographics) |
| `items` | Movies (`external_id`, title, genres, metadata) |
| `interactions` | User–item ratings with signal type and timestamp |
| `etl_runs` | Audit log of row counts at each ETL stage |

### Prerequisites

- Docker Desktop (for PostgreSQL)
- Python 3.10+

### Quick start

```bash
# 1. Start PostgreSQL
docker compose up -d postgres

# 2. Create virtual environment and install dependencies
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

# 3. Copy environment file
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux

# 4. Run ETL (downloads MovieLens 100K, cleans, loads)
python -m etl.etl --truncate

# 5. Validate data integrity
python -m etl.validate --strict
```

### ETL pipeline

The ETL script (`etl/etl.py`):

1. **Downloads** MovieLens 100K from GroupLens
2. **Extracts** `u.user`, `u.item`, `u.data`
3. **Cleans** data:
   - Flags and drops nulls in required fields
   - Deduplicates on natural keys
   - Removes orphaned interactions (invalid user/item FKs)
4. **Loads** into PostgreSQL with proper genre arrays
5. **Logs** row counts (raw → cleaned → loaded) to `etl_runs`

```bash
# Re-run without re-downloading
python -m etl.etl --truncate --skip-download
```

### Validation

The validation script (`etl/validate.py`) re-runs checks any time:

- Pandera schema validation (types, ranges, enums)
- Required-column null checks
- Foreign-key integrity (no orphaned interactions)
- Duplicate `external_id` detection

```bash
python -m etl.validate --strict
```

### Expected row counts (MovieLens 100K)

| Table | Approx. rows |
|-------|-------------|
| users | 943 |
| items | 1,682 |
| interactions | 100,000 |

### Project layout (Phase 1)

```
├── config/
│   └── services.yaml      # DB and ETL service config
├── sql/
│   └── schema.sql         # PostgreSQL DDL
├── etl/
│   ├── download_data.py   # MovieLens download helper
│   ├── etl.py             # Main ETL pipeline
│   ├── validate.py        # Pandera validation
│   └── utils.py           # DB helpers
├── docker-compose.yml     # PostgreSQL service
├── requirements.txt
└── README.md
```

---

## Phase 2 — Model Development (PyTorch NCF)

### Architecture

**Neural Collaborative Filtering (NCF)** with:
- User embedding tower (`nn.Embedding`)
- Item embedding tower (`nn.Embedding`)
- Concatenated vectors fed through an MLP
- Sigmoid output for implicit interaction likelihood (BCEWithLogitsLoss)

### Training pipeline

| Feature | Implementation |
|---------|----------------|
| Data source | PostgreSQL (falls back to raw MovieLens if DB unavailable) |
| Split | Temporal (default) or per-user — **not** random row split |
| Implicit signal | Ratings ≥ threshold (default 4) → positive label |
| Negative sampling | 4 negatives per positive during training |
| Metrics | NDCG@k, Precision@k, Recall@k, MAP, F1@k |
| Experiment tracking | MLflow (params, metrics, model artifact) |

### Run training

```bash
pip install -r requirements.txt

# Train (auto-detects PostgreSQL or uses raw files)
python -m model.train --source auto

# Force raw MovieLens files
python -m model.train --source raw

# Quick test run
python -m model.train --source raw --epochs 5
```

Hyperparameters live in `config/model.yaml` (separate from `config/services.yaml`).

### View MLflow runs

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Open http://localhost:5000 to inspect params, metrics, and artifacts.

### Project layout (Phase 2)

```
├── config/
│   ├── services.yaml
│   └── model.yaml           # model hyperparameters
├── model/
│   ├── ncf.py               # NCF architecture
│   ├── data.py              # splits, negative sampling
│   ├── metrics.py           # ranking metrics
│   └── train.py             # training + MLflow logging
├── artifacts/               # saved model checkpoints
└── mlruns/                  # MLflow experiment history
```

---

## Phase 3 — Vector Retrieval (Qdrant)

### Architecture & Serving Index

- Extracts item embedding weights from trained PyTorch NCF model (`model.get_item_embeddings()`).
- Populates Qdrant collection `movie_embeddings` with item vectors (64-dim) and metadata payload (`item_idx`, `external_id`, `title`, `genres`).
- Similarity queries use cosine vector distance for top-$k$ nearest item retrieval given user embedding vectors.

### Populate Vector Index

```bash
python -m retrieval.qdrant_indexer
```

### Test Vector Retrieval

```bash
python -c "from retrieval.vector_search import get_recommender; rec = get_recommender(); print(rec.recommend_for_user(1, top_k=5))"
```

---

## Phase 4 — API & Frontend (Django + Tailwind CSS)

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | `GET` | Single-page Tailwind CSS web UI |
| `/recommendations/<user_id>` | `GET` | Qdrant vector similarity recommendations for active users with cold-start popularity fallback |
| `/interactions` | `POST` | Log new explicit/implicit interaction (click, rating) |
| `/items/<id>` | `GET` | Detailed metadata for item |
| `/items` | `GET` | Search/list items |

### Cold-Start Fallback

If a `user_id` has no historical interaction data (e.g. User #9999), the API automatically falls back to popularity-based ranking (rating volume × score) and flags `"is_cold_start": true` in the API payload.

### Run Django App Locally

```bash
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

Open http://localhost:8000 to interact with the UI.

---

## Phase 5 — Containerization & Ops (Docker Compose)

### Services

- `postgres`: PostgreSQL 16 DB (port `5432`)
- `qdrant`: Qdrant Vector Engine (port `6333`)
- `mlflow`: MLflow Tracking Server (port `5000`)
- `django-app`: Django Web & Recommendation API (port `8000`)

### Run All Services

```bash
docker-compose up --build -d
```

### Access Points

- **Recommendation Web App**: http://localhost:8000
- **MLflow Tracking UI**: http://localhost:5000
- **Qdrant Vector DB**: http://localhost:6333
- **PostgreSQL**: `localhost:5432` (`user: recuser`, `pass: recpass`, `db: recommendations`)

### Project Architecture & Layout

```
├── config/
│   ├── services.yaml       # Database, Qdrant, ETL & port service configs
│   └── model.yaml          # PyTorch model hyperparameters
├── sql/
│   └── schema.sql          # PostgreSQL DDL
├── etl/
│   ├── download_data.py    # MovieLens 100K downlaod helper
│   ├── etl.py              # Cleaning & ingestion into Postgres
│   └── validate.py         # Pandera data validation
├── model/
│   ├── ncf.py              # Neural Collaborative Filtering architecture
│   ├── data.py             # Splits, dataset, negative sampling
│   ├── metrics.py          # NDCG@k, Precision@k, Recall@k, MAP, F1@k
│   └── train.py            # Model training & MLflow logging
├── retrieval/
│   ├── qdrant_indexer.py   # Extracts & loads vectors to Qdrant
│   └── vector_search.py    # Top-k vector similarity search
├── rec_api/
│   ├── views.py            # Django REST API handlers & cold-start fallback
│   ├── urls.py             # REST routes
│   └── templates/index.html # Tailwind CSS Dashboard UI
├── artifacts/              # Model checkpoints & ID mapping JSONs
├── docker-compose.yml      # Orchestration for postgres, qdrant, mlflow, django
├── Dockerfile              # Container image for Django app service
├── requirements.txt
└── README.md
```
