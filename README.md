# RainCheckAI

RainCheckAI is a production-grade machine learning system for forecasting public transit delays from transit state, weather context, and event pressure. The codebase is structured around strict separation of concerns so ingestion, feature engineering, training, online inference, and the API surface evolve independently without feature skew.

## Optimized Structure

```text
RainCheckAI/
├── raincheckai/
│   ├── api/
│   │   ├── app.py
│   │   └── schemas.py
│   ├── config.py
│   ├── contracts.py
│   ├── errors.py
│   ├── feature_engineering.py
│   ├── inference.py
│   ├── ingestion.py
│   ├── logging_utils.py
│   ├── telemetry.py
│   └── training.py
├── tests/
│   ├── test_api.py
│   ├── test_engineering.py
│   ├── test_ingest_data.py
│   └── test_training_integration.py
├── app.py
├── engineering.py
├── generate_synthetic_data.py
├── ingest_data.py
├── train_model.py
└── ui/streamlit_app.py
```

## Architecture

### 1. Ingestion
- `raincheckai.ingestion` normalizes raw CSVs into a canonical schema.
- Transport, weather, and event feeds are idempotently cleaned with timestamp coercion, schema validation, plausibility bounds, deterministic imputation, and duplicate removal.
- Empty event feeds are represented as a valid zero-row contract so downstream components never branch on missing tables.

### 2. Feature Engineering
- `raincheckai.feature_engineering` is the single source of truth for offline and online feature logic.
- Weather is aligned with `merge_asof`, events are attached via latest-active interval matching, and route-level lag features are computed without target leakage.
- Realtime inference requests are transformed through the same feature contract used during training.

### 3. Training
- `raincheckai.training` builds a reproducible `scikit-learn` pipeline with numeric imputation, quantile clipping, scaling, categorical encoding, and a fixed-seed `RandomForestRegressor`.
- Hyperparameters are selected with deterministic `RandomizedSearchCV` and time-series-aware validation.
- Persisted artifacts contain both the trained pipeline and versioned metadata for auditability.

### 4. Online Inference
- `raincheckai.inference` loads the model bundle once and serves predictions through a typed service layer.
- Missing weather or event context degrades gracefully to explicit fallback defaults.
- If model artifacts are unavailable or inference fails, a conservative heuristic predictor keeps the API available.

### 5. API and Observability
- `raincheckai.api.app` exposes `/predict`, `/health`, and `/metrics`.
- Logging is structured JSON with request-scoped correlation IDs.
- In-process telemetry records counters, gauges, and latencies for HTTP and inference paths.

## Quick Start

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Generate Demo Data

```bash
./.venv/bin/python generate_synthetic_data.py
```

This writes raw inputs to `data/raw/` and cleaned plus engineered outputs to `data/processed/`.

### Train the Model

```bash
./.venv/bin/python train_model.py \
  --engineered-csv data/processed/training_features.csv \
  --artifact-dir artifacts
```

### Run the API

```bash
./.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Run the Streamlit UI

```bash
./.venv/bin/streamlit run ui/streamlit_app.py
```

## Validation

```bash
./.venv/bin/ruff check .
./.venv/bin/python -m mypy
./.venv/bin/python -m pytest tests -q
```

## Design Trade-Offs

- `scikit-learn` pipeline over ad hoc preprocessing: slightly more boilerplate, but it guarantees training/inference parity and reproducibility.
- Heuristic fallback predictions: less accurate than the trained model, but they preserve API availability during artifact or upstream-context failures.
- Monorepo compatibility wrappers (`app.py`, `ingest_data.py`, `engineering.py`, `train_model.py`): they keep the developer UX simple while the real logic lives in the typed package.
