# RainCheckAI – Weather-Aware Transit Delay Prediction

RainCheckAI is an end-to-end, production-ready machine learning system that predicts urban public transport delays by fusing timetable data, stochastic weather events, and temporal structure. The project demonstrates how to take a realistic forecasting problem from raw CSVs all the way to a containerized API and an interactive dashboard.

The codebase is intentionally engineered to look like a professional, portfolio-grade project, with a clear pipeline, unit tests, CI, and modern serving patterns.

## Motivation & Background

RainCheckAI sits at the intersection of **Computer Science**, **Applied Mathematics**, and a long-standing interest in **Smart City urban optimization**. It builds on prior work exploring how local parklets, micro-mobility, and transit infrastructure interact, and asks a practical question:

> *Given current weather and upstream delays, how late is this vehicle likely to be at the next stop?*

The repository showcases:

- **Urban informatics** thinking (how weather and events propagate through a transit network),
- **Mathematical modeling** (time series, trigonometric encodings, and DAG-style delay propagation),
- **Modern MLOps practices** (clean modular code, tests, CI, Docker, and an API + UI for stakeholders).

## Architecture Overview

RainCheckAI is organized as a simple but realistic ML stack:

- **Data ingestion (`ingest_data.py`)**
  - Loads raw transport and weather CSVs (e.g., from Kaggle’s “Public Transport Delays with Weather and Events” dataset).
  - Standardizes timestamps into a single `timestamp` column using `pandas.to_datetime` with explicit UTC handling.
  - Performs numeric and categorical imputation to handle missing values.

- **Temporal & feature engineering (`engineering.py`)**
  - Rounds timestamps to an hourly `time_key` and performs a **left join** of transport on weather to avoid label leakage.
  - Adds **cyclical time encoding** using trigonometric functions (`sin_hour`, `cos_hour`) so 23:00 and 00:00 are close in feature space.
  - Implements **Directed Acyclic Graph (DAG)-style delay propagation** via a `prev_stop_delay_minutes` lag feature, computed per route and ordered by time.

- **Model training (`train_model.py`)**
  - Builds a numeric feature matrix from engineered data, excluding identifiers and textual columns.
  - Uses **TimeSeriesSplit** to respect temporal ordering during validation.
  - Integrates **Bayesian hyperparameter tuning with Optuna** (TPE sampler) to optimize XGBoost hyperparameters:
    - `learning_rate`, `max_depth`, `n_estimators`, `subsample`.
  - Trains a final `XGBRegressor` on the full dataset using the best hyperparameters.
  - Logs feature importances to highlight which weather and network factors drive delays.
  - Persists artifacts with `joblib`:
    - `artifacts/model.joblib`
    - `artifacts/feature_names.joblib`

- **Inference (`src/predict.py`)**
  - Loads the trained model and feature list.
  - Accepts JSON-like events, builds a single-row feature vector, and returns a predicted delay in minutes.

- **API (`app.py`)**
  - Modern **FastAPI** service using the `lifespan` context to load model artifacts once at startup into `app.state`.
  - Exposes a `POST /predict` endpoint that accepts a `{"features": {...}}` payload and returns:
    - `{"delay_minutes": <float>}`

- **UI (`ui/streamlit_app.py`)**
  - **Streamlit** dashboard for non-technical stakeholders.
  - Sliders and inputs for:
    - Hour of day,
    - Previous stop delay,
    - Precipitation, wind speed, temperature.
  - Computes `sin_hour` and `cos_hour` behind the scenes from the chosen hour.
  - Sends an HTTP `POST` to the FastAPI backend and visualizes the predicted delay in minutes.

- **Testing (`tests/`)**
  - `test_ingest_data.py`: verifies `standardise_timestamp` correctly parses valid timestamps and flags invalid entries as `NaT`.
  - `test_engineering.py`: checks that cyclical encodings for hour `0` and `23` are close in Euclidean space, validating the continuity of the circular time representation.

- **CI/CD (`.github/workflows/ci.yml`)**
  - GitHub Actions workflow triggered on `push` and `pull_request` to `main`.
  - Sets up Python 3.11, installs `requirements.txt`, and runs `python -m pytest tests/ -v`.

## Technical Highlights

- **Directed Acyclic Graph (DAG) Logic for Delay Propagation**
  - Delays are modeled as propagating forward along each route over time.
  - The `prev_stop_delay_minutes` feature is constructed by grouping by route, sorting by timestamp, and applying a one-step lag.
  - This is conceptually a DAG over stops and times: each node’s delay depends on its parent node (the previous stop), plus exogenous weather and event features.

- **Bayesian Hyperparameter Tuning with Optuna**
  - Uses Optuna’s `TPESampler` with a fixed random seed for reproducibility.
  - Objective function:
    - Samples candidate XGBoost hyperparameters.
    - Evaluates each configuration with **TimeSeriesSplit RMSE**.
    - Minimizes mean RMSE across folds.
  - This Bayesian optimization loop balances exploration and exploitation, usually outperforming naive grid or random search for structured models like gradient-boosted trees.

- **Cyclical Time Encoding via Trigonometry**
  - Raw hour-of-day values (0–23) are mapped into the unit circle:
    - \(\text{radians} = 2\pi \cdot \text{hour} / 24\)
    - \(\text{sin\_hour} = \sin(\text{radians})\)
    - \(\text{cos\_hour} = \cos(\text{radians})\)
  - This encoding ensures that 23:00 and 00:00 are adjacent in feature space, which is particularly important when modeling peak/off-peak patterns and late-night services.

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run tests

```bash
python -m pytest tests/ -v
```

### 3. Generate synthetic data, train the model, and generate artifacts

If you don't have real transport and weather data handy, you can still run the full pipeline end-to-end using a small synthetic dataset:

```bash
python generate_synthetic_data.py

python train_model.py \
  --engineered-csv data/processed/transport_with_features.csv \
  --target-col delay_minutes \
  --timestamp-col timestamp \
  --optuna-trials 30
```

These commands run feature engineering on synthetic data and then perform Optuna-based Bayesian tuning, writing the trained model and feature names into the `artifacts/` directory.

## Docker & Local Deployment

RainCheckAI ships with a `Dockerfile` that bundles the FastAPI service and all dependencies.

### 1. Build the Docker image

Make sure you have already trained a model so that `artifacts/model.joblib` and `artifacts/feature_names.joblib` exist locally.

```bash
docker build -t raincheckai .
```

### 2. Run the FastAPI backend in Docker

```bash
docker run -p 8000:8000 raincheckai
```

The FastAPI server will be available at `http://localhost:8000`.

You can test it directly:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": {"sin_hour": 0.0, "cos_hour": 1.0, "prev_stop_delay_minutes": 5.0}}'
```

## Streamlit Dashboard

The Streamlit UI in `ui/streamlit_app.py` is designed for quick exploration and stakeholder demos.

### 1. Start the backend (locally)

If you prefer to run FastAPI directly instead of via Docker:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Launch the Streamlit app

In a separate terminal:

```bash
streamlit run ui/streamlit_app.py
```

Then open the URL printed by Streamlit (typically `http://localhost:8501`) in your browser.

You can:

- Adjust hour of day, previous stop delay, and weather variables,
- See how the trigonometric time encoding and delay DAG logic interact,
- Receive real-time delay predictions from the FastAPI backend.

The Streamlit app also introspects the trained model's `feature_names` artifact and surfaces which features are expected; if any required model features are missing from the UI payload, you'll see a warning before the request is sent.

## Continuous Integration

Whenever code is pushed to `main` or a pull request targets `main`, GitHub Actions:

- Checks out the repository,
- Sets up Python 3.11,
- Installs dependencies via `requirements.txt`,
- Runs the test suite under `tests/` with verbose output.

This ensures that the ingestion, temporal engineering, and core modeling logic remain stable as the project evolves.

## Contributing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Run linting (install `ruff` first if needed):

```bash
python -m pip install ruff
python -m ruff check .
```

## Extending RainCheckAI

Potential extensions for future work include:

- Adding explicit event and disruption feeds (e.g., concerts, road closures).
- Modeling full-route propagation using graph or network models on top of the current DAG-style lags.
- Surfacing SHAP-based explanations for urban planners interested in “why” certain routes are fragile.

RainCheckAI is intentionally structured so that these additions can be layered on without rewriting the existing pipeline, making it a strong foundation for a Smart City–focused ML portfolio.
