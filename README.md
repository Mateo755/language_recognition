# Language detection — end-to-end ML deployment

This repository is an **end-to-end MLOps-style project**: **data handling** and exploratory analysis, **model training** with scikit-learn, a **FastAPI** inference service, an inference-only **Docker** image **published to Docker Hub**, and **local** validation of the container.

A **trial deployment on Render** was done to practice shipping the same Dockerized API to a managed host. That environment is **experimental**, **not** something I plan to keep running or supporting over time, so **any public URL may stop working** without notice. Details and a single reference link are in **[Deployment (Render)](#deployment-render)** below.

---

## Table of contents

- [Project journey](#project-journey)
- [Tech stack](#tech-stack)
- [Documentation map (README vs Docker Hub)](#documentation-map-readme-vs-docker-hub)
- [Getting started (local development)](#getting-started-local-development)
- [Training](#training)
- [Running the API locally](#running-the-api-locally)
- [Docker (inference image)](#docker-inference-image)
- [Exploratory analysis (EDA)](#exploratory-analysis-eda)
- [Deployment (Render)](#deployment-render)
- [Repository layout](#repository-layout)
- [License](#license)

---

## Project journey

1. **Data & EDA** — Tabular text data with labels; exploratory notebook under `notebooks/eda.ipynb`.
2. **Training** — Scikit-learn pipeline (vectorizer + classifier), training script and shared preprocessing in `src/` and `scripts/`.
3. **API** — FastAPI app in `app/`; **the same text cleaning** as in training (`src/text_cleaning.py`) so train/serve skew is minimized.
4. **Docker** — Slim **inference-only** image (`Dockerfile`, `requirements-api.txt`): API + frozen `.pkl` model; built and tested locally.
5. **Registry** — Image pushed to Docker Hub (`mateo755/language-detector-api`); Hub copy is documented in [`docker_docs.md`](docker_docs.md) (see below).
6. **Cloud (trial)** — The stack was deployed once on **Render** to verify end-to-end behavior in the cloud; see [Deployment (Render)](#deployment-render) for scope, limitations, and the (possibly temporary) public base URL.

---

## Tech stack

| Area            | Choice                                                                 |
| --------------- | ---------------------------------------------------------------------- |
| Language        | Python **3.10+** (development); Docker image uses **Python 3.12** slim |
| ML              | **scikit-learn** pipeline (serialized with pickle)                     |
| Data / training | **pandas** (training path)                                             |
| API             | **FastAPI**, **Pydantic**, **Uvicorn**                                 |
| Packaging       | **pip** / `pyproject.toml`; Docker for production inference            |
| Cloud (optional) | **Render** — one-off trial deploy of the Docker image (see below)   |

---

## Documentation map (README vs Docker Hub)

| Document                             | Purpose                                                                                                                                                                                                                          |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **This `README.md`**                 | Full project: setup, training, local API, Docker workflow, optional cloud trial notes, repository map.                                                                                                                           |
| **[`docker_docs.md`](docker_docs.md)** | **Canonical copy-paste** text for the [Docker Hub](https://hub.docker.com/r/mateo755/language-detector-api) repository **overview** for the **published inference image** only (includes Hub links for readers browsing GitHub). It does not duplicate training/EDA instructions. |

When you change API behavior (routes, env vars, limits), update **`docker_docs.md`** and paste the revised Markdown into Docker Hub so the Hub page stays aligned with the image.

---

## Getting started (local development)

### Requirements

- **Python 3.10+**
- Repository root (where `pyproject.toml` lives)

### Environment

```bash
cd /path/to/language_recognition
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -e .
```

Equivalent: `pip install -r requirements.txt` (editable install from `pyproject.toml`).

### Data

By default the training pipeline expects **`data/language_detection.csv`** with columns **`Text`** and **`Language`**. Adjust paths in `scripts/train_pipeline.py` if your layout differs.

---

## Training

From the project root with the virtual environment activated:

```bash
train-pipeline
```

Or:

```bash
python -m scripts.train_pipeline
# or
python scripts/train_pipeline.py
```

The trained artifact is written by default to **`models/trained_pipeline-0.1.0.pkl`** (the `models/` directory is created on save if missing).

---

## Running the API locally

The API loads the pickle at startup. Ensure the model file exists under **`app/model/trained_pipeline-0.1.0.pkl`** (e.g. copy from `models/` after training):

```bash
cp models/trained_pipeline-0.1.0.pkl app/model/
```

From the repository root:

```bash
export PYTHONPATH=.
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Health:** `GET http://localhost:8000/health`
- **Detect:** `POST http://localhost:8000/detect` with JSON `{"text":"..."}`
- **Docs:** `http://localhost:8000/docs`

Optional: set **`LANGUAGE_DETECTION_MODEL_PATH`** to an absolute path of a `.pkl` file if you override the default location (see `app/main.py`).

---

## Docker (inference image)

The production image contains **only inference**: FastAPI, shared `text_cleaning`, and the bundled `.pkl` — not the training stack (`scripts/`, pandas, notebooks).

```bash
cp models/trained_pipeline-0.1.0.pkl app/model/
docker build -t language-detector-api .
docker run --rm -p 8000:8000 language-detector-api
```

Published image (example tag): `mateo755/language-detector-api:v1`

**Docker Hub overview text** for visitors and `docker run` examples lives in **[`docker_docs.md`](docker_docs.md)** — keep it in sync with this README when the inference contract changes.

---

## Exploratory analysis (EDA)

Optional dev dependencies (Jupyter, matplotlib):

```bash
pip install -e ".[dev]"
jupyter notebook notebooks/eda.ipynb
```

---

## Deployment (Render)

The following describes a **test / learning deployment**, not a production commitment. The service **is not** expected to stay up indefinitely: the app may be removed, the URL may change, or the free tier may sleep or rate-limit traffic. **Do not rely on this endpoint** for anything critical; run the API **locally** or **via Docker** (or deploy your own instance) for stable use.

**Trial base URL (may break):** [https://language-detector-api-qjnu.onrender.com](https://language-detector-api-qjnu.onrender.com)

When it responds, behavior matches the repo: container listens on port **8000** inside the image; Render terminates TLS and forwards HTTPS to the service. Interactive OpenAPI/Swagger is at `/docs` on that host (append `/docs` to the base URL above). On a cold start, the first request can take noticeably longer while the instance wakes up.

Example (if the URL is still valid):

```bash
curl -s https://language-detector-api-qjnu.onrender.com/health
```

```bash
curl -s -X POST https://language-detector-api-qjnu.onrender.com/detect \
  -H "Content-Type: application/json" \
  -d '{"text":"Bonjour tout le monde"}'
```

**If you deploy yourself** on Render or elsewhere: point the platform at this repository’s `Dockerfile`, expose port **8000** to the process, and map public HTTPS to that port as required by the provider.

---

## Repository layout

```text
language_recognition/
├── app/                         # FastAPI app; bundled model: app/model/trained_pipeline-0.1.0.pkl
├── data/
│   └── language_detection.csv   # expected training data (columns Text, Language)
├── models/
│   └── trained_pipeline-0.1.0.pkl   # training output (optional in repo)
├── notebooks/
│   └── eda.ipynb                # exploratory data analysis
├── scripts/
│   └── train_pipeline.py        # training entry point
├── src/
│   ├── data_preprocessing.py
│   ├── data_split.py
│   ├── model_training.py
│   └── text_cleaning.py         # shared preprocessing: training + API
├── Dockerfile                   # inference-only production image
├── requirements-api.txt         # minimal deps for Docker image
├── docker_docs.md               # Docker Hub overview source (inference image); copy-paste for Hub
├── pyproject.toml               # package metadata; train-pipeline console script
├── requirements.txt             # pip install -e .
└── README.md
```

After `pip install -e .`, a local `*.egg-info/` directory may appear (ignored by git).

---

## License

This project is released under the [MIT License](LICENSE).

Copyright (c) 2026 Mateusz Sobczyński
