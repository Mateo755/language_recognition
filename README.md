# Language detection — end-to-end ML deployment

This repository is an **end-to-end MLOps-style project**: **data handling** and exploratory analysis, **model training** with scikit-learn, a **FastAPI** inference service, an inference-only **Docker** image **published to Docker Hub**, a root **`docker-compose.yml`** to run the **published** API and Streamlit UI together locally, and **local** validation of the container.

A **trial deployment on Render** was done to practice shipping the same Dockerized API to a managed host. That environment is **experimental**, **not** something I plan to keep running or supporting over time, so **any public URL may stop working** without notice. Details and a single reference link are in **[Deployment (Render)](#deployment-render)** below.

---

## Table of contents

- [Project journey](#project-journey)
- [Tech stack](#tech-stack)
- [Documentation map (README vs Docker Hub)](#documentation-map-readme-vs-docker-hub)
- [Getting started (local development)](#getting-started-local-development)
- [Training](#training)
- [Running the API locally](#running-the-api-locally)
- [Streamlit (web UI)](#streamlit-web-ui)
- [Trial deploy Streamlit on Render](#trial-deploy-streamlit-on-render)
- [Docker (inference image)](#docker-inference-image)
- [Docker Compose (full stack)](#docker-compose-full-stack)
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
7. **Docker Compose** — A root [`docker-compose.yml`](docker-compose.yml) file was added to run the **published** FastAPI and Streamlit images together on one local network (Streamlit calls the API by service name). See [Docker Compose (full stack)](#docker-compose-full-stack).

---

## Tech stack

| Area            | Choice                                                                 |
| --------------- | ---------------------------------------------------------------------- |
| Language        | Python **3.10+** (development); Docker image uses **Python 3.12** slim |
| ML              | **scikit-learn** pipeline (serialized with pickle)                     |
| Data / training | **pandas** (training path)                                             |
| API             | **FastAPI**, **Pydantic**, **Uvicorn**                                 |
| Optional UI     | **Streamlit** + **httpx** (optional `pip install -e ".[ui]"`)        |
| Packaging       | **pip** / `pyproject.toml`; Docker for production inference            |
| Cloud (optional) | **Render** — one-off trial deploy of the Docker image (see below)   |

---

## Documentation map (README vs Docker Hub)

| Document                             | Purpose                                                                                                                                                                                                                          |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **This `README.md`**                 | Full project: setup, training, local API, Docker workflow, optional cloud trial notes, repository map.                                                                                                                           |
| **[`docker_docs.md`](docker_docs.md)** | Markdown for the [Docker Hub](https://hub.docker.com/r/mateo755/language-detector-api) **overview** of the **inference image** (with Hub links for people opening the repo from GitHub). Training and EDA stay in this README only. |
| **[`docker_streamlit_hub.md`](docker_streamlit_hub.md)** | Markdown for the [language-detector-streamlit](https://hub.docker.com/r/mateo755/language-detector-streamlit) Hub **overview** (Streamlit UI). Keep it in sync with **`README.md`** if env vars or `Dockerfile.streamlit` change. |

After you edit endpoints, configuration, or limits documented for the API, refresh **`docker_docs.md`** and paste it into Docker Hub so the page matches the image. After Streamlit-related changes (**`Dockerfile.streamlit`**, `LANGUAGE_DETECTOR_API_URL`, and similar), do the same for **`docker_streamlit_hub.md`** on the Streamlit repository overview.

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

## Streamlit (web UI)

Optional browser UI that calls **only** the FastAPI HTTP API (`/health`, `/detect`); it does **not** load the pickled model.

From the repository root with the virtual environment activated:

```bash
pip install -e ".[ui]"
streamlit run streamlit_app.py
```

### Environment variables (Streamlit)

| Variable | Purpose |
| -------- | ------- |
| **`LANGUAGE_DETECTOR_API_URL`** | Default **remote** base URL when the app opens; you can change it for the session via the sidebar **“Set / change API URL”** popover. Trailing slashes are trimmed. |
| **`LANGUAGE_DETECTOR_LOCAL_API_URL`** | Overrides the **local** base URL when you choose **“Local (localhost)”** in the app (default: `http://127.0.0.1:8000`). |

### Choosing a backend in the app

In the sidebar:

- **Remote** — set the remote base URL in the sidebar **popover** **“Set / change API URL”**; open **“Info endpoint”** for protocol, host, full base URL, and a link to **`/docs`**. The first value comes from `LANGUAGE_DETECTOR_API_URL`. Public trial URLs may stop working; see [Deployment (Render)](#deployment-render).
- **Local (localhost)** — `httpx` talks to **127.0.0.1 on the machine running Streamlit**. That is useful when you run [`streamlit run streamlit_app.py`](#streamlit-web-ui) on your laptop with the API on the same host. **It does not reach your laptop when Streamlit is deployed on a remote host** (that “localhost” is inside the container).

If the remote API is down or slow, run the API [locally](#running-the-api-locally) or via [Docker](#docker-inference-image), and either use **Remote** in the UI with that reachable URL or run Streamlit locally with **Local**.

---

### Trial deploy Streamlit on Render

Second **Web Service** for a **trial Streamlit UI** ([`Dockerfile.streamlit`](Dockerfile.streamlit)): small image with [`requirements-streamlit.txt`](requirements-streamlit.txt) and [`streamlit_app.py`](streamlit_app.py) only — **not** the inference stack.

**Render (panel):**

1. Create a **Web Service** from this repo.
2. Set **Dockerfile path** to **`Dockerfile.streamlit`** (leave the API service on the root [`Dockerfile`](Dockerfile)).
3. Do **not** override the start command with a fixed port: the image uses **`PORT`** from Render (see `CMD` in [`Dockerfile.streamlit`](Dockerfile.streamlit)).
4. Set **`LANGUAGE_DETECTOR_API_URL`** to your **public HTTPS base URL** for the FastAPI service (no path; trailing `/` is OK — the app normalizes it). Requests are **server-to-server** from Streamlit to the API (**no browser CORS** requirement for this flow).
5. Optional: set **Health check path** to **`/_stcore/health`** (Streamlit core health).

**After testing:** **Suspend** or **Delete** this Web Service so it does not keep running on the free tier.

**Local Docker smoke test** (from repo root):

```bash
docker build -f Dockerfile.streamlit -t language-detector-streamlit .
docker run --rm -p 8501:8501 \
  -e LANGUAGE_DETECTOR_API_URL=https://your-fastapi-host.example.com \
  language-detector-streamlit
```

Then open `http://localhost:8501`. To mimic Render’s dynamic port: add `-e PORT=<port>` and map `-p <port>:<port>` accordingly.

---

## Docker (inference image)

The production image contains **only inference**: FastAPI, shared `text_cleaning`, and the bundled `.pkl` — not the training stack (`scripts/`, pandas, notebooks).

```bash
cp models/trained_pipeline-0.1.0.pkl app/model/
docker build -t language-detector-api .
docker run --rm -p 8000:8000 language-detector-api
```

Published image (example tag): `mateo755/language-detector-api:v1`

**Docker Hub overview text** for visitors and `docker run` examples lives in **[`docker_docs.md`](docker_docs.md)** (inference) and **[`docker_streamlit_hub.md`](docker_streamlit_hub.md)** (Streamlit UI) — keep them in sync with this README when the contract changes.

### Docker Compose (full stack)

[`docker-compose.yml`](docker-compose.yml) defines two services — **api** (FastAPI) and **streamlit** (UI) — using the default Hub images (`mateo755/language-detector-api:v1`, `mateo755/language-detector-streamlit:v1`). Streamlit receives `LANGUAGE_DETECTOR_API_URL=http://api:8000` on the Compose network.

From the repository root:

```bash
docker compose up -d
```

Then open **http://localhost:8501** (UI) and **http://localhost:8000/docs** (API Swagger). Host ports default to **8000** and **8501**; override with `LANGUAGE_DETECTOR_API_PORT`, `LANGUAGE_DETECTOR_STREAMLIT_PORT`, or image tags via `LANGUAGE_DETECTOR_API_IMAGE` / `LANGUAGE_DETECTOR_STREAMLIT_IMAGE` if needed (see comments at the top of `docker-compose.yml`).

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
├── Dockerfile.streamlit         # Streamlit-only UI image (trial / second deploy)
├── docker-compose.yml           # local full stack: published API + Streamlit images
├── requirements-api.txt         # minimal deps for Docker image
├── requirements-streamlit.txt   # minimal deps for Dockerfile.streamlit
├── docker_docs.md               # Docker Hub overview source (inference image); copy-paste for Hub
├── docker_streamlit_hub.md      # Docker Hub overview source (Streamlit UI); copy-paste for Hub
├── pyproject.toml               # package metadata; train-pipeline console script; optional [ui] extra
├── streamlit_app.py             # optional Streamlit client for the FastAPI /detect API
├── requirements.txt             # pip install -e .
└── README.md
```

After `pip install -e .`, a local `*.egg-info/` directory may appear (ignored by git).

---

## License

This project is released under the [MIT License](LICENSE).

Copyright (c) 2026 Mateusz Sobczyński
