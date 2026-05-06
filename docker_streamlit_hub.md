## Language Detector Streamlit (web UI image)

**Published image on Docker Hub:** [mateo755/language-detector-streamlit](https://hub.docker.com/r/mateo755/language-detector-streamlit) · [Tags](https://hub.docker.com/r/mateo755/language-detector-streamlit/tags)

---

**Streamlit** browser UI that calls a deployed **FastAPI Language Detector API** (`/health`, `/detect`) over HTTPS (`httpx`).

It **does not** contain the sklearn model or inference stack. For inference use **[mateo755/language-detector-api](https://hub.docker.com/r/mateo755/language-detector-api)**. For training, Docker workflow, and Streamlit-from-source workflows, see the repository **`README.md`**.

The **`docker_streamlit_hub.md`** file in this repo is the **canonical Markdown** to paste into **Docker Hub → Repository overview** for this image; keep it aligned with **`README.md`** and **`Dockerfile.streamlit`** when URLs, env vars, or ports change.

### Quick start

```bash
docker run --rm -p 8501:8501 \
  -e LANGUAGE_DETECTOR_API_URL=https://your-api-host.example.com \
  mateo755/language-detector-streamlit:v1
```

Open **http://localhost:8501** in your browser.

- **Port:** default listen port inside the container is **8501**; map as needed (**`-p host:container`** must match **`PORT`** if you override it — e.g. **`-e PORT=9000 -p 9000:9000`**).
- **Health:** **`/_stcore/health`** (used by the image `HEALTHCHECK`).

### Networking note

Inside the container, **`127.0.0.1`** is loopback **in the container**, not your laptop. Either set **`LANGUAGE_DETECTOR_API_URL`** to a **public** API URL, or reach FastAPI on the host via **`http://host.docker.internal:<port>`** (Docker Desktop), or use **`--add-host=host.docker.internal:host-gateway`** on Linux Docker.

### Environment variables

| Variable | Description |
| -------- | ----------- |
| **`LANGUAGE_DETECTOR_API_URL`** | **Recommended for deployed UI.** Base URL of the FastAPI service (scheme + host, **no** `/detect` suffix). Pre-fills **Remote** mode. Trailing `/` is acceptable. |
| **`LANGUAGE_DETECTOR_LOCAL_API_URL`** | Optional override for **Local (localhost)** in the app (**`http://127.0.0.1:8000`** default). Rarely useful for cloud-hosted Streamlit. |
| **`PORT`** | Listen port **inside** the container; platforms such as **Render** set this. Default **8501** when unset (`Dockerfile`). |

### Related inference image

- **`mateo755/language-detector-api`** — FastAPI service with bundled `.pkl` model (`/detect`, `/health`, `/docs`).

### Short description (Docker Hub one-line field)

Use this single line in the **repository description** field on Docker Hub:

`Streamlit UI for Language Detector FastAPI (/health, /detect); no model in image.`

### Source

Built from **`Dockerfile.streamlit`** with **`requirements-streamlit.txt`** and **`streamlit_app.py`** (see **`README.md`** → Streamlit section).
