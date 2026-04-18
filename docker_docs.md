## Language Detector API (inference image)

**Published image on Docker Hub:** [mateo755/language-detector-api](https://hub.docker.com/r/mateo755/language-detector-api) · [Tags](https://hub.docker.com/r/mateo755/language-detector-api/tags)

---

This Docker image ships **only the prediction (inference) stack**: a **FastAPI** service, the shared **text-cleaning** step (aligned with training), and a **frozen scikit-learn** pipeline loaded from a `.pkl` artifact at startup.

It does **not** include training code, datasets, or notebooks. For the full project (data prep, training, EDA, Docker workflow, and cloud deployment), see the repository **`README.md`**.

The **`docker_docs.md`** file in this repo is the **canonical Markdown** to paste into **Docker Hub → Repository overview** for the published image; keep it aligned with **`README.md`** when the inference API changes.

### Quick start

```bash
docker run --rm -p 8000:8000 mateo755/language-detector-api:v1
```

- **Port:** `8000` (map as needed, e.g. `-p 8000:8000`).
- **Health:** `GET /health` returns `{"status":"ok"}` (also used by the image `HEALTHCHECK`).

### Environment variables

| Variable | Description |
|----------|-------------|
| `LANGUAGE_DETECTION_MODEL_PATH` | Optional **absolute path** to a `.pkl` file **inside the container**. If unset, the model bundled in the image is used. Mount your own artifact and set this variable to override it. |

### Endpoints

**`POST /detect`** — JSON body:

```json
{ "text": "Any text to classify" }
```

Example with `curl`:

```bash
curl -s -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world"}'
```

Response:

```json
{ "text": "Hello world", "language": "<predicted_label>" }
```

The `text` field is limited to **50,000** characters per request.

### Interactive docs

Standard FastAPI OpenAPI/Swagger UI is available at **`/docs`** while the container is running.

### Source

Full source, training instructions, and deployment notes: see the project **`README.md`** in the Git repository.
