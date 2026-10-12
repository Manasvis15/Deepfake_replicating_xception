# Deepfake Detection API

A FastAPI service that serves DeepfakeBench's Xception detector (`xception_best.pth`,
trained on FF++ c23). Upload a face crop, get back P(fake), P(real) and a label.

## Architecture

Requests flow down through four layers. Each layer only depends on the ones below it.

```
HTTP  ─▶  api/        routes, validation, auth, error mapping, request IDs
              │
              ▼
          services/   business logic: preprocess, call the detector, apply threshold
              │
              ▼
          ml/base.py  Detector interface (Protocol)  ◀── the only thing services know
              │
              ▼
          ml/xception_detector.py   PyTorch implementation (loads weights once)
```

- **The API and service layers never import PyTorch.** They depend on the `Detector`
  interface. Tests use a fake detector, and the model could be swapped (Effort, ONNX)
  without touching any endpoint.
- **Domain errors know nothing about HTTP.** `core/errors.py` defines them;
  `api/errors.py` is the one place that maps them to status codes.

```
deepfake/
├── app/                     the API (production code)
│   ├── main.py              app factory + lifespan (model loaded once at startup)
│   ├── core/                config.py, errors.py, logging.py, policy.py
│   ├── api/                 deps.py, errors.py, middleware.py, routes/
│   ├── services/            prediction.py: orchestration + threshold decision
│   ├── schemas/             Pydantic response models (= the OpenAPI contract)
│   └── ml/                  base.py (interface), preprocessing.py, xception.py,
│                            xception_detector.py
├── tests/                   API, preprocessing and model tests
├── research/                evaluation and plotting scripts used for the analysis
├── plots/                   analysis figures (ROC, PCA, t-SNE, logits, …)
├── weights/                 xception_best.pth goes here (not committed)
└── data/                    real/ and fake/ face crops (not committed)
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
make install                        # CPU torch + app + dev tools
cp .env.example .env
# put xception_best.pth in weights/ (DeepfakeBench GitHub release v1.0.1)
make run                            # http://localhost:8000/docs
```

With Docker:

```bash
make docker-build
make docker-run                     # mounts ./weights read-only
```

## Research scripts

These reproduce the evaluation and the Effort paper's (arXiv:2411.15633) Xception analysis.
Run them from the repo root:

```bash
python3 research/xception.py eval --weights weights/xception_best.pth \
    --real data/real --fake data/fake
python3 research/xception_plot.py --weights weights/xception_best.pth \
    --real data/real --fake data/fake
```

Results on our 307-image set (171 real, 136 fake): frame-level AUC 0.846, accuracy 78.8%.
The first PCA component explains ~96% of the embedding variance and the logits lie on
y ≈ −x, reproducing the paper's finding that Xception's feature space collapses.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health/live` | Liveness: the process is up |
| GET | `/health/ready` | Readiness: the model is loaded (503 otherwise) |
| POST | `/api/v1/predict` | Score one image (form field `file`) |
| POST | `/api/v1/predict/batch` | Score several images in one forward pass (field `files`) |
| GET | `/api/v1/model` | Model details, default threshold and mode presets |

Interactive docs at `/docs` (disabled when `DFD_ENVIRONMENT=prod`).

```bash
curl -F "file=@face.png;type=image/png" localhost:8000/api/v1/predict

curl -F "files=@a.png;type=image/png" -F "files=@b.jpg;type=image/jpeg" \
     localhost:8000/api/v1/predict/batch
```

Example response:

```json
{
  "prediction": {"filename": "face.png", "label": "fake", "p_fake": 0.8137, "p_real": 0.1863},
  "decision": {"threshold": 0.5, "mode": null, "uncertain_margin": 0.0, "tta": false},
  "model": {"name": "xception-deepfakebench", "weights_sha256": "…", "device": "cpu",
            "input_size": 256},
  "inference_ms": 41.7
}
```

Every error has the same shape, with the request ID for tracing in the logs:

```json
{"error": {"code": "invalid_image", "message": "File could not be decoded as an image.",
           "details": {"filename": "x.png"}, "request_id": "6da46de0d0674949b279c3852bccbcd5"}}
```

| Status | Code | When |
|---|---|---|
| 401 | `unauthorized` | `DFD_API_KEY` is set and the `X-API-Key` header is missing or wrong |
| 413 | `payload_too_large` | file exceeds `DFD_MAX_UPLOAD_BYTES` |
| 415 | `unsupported_media_type` | content type isn't JPEG/PNG/WebP/BMP |
| 422 | `invalid_image` / `too_many_files` / `validation_error` | undecodable, empty or tiny image; too many files; bad request |
| 503 | `model_not_ready` | model not loaded |
| 500 | `internal_error` | unexpected failure (details logged, never returned) |

## Decision filters

Both prediction endpoints accept these optional query parameters. They change only how
the label is decided, never `p_fake` itself, and every response echoes them back in
`decision` so a label can always be interpreted.

| Parameter | Values | Effect |
|---|---|---|
| `threshold` | 0 to 1 | P(fake) at or above this is labelled `fake` (default `DFD_THRESHOLD`) |
| `mode` | `balanced`, `high_recall`, `high_precision` | named operating point; use instead of `threshold` |
| `uncertain_margin` | 0 to 0.5 | scores within ± this of the threshold are labelled `uncertain` |
| `tta` | `true` / `false` | average the score with the mirrored image (about 2× the compute) |

```bash
# Flag only confident fakes, and mark close calls for human review
curl -F "file=@face.png;type=image/png" \
  "localhost:8000/api/v1/predict?mode=high_precision&uncertain_margin=0.1"
```

- **Modes** are server-side presets (`DFD_MODE_THRESHOLDS`). Clients pick an intent and
  the thresholds can be recalibrated centrally. `balanced` (0.45) is the best threshold
  from our ROC curve; `high_recall` and `high_precision` are starting points to calibrate.
- **`uncertain`** exists because scores near the threshold are where this model is least
  reliable (see `plots/1_score_histogram.png`); those cases are better sent to a person.
- Sending both `threshold` and `mode` is rejected with `422 invalid_parameter`.

## Configuration

All settings are environment variables with the `DFD_` prefix (or a `.env` file).
See `.env.example`. The main ones:

| Variable | Default | Meaning |
|---|---|---|
| `DFD_WEIGHTS_PATH` | `weights/xception_best.pth` | checkpoint location |
| `DFD_DEVICE` | `auto` | `auto`, `cpu` or `cuda` |
| `DFD_THRESHOLD` | `0.5` | default threshold when a request sets neither `threshold` nor `mode` |
| `DFD_MODE_THRESHOLDS` | `{"balanced": 0.45, "high_recall": 0.25, "high_precision": 0.8}` | threshold behind each `mode` (JSON) |
| `DFD_MAX_CONCURRENT_INFERENCES` | `2` | forward passes allowed at once (memory guard) |
| `DFD_MAX_UPLOAD_BYTES` | `10485760` | per-file size limit |
| `DFD_MAX_BATCH_FILES` | `16` | files per batch request |
| `DFD_API_KEY` | unset | enables API-key auth on `/api/*` |

## Development

```bash
make test        # 56 tests (the 5 model tests need torch)
make lint        # ruff
make format
make typecheck   # mypy (strict)
```

The API tests inject a `FakeDetector`, so they run in under a second with no weights.

## Design decisions

- **The model loads once, at startup** (FastAPI lifespan), never per request. If it fails
  to load, the process exits: a container that can't serve shouldn't pretend to be healthy.
- **Inference runs in a worker thread** (`run_in_threadpool`) so the async event loop keeps
  accepting requests while the CPU is busy.
- **A semaphore caps concurrent forward passes**, so a burst of requests can't exhaust memory.
- **Liveness and readiness are separate.** Orchestrators restart a dead process but only
  route traffic to a ready one.
- **Uploads are read with a size cap** (limit + 1 bytes), so oversized files are rejected
  without being read entirely into memory.
- **`torch.load(weights_only=True)`** refuses to unpickle arbitrary objects from the checkpoint.
- **Every response carries the weights' SHA-256**, so any result can be traced to the exact model.
- **No per-request state on the model.** The research script stored `last_emb` on the model;
  that isn't safe under concurrent requests, so it was removed here.
- **One worker per container.** Each worker holds its own model copy; scale with containers.

## Limitations and next steps

- **Inputs must be face crops.** The model was trained on aligned faces; full frames give
  unreliable scores. A face-detection step (e.g. RetinaFace or dlib, matching DeepfakeBench's
  preprocessing) is the most valuable next addition.
- **Frame-level only.** Video upload with frame sampling and score averaging isn't supported yet.
- **No rate limiting.** Add it at the gateway / reverse proxy, or with a library like `slowapi`.
- **Generalization.** Xception is known to generalize poorly to unseen forgery methods
  (see the Effort paper, arXiv:2411.15633). Thanks to the `Detector` interface, a stronger
  model can be added as another implementation without changing the API.
