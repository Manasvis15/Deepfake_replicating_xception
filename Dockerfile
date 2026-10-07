# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OpenCV (headless) needs glib at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# CPU-only torch keeps the image ~1.5 GB smaller. For GPU, use a CUDA base image and
# drop the --index-url line.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies before copying source, so code changes don't bust this layer.
# Reads them from pyproject.toml (single source of truth); torch was installed above.
COPY pyproject.toml ./
RUN python -c "import tomllib, subprocess, sys; \
deps = [d for d in tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies'] \
        if not d.startswith('torch')]; \
subprocess.check_call([sys.executable, '-m', 'pip', 'install', *deps])"

COPY app ./app

# Never run as root inside the container.
RUN useradd --create-home --uid 10001 appuser
USER appuser

ENV DFD_ENVIRONMENT=prod \
    DFD_WEIGHTS_PATH=/srv/weights/xception_best.pth

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/ready || exit 1

# One worker per container: each worker holds its own copy of the model in memory.
# Scale by running more containers rather than more workers.
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
