# '>' instead of tabs for recipe lines, so copy-pasting can't break the file.
.RECIPEPREFIX = >
.PHONY: install test lint format typecheck run docker-build docker-run

install:
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -e ".[dev]"

test:
> pytest

lint:
> ruff check .
> ruff format --check .

format:
> ruff check --fix .
> ruff format .

typecheck:
> mypy app

run:
> uvicorn app.main:create_app --factory --reload --port 8000

docker-build:
> docker build -t deepfake-api:latest .

docker-run:
> docker run --rm -p 8000:8000 -v "$(PWD)/weights:/srv/weights:ro" deepfake-api:latest
