.PHONY: check test install

check:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

install:
	uv sync --dev
