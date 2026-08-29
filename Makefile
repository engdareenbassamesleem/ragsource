.PHONY: install run test lint format docker

install:
	python -m pip install -e ".[dev]"

run:
	uvicorn ragsource.api:app --reload

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

docker:
	docker compose up --build

