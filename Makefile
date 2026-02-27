.PHONY: install lint lint-fix check test test-unit test-int run

install:
	uv venv --clear
	uv sync

lint:
	uv run ruff check src test

lint-fix:
	uv run ruff check src test --fix

check:
	PYTHONPATH=. uv run pyright src test

test:
	PYTHONPATH=. uv run pytest --cov=src --cov-report=term-missing --tb=short

test-unit:
	PYTHONPATH=. uv run pytest test/unit/ -v --tb=short

test-int:
	PYTHONPATH=. uv run pytest test/integration/ -v --tb=short

run:
	uv run uvicorn src.main:app --host 0.0.0.0 --port 8001

docker-build:
	docker build -t evtexture .

docker-run:
	docker run --gpus all -p 8001:8001 evtexture

up:
	docker-compose up -d

up-build:
	docker-compose up -d --build

down:
	docker-compose down
