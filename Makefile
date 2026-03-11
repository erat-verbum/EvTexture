.PHONY: install lint lint-fix check run

install:
	uv venv --clear
	uv sync

lint:
	uv run ruff check src

lint-fix:
	uv run ruff check src --fix

check:
	PYTHONPATH=. uv run pyright src

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
