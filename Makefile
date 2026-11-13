.PHONY: help install lint test up load-vector load-outline down

help:
	@echo "make install       install dev dependencies"
	@echo "make lint          run ruff"
	@echo "make test          run the unit tests"
	@echo "make up            start both designs and the monitoring stack"
	@echo "make load-vector   send load to the vector design (5 minutes)"
	@echo "make load-outline  send load to the outline design (5 minutes)"
	@echo "make down          stop everything and remove volumes"

install:
	pip install -r requirements-dev.txt

lint:
	ruff check .
	ruff format --check .

test:
	python -m pytest -q

up:
	PREPARE=vector,outline docker compose --profile vector --profile outline up -d --build --wait
	@echo "Grafana: http://localhost:3000  (dashboard: RAG latency: vector vs outline)"

load-vector:
	DESIGN=vector docker compose --profile load run --rm k6

load-outline:
	DESIGN=outline docker compose --profile load run --rm k6

down:
	docker compose --profile vector --profile outline --profile load down -v
