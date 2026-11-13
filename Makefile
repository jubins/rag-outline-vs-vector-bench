.PHONY: help install lint test vector outline down

help:
	@echo "make install   install dev dependencies"
	@echo "make lint      run ruff"
	@echo "make test      run the unit tests"
	@echo "make vector    run the vector design with the full monitoring stack"
	@echo "make outline   run the outline design with the full monitoring stack"
	@echo "make down      stop everything and remove volumes"

install:
	pip install -r requirements-dev.txt

lint:
	ruff check .
	ruff format --check .

test:
	python -m pytest -q

vector:
	DESIGN=vector docker compose --profile vector up --build

outline:
	DESIGN=outline docker compose --profile outline up --build

down:
	docker compose --profile vector --profile outline down -v
