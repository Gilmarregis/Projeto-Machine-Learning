.PHONY: install test lint format clean docker-build docker-run

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

setup:
	python -m venv venv
	. venv/bin/activate && make install

docker-build:
	docker build -f docker/Dockerfile -t ml-api .

docker-run:
	docker-compose -f docker/docker-compose.yml up -d

train:
	python -m src.models.train

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

mlflow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000