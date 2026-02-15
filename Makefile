# =============================================================================
# Vaeda Development Makefile
# =============================================================================

.PHONY: help install install-dev lint format typecheck test \
        docker-dev docker-prod clean

# Default target
help:
	@echo "Vaeda Development Commands"
	@echo "=========================="
	@echo ""
	@echo "Local Development (requires uv):"
	@echo "  make install        Install core dependencies"
	@echo "  make install-dev    Install core + dev dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Run ruff linter"
	@echo "  make format         Format code with ruff"
	@echo "  make typecheck      Run mypy type checker"
	@echo "  make test           Run pytest"
	@echo "  make test-cov       Run pytest with coverage"
	@echo "  make check          Run all checks (lint, typecheck, test)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-prod    Build production image"
	@echo "  make docker-dev     Build and run development container"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          Remove build artifacts and caches"
	@echo "  make lock           Update uv.lock file"

# =============================================================================
# Local Installation
# =============================================================================

install:
	uv sync

install-dev:
	uv sync --group dev

# =============================================================================
# Code Quality
# =============================================================================

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy src/vaeda

test:
	uv run pytest

test-cov:
	uv run pytest --cov=vaeda --cov-report=term-missing --cov-report=html

check: lint typecheck test

# =============================================================================
# Docker
# =============================================================================

docker-prod:
	docker build --target production -t vaeda:prod .

docker-dev:
	docker compose run --rm dev bash

# =============================================================================
# Maintenance
# =============================================================================

lock:
	uv lock

clean:
	rm -rf .venv
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage
	rm -rf dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
