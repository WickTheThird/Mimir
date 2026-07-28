.PHONY: help install dev test lint fix typecheck eval serve ui doctor clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Create the venv and install MIMIR
	uv venv --python 3.13 .venv
	uv pip install -e ".[dev,search]"

dev: install ## Install everything including the web UI
	cd web && npm install

test: ## Run the test suite
	.venv/bin/python -m pytest tests/ -q

lint: ## Lint
	.venv/bin/ruff check src tests

fix: ## Lint and autofix
	.venv/bin/ruff check --fix src tests

typecheck: ## Type-check the package and the web UI
	.venv/bin/mypy src/mimir || true
	cd web && npx tsc --noEmit

eval: ## Run the deterministic evaluation corpus (ADR 21)
	.venv/bin/mimir evaluate --deterministic

serve: ## Run the API
	.venv/bin/mimir serve

ui: ## Run the web UI dev server
	cd web && npm run dev

doctor: ## Check dependencies
	.venv/bin/mimir doctor

clean:
	rm -rf .pytest_cache .ruff_cache dist build web/dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
