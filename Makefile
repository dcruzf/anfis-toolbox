PYTHONPATH := .:$(PYTHONPATH)
export PYTHONPATH
.DEFAULT_GOAL := all
sources = anfis_toolbox tests
project_dir = anfis_toolbox


.PHONY: .uv  ## Check that uv is installed
.uv:
	@uv -V || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.PHONY: rebuild-lockfiles  ## Rebuild lockfiles from scratch, updating all dependencies
rebuild-lockfiles: .uv
	uv lock --upgrade

.PHONY: install ## Install the package, dependencies, and pre-commit for local development
install: .uv
	uv sync
	uv tool install hatch
	uv tool update-shell
	uvx pre-commit install
	uvx pre-commit autoupdate

.PHONY: format  ## Auto-format python source files
format: .uv
	uvx ruff check --fix $(sources)
	uvx ruff format $(sources)

.PHONY: lint  ## Lint python source files
lint: .uv
	uvx ruff check $(sources)
	uvx ruff format --check $(sources)

.PHONY: .hatch  ## Check that hatch is installed
.hatch:
	@uv tool run hatch --version || echo 'Please install hatch: uv tool install hatch'

.PHONY: test ## Run tests
test: .hatch
	uv tool run hatch run test:test

.PHONY: test-cov ## Run tests with coverage
test-cov: .hatch
	uv tool run hatch run test:test-cov
	uv tool run hatch run test:cov-report

.PHONY: lf ## Run last failed tests
lf: .uv
	uv run pytest --lf -vv

.PHONY: all
all: format lint test

.PHONY: clean  ## Clear local caches and build artifacts
clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]'`
	rm -f `find . -type f -name '*~'`
	rm -f `find . -type f -name '.*~'`
	rm -rf .cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf *.egg-info
	rm -f .coverage
	rm -f .coverage.*
	rm -rf build
	rm -rf dist

.PHONY: help  ## Display this message
help:
	@grep -E \
		'^.PHONY: .*?## .*$$' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ".PHONY: |## "}; {printf "\033[36m%-19s\033[0m %s\n", $$2, $$3}'
