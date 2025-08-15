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
	uv self update
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
	uv tool run hatch test -c

.PHONY: test-all ## Run tests with coverage
test-all: .hatch
	uv tool run hatch test -c --all

.PHONY: lf ## Run last failed tests
lf: .uv
	uv run pytest --lf -vv

.PHONY: all
all: format lint test-all


.PHONY: build  ## Build wheel and sdist into dist/
build: .hatch
	uv tool run hatch build

.PHONY: docs  ## Serve the documentation at http://localhost:8000
docs:
	uvx --with mkdocs-material \
    --with mkdocstrings --with mkdocstrings-python \
    --with mkdocs-awesome-pages-plugin \
    --with mkdocs-git-revision-date-localized-plugin \
    --with ruff \
    mkdocs serve -a 127.0.0.1:8000

.PHONY: docs-build  ## Build static docs into site/
docs-build: .uv
	uvx --with mkdocs-material \
		--with mkdocstrings --with mkdocstrings-python \
		--with mkdocs-awesome-pages-plugin \
		--with mkdocs-git-revision-date-localized-plugin \
		--with ruff \
		mkdocs build

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
