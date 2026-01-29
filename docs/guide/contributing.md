## Contributing

Thank you for considering contributing to ANFIS Toolbox! This document outlines the guidelines and best practices to help you collaborate in an efficient and organized way. By following these recommendations, we aim to ensure a transparent, consistent, and productive contribution process for everyone involved—from opening issues to submitting pull requests.

### Ways to help

- Report bugs or request features via GitHub issues.
- Improve documentation, examples, or tests.
- Fix bugs or add features in the library.

## Project Goals

- Provide a batteries-included Adaptive Neuro-Fuzzy Inference System (ANFIS)
	implementation in Python.
- Offer high-level estimators (`ANFISRegressor`, `ANFISClassifier`) that feel
	familiar to scikit-learn users while remaining dependency-free.
- Support a wide range of membership function families and training regimes.
- Ship with reproducible examples, thorough tests, and easy-to-read docs.

### Quick start

#### 1. Clone the repository
```bash
git clone https://github.com/dcruzf/anfis-toolbox.git
cd anfis-toolbox
```

#### 2. Install the project in editable mode with development dependencies

```bash
pip install -e .[dev]
```

#### 3. Set up the pre-commit hooks

```bash
hatch run install
```

### Development workflow

1. Create a feature branch.
2. Make focused changes with tests and docs updates.
3. Run the checks below.
4. Open a pull request.

### Running checks

Formatting and linting:

```bash
hatch fmt
```

Run all pre-commit checks:

```bash
hatch run all
```

Type checks and security scan:

```bash
hatch run typing
hatch run security
```

Tests:

```bash
hatch test
```

Run the full suite (all Python versions configured in Hatch):

```bash
hatch test -a -c
```


### Documentation

Docs are built with MkDocs Material and include API reference generated from Google-style docstrings.

Serve docs locally:

```bash
hatch run docs:serve
```

Build static docs:

```bash
hatch run docs:build
```

### Style guidelines

- Keep changes small and focused.
- Add or update tests when behavior changes.
- Follow existing formatting (Ruff) and type-checking (mypy) conventions.
- Keep docstrings concise and in Google style.

### Reporting issues

When filing an issue, include:

- Expected vs. actual behavior
- Minimal reproducible example
- Python version and OS
- Any relevant logs or stack traces

### Submitting pull requests

PRs should include:

- A clear description of the change
- Tests and docs updates if applicable
- Notes about any breaking changes
- Confirmation that `hatch test -c --all` and `hatch run all` pass

We aim to review contributions promptly. Thank you for helping improve ANFIS Toolbox!
