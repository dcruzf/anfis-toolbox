## Contributing

Thanks for your interest in improving ANFIS Toolbox! This guide explains how to set up a local development environment, run checks, and submit changes.

### Ways to Help

* Report bugs or request features via GitHub issues.
* Improve documentation, examples, or tests.
* Fix bugs or add features to the library.

### Quick Start

This project uses **Hatch** for all tooling, providing a streamlined development experience with environment management and dependencies.

#### Setting up the Development Environment

To set up the development environment, run the following command to install it via **pip**:

```bash
pip install -e .[dev]
```

This will install the necessary development dependencies, including tools like:

* `pytest` – for running tests
* `mypy` – for static type checking
* `bandit` – for security linting
* `ruff` – for fast Python linting
* `pre-commit` – for managing pre-commit hooks
* `hatch` – for managing environments and dependencies

By installing the development environment with `pip install -e .[dev]`, you get direct access to these tools, offering flexibility in your workflow.

#### Recommended Approach: Hatch Environments

While it's possible to use the tools directly, we recommend managing the development environment through **Hatch environments** for better control and isolation. This ensures that each tool and dependency is correctly handled and isolated within each specific environment (e.g., `dev`, `test`). Running tools within controlled environments helps maintain consistency and avoid potential version conflicts across your setup.

However, if you prefer more freedom, you can opt to use the tools directly after installing them with `pip`. This gives you more flexibility, but may come at the cost of the strict control and environment isolation provided by Hatch.

For further details on how to manage environments and dependencies, refer to the [Hatch documentation](https://hatch.pypa.io).

### Development Workflow

1. Create a feature branch.
2. Make focused changes, including tests and documentation updates.
3. Run the checks below.
4. Open a pull request.

### Running Checks

Formatting and linting:

```bash
hatch fmt
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

Run the full test suite (across all Python versions configured in Hatch):

```bash
hatch test -a -c
```

### Documentation

The docs are built using MkDocs Material, and the API reference is generated from Google-style docstrings.

To serve docs locally:

```bash
hatch run docs:serve
```

To build static docs:

```bash
hatch run docs:build
```

### Style Guidelines

* Keep changes small and focused.
* Add or update tests when modifying behavior.
* Follow the existing formatting (Ruff) and type-checking (mypy) conventions.
* Keep docstrings concise and in Google style.

### Reporting Issues

When filing an issue, please include:

* Expected vs. actual behavior
* A minimal reproducible example
* Python version and OS
* Any relevant logs or stack traces

### Submitting Pull Requests

PRs should include:

* A clear description of the change
* Tests and documentation updates, if applicable
* Notes about any breaking changes

We aim to review contributions promptly. Thank you for helping improve ANFIS Toolbox!
