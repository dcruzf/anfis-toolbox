## Contributing

Thanks for your interest in improving ANFIS Toolbox! This guide explains how to set up a local development environment, run checks, and submit changes.

### Ways to help

- Report bugs or request features via GitHub issues.
- Improve documentation, examples, or tests.
- Fix bugs or add features in the library.

### Quick start

This project uses Hatch for all tooling.

```bash
hatch --version
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

We aim to review contributions promptly. Thank you for helping improve ANFIS Toolbox!
