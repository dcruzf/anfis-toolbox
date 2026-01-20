# Contributing to ANFIS Toolbox

Thank you for your interest in contributing to ANFIS Toolbox! This guide will help you get started.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) (recommended package manager)

### Development Setup

1. Fork the repository and clone your fork:

   ```bash
   git clone https://github.com/YOUR_USERNAME/anfis-toolbox.git
   cd anfis-toolbox
   ```

2. Install development dependencies:

   ```bash
   make install
   ```

   This will:
   - Install `uv` (if not already installed)
   - Install `hatch` for environment management
   - Sync all dependencies
   - Set up pre-commit hooks

3. Verify your setup by running the tests:

   ```bash
   make test
   ```

## Development Workflow

### Running Tests

```bash
# Run tests with coverage
make test

# Run tests for all Python versions
make test-all

# Run only the last failed tests
make lf
```

### Code Quality

This project uses several tools to maintain code quality:

```bash
# Format code
make format

# Run linting (via pre-commit)
make lint

# Run type checks
make type-check

# Run security checks
make bandit

# Run all checks
make all
```

### Building Documentation

```bash
# Serve docs locally at http://localhost:8000
make docs

# Build static docs
make docs-build
```

## Submitting Changes

1. Create a new branch for your changes:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure all tests pass:

   ```bash
   make all
   ```

3. Commit your changes with a descriptive message.

4. Push to your fork and open a pull request.

## Questions?

If you have questions or want to propose larger changes, please open an issue or discussion on GitHub.
