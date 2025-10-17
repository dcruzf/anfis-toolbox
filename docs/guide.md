# Developer Guide

Welcome to the ANFIS Toolbox developer guide. This document is intended for
contributors who want to understand the architecture, extend the library, or
work on the documentation and examples. If you are looking for the user-facing
API reference, consult the generated MkDocs site or the module docstrings.

## Project Goals

- Provide a batteries-included Adaptive Neuro-Fuzzy Inference System (ANFIS)
	implementation in pure Python.
- Offer high-level estimators (`ANFISRegressor`, `ANFISClassifier`) that feel
	familiar to scikit-learn users while remaining dependency-free.
- Support a wide range of membership function families and training regimes.
- Ship with reproducible examples, thorough tests, and easy-to-read docs.

## Repository Layout

```
anfis_toolbox/
├── model.py              # Low-level ANFIS graph (layers, forward passes)
├── regressor.py          # High-level regression estimator facade
├── classifier.py         # High-level classification estimator facade
├── membership.py         # Membership function implementations and helpers
├── builders.py           # Utilities that assemble models from configuration
├── optim/                # Optimizer and trainer implementations
├── metrics.py            # Regression and classification metric utilities
├── losses.py             # Loss definitions shared by optimizers
└── estimator_utils.py    # Mixins, validation helpers, sklearn-like machinery

docs/                     # MkDocs pages and reference material
examples/                 # Notebook-based walkthroughs
tests/                    # Pytest suite with full coverage
```

## High-Level Architecture

1. **Estimators** (`ANFISRegressor`, `ANFISClassifier`) expose a drop-in API
	 with `fit`, `predict`, `predict_proba` (classifier), `evaluate`, and
	 serialization helpers (`save`, `load`). They orchestrate membership
	 generation, rule configuration, optimizer instantiation, and evaluation.
2. **Builders** translate estimator configuration into the low-level model by
	 creating membership functions, calculating rule combinations, and producing
	 an object the trainers can consume.
3. **Low-level model** (`model.ANFIS`, `model.TSKANFISClassifier`) implements
	 the forward pass, rule firing strengths, normalization, and consequent
	 evaluation.
4. **Optimizers** (`optim.*`) encapsulate training strategies such as Hybrid
	 (OLS + gradient descent), Adam, RMSProp, SGD, and PSO. Each trainer accepts a
	 low-level model along with data and optional validation splits.
5. **Utilities** provide common infrastructure: `estimator_utils` for mixins
	 and input validation, `metrics` for reporting, `losses` for objective
	 functions, and `logging_config` for opt-in training logs.

The diagram below illustrates the runtime flow:

```
User code -> ANFISRegressor.fit ------------------------------.
							|                                             |
							v                                             |
				Membership config     Optimizer selection           |
							|                    |                        |
							v                    v                        |
				builders.ANFISBuilder  optim.<Trainer>              |
							|                    |                        |
							'------> model.ANFIS <------------------------'
																	 |
																	 v
														 Training loop
```

## Working With Estimators

- **Initialization**: Provide global defaults (`n_mfs`, `mf_type`, `init`,
	`overlap`, `margin`) plus optional `inputs_config` overrides for each input.
	`inputs_config` values can be dictionaries with membership parameters or
	explicit `MembershipFunction` instances.
- **Rules**: By default, all membership combinations form rules. Supply
	`rules=[(i1, i2, ...)]` to restrict to a subset of combinations.
- **Optimizers**: Pass `optimizer="adam"` (default classifier) or
	`optimizer="hybrid"` (default regressor). You can also supply instantiated
	trainers or custom subclasses of `BaseTrainer`.
- **Training**: `fit` accepts NumPy arrays, array-like objects, or pandas
	DataFrames. Use `validation_data=(X_val, y_val)` to monitor generalization.
- **Evaluation**: `evaluate` returns a metrics dictionary and optionally prints
	a nicely formatted summary. Use `return_dict=False` to suppress the return
	value.
- **Persistence**: `save` and `load` rely on `pickle`. Saved estimators capture
	fitted membership functions, rules, and optimizer state, enabling reuse.

## Membership Functions

Membership families live in `membership.py`. Key points:

- Each family subclasses `MembershipFunction` and implements `__call__`,
	`derivative`, and metadata accessors.
- Many families accept parameters such as centers, widths, slopes, or plateaus.
- Builders automatically infer parameters using grid spacing, fuzzy C-means
	clustering, or random sampling depending on `init`.
- Provide explicit membership functions via `inputs_config` to lock down
	shapes, e.g.:

	```python
	from anfis_toolbox.membership import GaussianMF

	inputs_config = {
			0: {"membership_functions": [GaussianMF(mean=-1, sigma=0.3), GaussianMF(mean=1, sigma=0.3)]},
			1: 4,  # shorthand for n_mfs=4 using estimator defaults
	}
	```

## Training Strategies

Optimizers live under `anfis_toolbox/optim/` and share a common interface:

- `BaseTrainer.fit(model, X, y, **kwargs)` drives epochs, batching, shuffling,
	and validation.
- Hybrid trainers combine gradient descent with ordinary least squares rule
	consequent updates, delivering fast convergence on regression tasks.
- Adam, RMSProp, and SGD offer familiar gradient-based alternatives.
- PSO provides a population-based search when gradient information is noisy.
- Trainers expose hooks for learning rate, epochs, batch size, shuffle, and
	optional loss overrides.

## Metrics and Evaluation

- `ANFISRegressor.evaluate` reports MSE, RMSE, MAE, and R² via
	`metrics.ANFISMetrics.regression_metrics`.
- `ANFISClassifier.evaluate` reports accuracy, balanced accuracy, macro/micro
	precision/recall/F1, and the confusion matrix.
- Metrics are returned as dictionaries to simplify logging or experiment
	tracking.

## Examples and Documentation

- Explore the `docs/examples/` notebooks for step-by-step tutorials covering
	regression, classification, time series, and membership customization.
- The MkDocs site (`mkdocs.yml`) assembles the `docs/` directory into a hosted
	documentation portal. Run `make docs` to build locally and serve via
	`mkdocs serve`.

## Development Workflow

1. **Install dependencies**: `make install`
2. **Run tests**: `make test`
3. **Lint**: `make lint`
4. **Format (optional)**: `make format`
5. **Docs preview**: `make docs`

Tests cover membership functions, optimizers, estimators, and integration with
scikit-learn-like patterns. New features should include corresponding tests.

## Contributing

- Fork the repository and create a feature branch.
- Keep pull requests focused; tie them to an issue when possible.
- Update or add documentation (including this guide) when behavior changes.
- Ensure `make test` and `make lint` pass before submitting.
- Include demo snippets or notebooks if the feature benefits from examples.

Thanks for contributing to ANFIS Toolbox! If you have questions, open a
discussion or issue on GitHub.
