# API Reference

This section documents the public surface of ANFIS Toolbox. Use it alongside the
user guides and examples when you need precise signatures, parameters, and
return types.

## Estimators

- **[`ANFISRegressor`](regressor.md)** – Scikit-learn style interface for
	Takagi–Sugeno–Kang regression.
- **[`ANFISClassifier`](classifier.md)** – Classification counterpart with
	probability predictions and evaluation helpers.

## Membership Functions

Thirteen membership function families covering Gaussian, bell, sigmoidal, and
piecewise-linear shapes are documented in
[`membership-functions.md`](membership-functions.md). Each entry includes
parameters, derivative support, and usage examples.

## Training

- **Optimizers** – Gradient-based, hybrid, and swarm trainers are described in
	[`optim.md`](optim.md) with configuration notes and supported hyper-parameters.
- **Losses** – Regression and classification objectives (and their gradients)
	are listed in [`losses.md`](losses.md).

## Metrics

Evaluation helpers for regression, classification, and clustering are grouped in
[`metrics.md`](metrics.md). Each function documents expected inputs and output
formats so you can integrate metrics into experiments or monitoring.

## Core Internals

- **Models** – Low-level ANFIS graph classes and their rule representations live
	in [`models.md`](models.md).
- **Layers** – Individual computational layers and their tensor operations are
	explained in [`layers.md`](layers.md).

These pages are useful when you need to inspect or extend the internal pipeline
that powers the high-level estimators.

## Utilities

- **Configuration** – Utilities for persisting and replaying setups appear in
	[`config.md`](config.md).
- **Logging** – Structured training logs and logging configuration are covered
	in [`logging.md`](logging.md).

## Advanced Topics

- **Builders** – Advanced model construction hooks are described in
	[`builders.md`](builders.md). Most users can rely on estimator defaults.
- **Clustering** – The fuzzy C-means implementation used for membership
	initialization is detailed in [`clustering.md`](clustering.md).

## Where to Start

- New to ANFIS Toolbox? Begin with the [models overview](../models/anfis.md).
- Looking for ready-to-run notebooks? Browse the Examples section in the
	navigation.
- Exploring code while reading docs? The “View source” actions in each page jump
	straight to the implementation.
