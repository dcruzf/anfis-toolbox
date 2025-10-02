# Losses API

::: anfis_toolbox.losses

This module provides loss functions and their gradients used during ANFIS model training.

## Regression Losses

Functions for regression tasks (continuous output prediction).

- [`mse_loss()`][anfis_toolbox.losses.mse_loss] - Mean squared error loss
- [`mse_grad()`][anfis_toolbox.losses.mse_grad] - Gradient of MSE loss

## Classification Losses

Functions for classification tasks (discrete output prediction).

- [`cross_entropy_loss()`][anfis_toolbox.losses.cross_entropy_loss] - Cross-entropy loss
- [`cross_entropy_grad()`][anfis_toolbox.losses.cross_entropy_grad] - Gradient of cross-entropy loss
