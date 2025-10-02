# Metrics API

::: anfis_toolbox.metrics

This module provides comprehensive metrics for evaluating ANFIS models across regression, classification, and clustering tasks.

## Regression Metrics

Functions for evaluating regression model performance:

- [`mean_squared_error()`][anfis_toolbox.metrics.mean_squared_error] - Mean squared error
- [`mean_absolute_error()`][anfis_toolbox.metrics.mean_absolute_error] - Mean absolute error
- [`root_mean_squared_error()`][anfis_toolbox.metrics.root_mean_squared_error] - Root mean squared error
- [`mean_absolute_percentage_error()`][anfis_toolbox.metrics.mean_absolute_percentage_error] - Mean absolute percentage error
- [`symmetric_mean_absolute_percentage_error()`][anfis_toolbox.metrics.symmetric_mean_absolute_percentage_error] - Symmetric MAPE
- [`r2_score()`][anfis_toolbox.metrics.r2_score] - Coefficient of determination
- [`pearson_correlation()`][anfis_toolbox.metrics.pearson_correlation] - Pearson correlation coefficient
- [`mean_squared_logarithmic_error()`][anfis_toolbox.metrics.mean_squared_logarithmic_error] - Mean squared logarithmic error

## Classification Metrics

Functions for evaluating classification model performance:

- [`softmax()`][anfis_toolbox.metrics.softmax] - Numerically stable softmax
- [`cross_entropy()`][anfis_toolbox.metrics.cross_entropy] - Cross-entropy loss
- [`log_loss()`][anfis_toolbox.metrics.log_loss] - Log loss
- [`accuracy()`][anfis_toolbox.metrics.accuracy] - Classification accuracy

## Clustering Validation

Functions for evaluating fuzzy clustering quality:

- [`partition_coefficient()`][anfis_toolbox.metrics.partition_coefficient] - Bezdek's partition coefficient
- [`classification_entropy()`][anfis_toolbox.metrics.classification_entropy] - Classification entropy
- [`xie_beni_index()`][anfis_toolbox.metrics.xie_beni_index] - Xie-Beni validity index
