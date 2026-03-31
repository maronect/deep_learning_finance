"""
Multi-Layer Perceptron model for predicting expected asset returns.

Wraps scikit-learn's MLPRegressor with the project's base interface, applying
walk-forward validation and reading hyperparameters from config/models.yaml.
"""
