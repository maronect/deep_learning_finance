"""
Responsible for constructing supervised learning feature matrices from return time series.

Generates lag features (t-1 … t-lag_window) for each asset, aligns targets,
and splits data into train/test windows for walk-forward validation.
"""
