"""
Responsible for blending ML model predictions with historical mean returns.

Applies the formula: final_mu = alpha * ml_prediction + (1 - alpha) * historical_mean
where alpha is read from config. Prevents extreme predictions from dominating optimization.
"""
