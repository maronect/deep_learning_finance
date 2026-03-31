"""
Defines the abstract base class and common interface for all return prediction models.

Every model (Ridge, MLP, RNN) must implement this interface so the pipeline
can swap models without changing orchestration logic.
"""
