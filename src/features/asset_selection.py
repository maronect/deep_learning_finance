"""
Responsible for selecting a subset of uncorrelated assets from the full ticker universe.

Implements four selection strategies: correlation threshold, stable pairs,
clustering, and variance-based filtering. Chosen strategy is read from config.
"""
