"""
Responsible for downloading raw OHLCV price data from market APIs (yfinance).

Provides a single entry point to fetch historical price series for a list of
B3 tickers over a configurable date range and cache results locally for
subsequent pipeline runs.
"""
