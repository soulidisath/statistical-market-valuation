"""Indicator normalization and aggregation for the BTC valuation research."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .normalization import gaussian_rank, rolling_zscore, smooth, static_zscore, transform


REQUIRED_COLUMNS = {
    "NUPL",
    "Retail",
    "Sentiment",
    "Market Power",
    "STH",
    "Open Interest",
    "SSR",
    "Funding",
}


def short_term_profit_loss(series: pd.Series) -> pd.Series:
    """Executable notebook specification: sqrt transform + 280-day rolling z-score."""
    return rolling_zscore(transform(series.dropna(), "sqrt"), 280)


def retail_interest(series: pd.Series) -> pd.Series:
    """Rank-Gaussian normalization with the original sign convention."""
    return gaussian_rank(series, invert=True)


def sentiment_index(series: pd.Series) -> pd.Series:
    return rolling_zscore(transform(series.dropna(), "log"), 110)


def market_power(series: pd.Series) -> pd.Series:
    smoothed = smooth(series, 8).dropna()
    return rolling_zscore(transform(smoothed, "yeojohnson"), 135)


def sth_sopr_multiple(series: pd.Series) -> pd.Series:
    """Follow the notebook's executable cell: 30-day mean, Box-Cox, 170-day z-score.

    The notebook markdown described a different -2 power / 465-day specification.
    The cleaned implementation intentionally follows the code that actually ran.
    """
    smoothed = series.astype(float).rolling(30).mean().dropna()
    return rolling_zscore(transform(smoothed, "boxcox"), 170)


def open_interest(series: pd.Series) -> pd.Series:
    changed = series.astype(float).diff(25).rolling(5).mean().dropna()
    scale = changed.rolling(120).std().replace(0.0, np.nan)
    normalized = (changed / scale).dropna()
    return gaussian_rank(normalized, invert=True)


def stablecoin_supply_ratio(series: pd.Series) -> pd.Series:
    return static_zscore(series, invert=True)


def funding_rate(series: pd.Series) -> pd.Series:
    smoothed = series.astype(float).rolling(20).mean()
    scale = smoothed.rolling(100, min_periods=50).std().replace(0.0, np.nan)
    normalized = (smoothed / scale).dropna()
    return gaussian_rank(normalized, invert=True)


def build_indicator_scores(data: pd.DataFrame) -> pd.DataFrame:
    """Transform the eight research inputs into comparable valuation scores."""
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    scores = {
        "short_term_profit_loss": short_term_profit_loss(data["NUPL"]),
        "retail_interest": retail_interest(data["Retail"]),
        "sentiment": sentiment_index(data["Sentiment"]),
        "market_power": market_power(data["Market Power"]),
        "sth_sopr": sth_sopr_multiple(data["STH"]),
        "open_interest": open_interest(data["Open Interest"]),
        "stablecoin_supply_ratio": stablecoin_supply_ratio(data["SSR"]),
        "funding": funding_rate(data["Funding"]),
    }
    return pd.DataFrame(scores).sort_index()


def aggregate_valuation(scores: pd.DataFrame, min_components: int = 4) -> pd.Series:
    """Average available component scores, requiring a minimum breadth.

    The original notebook averaged whichever indicators were available. Requiring
    ``min_components`` makes the varying early-history composition visible rather
    than silently treating a one- or two-signal mean as comparable to the full set.
    """
    count = scores.notna().sum(axis=1)
    aggregate = scores.mean(axis=1, skipna=True)
    aggregate[count < min_components] = np.nan
    aggregate.name = "valuation_score"
    return aggregate
