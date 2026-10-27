"""Statistical transformations used in the valuation research.

The original notebook explored power transforms, rolling z-scores, smoothing,
and rank-based Gaussian normalization. This module keeps those operations small,
explicit, and reusable.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

TransformName = Literal["identity", "sqrt", "cbrt", "log", "boxcox", "yeojohnson"]


def shift_positive(data: pd.Series) -> pd.Series:
    """Shift a series only when needed so strictly-positive transforms are valid."""
    series = data.astype(float).copy()
    minimum = float(series.min())
    if minimum <= 0:
        series = series + (1.0 - minimum)
    return series


def smooth(data: pd.Series, length: int) -> pd.Series:
    """Simple rolling mean used in several original indicator transforms."""
    return data.astype(float).rolling(length, min_periods=1).mean()


def rolling_zscore(data: pd.Series, length: int, invert: bool = True) -> pd.Series:
    """Rolling z-score.

    The original valuation notebook inverted its z-scores so that larger positive
    values represented relatively cheaper/more favorable conditions. ``invert``
    preserves that convention by default.
    """
    series = data.astype(float)
    mean = series.rolling(length, min_periods=1).mean()
    std = series.rolling(length, min_periods=2).std().replace(0.0, np.nan)
    score = (series - mean) / std
    return -score if invert else score


def gaussian_rank(data: pd.Series, invert: bool = False) -> pd.Series:
    """Map empirical ranks to a standard-normal scale with an inverse CDF."""
    series = data.astype(float).dropna()
    ranks = series.rank(method="average")
    scaled = ranks / (len(series) + 1.0)
    transformed = pd.Series(stats.norm.ppf(scaled), index=series.index, name=series.name)
    return -transformed if invert else transformed


def static_zscore(data: pd.Series, invert: bool = True) -> pd.Series:
    """Full-sample z-score retained for the SSR experiment."""
    series = shift_positive(data).dropna()
    values = pd.Series(stats.zscore(series), index=series.index, name=series.name)
    return -values if invert else values


def transform(data: pd.Series, method: TransformName) -> pd.Series:
    """Apply one of the transformations explored in the notebook."""
    series = data.astype(float).dropna()
    if method == "identity":
        return series
    if method == "sqrt":
        return np.sqrt(shift_positive(series))
    if method == "cbrt":
        return np.cbrt(shift_positive(series))
    if method == "log":
        return np.log(shift_positive(series))
    if method == "boxcox":
        positive = shift_positive(series)
        return pd.Series(stats.boxcox(positive)[0], index=series.index, name=series.name)
    if method == "yeojohnson":
        return pd.Series(stats.yeojohnson(series)[0], index=series.index, name=series.name)
    raise ValueError(f"Unsupported transform: {method}")


def average_normality_pvalue(data: pd.Series) -> float:
    """Average p-value of the three tests used in the exploratory notebook.

    This is retained as a search heuristic, not as a formal model-selection
    criterion. P-values from different tests are not generally interchangeable.
    """
    values = data.dropna().astype(float).to_numpy()
    if len(values) < 20:
        return float("nan")
    shapiro = stats.shapiro(values).pvalue
    ks = stats.kstest(values, "norm").pvalue
    dagostino = stats.normaltest(values).pvalue
    return float(np.mean([shapiro, ks, dagostino]))
