"""Exploratory search helpers from the notebook's automatic-normalization section."""
from __future__ import annotations

import pandas as pd

from .normalization import average_normality_pvalue, rolling_zscore, smooth, transform


def search_zscore_lengths(
    data: pd.Series,
    lengths: range = range(50, 500, 5),
) -> pd.DataFrame:
    """Rank rolling-z-score windows by the notebook's average normality p-value."""
    rows: list[dict[str, float | int]] = []
    for length in lengths:
        score = rolling_zscore(data, length).dropna()
        rows.append({"zscore_length": length, "normality_pvalue": average_normality_pvalue(score)})
    return pd.DataFrame(rows).sort_values("normality_pvalue", ascending=False, ignore_index=True)


def search_transformations(
    data: pd.Series,
    lengths: range = range(50, 500, 5),
) -> pd.DataFrame:
    """Compare the main transformations used in the original exploration."""
    frames = []
    for name in ("log", "sqrt", "cbrt", "boxcox", "yeojohnson"):
        transformed = transform(data.dropna(), name)
        result = search_zscore_lengths(transformed, lengths=lengths)
        result["transformation"] = name
        frames.append(result)
    return pd.concat(frames, ignore_index=True).sort_values(
        "normality_pvalue", ascending=False, ignore_index=True
    )


def search_smoothing_and_transformations(
    data: pd.Series,
    smoothing_lengths: tuple[int, ...] = (0, 5, 10, 15, 20, 25, 30),
    zscore_lengths: range = range(50, 500, 5),
) -> pd.DataFrame:
    """Search smoothing, transform, and rolling-window combinations.

    This mirrors the structure of the notebook's most expensive normalization
    search. It is exploratory: selecting the highest in-sample normality score is
    not evidence that a transform is economically useful or stable out of sample.
    """
    frames = []
    for smoothing in smoothing_lengths:
        candidate = data.dropna() if smoothing == 0 else smooth(data, smoothing).dropna()
        result = search_transformations(candidate, lengths=zscore_lengths)
        result["smoothing"] = smoothing
        frames.append(result)
    return pd.concat(frames, ignore_index=True).sort_values(
        "normality_pvalue", ascending=False, ignore_index=True
    )
