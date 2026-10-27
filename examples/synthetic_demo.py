"""Deterministic demonstration of the cleaned valuation pipeline.

The original project used proprietary / credentialed on-chain and derivatives
series. This example therefore uses synthetic indicator data to demonstrate the
normalization and aggregation mechanics without redistributing restricted data.
It is not a historical BTC valuation result.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.valuation import aggregate_valuation, build_indicator_scores  # noqa: E402


def make_synthetic_data(rows: int = 1300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=rows, freq="D")
    t = np.arange(rows)
    cycle = np.sin(t / 95.0) + 0.45 * np.sin(t / 31.0)
    slow = np.sin(t / 220.0)

    # Deliberately varied distributions so the transform pipeline has something
    # meaningful to normalize. These are illustrative, not calibrated market data.
    data = pd.DataFrame(index=index)
    data["NUPL"] = 1.3 + 0.45 * cycle + rng.lognormal(-2.2, 0.35, rows)
    data["Retail"] = np.exp(0.30 * cycle + rng.normal(0, 0.55, rows))
    data["Sentiment"] = np.exp(2.0 + 0.40 * cycle + rng.normal(0, 0.20, rows))
    data["Market Power"] = 0.8 * cycle + 0.4 * slow + rng.standard_t(5, rows) * 0.35
    data["STH"] = np.exp(0.10 + 0.12 * cycle + rng.normal(0, 0.08, rows))
    oi_level = 100 + np.cumsum(0.08 + 0.35 * cycle + rng.normal(0, 1.0, rows))
    data["Open Interest"] = oi_level - oi_level.min() + 20
    data["SSR"] = 8.0 + 1.1 * cycle + 0.5 * slow + rng.normal(0, 0.30, rows)
    data["Funding"] = 0.00015 * cycle + rng.normal(0, 0.00035, rows)
    return data


def main() -> None:
    data = make_synthetic_data()
    scores = build_indicator_scores(data)
    aggregate = aggregate_valuation(scores)

    output = Path(__file__).resolve().parent / "output"
    output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(aggregate.index, aggregate, linewidth=1.2)
    ax.axhline(0.0, linewidth=0.8)
    ax.axhline(1.0, linewidth=0.8, linestyle="--")
    ax.axhline(-1.0, linewidth=0.8, linestyle="--")
    ax.set_title("Illustrative aggregate valuation score (synthetic inputs)")
    ax.set_ylabel("Average normalized score")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "synthetic_aggregate.svg")
    plt.close(fig)

    visible = scores.dropna(how="all")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    image = ax.imshow(visible.T.to_numpy(), aspect="auto", interpolation="nearest", vmin=-2.5, vmax=2.5)
    ticks = np.linspace(0, len(visible) - 1, 7, dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([visible.index[i].strftime("%Y-%m") for i in ticks])
    ax.set_yticks(range(len(visible.columns)))
    ax.set_yticklabels([name.replace("_", " ") for name in visible.columns])
    ax.set_title("Illustrative normalized component scores (synthetic inputs)")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Normalized score")
    fig.tight_layout()
    fig.savefig(output / "synthetic_components.svg")
    plt.close(fig)

    print(f"Rows: {len(data):,}")
    print(f"Aggregate non-null observations: {aggregate.notna().sum():,}")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
