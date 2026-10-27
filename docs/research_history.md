# Research history

This note records the evolution of the original valuation experiments so the cleaned code is not mistaken for one frozen historical specification.

## Medium-term prototype

The earlier medium-term write-up started from a practical question: could a standardized valuation composite help reduce leverage when Bitcoin looked overbought, or provide context during corrections?

The prototype focused on transforming heterogeneous indicator series into more comparable standardized states. Four recurring data problems were identified:

1. variable volatility;
2. trend;
3. skewness; and
4. noise.

The working responses were moving z-scores, distributional transforms, and smoothing.

### Automatic normalization

The original `normalize` routine was organized into three increasingly expensive search levels:

- Level 1 — moving-z-score window search;
- Level 2 — transformation plus z-score-window search;
- Level 3 — smoothing plus transformation plus z-score-window search.

The candidate transform set included logarithms, roots, powers, Box-Cox, and Yeo-Johnson. The research ranked candidates partly with p-values from normality tests. The cleaned implementation retains this only as a heuristic comparison device; a high normality-test p-value is not evidence that an indicator has economic value.

### Short-Term MVRV/NUPL diagnostic

One saved diagnostic figure used four panels:

- **Indicator data with moving statistics** — the raw indicator together with a moving average and approximate dispersion bands;
- **BTC valuation view** — the standardized valuation state overlaid with Bitcoin price;
- **Raw histogram** — a visibly right-skewed original distribution; and
- **Transformed histogram / probability plot** — the post-transformation distribution and a probability-plot diagnostic.

The purpose of the figure was to show the full transformation workflow rather than a trading backtest.

### Aggregation

After individual indicators were normalized, their z-scores were averaged. The medium-term report's final example listed five component series in the aggregate.

The report also noted an important breadth problem: the earliest dates had only one or two available indicators, so the meaning of the average changed through time.

### Original conclusion

The original write-up did **not** conclude that the system had established alpha. It stated that no consistent actionable insight was visible from the composite alone and suggested that the valuation measure might instead be useful as confluence with the TPI/trend and liquidity work.

That conclusion is intentionally preserved in the public repository.

## Later notebook expansion

A later notebook expanded the valuation research to a broader set of eight on-chain, sentiment, and derivatives series. The cleaned `src/valuation.py` implementation primarily reconstructs that later executable pipeline.

This is why the main README distinguishes between:

- the five-indicator medium-term prototype; and
- the later eight-series notebook.

They are related stages of the same research process, not identical model specifications.

## Why the historical raw data are not included

Some original inputs came from third-party or authenticated analytics sources. The public repository therefore focuses on the transformation and aggregation mechanics and uses deterministic synthetic data for reproducibility rather than redistributing the original datasets or credentials.
