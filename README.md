# Statistical Market Valuation

[![Validate valuation pipeline](https://github.com/soulidisath/statistical-market-valuation/actions/workflows/validate.yml/badge.svg)](https://github.com/soulidisath/statistical-market-valuation/actions/workflows/validate.yml)

Personal research project exploring how heterogeneous Bitcoin on-chain, sentiment, and derivatives indicators can be transformed onto a more comparable statistical scale and combined into a descriptive valuation score.

The repository is a cleaned version of exploratory notebook work. It focuses on **normalization, transformation choice, signal aggregation, and the limitations of those choices**. It is not a production valuation model and it does not claim that the composite predicts returns or identifies repeatable market-timing opportunities.

## Research question

Different market indicators live on very different scales and distributions. The project asked a practical question:

> Can several structurally different Bitcoin indicators be transformed into comparable standardized scores and viewed together as a single descriptive market-state measure?

The original work experimented with rolling z-scores, smoothing, power transforms, Box-Cox, Yeo-Johnson, and rank-based Gaussian normalization before averaging the resulting component scores.

## Research evolution

There were two related stages of the work:

1. **Medium-term prototype** — an earlier write-up built a five-indicator valuation composite and focused heavily on making heterogeneous inputs easier to compare through moving z-scores, transformation, and smoothing.
2. **Later notebook expansion** — the subsequent exploratory notebook broadened the research to eight signal families and added more indicator-specific transformations.

The cleaned repository primarily reconstructs the later executable notebook while retaining the methodology and lessons from the earlier prototype. Keeping those stages separate matters: the five-indicator report and the eight-series notebook should not be read as one single frozen model specification.

## Original research inputs

The later notebook worked with eight signal families:

- short-term holder profit/loss / NUPL-style data;
- retail-interest data;
- a futures sentiment index;
- a market-power indicator;
- short-term-holder SOPR multiples;
- aggregate futures open interest;
- stablecoin supply ratio; and
- futures funding rates.

Some of the original series came from credentialed or third-party analytics sources. Their raw data and credentials are **not redistributed** in this repository. The cleaned code accepts a pandas DataFrame containing the required input series, while the included demo uses deterministic synthetic data purely to show the mechanics.

## What the original medium-term prototype did

The earlier research write-up identified four recurring problems in raw indicator series: **variable volatility, trend, skewness, and noise**. The workflow used:

- moving z-scores to adapt the reference mean and volatility through time;
- logarithms, roots, powers, Box-Cox, and Yeo-Johnson transforms to reduce skewness;
- moving averages to reduce noise; and
- an automatic search routine to compare candidate normalization choices.

The normalization search had three levels:

- **Level 1:** test multiple moving-z-score lengths;
- **Level 2:** test multiple transformations and corresponding z-score lengths;
- **Level 3:** additionally test smoothing lengths.

Candidate transformations were ranked using normality-test p-values. In the cleaned repository this is preserved only as a **heuristic search device**. A z-score itself does not require normally distributed data; normality becomes relevant when translating standardized values into probabilities under a normal model.

The original Short-Term MVRV/NUPL diagnostic illustrated the full idea visually: raw indicator and moving statistics, a strongly skewed raw histogram, the transformed distribution/probability plot, and the resulting standardized state plotted against Bitcoin price.

## What is implemented

### 1. Reusable statistical transforms

`src/normalization.py` contains the small building blocks used throughout the research:

- conditional positive shifting for transforms that require positive inputs;
- square-root, cube-root, logarithmic, Box-Cox, and Yeo-Johnson transforms;
- rolling z-scores;
- rank-based Gaussian normalization;
- full-sample z-scores; and
- simple smoothing.

The original sign convention is retained: many scores are inverted so that a larger positive value corresponds to a relatively more favorable / cheaper reading under the research interpretation.

### 2. Indicator-specific normalization pipeline

`src/valuation.py` reconstructs the executable transformations used in the later notebook:

| Component | Cleaned executable specification |
| --- | --- |
| Short-term profit/loss | square-root transform + 280-day rolling z-score |
| Retail interest | inverted rank-Gaussian normalization |
| Sentiment | log transform + 110-day rolling z-score |
| Market power | 8-day smoothing + Yeo-Johnson + 135-day rolling z-score |
| STH SOPR multiple | 30-day smoothing + Box-Cox + 170-day rolling z-score |
| Open interest | 25-day change, 5-day smoothing, 120-day volatility scaling, inverted rank-Gaussian normalization |
| Stablecoin supply ratio | inverted full-sample z-score |
| Funding | 20-day smoothing, 100-day volatility scaling, inverted rank-Gaussian normalization |

The component scores can then be averaged with `aggregate_valuation()`. The cleaned implementation adds a `min_components` requirement so early dates with very sparse indicator coverage are not silently treated as equivalent to dates with the full indicator set.

### 3. Automatic-normalization search

`src/search.py` retains the notebook's exploratory search idea: compare rolling-z-score windows, transformations, and smoothing choices using an average p-value from Shapiro-Wilk, Kolmogorov-Smirnov, and D'Agostino normality tests.

This is a **heuristic search tool**, not a statistically rigorous model-selection rule. Averaging p-values across different tests has no special theoretical justification; it was used as a practical way to rank candidate transformations during exploration.

## What the original research concluded

The most important result was not a spectacular backtest. The original medium-term write-up explicitly noted that:

- the earliest aggregate values were based on only one or two available indicators and were therefore unreliable;
- the prototype was still primitive;
- no consistent actionable insight was visible from the valuation composite by itself; and
- the measure might be more useful as **confluence** alongside the trend/TPI and liquidity research.

That negative result is retained intentionally. The project is more credible as an example of testing an idea and finding its limits than it would be if the historical chart were presented as evidence of a reliable trading edge.

## Reproducible demonstration

The original raw indicator history cannot be redistributed cleanly, so `examples/synthetic_demo.py` generates deterministic synthetic series with deliberately different distributions and runs them through the same cleaned transformation pipeline.

![Illustrative aggregate valuation score](examples/output/synthetic_aggregate.svg)

![Illustrative normalized components](examples/output/synthetic_components.svg)

These figures demonstrate code behavior only. They are **not historical Bitcoin valuation results**.

Run the example from the repository root:

```bash
python examples/synthetic_demo.py
```

## Important notebook discrepancy

One markdown note in the later notebook described the STH SOPR transformation as a **-2 power transform with a 465-day moving z-score**. The executable cell instead applied a **30-day rolling mean, Box-Cox transform, and 170-day moving z-score**.

The cleaned repository follows the executable code and documents the disagreement rather than silently choosing whichever description sounds cleaner.

## Important limitations

- Several transformations were selected by inspecting the same historical data they were later visualized on, so the choices are vulnerable to in-sample overfitting.
- Rank-Gaussian normalization and the full-sample SSR z-score, as originally implemented, use the **entire available sample**. They therefore contain look-ahead information if interpreted as historical walk-forward signals.
- The indicator set has unequal history. A simple average can change meaning as new components become available; the cleaned code makes breadth explicit but does not solve that structural issue.
- A more Gaussian-looking indicator is not automatically a better valuation signal.
- Normality-test p-values are not a measure of economic usefulness or predictive power.
- The aggregate score was exploratory and was not established as a robust forecasting or trading model.
- Third-party data definitions can change and may require licensed or authenticated access.

These limitations are central to the project: the useful output is the transformation and aggregation framework, together with a clearer understanding of where a visually appealing historical valuation score can become statistically misleading.

## Repository structure

```text
.
├── .github/
│   └── workflows/
│       └── validate.yml
├── README.md
├── requirements.txt
├── docs/
│   └── research_history.md
├── examples/
│   ├── README.md
│   ├── synthetic_demo.py
│   └── output/
│       ├── synthetic_aggregate.svg
│       └── synthetic_components.svg
└── src/
    ├── __init__.py
    ├── normalization.py
    ├── search.py
    └── valuation.py
```

## Running the code

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python examples/synthetic_demo.py
```

For real research data, pass a DataFrame containing the eight expected input columns to `build_indicator_scores()` and keep the data-source licensing / authentication outside the research module.
