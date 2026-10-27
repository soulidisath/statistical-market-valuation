# Examples

`synthetic_demo.py` is a deterministic smoke-test and visualization of the cleaned normalization pipeline.

The original notebook used credentialed / third-party on-chain and derivatives data. Those raw series are not redistributed here. The example therefore generates synthetic inputs with varied distributions and applies the same transformation structure used in the research code.

Run from the repository root:

```bash
python examples/synthetic_demo.py
```

It creates:

- `output/synthetic_aggregate.svg` — the average normalized valuation score; and
- `output/synthetic_components.svg` — the individual transformed component scores.

These figures demonstrate code behavior only. They are **not historical Bitcoin valuation results** and should not be interpreted as a performance or market-timing claim.
