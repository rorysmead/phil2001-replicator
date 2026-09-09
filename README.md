# Replicator Dynamics Explorer

Phase-portrait tool for the replicator dynamics — a teaching resource for PHIL 2001, *Ethics and Evolutionary Games*.

## Use it (nothing to install)

- **Explore with sliders:** open the widget → **https://rorysmead.github.io/phil2001-replicator/**
- **Read / edit the code:** open in Colab →
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rorysmead/phil2001-replicator/blob/main/notebook.ipynb)

## What it does

- 2-strategy (phase line), 3-strategy (simplex), and two-population 2×2 (unit square) games.
- Assortment slider `r`: positive = altruism lever, negative = spite mirror.
- Basins of attraction by Monte Carlo with 95% CIs.
- Six dynamics: replicator (continuous / discrete), BNN, logit best-response, and replicator–mutator in two forms (uniform mutation and the standard fitness-weighted form).

## Run locally

```bash
pip install numpy matplotlib marimo
python3 tests.py        # verify the model
marimo run app.py       # the widget
```

## Files

`replicator.py` (model) · `portraits.py` (plots) · `app.py` (widget) · `notebook.ipynb` (Colab) · `tests.py` (checks).

Build record, design rationale, and deployment notes: [`NOTES.md`](NOTES.md).


## License

MIT (c) 2026 Rory Smead and contributors. See [LICENSE](LICENSE). Contributions are welcome under the same license, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you use this tool in teaching or research, please cite it. See [CITATION.cff](CITATION.cff), or use the "Cite this repository" button on GitHub.
