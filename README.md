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
- Five dynamics: replicator (continuous / discrete), BNN, logit best-response, replicator–mutator.

## Run locally

```bash
pip install numpy matplotlib marimo
python3 tests.py        # verify the model
marimo run app.py       # the widget
```

## Files

`replicator.py` (model) · `portraits.py` (plots) · `app.py` (widget) · `notebook.ipynb` (Colab) · `tests.py` (checks).

Build record, design rationale, and deployment notes: [`NOTES.md`](NOTES.md).
