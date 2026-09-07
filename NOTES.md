# Replicator Dynamics Explorer (PHIL 2001)

A phase-portrait tool for the replicator dynamics, built as a teaching resource for
*Ethics and Evolutionary Games*. Two front-ends, **one shared model core**, so the
mathematics is defined in exactly one place.

## For students (nothing to install)

- **Explore the models:** open the widget in your browser → **https://OWNER.github.io/REPO/**
- **Read / edit the code:** open the notebook in Colab →
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OWNER/REPO/blob/main/notebook.ipynb)

*(Instructor: replace `OWNER`/`REPO` throughout this repo with your GitHub username and
repository name — one command in the "Deploying" section below.)*

## The two tracks

| For students who... | Use | File |
|---|---|---|
| just want to **explore** games with sliders | the marimo widget | `app.py` |
| want to **read and edit** the model | the Colab notebook | `notebook.ipynb` |

Both import the same core:

- **`replicator.py`** — the model. Continuous-time replicator dynamics, rest-point
  finding, Jacobian-eigenvalue stability, Monte-Carlo basins, correlation/assortment,
  and two-population dynamics. Heavily commented; this is the file to read first.
- **`portraits.py`** — the drawing layer (Matplotlib). No math here.
- **`tests.py`** — correctness checks against known games. Run `python3 tests.py`.

## What it does

- **2-strategy games**: phase line + speed profile (`ẋ` vs `x`), with rest points and
  stability read off the curve.
- **3-strategy games**: streamlines on the simplex, coloured by destination basin.
- **Two-population 2×2 games**: vector field + trajectories on the unit square
  (asymmetric games — Battle of the Sexes, Matching Pennies, Owner–Intruder).
- **Correlation / assortment** (`r`): positive `r` = like-meets-like (the altruism lever,
  `r > c/b`); negative `r` = anti-assortment (the mirror lever behind spite).
- **Basins of attraction** by Monte Carlo, reported with 95% confidence intervals.
- **Alternative dynamics** (single-population), selectable and grouped by family — the
  grouping is itself a lesson (a dynamic encodes a behavioural assumption):
  - *imitative*: replicator (continuous) and discrete-time replicator;
  - *innovative*: Brown–von Neumann–Nash (rest points = Nash; extinct strategies can return);
  - *perturbed best-response*: logit (rationality knob `β`, rest points → Nash as `β→∞`);
  - *selection + mutation*: replicator–mutator (mutation rate `μ`, rest points interior).

  Assortment `r` composes with all five. Comparing dynamics is the point: a conclusion that
  survives a change of dynamic is robust; one that does not is an artefact of the dynamic.
  The **two-population** setting supports all five dynamics too, with fixed-point markers on
  the square.
- **Equilibrium manifolds.** When a game is degenerate (a whole line/region of rest points,
  e.g. two interchangeable strategies), the tool represents it as an attracting/repelling
  **set** classified by its *transverse* stability, draws it on the portrait, and counts it
  as a single attractor for basins — rather than mislabelling scattered points.

## Running it

```bash
pip install numpy matplotlib marimo

python3 tests.py            # verify the model (should end "ALL TESTS PASSED")
marimo run app.py           # the student widget
marimo edit app.py          # edit the widget
jupyter notebook notebook.ipynb   # the coder walkthrough (or open in Colab)
```

## Deploying (one-time, instructor)

Hosting is automated: a GitHub Actions workflow (`.github/workflows/deploy.yml`) builds the
WASM widget and publishes it to GitHub Pages on every push. To set it up:

```bash
# 1. From this folder, fill in your GitHub username + repo name everywhere:
OWNER=your-github-username   REPO=replicator-dynamics-phil2001
grep -rl 'OWNER/REPO\|OWNER.github.io' . --include='*.md' --include='*.ipynb' \
  | xargs sed -i '' "s|OWNER/REPO|$OWNER/$REPO|g; s|OWNER.github.io/REPO|$OWNER.github.io/$REPO|g"

# 2. Install + authenticate the GitHub CLI (one-time browser login):
brew install gh
gh auth login

# 3. Create the public repo and push (this folder becomes the repo root):
git add -A && git commit -m "Replicator dynamics teaching tool"
gh repo create "$REPO" --public --source=. --push

# 4. Turn on Pages via Actions, then trigger the first deploy:
gh api -X POST "repos/$OWNER/$REPO/pages" -f build_type=workflow || true
git commit --allow-empty -m "trigger pages" && git push
```

The widget then lives at `https://OWNER.github.io/REPO/` and the notebook opens via the
Colab badge at the top. Every later `git push` rebuilds and redeploys automatically.

Preview the exact hosted page locally before publishing:

```bash
python3 -m marimo export html-wasm app.py -o build --mode run
cp replicator.py portraits.py build/          # REQUIRED -- see below
python3 -m http.server --directory build      # then open the printed URL
```

### ⚠ The one non-obvious deployment step

**`marimo export html-wasm` bundles the notebook, not the files it imports.** `replicator.py` and
`portraits.py` are *not* copied into the build, so in the browser `import replicator` fails, and the
failure is silent — you get a blank page and nothing in the console but a `ModuleNotFoundError` from
the import cell. The two files must be copied next to `index.html`, which the workflow and the preview
command above both do.

`app.py` also has to fetch them at runtime using the **absolute** URL from `mo.notebook_location()`;
a relative path resolves against marimo's virtual filesystem and quietly returns a few hundred bytes
of something else. And it loads them through `importlib` rather than a plain `import`, because
marimo's WASM loader scans the source for imports and tries to install anything it finds from PyPI —
a bare `import replicator` sends it hunting for a package by that name. The same scan is why `app.py`
contains an otherwise pointless `import matplotlib`: that is the only way matplotlib gets installed,
since the real import of it lives in `portraits.py` where the scanner never looks.

*(Found 2026-08-08. This repo had all four faults and would have deployed a blank page; the fix was
ported from `spatial-tool/`, where it is verified working.)*

## Design decisions (and why)

These differ deliberately from the original React prototype (`replicator-dynamics.jsx`,
kept for reference):

1. **Continuous time, not the discrete map.** The discrete multiplicative map
   `x' = x·f/f̄` is *not* invariant to adding a constant to all payoffs (the constant
   acts as background fitness), so "the same game" rendered differently depending on how
   the numbers were written. Worse, standard Rock-Paper-Scissors spiralled *out* to the
   boundary — the opposite of the textbook closed orbits. Continuous time fixes both:
   additive constants cancel, and RPS is correctly a neutral **center**.
2. **Jacobian eigenvalues for stability**, not perturb-and-threshold heuristics. It is
   rigorous and it teaches the linearization idea directly.
3. **One model core in Python.** The original mixed the 5-line model into ~800 lines of
   React + hand-rolled SVG, which students could not edit. Here the model is a small,
   commented module and the front-ends are thin.
4. **Monte-Carlo basins, not a closed form.** A basin boundary is the separatrix (the
   saddle's stable manifold); for a general game that curve has no closed form, so exact
   basins in a formula are impossible in general — a fact about nonlinear ODEs, not a gap.
   Monte Carlo with a confidence interval is the principled estimate.

## Provenance

Grew out of a React prototype dropped in the inbox (2026-07). Rebuilt in Python at Rory's
request so students can both use it (marimo) and edit it (notebook), and to fix the
discrete-time correctness issues above.
