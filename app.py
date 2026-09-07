"""
app.py -- the explorer WIDGET (marimo reactive notebook).

For students who want to *play* with the models, not read code. Everything
updates live as you change a control. The math is imported from replicator.py,
so this file only wires up controls to pictures.

Editing note: this is a marimo notebook. Each `@app.cell` function is a reactive
cell -- marimo re-runs a cell automatically whenever a value it reads (a slider,
dropdown, or another cell's output) changes. A cell's LAST expression is what it
displays. So to change the UI you edit these cells; there is no manual "refresh".

Run it locally:
    pip install marimo matplotlib numpy
    marimo run app.py            # read-only app mode (what students use)
    marimo edit app.py           # editable mode

Share with no install:
    marimo export html-wasm app.py -o build/ --mode run
    ...then host the build/ folder anywhere static (GitHub Pages, etc.). The whole
    thing then runs in the browser -- nothing for students to install.
"""
import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    # Loading the model modules, in the browser and out of it.
    #
    # Three things go wrong if you just write `import replicator` here, and all
    # three are silent -- the page simply stays blank:
    #
    #   1. A WASM export bundles the NOTEBOOK, not the files it imports, so
    #      replicator.py and portraits.py are not in the build at all. They have
    #      to be copied next to index.html at deploy time (the README and the
    #      Pages workflow both do this) and then fetched into Pyodide's
    #      filesystem.
    #   2. The fetch must use the ABSOLUTE url from mo.notebook_location(). A
    #      relative "./replicator.py" resolves against marimo's virtual
    #      filesystem and quietly returns a few hundred bytes of something else,
    #      which then fails to parse as Python.
    #   3. marimo's WASM loader scans this file for imports and installs them
    #      from PyPI. A bare `import replicator` makes it hunt for a package
    #      called "replicator" and fail. Hence importlib, which it cannot see --
    #      and hence the otherwise pointless `import matplotlib` below, which is
    #      how matplotlib gets installed at all, since the only real import of it
    #      is inside portraits.py where the scanner never looks.
    #
    # Outside the browser the emscripten branch is skipped and this is an
    # ordinary import.
    import importlib
    import sys

    import matplotlib  # noqa: F401  (declares the dependency for the WASM build)
    import numpy as np

    import marimo as mo

    if sys.platform == "emscripten":
        import pathlib

        import pyodide.http

        _base = str(mo.notebook_location()).rstrip("/")
        for _name in ("replicator.py", "portraits.py"):
            pathlib.Path(_name).write_text(
                pyodide.http.open_url(f"{_base}/{_name}").getvalue())
        if "." not in sys.path:
            sys.path.insert(0, ".")

    rp = importlib.import_module("replicator")
    P = importlib.import_module("portraits")
    return mo, np, rp, P


@app.cell
def _(mo):
    mode = mo.ui.radio(
        options=["2 strategies", "3 strategies", "Two populations"],
        value="2 strategies",
        label="Model",
        inline=True,
    )
    mode
    return (mode,)


@app.cell
def _(mode, rp):
    # choose the preset menu that matches the mode
    if mode.value == "2 strategies":
        presets = rp.GAMES_2x2
    elif mode.value == "3 strategies":
        presets = rp.GAMES_3x3
    else:
        presets = rp.GAMES_BIMATRIX
    return (presets,)


@app.cell
def _(mo, presets):
    preset = mo.ui.dropdown(
        options=list(presets.keys()),
        value=list(presets.keys())[0],
        label="Preset game",
    )
    preset
    return (preset,)


@app.cell
def _(mo, mode, np, preset, presets):
    # build editable payoff-matrix input(s), seeded from the chosen preset
    def make_grid(M):
        M = np.asarray(M, float)
        n = len(M)
        cells = [mo.ui.number(value=float(M[i][j]), step=0.5)
                 for i in range(n) for j in range(n)]
        return mo.ui.array(cells), n

    if mode.value == "Two populations":
        A0, B0 = presets[preset.value]
        gridA, n = make_grid(A0)
        gridB, _ = make_grid(B0)
    else:
        gridA, n = make_grid(presets[preset.value])
        gridB = None
    return gridA, gridB, n


@app.cell
def _(gridA, gridB, mo, mode, n):
    def show_grid(grid, title):
        rows = [mo.hstack([grid[i * n + j] for j in range(n)], justify="start")
                for i in range(n)]
        return mo.vstack([mo.md(f"**{title}**"), *rows])

    if mode.value == "Two populations":
        matrix_ui = mo.hstack([show_grid(gridA, "Population 1 payoffs (A)"),
                               show_grid(gridB, "Population 2 payoffs (B)")],
                              justify="start", gap=2)
    else:
        matrix_ui = show_grid(gridA, "Payoff matrix (row = focal strategy)")
    matrix_ui
    return


@app.cell
def _(mo):
    # controls (defined once; displayed conditionally in the next cell)
    r = mo.ui.slider(-0.9, 0.95, value=0.0, step=0.05, label="Assortment  r",
                     show_value=True)
    # dynamic selector, grouped by FAMILY (the grouping is itself the lesson)
    dyn_label = mo.ui.dropdown(
        options={
            "Replicator - continuous (imitative)": "replicator",
            "Replicator - discrete time (imitative)": "discrete",
            "Brown-von Neumann-Nash (innovative)": "bnn",
            "Logit best-response (perturbed best response)": "logit",
            "Replicator-mutator - uniform mutation (selection + mutation)": "replmut",
            "Replicator-mutator - fitness-weighted (selection + mutation)": "replmut_fw",
        },
        value="Replicator - continuous (imitative)", label="Dynamic")
    beta = mo.ui.slider(0.5, 30.0, value=5.0, step=0.5,
                        label="Logit rationality  beta", show_value=True)
    mu = mo.ui.slider(0.0, 0.3, value=0.05, step=0.01,
                      label="Mutation rate  mu", show_value=True)
    samples = mo.ui.slider(200, 3000, value=1000, step=100,
                           label="Basin samples (Monte Carlo)", show_value=True)
    compute = mo.ui.run_button(label="Estimate basins (Monte-Carlo)")
    return r, dyn_label, beta, mu, samples, compute


@app.cell
def _(mo, mode, r, dyn_label, beta, mu, samples, compute):
    extra = (beta if dyn_label.value == "logit"
             else mu if dyn_label.value in ("replmut", "replmut_fw") else None)
    items = [dyn_label]
    if mode.value != "Two populations":
        items.append(r)                       # assortment: single-population only
    if extra is not None:
        items.append(extra)                   # beta / mu apply to all modes
    if mode.value != "Two populations":
        items += [samples, compute]           # basin Monte-Carlo: single-population only
    controls = mo.vstack(items)
    controls
    return


@app.cell
def _(dyn_label, beta, mu, rp):
    # resolve the selected dynamic + its parameters (applies to every mode)
    dyn = rp.DYNAMICS.get(dyn_label.value, rp.DYNAMICS["replicator"])
    params = ({"beta": beta.value} if dyn_label.value == "logit"
              else {"mu": mu.value} if dyn_label.value in ("replmut", "replmut_fw") else {})
    return dyn, params


@app.cell
def _(gridA, gridB, mode, n, np):
    # read the live matrices back out of the grid controls
    A = np.array(gridA.value, float).reshape(n, n)
    B = (np.array(gridB.value, float).reshape(n, n)
         if (mode.value == "Two populations" and gridB is not None) else None)
    return A, B


@app.cell
def _(A, B, P, dyn, mode, params, r):
    # draw the phase portrait for the current model + dynamic
    if mode.value == "2 strategies":
        fig = P.portrait_1d(A, r=r.value, dyn=dyn, **params)
    elif mode.value == "3 strategies":
        fig = P.portrait_simplex(A, r=r.value, dyn=dyn, **params)
    else:
        fig = P.portrait_square(A, B, dyn=dyn, **params)
    fig  # last expression is what marimo renders
    return


@app.cell
def _(A, B, compute, dyn, mo, mode, np, params, r, rp, samples):
    def _pt_label(x):
        active = [i for i, v in enumerate(x) if v > 1e-3]
        if len(active) == 1:
            return f"all S{active[0] + 1}"
        return "mix (" + ", ".join(f"{v:.2f}" for v in x) + ")"

    if mode.value == "Two populations":
        rows = [{"fixed point": f"({z[0]:.2f}, {z[1]:.2f})", "stability": t}
                for z, t in rp.bimatrix_rest_points(dyn, A, B, **params)]
        report = mo.vstack([mo.md("**Fixed points on the square**"),
                            mo.ui.table(rows, selection=None)])
    else:
        # manifold-aware: rest objects can be isolated points OR equilibrium sets
        objs = rp.rest_objects(dyn, A, r=r.value, **params)

        def describe(o):
            if o["kind"] == "set":
                word = {"set-attract": "attracting", "set-repel": "repelling",
                        "set-mixed": "mixed", "set-neutral": "neutral"}[o["type"]]
                return f"line/region of rest points ({word})", o["type"].replace("set-", "set: ")
            return _pt_label(o["x"]), o["type"]

        base_rows = [dict(zip(("rest point", "stability"), describe(o))) for o in objs]
        # rest points are cheap and shown live; basins wait for the button
        mo.stop(not compute.value, mo.vstack([
            mo.ui.table(base_rows, selection=None),
            mo.md("Press **Estimate basins (Monte-Carlo)** to size the basins.")]))

        est = rp.estimate_basins_dyn(dyn, A, r=r.value, n_samples=int(samples.value),
                                     seed=0, **params)
        frac = {tuple(np.round(a["state"], 3)): a for a in est["attractors"]}
        rows = []
        for o in objs:
            name, stab = describe(o)
            a = frac.get(tuple(np.round(o["x"], 3)))
            basin = (f"{100*a['fraction']:.1f}%  (95% CI "
                     f"{100*a['ci_low']:.0f}-{100*a['ci_high']:.0f}%)" if a else "-")
            rows.append({"rest point": name, "stability": stab, "basin size": basin})
        report = mo.ui.table(rows, selection=None)
    report
    return


if __name__ == "__main__":
    app.run()
