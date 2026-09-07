"""
portraits.py -- the drawing layer.

Every function here takes a payoff matrix (and maybe an assortment r) and returns
a Matplotlib figure. The MATH is not here -- it all comes from replicator.py.
This file only decides how to put the dynamics on the page. Both the notebook and
the marimo widget call these, so a game looks the same everywhere.

Three views:
    portrait_1d       2-strategy game: the phase LINE plus a speed profile.
    portrait_simplex  3-strategy game: streamlines on the triangle.
    portrait_square   two-population 2x2 game: a vector field on the unit square.
"""
import numpy as np
import matplotlib.pyplot as plt
import replicator as rp

# a small, colour-blind-safe palette for attractors/basins
BASIN_COLOURS = ["#4C78A8", "#F58518", "#54A24B", "#E45756",
                 "#72B7B2", "#EECA3B", "#B279A2"]


def _dot(ax, xy, kind, colour, size=90):
    """Draw a rest-point marker: filled=attracting, open=repelling,
    half=saddle, ringed=center(neutral)."""
    x, y = xy
    if kind == "stable":
        ax.scatter([x], [y], s=size, facecolor=colour, edgecolor="k", zorder=5)
    elif kind == "unstable":
        ax.scatter([x], [y], s=size, facecolor="white", edgecolor=colour,
                   linewidths=2, zorder=5)
    elif kind == "saddle":
        ax.scatter([x], [y], s=size, facecolor=colour, edgecolor="k",
                   marker="D", zorder=5)
    else:  # center
        ax.scatter([x], [y], s=size, facecolor="none", edgecolor=colour,
                   linewidths=2, zorder=5)
        ax.scatter([x], [y], s=size * 3, facecolor="none", edgecolor=colour,
                   linewidths=1, linestyle=":", zorder=5)


def _draw_set(ax, obj, projected):
    """Draw an equilibrium SET (a line/region of rest points) as a thick band.
    Solid = attracting transversally, open/dashed = repelling, orange = mixed."""
    xy = np.array(projected)
    colour = {"set-attract": "#333", "set-repel": "#333",
              "set-mixed": "#F58518", "set-neutral": "#64748b"}[obj["type"]]
    filled = obj["type"] not in ("set-repel", "set-neutral")
    # order points along the set's long axis so the line is drawn cleanly
    c = xy.mean(0)
    ang = np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0])
    order = np.argsort(ang) if obj["type"] == "set-mixed" else np.argsort(xy[:, 0] + xy[:, 1])
    xy = xy[order]
    ax.plot(xy[:, 0], xy[:, 1], color=colour, lw=3.0,
            solid_capstyle="round", zorder=4,
            linestyle="-" if filled else (0, (2, 2)))
    # dots along the set to signal attract (filled) vs repel (open)
    idx = np.linspace(0, len(xy) - 1, min(7, len(xy))).astype(int)
    ax.scatter(xy[idx, 0], xy[idx, 1], s=45, zorder=5,
               facecolor=colour if filled else "white", edgecolor=colour, linewidths=1.6)


# ------------------------------------------------------------------------------
# 2-strategy game: phase line + speed profile   (feature #3, enriched)
# ------------------------------------------------------------------------------

def portrait_1d(A, r=0.0, labels=("S1", "S2"), title=None, dyn=None, **p):
    """A 2-strategy game lives on the interval x in [0,1] (x = fraction of S1).

    Top panel : the phase LINE -- rest points and flow-direction arrows.
    Bottom    : the SPEED profile (per-unit-time change in x) vs x. Where the curve
                crosses zero is a rest point; its sign is the flow direction; the
                SLOPE at a crossing is the stability (down-crossing = attracting).

    `dyn` selects the dynamic (default: continuous replicator); **p passes its
    parameters (e.g. beta for logit, mu for replicator-mutator)."""
    if dyn is None:
        dyn = rp.DYNAMICS["replicator"]
    xs = np.linspace(0, 1, 400)
    speed = np.array([dyn.motion([x, 1 - x], A, r, **p)[0] for x in xs])

    fig, (axl, axs) = plt.subplots(2, 1, figsize=(6.4, 3.6),
                                   gridspec_kw={"height_ratios": [1, 2.2]})

    # --- phase line ---
    axl.axhline(0, color="0.7", lw=1.5)
    for x in np.linspace(0.06, 0.94, 15):
        v = dyn.motion([x, 1 - x], A, r, **p)[0]
        if abs(v) < 1e-6:
            continue
        axl.annotate("", xy=(x + 0.03 * np.sign(v), 0), xytext=(x, 0),
                     arrowprops=dict(arrowstyle="->", color="0.5", lw=1.2))
    for x, t in rp.rest_points_classified_dyn(dyn, A, r, **p):
        _dot(axl, (x[0], 0), t, "#333")
    axl.set_xlim(-0.03, 1.03); axl.set_ylim(-1, 1)
    axl.set_yticks([]); axl.set_xticks([0, 1])
    axl.set_xticklabels([f"all {labels[1]}", f"all {labels[0]}"])
    axl.spines[["left", "right", "top"]].set_visible(False)

    # --- speed profile ---
    axs.axhline(0, color="0.7", lw=1)
    axs.plot(xs, speed, color="#4C78A8", lw=2)
    axs.fill_between(xs, speed, 0, where=speed > 0, alpha=0.12, color="#54A24B")
    axs.fill_between(xs, speed, 0, where=speed < 0, alpha=0.12, color="#E45756")
    for x, t in rp.rest_points_classified_dyn(dyn, A, r, **p):
        _dot(axs, (x[0], 0), t, "#333")
    axs.set_xlim(-0.03, 1.03)
    axs.set_xlabel(f"fraction playing {labels[0]}   (x)")
    axs.set_ylabel("change in x" + ("  (per step)" if dyn.discrete else "  (dx/dt)"))
    axs.spines[["right", "top"]].set_visible(False)

    if title is None:
        title = dyn.name + (f"   (assortment r = {r:g})" if r else "")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ------------------------------------------------------------------------------
# 3-strategy game: streamlines on the simplex triangle
# ------------------------------------------------------------------------------

def _bary(b, r=1.0, cx=0.0, cy=0.0):
    """Barycentric (b0,b1,b2) -> 2-D coords. S1 bottom-left, S2 top, S3 bottom-right."""
    v1 = np.array([cx - 0.866 * r, cy - 0.5 * r])
    v2 = np.array([cx,             cy + r])
    v3 = np.array([cx + 0.866 * r, cy - 0.5 * r])
    b = np.asarray(b)
    return b[0] * v1 + b[1] * v2 + b[2] * v3


def portrait_simplex(A, r=0.0, labels=("S1", "S2", "S3"), n_seeds=7,
                     title=None, dyn=None, **p):
    """3-strategy game on the 2-simplex. Trajectories are integrated from a grid
    of seed points and coloured by which attractor they reach.

    `dyn` selects the dynamic (default: continuous replicator); **p its parameters."""
    if dyn is None:
        dyn = rp.DYNAMICS["replicator"]
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.set_aspect("equal"); ax.axis("off")

    # triangle border + corner labels
    corners = [_bary([1, 0, 0]), _bary([0, 1, 0]), _bary([0, 0, 1])]
    tri = plt.Polygon(corners, fill=False, edgecolor="0.4", lw=1.5)
    ax.add_patch(tri)
    for c, lab in zip(corners, labels):
        off = 0.06 * np.sign(c - np.mean(corners, axis=0) + 1e-9)
        ax.text(c[0] + off[0], c[1] + off[1], lab, ha="center", va="center",
                fontsize=12, fontweight="bold")

    # manifold-aware rest set: isolated points AND attracting/repelling SETS
    objs = rp.rest_objects(dyn, A, r, **p)
    attractors = [o for o in objs if rp.is_attracting(o)]
    col_of = {id(o): BASIN_COLOURS[i % len(BASIN_COLOURS)]
              for i, o in enumerate(attractors)}

    def attractor_members(o):
        return np.array(rp._attractor_members(o))

    def nearest_attractor(end):
        best, bd = None, 0.06
        for o in attractors:
            d = np.min(np.linalg.norm(attractor_members(o) - end, axis=1))
            if d < bd:
                bd, best = d, o
        return best

    # streamlines from a triangular grid of seeds
    for i in range(1, n_seeds):
        for j in range(1, n_seeds - i):
            k = n_seeds - i - j
            if k < 1:
                continue
            x0 = np.array([i, j, k], float) / n_seeds
            path = rp.trajectory_dyn(dyn, x0, A, r, dt=0.02, t_max=300,
                                     record_every=3, **p)
            xy = np.array([_bary(pt) for pt in path])
            o = nearest_attractor(path[-1])
            colour = col_of.get(id(o), "0.6") if o is not None else "0.6"
            ax.plot(xy[:, 0], xy[:, 1], color=colour, lw=1.0, alpha=0.7, zorder=2)
            # a small arrowhead ~1/3 along to show direction of flow
            m = max(1, len(xy) // 3)
            if m + 1 < len(xy):
                ax.annotate("", xy=xy[m + 1], xytext=xy[m],
                            arrowprops=dict(arrowstyle="->", color=colour, lw=1.0),
                            zorder=3)

    # draw rest objects: isolated points as glyphs, sets as shaded curves
    for o in objs:
        if o["kind"] == "set":
            _draw_set(ax, o, [_bary(m) for m in o["members"]])
        else:
            _dot(ax, _bary(o["x"]), o["type"], "#222")

    if title is None:
        title = dyn.name + (f"   (assortment r = {r:g})" if r else "")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------
# Two-population 2x2 game: vector field on the unit square
# ------------------------------------------------------------------------------

def portrait_square(A, B, labels1=("S1", "S2"), labels2=("S1", "S2"),
                    n_grid=16, n_traj=5, title=None, dyn=None, **p):
    """Two-population dynamics. x = pop-1 share of its S1, y = pop-2 share of its S1.
    Vector field (arrows) + sample trajectories + fixed-point markers.
    `dyn` selects the dynamic (default continuous replicator); **p its parameters."""
    if dyn is None:
        dyn = rp.DYNAMICS["replicator"]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))

    # vector field (use the field, i.e. one small step, for arrow direction)
    gx = np.linspace(0.02, 0.98, n_grid)
    gy = np.linspace(0.02, 0.98, n_grid)
    X, Y = np.meshgrid(gx, gy)
    U = np.zeros_like(X); V = np.zeros_like(Y)
    for a in range(n_grid):
        for b in range(n_grid):
            U[a, b], V[a, b] = rp.bimatrix_motion(dyn, X[a, b], Y[a, b], A, B, **p)
    mag = np.hypot(U, V)
    ax.quiver(X, Y, U, V, mag, cmap="viridis", alpha=0.7,
              scale=None, width=0.004, pivot="mid")

    # a few trajectories
    rng = np.random.default_rng(0)
    for _ in range(n_traj):
        x0, y0 = rng.uniform(0.08, 0.92, 2)
        path = rp.bimatrix_trajectory(x0, y0, A, B, dyn=dyn, dt=0.02, t_max=200, **p)
        ax.plot(path[:, 0], path[:, 1], color="0.25", lw=1.0, alpha=0.8)

    # fixed-point markers
    for z, t in rp.bimatrix_rest_points(dyn, A, B, **p):
        _dot(ax, (z[0], z[1]), t, "#111", size=110)

    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
    ax.set_xlabel(f"pop 1: fraction playing {labels1[0]}")
    ax.set_ylabel(f"pop 2: fraction playing {labels2[0]}")
    if title is None:
        title = dyn.name + " (two populations)"
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


# quick manual smoke test
if __name__ == "__main__":
    portrait_1d(rp.GAMES_2x2["Stag Hunt"], labels=("Stag", "Hare"))
    portrait_simplex(rp.GAMES_3x3["Rock-Paper-Scissors"])
    A, B = rp.GAMES_BIMATRIX["Battle of the Sexes"]
    portrait_square(A, B, labels1=("Opera", "Fight"), labels2=("Opera", "Fight"))
    plt.show()
