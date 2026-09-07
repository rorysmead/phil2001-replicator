"""
replicator.py -- the model core for the PHIL 2001 replicator-dynamics explorer.
================================================================================

This single file is the whole model. Every front-end (the Colab notebook, the
marimo widget, the plotting helpers) imports from here, so there is exactly ONE
place where the dynamics live. If you want to understand or change how the model
behaves, this is the file to read and edit.

It has no dependencies beyond NumPy.

------------------------------------------------------------------------------
What the replicator dynamic says, in one sentence
------------------------------------------------------------------------------
A strategy grows in the population when it does BETTER THAN AVERAGE, and shrinks
when it does worse. That is the whole idea. Everything below is bookkeeping for
that sentence.

We work in CONTINUOUS time:

        dx_i/dt = x_i * ( f_i(x) - fbar(x) )

    x       state of the population: x_i is the fraction playing strategy i.
            The x_i are non-negative and sum to 1 -- a point on the SIMPLEX.
    f_i(x)  expected payoff (= "fitness") of strategy i in population x.
    fbar(x) average payoff in the whole population = sum_i x_i f_i.

Why continuous time, not the discrete map x_i' = x_i f_i / fbar ?
    Continuous time is what the textbooks (Skyrms, Weibull, Hofbauer-Sigmund)
    draw, AND it has a property students rely on: adding the SAME constant to
    every payoff changes nothing (it cancels in f_i - fbar). In discrete time
    that is false -- an additive constant acts like "background fitness" and
    silently rescales the dynamics -- which makes "the same game" look different
    depending on how you wrote the numbers down. Continuous time avoids that trap.

------------------------------------------------------------------------------
Correlation / assortment  (the `r` parameter)
------------------------------------------------------------------------------
By default players are matched at random: a strategy-i player expects to meet
strategy j with probability x_j. But interactions are often ASSORTED -- like
types meet like types more (positive assortment / correlation) or less
(negative assortment / anti-correlation) than chance. We use the standard
one-parameter linear model (Eshel & Cavalli-Sforza 1982; cf. Grafen 1979):

        f_i(x) = (1 - r) * (A x)_i  +  r * A_ii

    r = 0     random matching (ordinary replicator dynamics)
    r > 0     positive assortment: with "probability r" you meet your own type
    r < 0     negative assortment: you meet your own type LESS than chance

Positive r is the lever behind altruism (roughly r > c/b); negative r is the
mirror lever behind spite (roughly r^- > c/h). Same structure, opposite sign.

Feasibility note for r < 0: the clean probabilistic reading requires the implied
conditional distribution p(j|i) = (1-r)x_j + r*[i==j] to stay non-negative,
which needs x_i >= |r|/(1+|r|). Below that the field is still perfectly
well-defined mathematically, but "meets own type with probability ..." stops
being literally true. `feasible_r_range` reports the bound so the widget can
warn about it.

------------------------------------------------------------------------------
MAP OF THIS FILE  (sections are banner-commented below)
------------------------------------------------------------------------------
    1. The vector field dx/dt        the replicator equation itself
    2. Integrating a trajectory      RK4; where an orbit ends up
    3. Finding the rest points       equilibria, solved face by face
    4. Stability                     Jacobian eigenvalues -> stable/saddle/...
    5. Basins of attraction          Monte-Carlo estimate with confidence intervals
    6. Alternative dynamics          discrete / BNN / logit / replicator-mutator
    7. Equilibrium manifolds         lines/regions of rest points (degenerate games)
    8. Two-population (bimatrix)      dynamics on the unit square
    9. Feasibility of negative r
   10. A library of textbook games

Two API layers. Sections 1-5 are the plain CONTINUOUS-REPLICATOR functions
(`rest_points`, `classify`, `estimate_basins`, ...). Section 6 generalises them
to ANY dynamic: the same names with a `_dyn` suffix take a `Dynamics` object as
their first argument (`rest_points_dyn`, `classify_dyn`, ...). Read the plain
ones first; reach for `_dyn` only when you want to change the dynamic.
"""

import numpy as np

# ------------------------------------------------------------------------------
# 1. The vector field  dx/dt
# ------------------------------------------------------------------------------

def effective_payoffs(x, A, r=0.0):
    """Expected payoff of each strategy in population x, under assortment r.

    f_i = (1 - r) * (A x)_i + r * A_ii
    """
    x = np.asarray(x, dtype=float)
    A = np.asarray(A, dtype=float)
    return (1.0 - r) * (A @ x) + r * np.diag(A)


def mean_payoff(x, A, r=0.0):
    """Population-average payoff  fbar = sum_i x_i f_i."""
    return float(np.asarray(x, float) @ effective_payoffs(x, A, r))


def replicator_field(x, A, r=0.0):
    """The continuous-time replicator vector field dx/dt at state x.

    Note it is automatically tangent to the simplex: sum_i (dx/dt)_i = 0
    whenever sum_i x_i = 1, so a trajectory that starts on the simplex stays on it.
    """
    x = np.asarray(x, dtype=float)
    f = effective_payoffs(x, A, r)
    return x * (f - x @ f)


# ------------------------------------------------------------------------------
# 2. Integrating a trajectory
# ------------------------------------------------------------------------------
# We push a starting point forward in time with classic 4th-order Runge-Kutta.
# RK4 is far more accurate per step than the naive x <- x + dt*field used by many
# quick demos, so trajectories that should sit on a closed orbit (e.g. standard
# Rock-Paper-Scissors) actually stay on it instead of slowly drifting off.

def _rk4_step(x, A, r, dt):
    k1 = replicator_field(x, A, r)
    k2 = replicator_field(x + 0.5 * dt * k1, A, r)
    k3 = replicator_field(x + 0.5 * dt * k2, A, r)
    k4 = replicator_field(x + dt * k3, A, r)
    x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    # project back onto the simplex to kill any tiny numerical leak
    x = np.clip(x, 0.0, None)
    s = x.sum()
    return x / s if s > 0 else x


def trajectory(x0, A, r=0.0, dt=0.01, t_max=200.0, tol=1e-10, record_every=1):
    """Integrate from x0 and return the full path as an array of shape (T, n).

    Stops early once the state stops moving (change per step < tol).
    `record_every` sub-samples the path so plots stay light.
    """
    x = np.asarray(x0, dtype=float).copy()
    x = x / x.sum()
    n_steps = int(t_max / dt)
    path = [x.copy()]
    for t in range(n_steps):
        nx = _rk4_step(x, A, r, dt)
        moved = np.abs(nx - x).sum()
        x = nx
        if (t + 1) % record_every == 0:
            path.append(x.copy())
        if moved < tol:
            break
    if not np.allclose(path[-1], x):
        path.append(x.copy())
    return np.array(path)


def endpoint(x0, A, r=0.0, dt=0.01, t_max=200.0, tol=1e-10):
    """Where does the orbit from x0 end up? (Just the final state.)"""
    x = np.asarray(x0, dtype=float).copy()
    x = x / x.sum()
    n_steps = int(t_max / dt)
    for _ in range(n_steps):
        nx = _rk4_step(x, A, r, dt)
        if np.abs(nx - x).sum() < tol:
            return nx
        x = nx
    return x


# ------------------------------------------------------------------------------
# 3. Finding the rest points (equilibria)
# ------------------------------------------------------------------------------
# A rest point is a state where dx/dt = 0. For the replicator dynamics these are
# exactly the states where every strategy that is PRESENT earns the same payoff
# (strategies that are absent stay absent, so they impose no condition).
#
# Strategy: enumerate every "face" of the simplex -- i.e. every non-empty subset
# S of strategies that could be the support (the present strategies) -- and solve
# the linear "equal payoff on S" system. This finds all isolated rest points
# cleanly (corners, edges, interior) without a grid search.

def _support_subsets(n):
    """All non-empty subsets of {0, ..., n-1}, as lists of indices."""
    out = []
    for mask in range(1, 1 << n):
        out.append([i for i in range(n) if mask & (1 << i)])
    return out


def _equilibrium_on_support(A, S, r=0.0):
    """Solve for the rest point whose support is exactly S, or return None.

    On S we need all payoffs equal to some common value lambda:
        (1-r) * sum_{j in S} A_ij x_j + r*A_ii = lambda   for each i in S
        sum_{j in S} x_j = 1
    That is a linear system in the unknowns (x_S, lambda). We solve it and reject
    the solution if any component falls outside [0, 1] (i.e. it is not really on
    this face).
    """
    A = np.asarray(A, dtype=float)
    k = len(S)
    if k == 1:                       # a pure strategy is always a rest point
        x = np.zeros(len(A))
        x[S[0]] = 1.0
        return x

    sub = A[np.ix_(S, S)]            # payoffs among the present strategies
    # Unknown vector is [x_S (k of them), lambda]. Build (k+1)x(k+1) system.
    M = np.zeros((k + 1, k + 1))
    b = np.zeros(k + 1)
    for row in range(k):
        M[row, :k] = (1.0 - r) * sub[row]     # (1-r) * (A x)_i
        M[row, k] = -1.0                       # - lambda
        b[row] = -r * sub[row, row]            # move the constant r*A_ii to RHS
    M[k, :k] = 1.0                             # normalisation sum x = 1
    b[k] = 1.0
    try:
        sol = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        return None                            # degenerate face; grid scan can catch it
    xS = sol[:k]
    if np.any(xS < -1e-7) or np.any(xS > 1 + 1e-7):
        return None
    x = np.zeros(len(A))
    x[S] = np.clip(xS, 0.0, None)
    s = x.sum()
    return x / s if s > 0 else None


def rest_points(A, r=0.0, dedup_tol=1e-4):
    """All isolated rest points of the dynamics, as a list of state vectors."""
    A = np.asarray(A, dtype=float)
    n = len(A)
    found = []
    for S in _support_subsets(n):
        eq = _equilibrium_on_support(A, S, r)
        if eq is None:
            continue
        # verify it really rests (guards against the linear solve returning a
        # point that is not actually a fixed point of the full dynamics)
        if np.abs(replicator_field(eq, A, r)).sum() > 1e-6:
            continue
        if not any(np.abs(eq - p).sum() < dedup_tol for p in found):
            found.append(eq)
    return found


def has_equilibrium_manifold(A, r=0.0, tol=1e-9):
    """True if the game is DEGENERATE -- some face carries a whole line/region of
    rest points instead of isolated ones (e.g. two payoff-identical strategies, or
    an all-neutral game).

    `rest_points` finds isolated equilibria by solving each face's equal-payoff
    system. When that system is rank-deficient BUT still consistent, the solutions
    form a continuum that the isolated-point solver cannot represent -- so
    front-ends should warn the user that the rest-point picture is incomplete.
    This flags exactly that situation.
    """
    A = np.asarray(A, dtype=float)
    n = len(A)
    for S in _support_subsets(n):
        k = len(S)
        if k < 2:
            continue
        sub = A[np.ix_(S, S)]
        M = np.zeros((k + 1, k + 1))
        b = np.zeros(k + 1)
        for row in range(k):
            M[row, :k] = (1.0 - r) * sub[row]
            M[row, k] = -1.0
            b[row] = -r * sub[row, row]
        M[k, :k] = 1.0
        b[k] = 1.0
        rank_M = np.linalg.matrix_rank(M, tol=tol)
        rank_aug = np.linalg.matrix_rank(np.column_stack([M, b]), tol=tol)
        # rank-deficient (< k+1) and consistent (aug rank == M rank) => a continuum
        if rank_M < k + 1 and rank_aug == rank_M:
            return True
    return False


# ------------------------------------------------------------------------------
# 4. Stability -- the linearization (Jacobian) method
# ------------------------------------------------------------------------------
# Near a rest point x*, the dynamics look like their linear approximation
# dv/dt = J v, where J is the Jacobian of the field. The SIGN of the real parts
# of J's eigenvalues tells the story:
#     all real parts < 0   -> everything nearby flows IN     -> attracting (sink)
#     all real parts > 0   -> everything nearby flows OUT    -> repelling (source)
#     mixed signs          -> in along some directions, out along others -> saddle
#     real parts ~ 0 with imaginary parts -> orbits circle   -> center (neutral)
#
# One subtlety: the state is confined to the simplex, so we only care about
# perturbations that KEEP the population summing to 1 (the "sum-zero" directions).
# We build an orthonormal basis of that tangent space and compute the Jacobian
# restricted to it -- giving n-1 eigenvalues, exactly the meaningful ones.

def _tangent_basis(n):
    """Orthonormal basis (columns) of the sum-zero subspace of R^n."""
    ones = np.ones((1, n))
    # rows 1.. of V^T from the SVD of the all-ones row span the null space
    _, _, vt = np.linalg.svd(ones)
    return vt[1:].T                # shape (n, n-1), orthonormal columns


def _reduced_jacobian_matrix(fn, x, h=1e-6):
    """Jacobian of ANY map fn: R^n -> R^n, restricted to the simplex tangent space.

    Central finite differences along the tangent basis -- robust and model-agnostic
    (works for any dynamic without hand-derivatives). Returns (Jred, B) where Jred is
    the (n-1)x(n-1) reduced Jacobian and B is the tangent basis used.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    B = _tangent_basis(n)
    m = n - 1
    Jred = np.zeros((m, m))
    for col in range(m):
        d = B[:, col]
        deriv = (fn(x + h * d) - fn(x - h * d)) / (2 * h)
        Jred[:, col] = B.T @ deriv
    return Jred, B


def _reduced_eigs(fn, x, h=1e-6):
    """Eigenvalues of fn's Jacobian on the simplex tangent space."""
    Jred, _ = _reduced_jacobian_matrix(fn, x, h)
    return np.linalg.eigvals(Jred)


def jacobian_eigenvalues(x, A, r=0.0, h=1e-6):
    """Eigenvalues of the (continuous) replicator field at x, on the tangent space."""
    return _reduced_eigs(lambda y: replicator_field(y, A, r), np.asarray(x, float), h)


def classify(x, A, r=0.0, tol=1e-6):
    """Label a rest point from its eigenvalues.

    Returns one of: 'stable', 'unstable', 'saddle', 'center'.
        stable  -- attracting (a candidate ESS / sink)
        unstable-- repelling (source)
        saddle  -- attracts along some directions, repels along others
        center  -- neutral, orbits circle it without converging (e.g. RPS)
    """
    ev = jacobian_eigenvalues(x, A, r)
    re = ev.real
    im = ev.imag
    max_re, min_re = re.max(), re.min()
    if max_re < -tol:
        return "stable"
    if min_re > tol:
        return "unstable"
    if max_re > tol and min_re < -tol:
        return "saddle"
    # leading real part is ~0: neutral. Circling if there is an imaginary part.
    if np.any(np.abs(im) > tol):
        return "center"
    return "stable"                # marginal, treat as (weakly) attracting


def rest_points_classified(A, r=0.0):
    """Convenience: every rest point paired with its stability label."""
    return [(x, classify(x, A, r)) for x in rest_points(A, r)]


# ------------------------------------------------------------------------------
# 5. Basins of attraction, estimated by Monte Carlo
# ------------------------------------------------------------------------------
# "If we drop the population at a RANDOM starting mix, where does it end up, and
# how often?" We sample many starting points uniformly on the simplex (that is a
# Dirichlet(1,...,1) draw), run each to its resting place, match it to the
# nearest attractor, and report the fraction landing in each -- with a proper
# confidence interval so students can see the estimate is statistical, not exact.

def _wilson_interval(k, N, z=1.96):
    """95% Wilson score interval for a proportion k/N. Better than the normal
    approximation when the proportion is near 0 or 1 or N is small."""
    if N == 0:
        return (0.0, 0.0)
    p = k / N
    denom = 1 + z * z / N
    center = (p + z * z / (2 * N)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / N + z * z / (4 * N * N))
    return (max(0.0, center - half), min(1.0, center + half))


def estimate_basins(A, r=0.0, n_samples=2000, seed=None, dt=0.01,
                    t_max=200.0, match_tol=0.05):
    """Monte-Carlo basin sizes for the attracting rest points.

    Returns a list of dicts, one per attractor:
        {'state', 'type', 'count', 'fraction', 'ci_low', 'ci_high'}
    plus a final 'unresolved' fraction for samples that did not settle near any
    classified attractor (rare; usually indicates a rest-point manifold, e.g. a
    whole edge of equilibria).

    Only ASYMPTOTICALLY STABLE points have a basin. A center is Lyapunov-stable but
    neutral -- orbits circle it without converging -- so it captures nothing and is
    excluded (this is the continuous-time replicator; see estimate_basins_dyn for
    the general version).
    """
    return estimate_basins_dyn(DYNAMICS["replicator"], A, r, n_samples=n_samples,
                               seed=seed, dt=dt, t_max=t_max, match_tol=match_tol)


# ------------------------------------------------------------------------------
# 6. Alternative dynamics  (the general `Dynamics` object + the `*_dyn` API)
# ------------------------------------------------------------------------------
# The replicator dynamic is one story about how strategy frequencies change, but
# not the only one. A "dynamic" encodes a behavioural assumption -- how players
# revise what they do -- and different assumptions give different pictures for the
# SAME game. Comparing them is the point: a conclusion that survives a change of
# dynamic is robust; one that does not is an artefact of the dynamic.
#
# Every dynamic here is a vector field (or, for discrete time, a map) on the
# simplex. Assortment `r` enters them all identically -- through the payoff f_i --
# so correlation composes with any dynamic for free.
#
# Families (this grouping is itself the lesson):
#   imitative           -- copy successful others. Cannot revive an extinct
#                          strategy (if x_i = 0 it stays 0). [replicator, discrete]
#   innovative          -- above-average strategies get *introduced*, so extinct
#                          strategies can come back; rest points are exactly the
#                          Nash equilibria. [BNN]
#   perturbed best-resp -- players approximately best-respond with noise; rest
#                          points are quantal-response equilibria, interior,
#                          approaching Nash as rationality beta -> infinity. [logit]
#   selection-mutation  -- replicator plus mutation, which keeps every strategy
#                          present; rest points move into the interior. Two forms:
#                          uniform ADDITIVE decay [replmut] (a clean -mu*I shift,
#                          always stabilising) and the standard FITNESS-WEIGHTED
#                          form [replmut_fw] (mutation coupled to reproduction, so
#                          its effect on stability is game-dependent).

def _disc_replicator_map(x, A, r=0.0):
    """One step of the DISCRETE-time (Maynard Smith) replicator map: x' = x f / fbar.

    The multiplicative map needs positive fitness, so we shift the effective payoff
    up to be positive each step. That shift is exactly why discrete time -- unlike
    continuous time -- is NOT invariant to adding a constant to all payoffs: the
    constant changes the ratio f_i/fbar. (That sensitivity is a lesson, not a bug.)
    Written clip-free so its Jacobian is smooth; `Dynamics.advance` adds a guard.
    """
    x = np.asarray(x, dtype=float)
    f = effective_payoffs(x, A, r)
    f = f - min(0.0, f.min()) + 1.0
    fbar = x @ f
    return x * f / fbar            # sum is analytically 1 on the simplex


def _bnn_field(x, A, r=0.0):
    """Brown-von Neumann-Nash: an innovative dynamic. k_i = excess payoff over the
    mean; strategies doing better than average gain share, and (crucially) a
    strategy at zero with above-average payoff has k_i > 0, so it re-enters."""
    x = np.asarray(x, dtype=float)
    f = effective_payoffs(x, A, r)
    k = np.maximum(0.0, f - x @ f)
    return k - x * k.sum()


def _logit_field(x, A, r=0.0, beta=5.0):
    """Logit (perturbed best-response) dynamic: drift toward the logit/quantal
    best response softmax(beta * f). beta is rationality: beta -> 0 gives uniform
    randomisation (simplex centre), beta -> infinity gives sharp best response."""
    x = np.asarray(x, dtype=float)
    f = effective_payoffs(x, A, r)
    z = beta * (f - f.max())       # subtract max for numerical stability
    e = np.exp(z)
    return e / e.sum() - x


def _replicator_mutator_field(x, A, r=0.0, mu=0.05):
    """Replicator plus uniform mutation at rate mu, in the simple ADDITIVE form:
    every strategy decays toward the uniform mix at a flat rate, independent of
    fitness. Mutation keeps every strategy present, pushing rest points into the
    interior. Because the mutation term is fitness-INDEPENDENT the Jacobian shifts
    by exactly -mu*I, so mutation is uniformly STABILISING: on RPS the neutral
    centre becomes a stable spiral for any mu > 0. Contrast
    `_replicator_mutator_fw_field`, the standard fitness-weighted form, whose
    mutation is fitness-coupled and so is NOT uniformly stabilising. Both are
    offered so the two can be compared."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    return replicator_field(x, A, r) + mu * (np.ones(n) / n - x)


def _replicator_mutator_fw_field(x, A, r=0.0, mu=0.05):
    """Fitness-weighted replicator-mutator -- the STANDARD form (Nowak 2006;
    Page & Nowak 2002). Offspring are produced in proportion to fitness and THEN
    mutate uniformly at rate mu, i.e. with mutation matrix Q_ji = (1-mu)[i=j] + mu/n,

        dx_i/dt = (1 - mu) * x_i * f_i  +  (mu/n) * sum_j x_j f_j  -  x_i * fbar

    Here mutation is COUPLED to fitness (it is the reproduction step that mutates),
    so it does NOT shift the Jacobian by a clean -mu*I the way the additive form
    does. Its effect on stability is therefore GAME-DEPENDENT rather than uniformly
    stabilising: on the symmetric zero-sum RPS shipped here it leaves the interior
    a neutral CENTRE for every mu (it only slows the rotation), where the additive
    form turns that same centre into a stable spiral. The fitness coupling is also
    what makes genuinely non-replicator behaviour -- Hopf bifurcations, limit
    cycles -- possible at all in other games; the additive form, being a pure
    -mu*I shift, can never produce them. At mu = 0 it is exactly the replicator
    dynamic; tangency holds because the three terms sum to
    (1-mu)*fbar + mu*fbar - fbar = 0."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    f = effective_payoffs(x, A, r)
    fbar = x @ f
    return (1.0 - mu) * x * f + (mu / n) * fbar - x * fbar


class Dynamics:
    """A named dynamic on the simplex. Bundles the vector field / map with the
    right rest-point and stability machinery for its type."""

    def __init__(self, key, name, family, discrete, fn,
                 rest_method, stability_method, params=None):
        self.key = key
        self.name = name
        self.family = family
        self.discrete = discrete        # True => `fn` is a map x'; False => field dx/dt
        self._fn = fn
        self.rest_method = rest_method          # "face" or "numerical"
        self.stability_method = stability_method  # "jacobian" or "numerical"
        self.default_params = params or {}

    def _p(self, p):
        return {**self.default_params, **p}

    def motion(self, x, A, r=0.0, **p):
        """The displacement whose zeros are the rest points: the field (continuous)
        or x' - x (discrete)."""
        x = np.asarray(x, dtype=float)
        val = self._fn(x, A, r, **self._p(p))
        return val - x if self.discrete else val

    def advance(self, x, A, r=0.0, dt=0.01, **p):
        """One integration/iteration step (for trajectories)."""
        x = np.asarray(x, dtype=float)
        pr = self._p(p)
        if self.discrete:
            nx = self._fn(x, A, r, **pr)
        else:
            f = lambda y: self._fn(y, A, r, **pr)
            k1 = f(x); k2 = f(x + 0.5 * dt * k1)
            k3 = f(x + 0.5 * dt * k2); k4 = f(x + dt * k3)
            nx = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        nx = np.clip(nx, 0.0, None)
        s = nx.sum()
        return nx / s if s > 0 else x


DYNAMICS = {
    "replicator": Dynamics("replicator", "Replicator (continuous)", "imitative",
                           False, lambda x, A, r: replicator_field(x, A, r),
                           "face", "jacobian"),
    "discrete":   Dynamics("discrete", "Replicator (discrete-time)", "imitative",
                           True, _disc_replicator_map, "face", "jacobian"),
    "bnn":        Dynamics("bnn", "Brown-von Neumann-Nash", "innovative",
                           False, _bnn_field, "face", "numerical"),
    "logit":      Dynamics("logit", "Logit best-response", "perturbed best-response",
                           False, _logit_field, "numerical", "jacobian",
                           params={"beta": 5.0}),
    "replmut":    Dynamics("replmut", "Replicator-mutator (uniform)", "selection-mutation",
                           False, _replicator_mutator_field, "numerical", "jacobian",
                           params={"mu": 0.05}),
    "replmut_fw": Dynamics("replmut_fw", "Replicator-mutator (fitness-weighted)",
                           "selection-mutation", False, _replicator_mutator_fw_field,
                           "numerical", "jacobian", params={"mu": 0.05}),
}


def _newton_rest_points(g, n, seed=0, n_starts=16, tol=1e-9):
    """Find interior rest points of a field g (g: R^n -> R^n) by multi-start Newton
    on the tangent space. Used for dynamics whose equilibria are NOT on the faces
    (logit, replicator-mutator). Multi-start catches attractors, repellers, saddles."""
    rng = np.random.default_rng(seed)
    starts = [np.ones(n) / n] + [rng.dirichlet(np.ones(n)) for _ in range(n_starts)]
    found = []
    for x0 in starts:
        x = x0.copy()
        converged = False
        for _ in range(80):
            Jred, B = _reduced_jacobian_matrix(g, x)
            gr = B.T @ g(x)
            if np.linalg.norm(gr) < tol:
                converged = True
                break
            try:
                du = np.linalg.solve(Jred, -gr)
            except np.linalg.LinAlgError:
                break
            step = B @ du
            nrm = np.linalg.norm(step)
            if nrm > 0.3:                      # damp large Newton steps
                step *= 0.3 / nrm
            x = np.clip(x + step, 1e-12, None)
            x = x / x.sum()
        if converged and np.abs(g(x)).sum() < 1e-6:
            xr = np.clip(x, 0.0, None)
            xr = xr / xr.sum()
            if not any(np.abs(xr - q).sum() < 1e-3 for q in found):
                found.append(xr)
    return found


def rest_points_dyn(dyn, A, r=0.0, dedup_tol=1e-4, **p):
    """Rest points of an arbitrary dynamic."""
    A = np.asarray(A, dtype=float)
    n = len(A)
    g = lambda x: dyn.motion(x, A, r, **p)
    if dyn.rest_method == "face":
        # candidates are the equal-payoff face solutions; keep those the dynamic
        # actually rests at. For BNN this filter reduces the set to the Nash equilibria.
        found = []
        for S in _support_subsets(n):
            eq = _equilibrium_on_support(A, S, r)
            if eq is None:
                continue
            if np.abs(g(eq)).sum() > 1e-6:
                continue
            if not any(np.abs(eq - q).sum() < dedup_tol for q in found):
                found.append(eq)
        return found
    return _newton_rest_points(g, n)


def classify_dyn(dyn, A, x, r=0.0, tol=1e-6, **p):
    """Stability label for a rest point under an arbitrary dynamic.

    Continuous dynamics use the sign of the eigenvalues' real parts; DISCRETE
    dynamics use whether the eigenvalues sit inside the unit circle (|lambda|<1).
    BNN's field is non-smooth, so it uses a numerical perturb-and-watch test."""
    x = np.asarray(x, dtype=float)
    if dyn.stability_method == "numerical":
        return _classify_numerical(dyn, A, x, r, **p)

    fn = lambda y: dyn._fn(y, A, r, **dyn._p(p))
    ev = _reduced_eigs(fn, x)
    if dyn.discrete:
        mod = np.abs(ev)                         # stable iff inside the unit circle
        if mod.max() < 1 - tol:
            return "stable"
        if mod.min() > 1 + tol:
            return "unstable"
        if mod.max() > 1 + tol and mod.min() < 1 - tol:
            return "saddle"
        return "center" if np.any(np.abs(ev.imag) > tol) else "stable"
    re, im = ev.real, ev.imag                    # continuous: sign of real parts
    if re.max() < -tol:
        return "stable"
    if re.min() > tol:
        return "unstable"
    if re.max() > tol and re.min() < -tol:
        return "saddle"
    return "center" if np.any(np.abs(im) > tol) else "stable"


def _classify_numerical(dyn, A, x, r=0.0, eps=0.06, steps=1000, dt=0.05, **p):
    """Perturb in each tangent direction, run the dynamic, and see whether the
    orbit returns (in), leaves (out), or holds its distance (neutral). Used for
    non-smooth fields (BNN) where a finite-difference Jacobian is unreliable.

    BNN's strict-Nash rest points attract only QUADRATICALLY (eigenvalue 0), so a
    tiny perturbation barely moves in finite time. We use a sizeable perturbation
    (eps ~ 0.06): quadratic contraction has timescale ~1/|perturbation|, so a
    larger kick makes genuine attraction visible while a true center still returns
    to the same radius (ratio ~ 1). Calibrated thresholds: <0.7 in, >1.5 out."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    B = _tangent_basis(n)
    verdicts = []
    for c in range(n - 1):
        for sgn in (1, -1):
            xp = np.clip(x + sgn * eps * B[:, c], 0.0, None)
            s = xp.sum()
            if s <= 0:
                continue
            xp = xp / s
            d0 = np.linalg.norm(xp - x)
            if d0 < 1e-9:
                continue
            xt = xp.copy()
            for _ in range(steps):
                xt = dyn.advance(xt, A, r, dt=dt, **p)
            ratio = np.linalg.norm(xt - x) / d0
            verdicts.append("in" if ratio < 0.7 else "out" if ratio > 1.5 else "neutral")
    if not verdicts or all(v == "in" for v in verdicts):
        return "stable"
    if all(v == "out" for v in verdicts):
        return "unstable"
    if any(v == "out" for v in verdicts) and any(v == "in" for v in verdicts):
        return "saddle"
    return "center"


def rest_points_classified_dyn(dyn, A, r=0.0, **p):
    """Every rest point of `dyn` paired with its stability label."""
    return [(x, classify_dyn(dyn, A, x, r, **p)) for x in rest_points_dyn(dyn, A, r, **p)]


# ------------------------------------------------------------------------------
# 7. Equilibrium manifolds  (neutral stability across a set of rest points)
# ------------------------------------------------------------------------------
# When a game is degenerate a whole LINE or REGION of the simplex can be rest
# points (e.g. two strategies that always earn the same). The isolated-point
# solver cannot represent that, and the eigenvalue classifier gets a ~0 eigenvalue
# ALONG the set, which used to be mislabelled "stable". The fix: when degeneracy
# is detected, grid-scan the simplex for resting states, cluster them into
# connected sets, and classify each set by its TRANSVERSE stability -- the
# behaviour in the directions OFF the set, ignoring the neutral along-set direction.
#
# `rest_objects` is the manifold-aware replacement for `rest_points_classified_dyn`:
# it returns a list of dicts, each either
#     {"kind": "point", "x": state, "type": stable/unstable/saddle/center}
# or  {"kind": "set",   "members": [states...], "x": centroid,
#      "type": set-attract / set-repel / set-mixed}

def _simplex_grid(n, G):
    if n == 2:
        return [np.array([i / G, 1 - i / G]) for i in range(G + 1)]
    return [np.array([i / G, j / G, (G - i - j) / G])
            for i in range(G + 1) for j in range(G + 1 - i)]


def _cluster(pts, link):
    """Group points into connected components (single-linkage within `link`)."""
    m = len(pts)
    used = [False] * m
    comps = []
    for i in range(m):
        if used[i]:
            continue
        stack, comp, used[i] = [i], [], True
        while stack:
            c = stack.pop()
            comp.append(pts[c])
            for k in range(m):
                if not used[k] and np.linalg.norm(pts[c] - pts[k]) < link:
                    used[k] = True
                    stack.append(k)
        comps.append(comp)
    return comps


def _centroid(members):
    return np.mean(np.array(members), axis=0)


def _spread(members):
    """Largest pairwise distance in a cluster (subsampled for speed)."""
    if len(members) < 2:
        return 0.0
    arr = np.array(members)
    idx = range(0, len(arr), max(1, len(arr) // 30))
    best = 0.0
    for a in idx:
        for b in idx:
            best = max(best, np.linalg.norm(arr[a] - arr[b]))
    return best


def _transverse_type(dyn, A, x, r=0.0, zero_tol=1e-4, **p):
    """Stability OFF a manifold at state x: classify by the eigenvalues that are
    NOT ~0 (the ~0 ones are the neutral along-manifold directions)."""
    fn = lambda y: dyn._fn(y, A, r, **dyn._p(p))
    ev = _reduced_eigs(fn, x)
    if dyn.discrete:
        off = ev[np.abs(np.abs(ev) - 1.0) > zero_tol]     # off the unit circle
        if len(off) == 0:
            return "neutral"
        mod = np.abs(off)
        return "attract" if mod.max() < 1 else "repel" if mod.min() > 1 else "mixed"
    off = ev[np.abs(ev.real) > zero_tol]                  # nonzero real part
    if len(off) == 0:
        return "neutral"
    return "attract" if off.real.max() < 0 else "repel" if off.real.min() > 0 else "mixed"


def rest_objects(dyn, A, r=0.0, **p):
    """Manifold-aware rest set: isolated points AND equilibrium sets (see above)."""
    A = np.asarray(A, dtype=float)
    n = len(A)
    if n not in (2, 3) or not has_equilibrium_manifold(A, r):
        return [{"kind": "point", "x": x, "type": t}
                for x, t in rest_points_classified_dyn(dyn, A, r, **p)]

    G = 200 if n == 2 else 70
    resting = [x for x in _simplex_grid(n, G)
               if np.abs(dyn.motion(x, A, r, **p)).sum() < 2e-4]
    if not resting:
        return [{"kind": "point", "x": x, "type": t}
                for x, t in rest_points_classified_dyn(dyn, A, r, **p)]

    objs = []
    for members in _cluster(resting, link=4.0 / G):
        if _spread(members) > 0.08 and len(members) > 4:
            step = max(1, len(members) // 12)
            types = [_transverse_type(dyn, A, m, r, **p) for m in members[::step]]
            has_a = any(t == "attract" for t in types)
            has_r = any(t == "repel" for t in types)
            kind = ("set-mixed" if has_a and has_r
                    else "set-attract" if has_a
                    else "set-repel" if has_r
                    else "set-neutral")   # every direction neutral (fully degenerate)
            objs.append({"kind": "set", "members": members,
                         "x": _centroid(members), "type": kind})
        else:
            c = _centroid(members)
            objs.append({"kind": "point", "x": c, "type": classify_dyn(dyn, A, c, r, **p)})
    return objs


def _attractor_members(obj, cap=40):
    """Representative points of an attracting rest object (for basin proximity)."""
    if obj["kind"] == "set":
        mem = obj["members"]
        step = max(1, len(mem) // cap)
        return mem[::step]
    return [obj["x"]]


def is_attracting(obj):
    return (obj["kind"] == "point" and obj["type"] == "stable") or \
           (obj["kind"] == "set" and obj["type"] in ("set-attract", "set-mixed"))


def trajectory_dyn(dyn, x0, A, r=0.0, dt=0.01, t_max=200.0, tol=1e-10,
                   record_every=1, **p):
    """Integrate/iterate an orbit under `dyn`; returns the path, shape (T, n)."""
    x = np.asarray(x0, dtype=float).copy()
    x = x / x.sum()
    n_steps = int(t_max / dt)
    path = [x.copy()]
    for t in range(n_steps):
        nx = dyn.advance(x, A, r, dt=dt, **p)
        moved = np.abs(nx - x).sum()
        x = nx
        if (t + 1) % record_every == 0:
            path.append(x.copy())
        if moved < tol:
            break
    if not np.allclose(path[-1], x):
        path.append(x.copy())
    return np.array(path)


def endpoint_dyn(dyn, x0, A, r=0.0, dt=0.01, t_max=200.0, tol=1e-10, **p):
    """Where the orbit under `dyn` ends up."""
    x = np.asarray(x0, dtype=float).copy()
    x = x / x.sum()
    for _ in range(int(t_max / dt)):
        nx = dyn.advance(x, A, r, dt=dt, **p)
        if np.abs(nx - x).sum() < tol:
            return nx
        x = nx
    return x


def estimate_basins_dyn(dyn, A, r=0.0, n_samples=2000, seed=None, dt=0.02,
                        t_max=200.0, match_tol=0.05, tol=1e-9, **p):
    """Monte-Carlo basin sizes under an arbitrary dynamic (see `estimate_basins`).
    Uses a coarser step than trajectory plotting -- basin ASSIGNMENT is robust to
    step size, and it keeps the Monte-Carlo loop responsive."""
    rng = np.random.default_rng(seed)
    n = len(A)
    # manifold-aware: an attractor may be an isolated point OR a whole attracting
    # SET (line/region). An orbit is "in" a set's basin if it lands near ANY member.
    objs = [o for o in rest_objects(dyn, A, r, **p) if is_attracting(o)]
    if not objs:
        return {"attractors": [], "unresolved": 1.0, "n_samples": 0,
                "note": "no asymptotically stable rest point/set under this dynamic."}
    att_members = [np.array(_attractor_members(o)) for o in objs]  # per-attractor point cloud

    def nearest(x):
        best, bd = -1, np.inf
        for j, mem in enumerate(att_members):
            d = np.min(np.linalg.norm(mem - x, axis=1))
            if d < bd:
                bd, best = d, j
        return best, bd

    counts = np.zeros(len(objs), dtype=int)
    unresolved = 0
    n_steps = int(t_max / dt)
    for _ in range(n_samples):
        x = rng.dirichlet(np.ones(n))
        matched = -1
        for step in range(n_steps):
            nx = dyn.advance(x, A, r, dt=dt, **p)
            moved = np.abs(nx - x).sum()
            x = nx
            if step % 25 == 0:
                j, d = nearest(x)
                if d < match_tol:
                    matched = j
                    break
            if moved < tol:
                break
        if matched < 0:
            j, d = nearest(x)
            matched = j if d < match_tol else -1
        if matched >= 0:
            counts[matched] += 1
        else:
            unresolved += 1
    out = []
    for j, o in enumerate(objs):
        lo, hi = _wilson_interval(counts[j], n_samples)
        out.append({"state": o["x"], "type": o["type"], "kind": o["kind"],
                    "members": o.get("members"), "count": int(counts[j]),
                    "fraction": counts[j] / n_samples, "ci_low": lo, "ci_high": hi})
    return {"attractors": out, "unresolved": unresolved / n_samples, "n_samples": n_samples}


# ------------------------------------------------------------------------------
# 8. Two-population (bimatrix) dynamics on the unit square
# ------------------------------------------------------------------------------
# Some games are ASYMMETRIC: two roles with possibly different payoffs (buyer /
# seller, owner / intruder, the two sides of a Nash demand game). With two
# strategies per role the state is a point (x, y) in the unit SQUARE, not the
# simplex:
#     x = fraction of population 1 playing its strategy 1
#     y = fraction of population 2 playing its strategy 1
# Population 1 earns from matrix A (rows = pop-1 strategy, cols = pop-2 strategy);
# population 2 earns from matrix B (rows = pop-2 strategy, cols = pop-1 strategy).

# The SAME five dynamics apply here. Each population revises its own frequency by
# the chosen rule, responding to the other population's current mix. So the two-
# population motion is just the dynamic's per-population "response" applied twice.

def _pop_response(dyn, pvec, f, **params):
    """Displacement of a 2-strategy population with frequencies `pvec` and payoff
    vector `f`, under `dyn`: the field (continuous) or next-state minus current
    (discrete). This is the two-population echo of `Dynamics.motion`."""
    pvec = np.asarray(pvec, dtype=float)
    f = np.asarray(f, dtype=float)
    pr = dyn._p(params)
    key = dyn.key
    if key == "discrete":
        fs = f - min(0.0, f.min()) + 1.0        # positive fitness for the map
        return pvec * fs / (pvec @ fs) - pvec
    if key == "bnn":
        k = np.maximum(0.0, f - pvec @ f)
        return k - pvec * k.sum()
    if key == "logit":
        z = pr["beta"] * (f - f.max())
        e = np.exp(z)
        return e / e.sum() - pvec
    if key == "replmut":
        n = len(pvec)
        return pvec * (f - pvec @ f) + pr["mu"] * (np.ones(n) / n - pvec)
    if key == "replmut_fw":
        n = len(pvec)
        fbar = pvec @ f
        return (1.0 - pr["mu"]) * pvec * f + (pr["mu"] / n) * fbar - pvec * fbar
    return pvec * (f - pvec @ f)                 # replicator (also for continuous default)


def bimatrix_motion(dyn, x, y, A, B, **p):
    """(motion in x, motion in y) for the two-population dynamic `dyn`.
    'Motion' is the field for continuous dynamics, or next-state-minus-current for
    the discrete map -- so a step is always `state += motion` (Euler/RK for
    continuous, one iteration for discrete)."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    p1 = np.array([x, 1 - x])
    p2 = np.array([y, 1 - y])
    m1 = _pop_response(dyn, p1, A @ p2, **p)
    m2 = _pop_response(dyn, p2, B @ p1, **p)
    return m1[0], m2[0]


def bimatrix_field(x, y, A, B):
    """Two-population continuous replicator field (dx/dt, dy/dt). Kept for
    backward compatibility; `bimatrix_motion` is the general version."""
    return bimatrix_motion(DYNAMICS["replicator"], x, y, A, B)


def bimatrix_trajectory(x0, y0, A, B, dyn=None, dt=0.01, t_max=200.0, tol=1e-10, **p):
    """Integrate/iterate a two-population orbit from (x0, y0); returns array of (x, y).
    `dyn` selects the dynamic (default continuous replicator)."""
    if dyn is None:
        dyn = DYNAMICS["replicator"]
    x, y = float(x0), float(y0)
    path = [(x, y)]
    clip = lambda v: min(1.0, max(0.0, v))
    for _ in range(int(t_max / dt)):
        if dyn.discrete:
            mx, my = bimatrix_motion(dyn, x, y, A, B, **p)   # motion = map - state
            nx, ny = clip(x + mx), clip(y + my)
        else:
            k1 = bimatrix_motion(dyn, x, y, A, B, **p)
            k2 = bimatrix_motion(dyn, x + 0.5 * dt * k1[0], y + 0.5 * dt * k1[1], A, B, **p)
            k3 = bimatrix_motion(dyn, x + 0.5 * dt * k2[0], y + 0.5 * dt * k2[1], A, B, **p)
            k4 = bimatrix_motion(dyn, x + dt * k3[0], y + dt * k3[1], A, B, **p)
            nx = clip(x + (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]))
            ny = clip(y + (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))
        moved = abs(nx - x) + abs(ny - y)
        x, y = nx, ny
        path.append((x, y))
        if moved < tol:
            break
    return np.array(path)


def _bimatrix_classify(dyn, A, B, z, eps=0.06, steps=800, dt=0.05, **p):
    """Stability of a square fixed point by perturb-and-watch (same calibrated
    method as the single-population classifier: return-ratio <0.7 in, >1.5 out).
    Robust to non-hyperbolic points (centers) and BNN's non-smooth field."""
    verdicts = []
    s = 0.70710678  # diagonals too, so a saddle's diagonal stable manifold is seen
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                   (s, s), (-s, -s), (s, -s), (-s, s)]:
        x0 = min(1.0, max(0.0, z[0] + eps * dx))
        y0 = min(1.0, max(0.0, z[1] + eps * dy))
        d0 = np.hypot(x0 - z[0], y0 - z[1])
        if d0 < 1e-9:
            continue
        end = bimatrix_trajectory(x0, y0, A, B, dyn=dyn, dt=dt, t_max=steps * dt, **p)[-1]
        ratio = np.hypot(end[0] - z[0], end[1] - z[1]) / d0
        verdicts.append("in" if ratio < 0.7 else "out" if ratio > 1.5 else "neutral")
    if not verdicts or all(v == "in" for v in verdicts):
        return "stable"
    if all(v == "out" for v in verdicts):
        return "unstable"
    if any(v == "out" for v in verdicts) and any(v == "in" for v in verdicts):
        return "saddle"
    return "center"


def bimatrix_rest_points(dyn, A, B, **p):
    """Rest points of the two-population dynamic on the unit square, each paired
    with a stability label (stable / unstable / saddle / center). Corners are
    always candidates; interior/edge fixed points are found by multi-start Newton."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    g = lambda xy: np.array(bimatrix_motion(dyn, xy[0], xy[1], A, B, **p))

    cands = [np.array([a, b]) for a in (0.0, 1.0) for b in (0.0, 1.0)]  # corners
    grid = np.linspace(0.1, 0.9, 5)                                     # includes 0.5
    for sx in grid:
        for sy in grid:
            z = np.array([sx, sy])
            for _ in range(60):
                gv = g(z)
                if np.linalg.norm(gv) < 1e-11:
                    break
                J = np.zeros((2, 2))
                for c in range(2):
                    e = np.zeros(2); e[c] = 1e-6
                    J[:, c] = (g(z + e) - g(z - e)) / 2e-6
                try:
                    z = np.clip(z - np.linalg.solve(J, gv), 0.0, 1.0)
                except np.linalg.LinAlgError:
                    break
            if np.linalg.norm(g(z)) < 1e-7:
                cands.append(z)

    found = []
    for z in cands:
        if np.linalg.norm(g(z)) < 1e-6 and not any(np.linalg.norm(z - q) < 1e-3 for q in found):
            found.append(z)

    return [(z, _bimatrix_classify(dyn, A, B, z, **p)) for z in found]


# ------------------------------------------------------------------------------
# 9. Feasibility helper for negative assortment
# ------------------------------------------------------------------------------

def feasible_r_range(x):
    """Range of r for which the assortment model keeps a valid probability
    reading at state x. Positive assortment is always fine (up to r=1); negative
    assortment is limited by the RAREST present strategy."""
    x = np.asarray(x, dtype=float)
    present = x[x > 1e-9]
    if len(present) == 0:
        return (-1.0, 1.0)
    xmin = present.min()
    # need x_i >= |r|/(1+|r|)  ->  |r| <= xmin/(1-xmin)
    r_neg = -xmin / (1 - xmin) if xmin < 1 else -1.0
    return (max(-1.0, r_neg), 1.0)


# ------------------------------------------------------------------------------
# 10. A small library of textbook games (payoff matrices)
# ------------------------------------------------------------------------------
# Row = focal player's strategy, column = opponent's strategy. Entry = focal's payoff.

GAMES_2x2 = {
    "Prisoner's Dilemma": np.array([[3.0, 0.0], [5.0, 1.0]]),   # S1=Cooperate, S2=Defect
    "Hawk-Dove":          np.array([[0.0, 3.0], [1.0, 2.0]]),   # S1=Hawk, S2=Dove
    "Stag Hunt":          np.array([[3.0, 0.0], [1.0, 1.0]]),   # S1=Stag, S2=Hare
    "Coordination":       np.array([[2.0, 0.0], [0.0, 2.0]]),
}

GAMES_3x3 = {
    "Rock-Paper-Scissors":      np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]]),
    "RPS (attracting)":         np.array([[0.0, -1.0, 2.0], [2.0, 0.0, -1.0], [-1.0, 2.0, 0.0]]),
    "Dominance chain":          np.array([[3.0, 0.0, 0.0], [2.0, 2.0, 0.0], [1.0, 1.0, 1.0]]),
}

# Two-population examples: (A, B) with A = pop-1 payoffs, B = pop-2 payoffs.
GAMES_BIMATRIX = {
    # Battle of the Sexes: both want to coordinate but on different options.
    "Battle of the Sexes": (np.array([[2.0, 0.0], [0.0, 1.0]]),
                            np.array([[1.0, 0.0], [0.0, 2.0]])),
    # Owner-Intruder (asymmetric Hawk-Dove): the classic "respect ownership" game.
    "Owner-Intruder":      (np.array([[0.0, 3.0], [1.0, 2.0]]),
                            np.array([[0.0, 3.0], [1.0, 2.0]])),
    "Matching Pennies":    (np.array([[1.0, -1.0], [-1.0, 1.0]]),
                            np.array([[-1.0, 1.0], [1.0, -1.0]])),
}
