"""
tests.py -- correctness checks for the replicator core.

These check the model against results you can look up or derive by hand. Run:

    python3 tests.py

Every line should end in PASS. This is the safety net: if you edit replicator.py
and a dynamical fact breaks, you find out here rather than in front of a class.
"""
import numpy as np
import replicator as rp


def approx(a, b, tol=1e-3):
    return np.abs(np.asarray(a) - np.asarray(b)).max() < tol


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    return bool(cond)


def main():
    ok = True
    A_pd = rp.GAMES_2x2["Prisoner's Dilemma"]
    A_hd = rp.GAMES_2x2["Hawk-Dove"]
    A_sh = rp.GAMES_2x2["Stag Hunt"]
    rps = rp.GAMES_3x3["Rock-Paper-Scissors"]
    rps_a = rp.GAMES_3x3["RPS (attracting)"]

    # --- 1. Additive-constant invariance (the property discrete time lacks) ---
    x = np.array([0.4, 0.6])
    f1 = rp.replicator_field(x, A_pd)
    f2 = rp.replicator_field(x, A_pd + 100.0)
    ok &= check("adding a constant to all payoffs leaves the field unchanged",
                approx(f1, f2))

    # --- 2. Prisoner's Dilemma: Defect (S2) dominates, takes the whole population ---
    end = rp.endpoint([0.9, 0.1], A_pd)
    ok &= check("PD: all-Defect is the outcome", approx(end, [0.0, 1.0]))
    types = dict((tuple(np.round(x, 3)), t) for x, t in rp.rest_points_classified(A_pd))
    ok &= check("PD: all-Defect (0,1) is stable", types.get((0.0, 1.0)) == "stable")
    ok &= check("PD: all-Cooperate (1,0) is unstable", types.get((1.0, 0.0)) == "unstable")

    # --- 3. Hawk-Dove: interior ESS. Hawk=0,Dove=3,payoffs give x*_Hawk = 1/2 ---
    # Hawk share x solves 3(1-x) = x + 2(1-x) -> mixed point. Known ESS here is x*=0.5.
    pts = rp.rest_points_classified(A_hd)
    interior = [(x, t) for x, t in pts if x.min() > 1e-3]
    ok &= check("Hawk-Dove: an interior rest point exists", len(interior) == 1)
    if interior:
        xi, ti = interior[0]
        ok &= check("Hawk-Dove: interior point is at (1/2, 1/2)", approx(xi, [0.5, 0.5]))
        ok &= check("Hawk-Dove: interior point is stable (an ESS)", ti == "stable")

    # --- 4. Stag Hunt: two stable pure states separated by an interior REPELLER ---
    # (On a 2-strategy game the simplex is a line, so the basin-boundary point has
    # a single eigenvalue -- it is a repeller/'unstable', not a 2-D saddle.)
    pts = rp.rest_points_classified(A_sh)
    stable_pures = [x for x, t in pts if t == "stable" and x.max() > 0.999]
    ok &= check("Stag Hunt: both pure states are stable", len(stable_pures) == 2)
    interior_sh = [(x, t) for x, t in pts if x.min() > 1e-3]
    ok &= check("Stag Hunt: interior point at (1/3, 2/3)",
                len(interior_sh) == 1 and approx(interior_sh[0][0], [1/3, 2/3]))
    ok &= check("Stag Hunt: interior point is a repeller (basin boundary)",
                len(interior_sh) == 1 and interior_sh[0][1] == "unstable")

    # --- 5. Standard RPS: interior point is a CENTER (neutral orbits) ---
    # This is THE case discrete-time replicator gets qualitatively wrong.
    center = rp.classify([1/3, 1/3, 1/3], rps)
    ok &= check("RPS: interior (1/3,1/3,1/3) is a center, not a spiral", center == "center")

    # --- 6. Attracting RPS: same interior point is now stable ---
    catt = rp.classify([1/3, 1/3, 1/3], rps_a)
    ok &= check("RPS(attracting): interior point is stable", catt == "stable")

    # --- 7. Assortment payoff matches the Bergstrom-index formula exactly,
    #        including NEGATIVE r (anti-assortment). This is the ground truth. ---
    Ag = np.array([[3., 0., 1.], [5., 1., 2.], [0., 4., 2.]])
    xg = np.array([0.5, 0.3, 0.2])
    ground_ok = True
    for rr in (-0.5, -0.2, 0.0, 0.4, 0.9):
        f_code = rp.effective_payoffs(xg, Ag, rr)
        f_hand = np.array([rr * Ag[i, i] + (1 - rr) * (xg @ Ag[i]) for i in range(3)])
        ground_ok &= approx(f_code, f_hand)
    ok &= check("assortment payoff = r*A_ii + (1-r)*(Ax)_i for all r (incl. negative)",
                ground_ok)

    # --- 7b. ALTRUISM threshold: cooperators invade all-Defect iff r > 1/3 (=c/b here) ---
    near_D = np.array([1e-4, 1 - 1e-4])
    below = rp.replicator_field(near_D, A_pd, r=0.30)[0]   # r<1/3: shrink
    above = rp.replicator_field(near_D, A_pd, r=0.34)[0]   # r>1/3: grow
    ok &= check("PD: cooperation invasion threshold is r = 1/3", below < 0 < above)

    # --- 7c. SPITE (anti-assortment) mirror: a strategy that harms others is
    #        dominated at r=0 but invades once r < -1/3. ---
    # S pays cost 1 to harm the opponent by 3; N is neutral. Row = focal payoff.
    spite = np.array([[-4., -1.], [-3., 0.]])
    mix = np.array([0.5, 0.5])
    dS_random = rp.replicator_field(mix, spite, r=0.0)[0]   # dies
    dS_anti = rp.replicator_field(mix, spite, r=-0.5)[0]    # invades
    ok &= check("spite: dominated at r=0 but invades under negative assortment",
                dS_random < 0 < dS_anti)

    # --- 7d. feasible_r_range: negative bound set by the rarest present strategy ---
    lo, hi = rp.feasible_r_range(np.array([0.4, 0.35, 0.25]))
    ok &= check("feasible_r_range negative bound = -x_min/(1-x_min)",
                approx(lo, -0.25 / 0.75) and hi == 1.0)

    # --- 8. Basin estimate sanity: fractions + unresolved sum to ~1 ---
    b = rp.estimate_basins(A_sh, n_samples=400, seed=0)
    total = sum(a["fraction"] for a in b["attractors"]) + b["unresolved"]
    ok &= check("basin fractions + unresolved sum to 1", abs(total - 1.0) < 1e-9)
    ok &= check("Stag Hunt basins: two attractors found", len(b["attractors"]) == 2)

    # --- 8b. RPS has no asymptotic attractor: estimate_basins short-circuits
    #         (regression test -- this used to grind for ~80s on non-converging orbits) ---
    import time as _time
    t0 = _time.time()
    b_rps = rp.estimate_basins(rps, n_samples=300, seed=0)
    elapsed = _time.time() - t0
    ok &= check("RPS basins: no attractors, returns immediately (< 1s)",
                len(b_rps["attractors"]) == 0 and elapsed < 1.0)

    # --- 8c. Degeneracy detection: a game with two identical strategies has a LINE
    #         of equilibria; flag it. A generic game does not. ---
    dup = np.array([[1., 1., 0.], [1., 1., 0.], [0., 0., 2.]])  # S1, S2 identical
    ok &= check("degenerate game (duplicate strategies) is flagged",
                rp.has_equilibrium_manifold(dup))
    ok &= check("generic game (Stag Hunt) is not flagged as degenerate",
                not rp.has_equilibrium_manifold(A_sh))

    # --- 9. Two-population Matching Pennies: interior fixed point (1/2,1/2), orbits circle ---
    Amp, Bmp = rp.GAMES_BIMATRIX["Matching Pennies"]
    path = rp.bimatrix_trajectory(0.6, 0.5, Amp, Bmp, t_max=50)
    # should orbit the center, not converge to it: end still far from a corner
    ok &= check("Matching Pennies (2-pop): orbit stays interior (does not collapse)",
                path[-1].min() > 0.05 and path[-1].max() < 0.95)

    # ---------------------------------------------------------------------------
    # 10. ALTERNATIVE DYNAMICS
    # ---------------------------------------------------------------------------
    D = rp.DYNAMICS
    center = [1/3, 1/3, 1/3]

    # 10a. Discrete-time replicator: RPS interior REPELS (spirals out) where the
    #      continuous center is neutral -- the classic discrete artefact.
    ok &= check("discrete RPS interior repels (vs continuous center)",
                rp.classify_dyn(D["discrete"], rps, center) == "unstable"
                and rp.classify_dyn(D["replicator"], rps, center) == "center")

    # 10b. Discrete time is NOT additive-invariant; continuous time IS.
    x3 = np.array([0.4, 0.3, 0.3])
    disc_moves = np.abs(D["discrete"].advance(x3, rps, 0.0)
                        - D["discrete"].advance(x3, rps + 10, 0.0)).sum()
    cont_moves = np.abs(D["replicator"].advance(x3, rps, 0.0)
                        - D["replicator"].advance(x3, rps + 10, 0.0)).sum()
    ok &= check("discrete time is payoff-shift sensitive; continuous is not",
                disc_moves > 1e-6 and cont_moves < 1e-9)

    # 10c. BNN revives an extinct strategy; the imitative replicator cannot.
    from_rock_bnn = rp.endpoint_dyn(D["bnn"], [1., 0., 0.], rps, t_max=120)
    from_rock_rep = rp.endpoint_dyn(D["replicator"], [1., 0., 0.], rps, t_max=120)
    ok &= check("BNN revives extinct strategies (replicator stays stuck at pure Rock)",
                from_rock_bnn.min() > 0.1 and approx(from_rock_rep, [1., 0., 0.]))

    # 10d. BNN rest points are the Nash equilibria (PD: only all-Defect).
    bnn_pd = rp.rest_points_dyn(D["bnn"], A_pd)
    ok &= check("BNN rest points of PD = {all-Defect} (Nash), not both corners",
                len(bnn_pd) == 1 and approx(bnn_pd[0], [0., 1.]))

    # 10e. Logit: interior QRE that sharpens to the Nash equilibrium as beta grows.
    lo_beta = rp.rest_points_dyn(D["logit"], A_pd, beta=0.5)[0]
    hi_beta = rp.rest_points_dyn(D["logit"], A_pd, beta=20.0)[0]
    ok &= check("logit QRE is interior at low beta, -> Nash (all-Defect) at high beta",
                lo_beta[0] > 0.2 and hi_beta[1] > 0.98)

    # 10f. Replicator-mutator (both forms): mutation keeps every strategy present.
    for key in ("replmut", "replmut_fw"):
        rm_pd = rp.rest_points_dyn(D[key], A_pd, mu=0.05)
        ok &= check(f"{key} keeps cooperators alive (interior rest point)",
                    len(rm_pd) == 1 and 1e-3 < rm_pd[0][0] < 0.5)

    # 10f'. At mu=0 BOTH mutator forms reduce EXACTLY to the replicator; at mu>0
    #       they differ from each other (uniform-additive vs fitness-weighted).
    xr = np.array([0.2, 0.3, 0.5])
    rep0 = rp.replicator_field(xr, rps)
    ok &= check("both mutator forms reduce to replicator at mu=0",
                np.allclose(D["replmut"]._fn(xr, rps, 0.0, mu=0.0), rep0)
                and np.allclose(D["replmut_fw"]._fn(xr, rps, 0.0, mu=0.0), rep0))
    ok &= check("the two mutator forms genuinely differ at mu>0",
                np.abs(D["replmut"]._fn(xr, rps, 0.0, mu=0.1)
                       - D["replmut_fw"]._fn(xr, rps, 0.0, mu=0.1)).sum() > 1e-6)

    # 10f''. The distinguishing behaviour on zero-sum RPS: uniform mutation shifts
    #        the Jacobian by -mu*I (centre -> stable spiral); fitness-weighted
    #        mutation is fitness-coupled and leaves the centre marginal (Re ~ 0).
    ev_u = rp._reduced_eigs(lambda y: D["replmut"]._fn(y, rps, 0.0, mu=0.1), center)
    ev_f = rp._reduced_eigs(lambda y: D["replmut_fw"]._fn(y, rps, 0.0, mu=0.1), center)
    ok &= check("uniform mutation stabilises the RPS centre (Re ~ -mu)",
                abs(ev_u.real.max() + 0.1) < 1e-3)
    ok &= check("fitness-weighted mutation leaves the RPS centre marginal (Re ~ 0)",
                abs(ev_f.real.max()) < 1e-3)

    # 10g. Assortment composes with every dynamic (it changes the payoff input).
    xh = np.array([0.5, 0.5])
    composes = all(np.abs(D[k].motion(xh, A_pd, 0.0) - D[k].motion(xh, A_pd, 0.4)).sum() > 1e-9
                   for k in ("replicator", "discrete", "bnn", "logit", "replmut", "replmut_fw"))
    ok &= check("assortment r composes with all six dynamics", composes)

    # ---------------------------------------------------------------------------
    # 11. EQUILIBRIUM MANIFOLDS (neutral stability across a set)
    # ---------------------------------------------------------------------------
    # Clean single manifold: the S1-S2 edge is a line of rest points, attracting
    # transversally; S3 is not an attractor here. Expect one attracting SET, and
    # basins fully resolved onto it (no fragmentation into "unresolved").
    clean = np.array([[2., 3., 0.], [2., 3., 5.], [0., 0., 1.]])
    objs = rp.rest_objects(D["replicator"], clean)
    sets = [o for o in objs if o["kind"] == "set"]
    ok &= check("manifold game: the S1-S2 edge is found as ONE attracting set",
                len(sets) == 1 and sets[0]["type"] == "set-attract")
    bm = rp.estimate_basins_dyn(D["replicator"], clean, n_samples=300, seed=0)
    ok &= check("manifold game: basins fully resolved (no fragmentation)",
                bm["unresolved"] < 0.02 and len(bm["attractors"]) == 1)

    # Duplicate-strategy game: bistable between the S3 point attractor and the
    # S1-S2 line attractor, separated by a repelling set. Both basins found.
    dup = np.array([[1., 1., 0.], [1., 1., 0.], [0., 0., 1.]])
    dobjs = rp.rest_objects(D["replicator"], dup)
    has_attract_set = any(o["kind"] == "set" and o["type"] == "set-attract" for o in dobjs)
    has_point_attractor = any(o["kind"] == "point" and o["type"] == "stable" for o in dobjs)
    ok &= check("bistable manifold game: an attracting SET and a point attractor coexist",
                has_attract_set and has_point_attractor)
    bd = rp.estimate_basins_dyn(D["replicator"], dup, n_samples=300, seed=0)
    ok &= check("bistable manifold game: two basins, fully resolved",
                len(bd["attractors"]) == 2 and bd["unresolved"] < 0.05)

    # ---------------------------------------------------------------------------
    # 12. TWO-POPULATION dynamics (all dynamics on the unit square)
    # ---------------------------------------------------------------------------
    Abos, Bbos = rp.GAMES_BIMATRIX["Battle of the Sexes"]
    Amp, Bmp = rp.GAMES_BIMATRIX["Matching Pennies"]

    def interior(pts):
        return [(z, t) for z, t in pts if 0.05 < z[0] < 0.95 and 0.05 < z[1] < 0.95]

    # BoS: two coordinated corner attractors + an interior saddle (replicator).
    bos = rp.bimatrix_rest_points(D["replicator"], Abos, Bbos)
    corner_stable = sum(1 for z, t in bos if t == "stable" and (z.min() > 0.99 or z.max() < 0.01))
    ok &= check("BoS (2-pop replicator): two coordinated corners are stable",
                corner_stable == 2)
    ok &= check("BoS (2-pop replicator): interior mixed point is a saddle",
                len(interior(bos)) == 1 and interior(bos)[0][1] == "saddle")

    # Matching Pennies: interior (1/2,1/2) is a CENTER under replicator, but mutation
    # stabilises it (replicator-mutator) -- the two-population echo of RPS.
    mp_rep = interior(rp.bimatrix_rest_points(D["replicator"], Amp, Bmp))
    mp_rm = interior(rp.bimatrix_rest_points(D["replmut"], Amp, Bmp, mu=0.05))
    ok &= check("Matching Pennies (2-pop): interior is a center under replicator",
                len(mp_rep) == 1 and mp_rep[0][1] == "center")
    ok &= check("Matching Pennies (2-pop): mutation stabilises the interior point",
                len(mp_rm) == 1 and mp_rm[0][1] == "stable")

    # The dynamic actually changes the two-population motion.
    m_rep = rp.bimatrix_motion(D["replicator"], 0.6, 0.4, Abos, Bbos)
    m_log = rp.bimatrix_motion(D["logit"], 0.6, 0.4, Abos, Bbos, beta=3.0)
    ok &= check("two-population motion depends on the chosen dynamic",
                abs(m_rep[0] - m_log[0]) > 1e-6)

    # BNN's two-population rest points are the Nash equilibria (dominance -> 1).
    A_dom = np.array([[3., 0.], [5., 1.]])
    bnn_dom = rp.bimatrix_rest_points(D["bnn"], A_dom, A_dom)
    ok &= check("BNN (2-pop) rest points of a dominance game = the single Nash",
                len(bnn_dom) == 1 and approx(bnn_dom[0][0], [0., 0.]))

    # ---------------------------------------------------------------------------
    # 13. DEGENERATE / FULLY-NEUTRAL edge cases (regressions)
    # ---------------------------------------------------------------------------
    # A fully-neutral game (all payoffs equal) is one neutral set, NOT an attractor,
    # and must not crash the type lookups. (Regression: it was mislabelled set-mixed.)
    neutral = np.ones((3, 3))
    nobjs = rp.rest_objects(D["replicator"], neutral)
    ok &= check("fully-neutral game -> a single 'set-neutral' object",
                len(nobjs) == 1 and nobjs[0]["type"] == "set-neutral")
    nb = rp.estimate_basins_dyn(D["replicator"], neutral, n_samples=100, seed=0)
    ok &= check("fully-neutral game -> no attractors (neutral set has no basin)",
                len(nb["attractors"]) == 0)

    # Every dynamic handles a payoff-degenerate game without crashing.
    dupg = np.array([[1., 1., 0.], [1., 1., 0.], [0., 0., 1.]])
    degen_ok = True
    for k in ("replicator", "discrete", "bnn", "logit", "replmut", "replmut_fw"):
        kw = {"beta": 4.0} if k == "logit" else {"mu": 0.05} if k in ("replmut", "replmut_fw") else {}
        try:
            rp.rest_objects(D[k], dupg, **kw)
        except Exception:
            degen_ok = False
    ok &= check("all six dynamics handle a degenerate game without crashing", degen_ok)

    print("\n" + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
