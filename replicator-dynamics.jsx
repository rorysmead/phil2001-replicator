import { useState } from "react";

// ── palette ──────────────────────────────────────────────────────────────────
const C = {
  bg:      "#0f1117",
  panel:   "#161b27",
  border:  "#252d3f",
  text:    "#e2e8f0",
  muted:   "#64748b",
  accent:  "#38bdf8",
  basin: ["#f97316","#22d3ee","#a78bfa","#34d399","#fb7185","#fbbf24","#c084fc"],
};

// ── replicator dynamics core ──────────────────────────────────────────────────
function scaleMatrix(M) {
  const min = Math.min(...M.flat());
  if (min < 0) return M.map(r => r.map(v => v - min + 1));
  return M.map(r => r.map(v => v + 1e-9));
}

function fitness(x, A) {
  return A.map(row => row.reduce((s, a, j) => s + a * x[j], 0));
}

function step(x, A) {
  const n = x.length;
  const f = fitness(x, A);
  const fbar = x.reduce((s, xi, i) => s + xi * f[i], 0);
  if (fbar <= 0) return [...x];
  let xp = x.map((xi, i) => xi * f[i] / fbar);
  // clamp to simplex (guard against numerical drift)
  xp = xp.map(v => (v < 0 ? 0 : v));
  const sum = xp.reduce((a, b) => a + b, 0);
  return sum > 0 ? xp.map(v => v / sum) : [...x];
}

function iterate(x0, A, steps = 700) {
  let x = [...x0];
  for (let t = 0; t < steps; t++) {
    const xp = step(x, A);
    const diff = xp.reduce((s, v, i) => s + Math.abs(v - x[i]), 0);
    x = xp;
    if (diff < 1e-11) break;
  }
  return x;
}

// ── rest-point finding ─────────────────────────────────────────────────────────
// A rest point of the replicator dynamics: on the support S, all active strategies
// share equal payoff, and no strategy outside S has higher payoff (Nash-like check
// only used for stability, not for existence). We enumerate every face (support
// subset) and solve for the equilibrium on that face.

function combos(n) {
  // all non-empty subsets of {0..n-1}
  const out = [];
  for (let mask = 1; mask < (1 << n); mask++) {
    const s = [];
    for (let i = 0; i < n; i++) if (mask & (1 << i)) s.push(i);
    out.push(s);
  }
  return out;
}

// solve for interior equilibrium on a given support set S
function equilibriumOnSupport(A, S) {
  const k = S.length;
  if (k === 1) {
    const x = A.map(() => 0);
    x[S[0]] = 1;
    return x;
  }
  // Equal-payoff condition among active strategies + normalisation.
  // Build (k) equations: for active i, sum_j A[i][j]*x[j] = lambda; sum x = 1.
  // Unknowns: x[S] (k) and lambda → k+1 unknowns, k+1 equations.
  const sub = S.map(i => S.map(j => A[i][j]));
  // rows: payoff_i - lambda = 0  → [sub_i | -1] · [x;lambda] = 0
  // last row: sum x = 1
  const M = [];
  const b = [];
  for (let r = 0; r < k; r++) {
    M.push([...sub[r], -1]);
    b.push(0);
  }
  M.push([...Array(k).fill(1), 0]);
  b.push(1);
  const sol = solveLinear(M, b);
  if (!sol) return null;
  const xS = sol.slice(0, k);
  if (xS.some(v => v < -1e-7 || v > 1 + 1e-7)) return null; // outside face
  const x = A.map(() => 0);
  S.forEach((idx, r) => { x[idx] = Math.max(0, xS[r]); });
  const sum = x.reduce((a, b) => a + b, 0);
  return sum > 0 ? x.map(v => v / sum) : null;
}

function solveLinear(Min, bin) {
  const n = bin.length;
  const M = Min.map((r, i) => [...r, bin[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++)
      if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-12) return null;
    [M[col], M[piv]] = [M[piv], M[col]];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col] / M[col][col];
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((r, i) => r[n] / r[i]);
}

function sameState(a, b, tol = 1e-3) {
  return a.every((v, i) => Math.abs(v - b[i]) < tol);
}

// drift magnitude of one replicator step at x
function drift(x, A) {
  const nx = step(x, A);
  return nx.reduce((s, v, i) => s + Math.abs(v - x[i]), 0);
}

// Scan the simplex for ALL points where the dynamics rest, then cluster.
// Isolated clusters → point rest points. Elongated clusters → rest-point sets
// (lines/segments), which the algebraic solver misses when the system is
// degenerate (e.g. a whole edge of equilibria).
function findRestPoints(A) {
  const n = A.length;
  const found = [];

  // 1) algebraic equilibria on every face (catches isolated points cleanly)
  for (const S of combos(n)) {
    const eq = equilibriumOnSupport(A, S);
    if (eq && drift(eq, A) < 1e-6 && !found.some(p => sameState(p, eq)))
      found.push(eq);
  }

  // 2) dense grid scan for resting points (catches manifolds)
  const G = n === 2 ? 400 : 90;
  const grid = [];
  if (n === 2) {
    for (let i = 0; i <= G; i++) grid.push([i/G, 1 - i/G]);
  } else {
    for (let i = 0; i <= G; i++)
      for (let j = 0; j <= G - i; j++)
        grid.push([i/G, j/G, (G-i-j)/G]);
  }
  // tighter resting tolerance so we don't pick up merely-slow (non-resting) states
  const resting = grid.filter(x => drift(x, A) < 2e-4);

  // 3) cluster resting points by proximity. Use a generous link distance so a line
  // that runs diagonally across the simplex (sparser neighbours) stays connected.
  const link = 4.5 / G;
  const clusters = [];
  const used = new Array(resting.length).fill(false);
  for (let i = 0; i < resting.length; i++) {
    if (used[i]) continue;
    const stack = [i]; used[i] = true;
    const members = [];
    while (stack.length) {
      const c = stack.pop();
      members.push(resting[c]);
      for (let k = 0; k < resting.length; k++)
        if (!used[k] && dist(resting[c], resting[k]) < link) { used[k] = true; stack.push(k); }
    }
    clusters.push(members);
  }

  // 4) turn clusters into rest objects; merge in algebraic points
  const objs = [];
  for (const m of clusters) {
    let maxSpread = 0;
    const st = Math.max(1, Math.floor(m.length / 20));
    for (let a = 0; a < m.length; a += st)
      for (let b = a + 1; b < m.length; b += st)
        maxSpread = Math.max(maxSpread, dist(m[a], m[b]));
    if (maxSpread > 0.06 && m.length > 4) {
      objs.push({ kind: "set", x: centroid(m), members: m, spread: maxSpread });
    } else {
      // small cluster → an isolated point, but only keep it if it's a genuine
      // equilibrium (matches an algebraic solution or rests very tightly). This
      // discards slow-dynamics false positives that would otherwise litter the plot.
      const c = centroid(m);
      const real = found.some(eq => sameState(eq, c, 0.03)) || drift(c, A) < 2e-5;
      if (real) objs.push({ kind: "point", x: c });
    }
  }
  // ensure algebraic isolated points are represented (in case grid missed them),
  // but not if they already lie on a detected set
  for (const eq of found)
    if (!objs.some(o => o.kind === "set"
          ? o.members.some(mm => sameState(mm, eq, 0.05))
          : sameState(o.x, eq, 0.03)))
      objs.push({ kind: "point", x: eq });

  return objs;
}

function centroid(pts) {
  const n = pts[0].length;
  const c = Array(n).fill(0);
  pts.forEach(p => p.forEach((v, i) => c[i] += v));
  return c.map(v => v / pts.length);
}

// ── stability classification ───────────────────────────────────────────────────
// Classify the transverse behaviour at a single state: perturb in every admissible
// simplex direction, iterate, and measure whether the orbit returns toward the
// state (attracting), leaves it (repelling), or stays nearby without converging.
function classifyPoint(A, probe) {
  const n = probe.length;
  const eps = 1e-3;
  let anyEscape = false, allReturn = true, anyNeutral = false;

  const dirs = [];
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      if (i !== j) { const d = Array(n).fill(0); d[i] = 1; d[j] = -1; dirs.push(d); }

  for (const d of dirs) {
    let x = probe.map((v, i) => v + eps * d[i]);
    if (x.some(v => v < -1e-9)) continue;
    const sum = x.reduce((a, b) => a + b, 0);
    x = x.map(v => v / sum);
    const d0 = dist(x, probe);
    // single pass: track both max excursion and final distance
    let xt = [...x], maxDist = d0;
    for (let t = 0; t < 250; t++) { xt = step(xt, A); maxDist = Math.max(maxDist, dist(xt, probe)); }
    const dEnd = dist(xt, probe);

    if (dEnd > d0 * 2.5) { anyEscape = true; allReturn = false; }
    else if (dEnd < d0 * 0.3) { /* attracting in this dir */ }
    else if (maxDist < d0 * 2.0 && dEnd > d0 * 0.5) { anyNeutral = true; allReturn = false; }
    else { allReturn = false; }
  }
  if (anyEscape) return "unstable";
  if (allReturn) return "stable";
  if (anyNeutral) return "lyapunov";
  return "stable";
}

// Classify a rest object. For a SET, classify EVERY member transversally — a line
// of equilibria can attract along part of its length and repel along the rest
// (e.g. PD with punishment). We tag each member attracting/repelling and report
// the set as mixed/attracting/repelling.
function classifyStability(A, p) {
  if (p.kind !== "set") return classifyPoint(A, p.x);

  // Classifying every member is too costly for large sets. Probe a downsampled
  // subset (~24 points) then assign each member the type of its nearest probe.
  const M = p.members.length;
  const maxProbes = 24;
  const stride = Math.max(1, Math.floor(M / maxProbes));
  const probeIdx = [];
  for (let i = 0; i < M; i += stride) probeIdx.push(i);
  if (probeIdx[probeIdx.length - 1] !== M - 1) probeIdx.push(M - 1);

  const probeType = probeIdx.map(i => {
    const c = classifyPoint(A, p.members[i]);
    return c === "unstable" ? "repel" : "attract";
  });

  p.memberTypes = p.members.map((m, i) => {
    let best = 0, bd = Infinity;
    probeIdx.forEach((pi, k) => {
      const d = dist(m, p.members[pi]);
      if (d < bd) { bd = d; best = k; }
    });
    return probeType[best];
  });

  const nAttract = p.memberTypes.filter(t => t === "attract").length;
  const nRepel = p.memberTypes.length - nAttract;
  if (nRepel === 0) return "set-attract";
  if (nAttract === 0) return "set-repel";
  return "set-mixed";
}

function dist(a, b) {
  return Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0));
}

// ── basin computation (by simulating a grid to nearest stable rest point) ───────
function computeBasins(A, restPoints, samples) {
  const attractors = restPoints
    .map((p, idx) => ({ ...p, idx }))
    .filter(p =>
      p.type === "stable" || p.type === "lyapunov" ||
      p.type === "set-attract" || p.type === "set-mixed");
  if (!attractors.length) return { basinOf: samples.map(() => -1), sizes: [] };

  // for a set, only the ATTRACTING members can capture orbits
  const attractMembers = a => a.kind === "set"
    ? a.members.filter((m, i) => a.memberTypes[i] === "attract")
    : null;

  // distance from a state to an attractor (point, or attracting part of a set)
  const distTo = (a, x) => {
    if (a.kind === "set") {
      const mem = attractMembers(a);
      return mem.length ? Math.min(...mem.map(m => dist(x, m))) : Infinity;
    }
    return dist(x, a.x);
  };

  const counts = attractors.map(() => 0);
  const basinOf = samples.map(x0 => {
    const end = iterate(x0, A);
    let best = -1, bd = Infinity;
    attractors.forEach((a, ai) => {
      const d = distTo(a, end);
      if (d < bd) { bd = d; best = ai; }
    });
    if (bd < 0.08) { counts[best]++; return attractors[best].idx; }
    return -1;
  });

  const total = samples.length;
  const sizes = attractors.map((a, ai) => ({
    idx: a.idx, x: a.x, kind: a.kind, members: a.members, memberTypes: a.memberTypes,
    pct: (counts[ai] / total) * 100,
  }));
  return { basinOf, sizes };
}

// ── labels ──────────────────────────────────────────────────────────────────
function restLabel(x) {
  const tol = 1e-3;
  const active = x.map((v, i) => (v > tol ? i : -1)).filter(i => i >= 0);
  if (active.length === 1) return `S${active[0] + 1} (pure)`;
  if (active.length === x.length)
    return `interior (${x.map(v => v.toFixed(2)).join(", ")})`;
  return `${active.map(i => `S${i + 1}`).join("–")} mix (${active.map(i => x[i].toFixed(2)).join(", ")})`;
}

// ── 2×2 portrait ────────────────────────────────────────────────────────────
function Portrait2x2({ A, restPoints, basins }) {
  const W = 342, H = 63, PAD = 25;
  const lineLen = W - 2 * PAD;
  const toX = p => PAD + p * lineLen;       // p = x[0]
  const N = 240;

  // flow direction: sign of (x0' - x0)
  const arrows = [];
  for (let i = 1; i < N; i += 12) {
    const p = i / N;
    const x = [p, 1 - p];
    const dp = step(x, A)[0] - p;
    if (Math.abs(dp) < 1e-6) continue;
    arrows.push({ p, dir: dp > 0 ? 1 : -1 });
  }

  // basin shading
  const bandColour = {};
  basins.sizes.forEach((b, i) => { bandColour[b.idx] = C.basin[i % C.basin.length]; });
  const grid = [];
  for (let i = 0; i <= N; i++) {
    const p = i / N;
    const end = iterate([p, 1 - p], A);
    let nearest = -1, bd = Infinity;
    basins.sizes.forEach(b => {
      const d = Math.abs(end[0] - b.x[0]);
      if (d < bd) { bd = d; nearest = b.idx; }
    });
    grid.push(bd < 0.08 ? nearest : -1);
  }

  return (
    <svg width={W} height={H} style={{ display: "block", margin: "0 auto", maxWidth: "100%" }}>
      {grid.map((g, i) => g < 0 ? null : (
        <rect key={i} x={toX(i / N)} y={H / 2 - 7}
          width={Math.max(1, lineLen / N + 1)} height={14}
          fill={bandColour[g]} opacity={0.22} />
      ))}
      <line x1={PAD} y1={H / 2} x2={W - PAD} y2={H / 2} stroke={C.border} strokeWidth={2} />
      {arrows.map((a, i) => {
        const cx = toX(a.p);
        return (
          <path key={i}
            d={a.dir > 0 ? `M${cx - 4},${H/2 - 4} L${cx + 4},${H/2} L${cx - 4},${H/2 + 4}`
                          : `M${cx + 4},${H/2 - 4} L${cx - 4},${H/2} L${cx + 4},${H/2 + 4}`}
            fill="none" stroke={C.muted} strokeWidth={1.4} />
        );
      })}
      {restPoints.map((rp, i) => {
        if (rp.kind === "set") {
          return <RestSet key={i} members={rp.members} memberTypes={rp.memberTypes}
            project={x => ({ x: toX(x[0]), y: H / 2 })}
            colour={bandColour[i] || C.text} />;
        }
        return <RestDot key={i} cx={toX(rp.x[0])} cy={H / 2} type={rp.type}
          colour={bandColour[i] || C.text} />;
      })}
      <text x={PAD - 8} y={H/2 + 4} fill={C.muted} fontSize={11} textAnchor="end" fontFamily="monospace">S₁</text>
      <text x={W - PAD + 8} y={H/2 + 4} fill={C.muted} fontSize={11} textAnchor="start" fontFamily="monospace">S₂</text>
    </svg>
  );
}

// ── 3×3 portrait ────────────────────────────────────────────────────────────
function bary(b, cx, cy, r) {
  const v1 = { x: cx - r * 0.866, y: cy + r * 0.5 };  // S1 left
  const v2 = { x: cx,             y: cy - r };          // S2 top
  const v3 = { x: cx + r * 0.866, y: cy + r * 0.5 };  // S3 right
  return {
    x: b[0]*v1.x + b[1]*v2.x + b[2]*v3.x,
    y: b[0]*v1.y + b[1]*v2.y + b[2]*v3.y,
  };
}

function Portrait3x3({ A, restPoints, basins, samples, basinOf }) {
  const S = 340;
  const cx = S/2, cy = S/2 + 6, r = 140;
  const v1 = bary([1,0,0], cx, cy, r);
  const v2 = bary([0,1,0], cx, cy, r);
  const v3 = bary([0,0,1], cx, cy, r);

  const bandColour = {};
  basins.sizes.forEach((b, i) => { bandColour[b.idx] = C.basin[i % C.basin.length]; });

  // trajectory streamlines from a grid of seed points
  const SEED = 9;            // seeds per simplex edge
  const TSTEPS = 600;        // integration length
  const trajectories = [];
  for (let i = 1; i < SEED; i++)
    for (let j = 1; j < SEED - i; j++) {
      const k = SEED - i - j;
      if (k < 1) continue;
      const x0 = [i/SEED, j/SEED, k/SEED];
      // integrate, recording points; sub-sample for a smooth but light path
      let x = [...x0];
      const pts = [bary(x, cx, cy, r)];
      let moved = 0;
      for (let t = 0; t < TSTEPS; t++) {
        const nx = step(x, A);
        const d = nx.reduce((s, v, q) => s + Math.abs(v - x[q]), 0);
        x = nx;
        moved += d;
        // record a point every time the orbit has travelled enough
        if (moved > 0.012) { pts.push(bary(x, cx, cy, r)); moved = 0; }
        if (d < 1e-9) break;
      }
      pts.push(bary(x, cx, cy, r));
      // colour by destination basin
      let col = C.muted, bd = Infinity;
      basins.sizes.forEach(b => {
        let dd;
        if (b.kind === "set") {
          const mem = b.members.filter((m, mi) => b.memberTypes[mi] === "attract");
          dd = mem.length ? Math.min(...mem.map(m => dist(x, m))) : Infinity;
        } else dd = dist(x, b.x);
        if (dd < bd) { bd = dd; col = bd < 0.08 ? bandColour[b.idx] : C.muted; }
      });
      if (pts.length > 2) trajectories.push({ pts, col });
    }

  // smooth polyline → SVG path
  const pathD = pts => pts.map((p, i) =>
    `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  // a direction arrowhead at ~35% along each trajectory
  const midArrow = pts => {
    if (pts.length < 4) return null;
    const i = Math.max(1, Math.floor(pts.length * 0.35));
    const a = pts[i - 1], b = pts[i];
    const ang = Math.atan2(b.y - a.y, b.x - a.x);
    return { x: b.x, y: b.y, ang: (ang * 180) / Math.PI };
  };

  return (
    <svg width={S} height={S} style={{ display: "block", margin: "0 auto", maxWidth: "100%" }}>
      <defs>
        <clipPath id="clip3">
          <polygon points={`${v1.x},${v1.y} ${v2.x},${v2.y} ${v3.x},${v3.y}`} />
        </clipPath>
      </defs>

      <g clipPath="url(#clip3)">
        {samples.map((x0, idx) => {
          const b = basinOf[idx];
          if (b < 0) return null;
          const p = bary(x0, cx, cy, r);
          return <circle key={idx} cx={p.x} cy={p.y} r={3.2} fill={bandColour[b]} opacity={0.18} />;
        })}
      </g>

      <polygon points={`${v1.x},${v1.y} ${v2.x},${v2.y} ${v3.x},${v3.y}`}
        fill="none" stroke={C.border} strokeWidth={2} />

      {/* trajectory streamlines */}
      <g clipPath="url(#clip3)">
        {trajectories.map((tr, i) => (
          <path key={i} d={pathD(tr.pts)} fill="none"
            stroke={tr.col} strokeWidth={1.3} opacity={0.7}
            strokeLinecap="round" strokeLinejoin="round" />
        ))}
        {trajectories.map((tr, i) => {
          const a = midArrow(tr.pts);
          if (!a) return null;
          return (
            <path key={`a${i}`}
              d="M-3,-2.4 L2.4,0 L-3,2.4 Z"
              fill={tr.col} opacity={0.85}
              transform={`translate(${a.x},${a.y}) rotate(${a.ang})`} />
          );
        })}
      </g>

      {restPoints.map((rp, i) => {
        if (rp.kind === "set") {
          return <RestSet key={i} members={rp.members} memberTypes={rp.memberTypes}
            project={x => bary(x, cx, cy, r)}
            colour={bandColour[i] || C.text} />;
        }
        const p = bary(rp.x, cx, cy, r);
        return <RestDot key={i} cx={p.x} cy={p.y} type={rp.type}
          colour={bandColour[i] || C.text} />;
      })}

      <text x={v1.x - 16} y={v1.y + 6} fill={C.text} fontSize={13} fontFamily="monospace" fontWeight="bold">S₁</text>
      <text x={v2.x - 7}  y={v2.y - 8} fill={C.text} fontSize={13} fontFamily="monospace" fontWeight="bold">S₂</text>
      <text x={v3.x + 4}  y={v3.y + 6} fill={C.text} fontSize={13} fontFamily="monospace" fontWeight="bold">S₃</text>
    </svg>
  );
}

// rest-point SET: dots spaced along the set, each coloured by the LOCAL transverse
// stability at that position — solid = attracting, open = repelling. A single set
// can be partly attracting and partly repelling (e.g. PD with punishment).
function RestSet({ members, memberTypes, project, colour }) {
  const proj = members.map((m, i) => ({ p: project(m), type: memberTypes ? memberTypes[i] : "attract" }));
  // dominant axis via farthest pair
  let a = 0, b = 0, far = 0;
  const stride = Math.max(1, Math.floor(proj.length / 30));
  for (let i = 0; i < proj.length; i += stride)
    for (let j = i + 1; j < proj.length; j += stride) {
      const d = (proj[i].p.x - proj[j].p.x) ** 2 + (proj[i].p.y - proj[j].p.y) ** 2;
      if (d > far) { far = d; a = i; b = j; }
    }
  const ax = proj[b].p.x - proj[a].p.x, ay = proj[b].p.y - proj[a].p.y;
  const axis = m => (m.p.x - proj[a].p.x) * ax + (m.p.y - proj[a].p.y) * ay;
  const ordered = [...proj].sort((p, q) => axis(p) - axis(q));
  const p0 = ordered[0].p, p1 = ordered[ordered.length - 1].p;

  // sample evenly along the line; pick the nearest member's type for each sample
  const nDots = 9;
  const dots = [];
  for (let i = 0; i < nDots; i++) {
    const t = i / (nDots - 1);
    const x = p0.x + t * (p1.x - p0.x), y = p0.y + t * (p1.y - p0.y);
    let best = ordered[0], bd = Infinity;
    ordered.forEach(o => {
      const dd = (o.p.x - x) ** 2 + (o.p.y - y) ** 2;
      if (dd < bd) { bd = dd; best = o; }
    });
    dots.push({ x, y, repel: best.type === "repel" });
  }

  return (
    <g>
      <line x1={p0.x} y1={p0.y} x2={p1.x} y2={p1.y}
        stroke={colour} strokeWidth={1.6} opacity={0.45} />
      {dots.map((d, i) => (
        <circle key={i} cx={d.x} cy={d.y} r={3.9}
          fill={d.repel ? C.bg : colour}
          stroke={colour} strokeWidth={d.repel ? 1.8 : 1} />
      ))}
    </g>
  );
}

// ── rest-point glyphs ──────────────────────────────────────────────────────────
// stable = solid; unstable = open; lyapunov = ring of spaced solid dots
function RestDot({ cx, cy, type, colour }) {
  if (type === "lyapunov") {
    const ring = [];
    const R = 9, n = 6;
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2;
      ring.push(<circle key={i} cx={cx + R * Math.cos(a)} cy={cy + R * Math.sin(a)}
        r={2.4} fill={colour} />);
    }
    return <g>{ring}<circle cx={cx} cy={cy} r={R} fill="none" stroke={colour}
      strokeWidth={1} strokeDasharray="2 3" opacity={0.5} /></g>;
  }
  if (type === "unstable")
    return <circle cx={cx} cy={cy} r={6} fill={C.bg} stroke={colour} strokeWidth={2.2} />;
  return <circle cx={cx} cy={cy} r={6.5} fill={colour} stroke={C.bg} strokeWidth={1.5} />;
}

// ── basin report ──────────────────────────────────────────────────────────────
function Report({ restPoints, basins }) {
  const colourFor = {};
  basins.sizes.forEach((b, i) => { colourFor[b.idx] = C.basin[i % C.basin.length]; });

  return (
    <div style={{ marginTop: 18 }}>
      <Legend />
      <div style={{ marginTop: 14, padding: "14px 16px", background: C.bg,
        borderRadius: 8, border: `1px solid ${C.border}` }}>
        <div style={{ fontSize: 10, letterSpacing: ".1em", color: C.muted,
          textTransform: "uppercase", marginBottom: 10 }}>Rest points & basins</div>
        {restPoints.map((rp, i) => {
          const basin = basins.sizes.find(b => b.idx === i);
          const col = colourFor[i] || C.muted;
          return (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 7 }}>
              <span style={{ width: 14, textAlign: "center", flexShrink: 0 }}>
                {rp.kind === "set"
                  ? (rp.type === "set-repel" ? "⊝" : rp.type === "set-mixed" ? "◐" : "⊙")
                  : rp.type === "unstable" ? "○" : rp.type === "lyapunov" ? "◌" : "●"}
              </span>
              <span style={{ flex: 1, fontSize: 12.5, color: C.text, fontFamily: "monospace" }}>
                {rp.kind === "set"
                  ? `rest-point line (${rp.type === "set-mixed"
                      ? "partly attracting / repelling"
                      : rp.type === "set-repel" ? "repelling" : "attracting"})`
                  : restLabel(rp.x)}
                <span style={{ color: C.muted, marginLeft: 6 }}>· {rp.type.replace("set-", "")}</span>
              </span>
              {basin && (
                <>
                  <span style={{ fontSize: 12.5, color: C.accent, fontFamily: "monospace",
                    minWidth: 48, textAlign: "right" }}>{basin.pct.toFixed(1)}%</span>
                  <span style={{ width: 70, height: 5, background: C.border, borderRadius: 3 }}>
                    <span style={{ display: "block", width: `${basin.pct}%`, height: "100%",
                      background: col, borderRadius: 3 }} />
                  </span>
                </>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Legend() {
  const items = [
    ["●", "stable / attracting"],
    ["○", "unstable / repelling"],
    ["◌", "Lyapunov stable (neutral)"],
    ["●—○", "rest-point line (solid=attract, open=repel)"],
  ];
  return (
    <div style={{ display: "flex", gap: 16, justifyContent: "center", flexWrap: "wrap" }}>
      {items.map(([g, l]) => (
        <span key={l} style={{ fontSize: 11, color: C.muted, display: "flex",
          alignItems: "center", gap: 5 }}>
          <span style={{ fontSize: 13 }}>{g}</span>{l}
        </span>
      ))}
    </div>
  );
}

// ── presets ──────────────────────────────────────────────────────────────────
const PRESETS_2x2 = {
  "Prisoner's Dilemma": [[3,0],[5,1]],
  "Hawk-Dove":          [[0,3],[1,2]],
  "Stag Hunt":          [[3,0],[1,1]],
  "Coordination":       [[2,0],[0,2]],
};
const PRESETS_3x3 = {
  "Rock-Paper-Scissors": [[0,-1,1],[1,0,-1],[-1,1,0]],
  "RPS (attracting)":    [[0,-1,2],[2,0,-1],[-1,2,0]],
  "Dominance":           [[3,0,0],[2,2,0],[1,1,1]],
  "PD w/ Punishment":    [[4,1,4],[5,2,2],[4,0,4]],
};

const defaultMatrix = n =>
  Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 2 : 1)));

// ── main app ──────────────────────────────────────────────────────────────────
export default function App() {
  const [gameSize, setGameSize] = useState(2);
  const [matrix, setMatrix] = useState(defaultMatrix(2));
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);

  const setSize = n => { setGameSize(n); setMatrix(defaultMatrix(n)); setResult(null); };
  const setCell = (i, j, v) => {
    setMatrix(p => { const m = p.map(r => [...r]); m[i][j] = v; return m; });
    setResult(null);
  };
  const loadPreset = name => {
    const src = gameSize === 2 ? PRESETS_2x2 : PRESETS_3x3;
    setMatrix(src[name].map(r => [...r])); setResult(null);
  };

  const run = () => {
    setBusy(true);
    setTimeout(() => {
      const A = scaleMatrix(matrix.map(r => r.map(v => parseFloat(v) || 0)));
      const rps = findRestPoints(A).map(p => {
        const type = classifyStability(A, p); // mutates p.memberTypes for sets
        return { ...p, type };
      });

      // sample grid for basins
      let samples = [];
      if (gameSize === 2) {
        for (let i = 0; i <= 240; i++) samples.push([i/240, 1 - i/240]);
      } else {
        const N = 44;
        for (let i = 0; i <= N; i++)
          for (let j = 0; j <= N - i; j++)
            samples.push([i/N, j/N, (N-i-j)/N]);
      }
      const basins = computeBasins(A, rps, samples);
      const basinOf = gameSize === 3 ? basins.basinOf : [];

      setResult({ A, rps, basins, samples, basinOf });
      setBusy(false);
    }, 30);
  };

  const presets = gameSize === 2 ? PRESETS_2x2 : PRESETS_3x3;
  const labels = ["S₁","S₂","S₃"].slice(0, gameSize);

  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: C.text,
      fontFamily: "'Inter', system-ui, sans-serif", padding: "24px 16px 48px" }}>
      <div style={{ maxWidth: 460, margin: "0 auto" }}>
        <div style={{ marginBottom: 24 }}>
          <div style={{ fontSize: 11, letterSpacing: ".15em", color: C.accent,
            textTransform: "uppercase", marginBottom: 6 }}>Evolutionary Game Theory</div>
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, letterSpacing: "-.02em" }}>
            Replicator Dynamics</h1>
          <p style={{ margin: "6px 0 0", fontSize: 13, color: C.muted }}>
            Discrete-time phase portraits · rest points · basins of attraction</p>
        </div>

        <div style={{ display: "flex", gap: 8, marginBottom: 18 }}>
          {[2,3].map(n => (
            <button key={n} onClick={() => setSize(n)} style={{
              padding: "8px 20px", borderRadius: 6, fontSize: 13, fontWeight: 600,
              cursor: "pointer", border: "none",
              background: gameSize === n ? C.accent : C.panel,
              color: gameSize === n ? C.bg : C.muted }}>{n}×{n}</button>
          ))}
        </div>

        <div style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 10, color: C.muted, marginBottom: 6,
            letterSpacing: ".08em", textTransform: "uppercase" }}>Presets</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {Object.keys(presets).map(name => (
              <button key={name} onClick={() => loadPreset(name)} style={{
                padding: "5px 11px", borderRadius: 5, fontSize: 12, background: "transparent",
                border: `1px solid ${C.border}`, color: C.muted, cursor: "pointer" }}>{name}</button>
            ))}
          </div>
        </div>

        <div style={{ background: C.panel, border: `1px solid ${C.border}`,
          borderRadius: 10, padding: "16px 18px", marginBottom: 14 }}>
          <div style={{ fontSize: 10, color: C.muted, marginBottom: 10,
            letterSpacing: ".08em", textTransform: "uppercase" }}>Payoff matrix (row = focal)</div>
          <table style={{ borderCollapse: "collapse", margin: "0 auto" }}>
            <thead><tr><td style={{ width: 30 }} />
              {labels.map(l => <th key={l} style={{ padding: "2px 6px", color: C.muted,
                fontSize: 12, fontFamily: "monospace", fontWeight: 500 }}>{l}</th>)}
            </tr></thead>
            <tbody>
              {labels.map((rl, i) => (
                <tr key={i}>
                  <td style={{ color: C.muted, fontSize: 12, fontFamily: "monospace",
                    paddingRight: 6, textAlign: "right" }}>{rl}</td>
                  {labels.map((_, j) => (
                    <td key={j} style={{ padding: 3 }}>
                      <input type="number" step="0.1" value={matrix[i][j]}
                        onChange={e => setCell(i, j, e.target.value)} style={{
                          width: 60, padding: "6px", background: C.bg,
                          border: `1px solid ${C.border}`, borderRadius: 5, color: C.text,
                          fontSize: 14, fontFamily: "monospace", textAlign: "center", outline: "none" }} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ fontSize: 11, color: C.muted, marginTop: 9, textAlign: "center" }}>
            Negative payoffs shift up automatically</div>
        </div>

        <button onClick={run} disabled={busy} style={{
          width: "100%", padding: "11px", borderRadius: 8, background: C.accent,
          color: C.bg, fontSize: 14, fontWeight: 700, border: "none",
          cursor: busy ? "default" : "pointer", opacity: busy ? 0.6 : 1 }}>
          {busy ? "Computing…" : "Compute phase portrait"}</button>

        {result && (
          <div style={{ marginTop: 24, background: C.panel, border: `1px solid ${C.border}`,
            borderRadius: 10, padding: "18px 14px" }}>
            <div style={{ fontSize: 10, color: C.muted, marginBottom: 14,
              letterSpacing: ".08em", textTransform: "uppercase", textAlign: "center" }}>Phase portrait</div>
            {gameSize === 2
              ? <Portrait2x2 A={result.A} restPoints={result.rps} basins={result.basins} />
              : <Portrait3x3 A={result.A} restPoints={result.rps} basins={result.basins}
                  samples={result.samples} basinOf={result.basinOf} />}
            <Report restPoints={result.rps} basins={result.basins} />
          </div>
        )}
      </div>
    </div>
  );
}
