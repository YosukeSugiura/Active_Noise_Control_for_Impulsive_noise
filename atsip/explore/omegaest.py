#!/usr/bin/env python3
"""Online estimate of the schedule omega_i from the error decay slope (replaces geosched's ramp).

Normaliser (as geosched): Lambda_i(k) = (1-w_i) Srr(k) + w_i Srr(k)^a Pbar^(1-a) + eps, a = 0.5,
step mu_i = mu (1 + (m0-1)(1-w_i)). Framing E_M throughout.

Estimator, per block i (causal, uses the error up to block i):
  See_i(k): EMA of |E_M(k)|^2 with the Srr smoothing (Shared.See).
  J_i = mean over bins 1..511 of log(See_i(k)/Srr_i(k)).
  g_i = (J_{i-L} - J_i)/L, smoothed by EMA (factor beta).
  w_i = clip(1 - g_i/g0, 0, 1), w_i = 0 for i < L. monotone: w_i = max(w_i, w_{i-1}).
  g0 mode A: g0 = raw decay over the first L blocks (measured at i = L, then frozen).
  g0 mode B: g0 = c mu, c grid-searched once on a145 (reference config L=64, beta=0.97, free, m0=3),
             then frozen for every source.

Grid: L in {32,64,128} x beta in {0.9,0.97} x gmode {A,B} x {monotone,free} x m0 in {1,2,3} (72 configs).
mu per config: best six-source white whole mean (as geosched). Baselines B0, T3, W0.5 and the manual
ramp (a=0.5, T=4, m0=3) are rerun here on the same ten sources.

Robustness: mid-run abrupt change at 15 s (primary path replaced by 0.8x scaled, 8-sample delayed copy)
on three white sources. Only the output d changes; the reference path and the norm inputs do not.

Reuses geosched.py (sim_sched, D.simulate, metrics conventions), dissect.py (data, MUS, sources),
explore.py (Shared, metrics), oracle.py (anr_curve). Memory: one source per worker, no all-pairs work.

Usage:  python3 atsip/explore/omegaest.py
Writes  atsip/explore/out/omegaest_*.{csv,json,png} and atsip/explore/RESULTS_omegaest.md.
"""
import os
import sys
import csv
import json
import time
import warnings
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dissect as D  # noqa: E402
import explore as X  # noqa: E402
import mixlam as ML  # noqa: E402
import geosched as G  # noqa: E402
import oracle as O  # noqa: E402

FS, M, N, B, NB = D.FS, D.M, D.N, D.B, D.NB
OUT = D.OUT
MUS = D.MUS
SIX, TWO, COL, ALLTAGS, KIND = D.SIX, D.TWO, D.COL, D.ALLTAGS, D.KIND
A_NORM = 0.5
L_LIST, BETA_LIST, GMODES, MONOS, M0_LIST = [32, 64, 128], [0.9, 0.97], ["A", "B"], ["mono", "free"], [1, 2, 3]
EST_CFGS = [(Lw, beta, gm, mo, m0) for Lw in L_LIST for beta in BETA_LIST for gm in GMODES
            for mo in MONOS for m0 in M0_LIST]
C_GRID = [float(10 ** x) for x in np.arange(-3.0, 1.01, 0.5)]
REF_CFG = (64, 0.97, "B", "free", 3)
MAN_CFG = ("S", 0.5, 4, 3)
MAN_LAB = G.slab(0.5, 4, 3)
BASES = ["B0", "T3", "W0.5"]
TSW = 15.0                       # abrupt change time [s]
ABRUPT_SRCS = SIX[:3]
TRAJ_SRCS = ["a145", "cms145_s0"]
TOL_DB = 1.0

# set in main() before the phase-2 pool (inherited by fork)
C_FIT = None
BEST_CFG = None
BEST_MU = None
MU_REF = {}


def lab_of(cfg):
    Lw, beta, gm, mo, m0 = cfg
    return f"E_L{Lw}_b{beta}_{gm}_{mo}_m{m0}"


# ----------------------------------------------------------------------------
# Estimator simulation (same loop as geosched.sim_sched, w_i from the decay estimate)
# ----------------------------------------------------------------------------
def sim_est(data, Rall, cfg, mu, c):
    Lw, beta, gmode, mono, m0 = cfg
    d, xq, hS = data["d"], data["xq"], data["hS"]
    Ls = len(d)
    nb = Ls // B
    swv = sliding_window_view(xq, M)
    w = np.zeros(M)
    zi = np.zeros(len(hS) - 1)
    e_all = np.zeros(Ls)
    shared = X.Shared(data["S2"])
    Jh = np.zeros(nb)
    wtraj = np.zeros(nb)
    g0 = None if gmode == "A" else c * mu
    gs, w_prev = None, 0.0
    diverged = False
    for i in range(nb):
        s0 = i * B
        Xb = swv[s0 + N - M + 1: s0 + N - M + 1 + B][:, ::-1]
        y = Xb @ w
        ys, zi = signal.lfilter(hS, [1.0], y, zi=zi)
        e_all[s0:s0 + B] = d[s0:s0 + B] + ys
        E = ML.frame_E(e_all, s0, M)
        s = shared.step(Rall[i], E)
        Srr, Pbar, eps = s["Srr"], s["Pbar"], s["eps"]
        See = shared.See
        Jh[i] = np.mean(np.log(np.maximum(See[1:512], 1e-30) / np.maximum(Srr[1:512], 1e-30)))
        if i >= Lw:
            graw = (Jh[i - Lw] - Jh[i]) / Lw
            if gmode == "A" and g0 is None:
                g0 = graw                       # initial decay rate over the first L blocks, then frozen
            gs = graw if gs is None else beta * gs + (1 - beta) * graw
            g0e = max(g0, 1e-6)                 # guard: no initial decay -> w reacts to sign of gs
            wr = min(1.0, max(0.0, 1.0 - gs / g0e))
            w_i = max(wr, w_prev) if mono == "mono" else wr
        else:
            w_i = 0.0
        w_prev = w_i
        wtraj[i] = w_i
        Lam = (1 - w_i) * Srr + w_i * Srr ** A_NORM * Pbar ** (1 - A_NORM) + eps
        mu_i = mu * (1 + (m0 - 1) * (1 - w_i))
        g = np.fft.irfft(np.conj(Rall[i]) * E / Lam, N)[:M]
        w = w - 2 * mu_i * B / M * g
        if not np.isfinite(w).all():
            e_all[s0 + B:] = np.nan
            diverged = True
            break
    return dict(e=e_all, w=w, diverged=diverged, wtraj=wtraj[:i + 1])


def score(data, sim, lab, mu):
    """Same metric and divergence rule as geosched.one_run (task rule)."""
    m = X.metrics(sim["e"], sim["w"], data["d"], data["w_opt"])
    w_fin = bool(np.isfinite(sim["w"]).all())
    div = (not w_fin) or (not np.isfinite(m["last10"])) or (not np.isfinite(m["whole"])) or m["last10"] > 0.0
    nanv = float("nan")
    return dict(label=lab, mu=mu, whole=nanv if div else float(m["whole"]),
                first5=nanv if div else float(m["first5"]), last10=nanv if div else float(m["last10"]),
                msd=nanv if div else float(m["msd"]), diverged=int(div),
                raw_whole=float(m["whole"]) if np.isfinite(m["whole"]) else nanv)


# ----------------------------------------------------------------------------
# Workers
# ----------------------------------------------------------------------------
def unit(args):
    tag, c = args
    t0 = time.time()
    kind, arg = KIND[tag]
    data = D.load_data(kind, arg)
    Rall = ML.prepass(data)["Rall"]
    rows = []
    for cfg in EST_CFGS:
        for mu in MUS:
            rows.append(dict(src=tag, **score(data, sim_est(data, Rall, cfg, mu, c), lab_of(cfg), mu)))
        print(f"[cfg] {tag} {lab_of(cfg)}", flush=True)
    for mu in MUS:
        rows.append(dict(src=tag, **G.one_run(data, Rall, MAN_CFG, mu)))
    print(f"[cfg] {tag} {MAN_LAB}", flush=True)
    for lab in BASES:
        for mu in MUS:
            rows.append(dict(src=tag, **G.one_run(data, Rall, ("D", lab), mu)))
        print(f"[cfg] {tag} {lab}", flush=True)
    return dict(tag=tag, rows=rows, runtime=time.time() - t0)


def abrupt_stats(anr, w_tr=None):
    """anr has one entry per 64 samples (sample index 64 j + 63)."""
    idx = np.arange(63, 64 * len(anr), 64)[:len(anr)]
    t = (idx + 1) / FS

    def win(a, b):
        sel = (t >= a) & (t < b)
        return float(anr[sel].mean()) if sel.any() else float("nan")
    out = dict(pre=win(10.0, TSW), post_2s=win(TSW, TSW + 2.0), last5=win(25.0, 30.0))
    out["recov_gap"] = out["last5"] - out["pre"]
    return out


def unit2(tag):
    kind, arg = KIND[tag]
    data = D.load_data(kind, arg)
    Rall = ML.prepass(data)["Rall"]
    out = dict(tag=tag)
    if tag in TRAJ_SRCS:
        sim = sim_est(data, Rall, BEST_CFG, BEST_MU, C_FIT)
        out["w_est"] = sim["wtraj"].tolist()
        out["est_whole"] = score(data, sim, lab_of(BEST_CFG), BEST_MU)["whole"]
        print(f"[traj] {tag} done", flush=True)
    if tag in ABRUPT_SRCS:
        hP = O.pad_or_trim(np.loadtxt(os.path.join(O.ROOT, "impulse_response/primary_ir.dat"), ndmin=1), M)
        hP2 = 0.8 * np.concatenate([np.zeros(8), hP])[:M]
        Ls = len(data["d"])
        d_new = signal.lfilter(hP2, [1.0], data["x"])[:Ls]
        t_sw = int(round(TSW * FS))
        d_sw = data["d"].copy()
        d_sw[t_sw:] = d_new[t_sw:]
        dd = dict(data)
        dd["d"] = d_sw
        runs = {
            "EST": sim_est(dd, Rall, BEST_CFG, BEST_MU, C_FIT),
            "W0.5": D.simulate(dd, Rall, D.spec_of("W0.5"), MU_REF["W0.5"], M),
            "MAN": G.sim_sched(dd, Rall, 0.5, 4, 3, MU_REF[MAN_LAB]),
        }
        res = {}
        for name, sim in runs.items():
            fin = bool(np.isfinite(sim["e"]).all())
            anr = O.anr_curve(sim["e"], d_sw) if fin else np.full(Ls // 64, np.nan)
            if not np.isfinite(anr).all():
                res[name] = dict(pre=np.nan, post_2s=np.nan, last5=np.nan, recov_gap=np.nan, diverged=1)
            else:
                res[name] = dict(abrupt_stats(anr), diverged=0)
            if name == "EST":
                wt = sim["wtraj"]
                tb = (np.arange(len(wt)) + 1) * B / FS
                pre = wt[(tb >= TSW - 0.1) & (tb < TSW)]
                post = wt[(tb >= TSW) & (tb < TSW + 2.0)]
                last = wt[(tb >= 25.0)]
                res["EST_w"] = dict(pre=float(pre.mean()) if pre.size else np.nan,
                                    post_peak=float(post.max()) if post.size else np.nan,
                                    last5=float(last.mean()) if last.size else np.nan)
                out["abrupt_w"] = wt.tolist()
        out["abrupt"] = res
        print(f"[abrupt] {tag} done", flush=True)
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    global C_FIT, BEST_CFG, BEST_MU, MU_REF
    os.makedirs(OUT, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()

    # ---- c fit on a145 only (grid search on the reference config, then frozen)
    kind, arg = KIND["a145"]
    dref = D.load_data(kind, arg)
    Rref = ML.prepass(dref)["Rall"]
    cgrid_rows = []
    best = None
    for c in C_GRID:
        for mu in MUS:
            r = score(dref, sim_est(dref, Rref, REF_CFG, mu, c), lab_of(REF_CFG), mu)
            cgrid_rows.append(dict(c=c, mu=mu, whole=r["whole"], diverged=r["diverged"]))
            if np.isfinite(r["whole"]) and (best is None or r["whole"] < best[0]):
                best = (r["whole"], c, mu)
        print(f"[cfit] c={c:.3g} done", flush=True)
    C_FIT = best[1]
    del dref, Rref
    with open(os.path.join(OUT, "omegaest_cfit.csv"), "w", newline="") as fh:
        wr_ = csv.writer(fh)
        wr_.writerow(["c", "mu", "a145_whole_db", "diverged"])
        for r in cgrid_rows:
            wr_.writerow([r["c"], r["mu"], r["whole"], r["diverged"]])
    print(f"[cfit] chosen c={C_FIT:.4g} (a145 whole {best[0]:.2f} dB at mu={best[2]})", flush=True)

    # ---- phase 1: grid, baselines, manual schedule on the ten sources
    res = []
    with Pool(processes=2) as pool:
        for rr in pool.imap_unordered(unit, [(t, C_FIT) for t in ALLTAGS], chunksize=1):
            res.append(rr)
            print(f"[done] {rr['tag']} {rr['runtime']:.1f} s ({len(res)}/{len(ALLTAGS)})", flush=True)
            with open(os.path.join(OUT, f"omegaest_src_{rr['tag']}.json"), "w") as fh:
                json.dump(dict(rows=rr["rows"]), fh, default=float)

    rows = [r for rr in res for r in rr["rows"]]
    cell = {(r["label"], r["mu"], r["src"]): r for r in rows}
    est_labels = [lab_of(c) for c in EST_CFGS]
    all_labels = est_labels + [MAN_LAB] + BASES

    def allmean(lab, mu, srcs, key):
        vals = [cell[(lab, mu, t)][key] for t in srcs]
        return float(np.mean(vals)) if np.isfinite(vals).all() else float("nan")

    def ndiv(lab, mu, srcs):
        return int(sum(cell[(lab, mu, t)]["diverged"] for t in srcs))

    chosen = {}
    for lab in all_labels:
        ok = {mu: allmean(lab, mu, SIX, "whole") for mu in MUS}
        ok = {mu: v for mu, v in ok.items() if np.isfinite(v)}
        chosen[lab] = min(ok, key=ok.get) if ok else None
    MU_REF = {lab: chosen[lab] for lab in ["W0.5", MAN_LAB]}

    # best estimator config by white whole (at its own mu)
    cands = [(allmean(lab, chosen[lab], SIX, "whole"), lab) for lab in est_labels if chosen[lab] is not None]
    cands = [c for c in cands if np.isfinite(c[0])]
    _, best_lab = min(cands)
    BEST_CFG = EST_CFGS[est_labels.index(best_lab)]
    BEST_MU = chosen[best_lab]
    print(f"[best] {best_lab} mu={BEST_MU}", flush=True)

    # ---- phase 2: w trajectories and abrupt-change runs
    p2 = {}
    with Pool(processes=2) as pool:
        for rr in pool.imap_unordered(unit2, TRAJ_SRCS + [t for t in ABRUPT_SRCS if t not in TRAJ_SRCS],
                                      chunksize=1):
            p2[rr["tag"]] = rr
            print(f"[phase2] {rr['tag']} done", flush=True)

    # ---- CSVs
    with open(os.path.join(OUT, "omegaest_grid.csv"), "w", newline="") as fh:
        wr_ = csv.writer(fh)
        wr_.writerow(["label", "source", "mu", "whole_db", "first5_db", "last10_db", "final_msd_db",
                      "diverged_task", "raw_whole_db"])
        for r in sorted(rows, key=lambda r: (r["label"], r["src"], r["mu"])):
            wr_.writerow([r["label"], r["src"], r["mu"], r["whole"], r["first5"], r["last10"], r["msd"],
                          r["diverged"], r["raw_whole"]])

    # ---- tables
    F = X.fmt
    def line(lab):
        mu = chosen[lab]
        if mu is None:
            return f"| {lab} | nan | " + " | ".join(["nan"] * 6) + " |"
        return (f"| {lab} | {mu} | {F(allmean(lab, mu, SIX, 'whole'))} | {F(allmean(lab, mu, SIX, 'first5'))} | "
                f"{F(allmean(lab, mu, SIX, 'last10'))} | {F(allmean(lab, mu, TWO, 'whole'))} | "
                f"{F(allmean(lab, mu, COL, 'whole'))} | {ndiv(lab, mu, ALLTAGS)} | "
                f"{sum(ndiv(lab, m, ALLTAGS) for m in MUS)} |")

    L = []
    L.append("# Online omega estimate from the error decay slope (atsip/explore/omegaest.py)\n")
    L.append("Numbers only. Generated by `atsip/explore/omegaest.py`.\n")
    L.append(f"Runtime (2 worker processes): {(time.time() - t0) / 60:.1f} min. Ten sources as in dissect.py "
             "(six white alpha=1.45 sets, two alpha=1.2, two colored). Framing E_M. Means over sources; nan if any "
             "run in the set diverged. Divergence: non-finite output or end ANR > 0 dB (task rule).\n")
    L.append(f"g0 mode B constant: c = {C_FIT:.4g} (grid over 10^(-3..1, step 0.5), fitted on a145 only, "
             f"reference config L=64, beta=0.97, free, m0=3).\n")
    L.append("## Table 1. Comparison (mu per row chosen by six-source white whole mean)\n")
    L.append("| row | mu | white whole | white first5 | white last10 | a1.2 whole | colored whole | "
             "#div at mu (of 10) | #div sweep (of 60) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for lab in BASES + [MAN_LAB]:
        L.append(line(lab))
    groups = {}
    for lab in est_labels:
        if chosen[lab] is None:
            continue
        cfg = EST_CFGS[est_labels.index(lab)]
        key = (cfg[2], cfg[3])
        v = allmean(lab, chosen[lab], SIX, "whole")
        if np.isfinite(v) and (key not in groups or v < groups[key][0]):
            groups[key] = (v, lab)
    top = sorted([(allmean(lab, chosen[lab], SIX, "whole"), lab) for lab in est_labels
                  if chosen[lab] is not None and np.isfinite(allmean(lab, chosen[lab], SIX, "whole"))])[:3]
    sel = []
    for key in [("A", "mono"), ("A", "free"), ("B", "mono"), ("B", "free")]:
        if key in groups and groups[key][1] not in sel:
            sel.append(groups[key][1])
    for _, lab in top:
        if lab not in sel:
            sel.append(lab)
    for lab in sel:
        L.append(line(lab))
    L.append("")
    L.append("Estimator rows: best config per (gmode, monotone) group, plus the three best overall by white whole. "
             "All 72 configs are in `out/omegaest_grid.csv`.\n")

    L.append("## Table 2. Best white whole per factor level (over configs at that level, own mu)\n")
    L.append("| factor | level | best white whole | config |")
    L.append("|---|---|---|---|")
    for name, idx, levels in [("L", 0, L_LIST), ("beta", 1, BETA_LIST), ("g0 mode", 2, GMODES),
                              ("monotone", 3, MONOS), ("m0", 4, M0_LIST)]:
        for lv in levels:
            bl = [(allmean(lab, chosen[lab], SIX, "whole"), lab) for lab in est_labels
                  if EST_CFGS[est_labels.index(lab)][idx] == lv and chosen[lab] is not None]
            bl = [b for b in bl if np.isfinite(b[0])]
            if bl:
                v, lab = min(bl)
                L.append(f"| {name} | {lv} | {F(v)} | {lab} |")
            else:
                L.append(f"| {name} | {lv} | nan | none |")
    L.append("")

    L.append("## Table 3. Abrupt change at 15 s (primary IR -> 0.8x, 8-sample delay), three white sources\n")
    L.append("ANR mean over 10-15 s (pre), 15-17 s (post), 25-30 s (recovered), and recovered minus pre. "
             "Mu per row as in Table 1. w from the best estimator config.\n")
    L.append("| row | pre [dB] | post 15-17 s [dB] | recovered 25-30 s [dB] | recovered - pre [dB] | diverged runs |")
    L.append("|---|---|---|---|---|---|")
    names = ["EST", "W0.5", "MAN"]
    ab = {n: {k: [] for k in ["pre", "post_2s", "last5", "recov_gap"]} for n in names}
    ndv = {n: 0 for n in names}
    for t in ABRUPT_SRCS:
        for n in names:
            st = p2[t]["abrupt"][n]
            ndv[n] += st["diverged"]
            for k in ab[n]:
                ab[n][k].append(st[k])
    for n in names:
        lbl = {"EST": f"EST {best_lab}", "W0.5": "W0.5", "MAN": MAN_LAB}[n]
        vals = {k: float(np.mean(v)) if np.isfinite(v).all() else float("nan") for k, v in ab[n].items()}
        L.append(f"| {lbl} | {F(vals['pre'])} | {F(vals['post_2s'])} | {F(vals['last5'])} | "
                 f"{F(vals['recov_gap'])} | {ndv[n]} |")
    wsum = []
    for t in ABRUPT_SRCS:
        e = p2[t]["abrupt"]["EST_w"]
        wsum.append(e)
    L.append("")
    L.append("Estimator w around the change (mean over the three sources): "
             f"pre {F(float(np.mean([e['pre'] for e in wsum])), 3)}, "
             f"peak in 15-17 s {F(float(np.mean([e['post_peak'] for e in wsum])), 3)}, "
             f"25-30 s {F(float(np.mean([e['last5'] for e in wsum])), 3)}.\n")

    L.append("## Files\n")
    L.append("- `out/omegaest_grid.csv`: every run (label, source, mu), task divergence rule.")
    L.append("- `out/omegaest_cfit.csv`: c grid on a145.")
    L.append("- `out/omegaest_w_traj.csv`: w_i vs time for two sources and the manual ramp.")
    L.append("- `out/omegaest_w.png`: w_i vs time next to the manual ramp, and the abrupt-change w trace.")
    L.append("")
    with open(os.path.join(HERE, "RESULTS_omegaest.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    # ---- w trajectories and figure
    tb = np.arange(len(p2["a145"]["w_est"])) * B / FS
    n = len(tb)
    ramp = np.minimum(1.0, tb / 4.0)
    tr = {t: np.array(p2[t]["w_est"]) for t in TRAJ_SRCS}
    with open(os.path.join(OUT, "omegaest_w_traj.csv"), "w", newline="") as fh:
        wr_ = csv.writer(fh)
        wr_.writerow(["time_s", "w_est_a145", "w_est_cms145_s0", "w_manual_T4"])
        for j in range(n):
            wr_.writerow([round(tb[j], 4), tr["a145"][j], tr["cms145_s0"][j], ramp[j]])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(tb, tr["a145"], label=f"estimate a145 ({best_lab})")
    ax[0].plot(tb, tr["cms145_s0"], label="estimate cms145_s0")
    ax[0].plot(tb, ramp, "k--", label="manual ramp T=4 s")
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("w")
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].legend(fontsize=7)
    ax[0].set_title("w_i, no change")
    wab = np.array(p2[ABRUPT_SRCS[0]]["abrupt_w"])
    tab = np.arange(len(wab)) * B / FS
    ax[1].plot(tab, wab, label=f"estimate {ABRUPT_SRCS[0]}")
    ax[1].axvline(TSW, color="r", ls=":", label="change at 15 s")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].legend(fontsize=7)
    ax[1].set_title("w_i, abrupt primary-path change")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "omegaest_w.png"), dpi=120)
    plt.close(fig)

    summ = dict(runtime_s=time.time() - t0, c_fit=C_FIT, best=best_lab, best_mu=BEST_MU,
                chosen_mu=chosen, abrupt={t: p2[t]["abrupt"] for t in ABRUPT_SRCS})
    with open(os.path.join(OUT, "omegaest_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1, default=float)
    print(f"[total] {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
