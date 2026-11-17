#!/usr/bin/env python3
"""MFBLMS, Wang-Xue-Lu switching and Wu-Yang flatness control in the delayless FD-FxNLMS framework.

Baselines (M = 512, N = 1024, B = 128, E_M framing, same loop and divergence rule as specfac.py):
  MFBLMS    (Lu, Qiu, Zou): g = irfft(conj(R) E)[:M] (unnormalised, constrained), then
            w <- w - mu_b irfft(rfft(pad_N(g)) / Lam)[:M], Lam = Srr + eps. One extra rfft/irfft pair over B0.
  Switch    (Wang, Xue, Lu, simplified): B0 until the relative smoothed-error decrease
            dJ/J < TAU_SW, then MFBLMS; back to B0 when J / min(J since switch) > 4 (6 dB).
            Every mode is held for >= 32 blocks. J is the EMA (0.05 per block) of the block mean of e^2.
  WY g=gmax (Wu, Yang style, adapted to FxNLMS): Lam = Srr + gamma(n) Pbar + eps, gamma starts at 0 and
            rises by 0.1 gamma_max when dJ/J < TAU_WY (held >= 32 blocks), capped at gamma_max. Reset to 0
            when J / min(J since reset) > 4.
Existing references, reused unchanged: B0, W0.5 (dissect.simulate), GS (geosched.sim_sched),
SF and SF+ramp (specfac.sim_sf).

Tuning (lightly, on a145 only): TAU_SW and TAU_WY are chosen by the lowest a145 whole-curve L1 over the mu grid.
The mu of every row is chosen as in specfac (six-source white whole mean). Conditions B (inclass inputs,
2 seeds) and C (path change at 15 s, three white sources) use the mu of condition A.

Usage:
  python3 atsip/explore/mfblms.py check   analytic expected-value fixed point of MFBLMS vs the M-tap Wiener, mode spread
  python3 atsip/explore/mfblms.py diag    a145, B0 at mu 0.1: quantiles of dJ/J and J/min(J), to set the TAU grid
  python3 atsip/explore/mfblms.py         full run (tune, condition A, B, C, timing) -> out/mfblms_*, RESULTS_mfblms.md
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
from scipy import signal, linalg

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dissect as D     # noqa: E402
import explore as X     # noqa: E402
import mixlam as ML     # noqa: E402
import geosched as G    # noqa: E402
import specfac as SF    # noqa: E402
import inclass as IC    # noqa: E402

O = X.O
FS, M, N, B, NB = D.FS, D.M, D.N, D.B, D.NB
OUT = D.OUT
MUS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.36, 0.4, 0.6, 1.0]
SIX, TWO, COL, ALLTAGS = D.SIX, D.TWO, D.COL, D.ALLTAGS
N5 = X.N5
J_ALPHA = 0.05
HOLD = 32
RATIO = 4.0
RATIO_DB = 10 * np.log10(RATIO)      # 6.02 dB
DG = 0.1                                     # gamma step = DG * gamma_max
TAUS = [0.0, 0.3, 1.0]                       # candidate decrease thresholds over HOLD blocks [dB] (set after `diag`)
METHODS = ["B0", "W0.5", "GS", "SF", "SF+ramp", "MFBLMS", "MFBLMS+ramp", "Switch", "WY g=1", "WY g=10"]
TAG_B = [("A", "Gaussian"), ("B", "steps +-20 dB"), ("C", "bursts +20 dB")]   # inclass names
SEEDS_B = [0, 1]
PC_SEEDS = ["cms145_s0", "cms145_s1", "cms145_s2"]   # three white (CMS alpha 1.45) sources
PC_T = 15.0                                   # path change time [s]
PC_SCALE, PC_DELAY = 0.8, 8


# ----------------------------------------------------------------------------
# Directions
# ----------------------------------------------------------------------------
def b0_dir(R, E, Lam):
    """Binwise (pre-constraint) direction, irfft(conj(R) E / Lam)[:M]."""
    return np.fft.irfft(np.conj(R) * E / Lam, N)[:M]


def mf_dir(R, E, Lam):
    """MFBLMS (post-constraint): constrain the gradient in time, then normalise in frequency."""
    g = np.fft.irfft(np.conj(R) * E, N)[:M]
    return np.fft.irfft(np.fft.rfft(g, N) / Lam, N)[:M]


# ----------------------------------------------------------------------------
# Simulation: MFBLMS, switch, WY (same loop as specfac.sim_sf, with a per-block detector)
# ----------------------------------------------------------------------------
def sim_new(data, Rall, kind, mu, T=0, m0=1, tau=0.0, gmax=0.0, rec=None):
    d, xq, hS, S2 = data["d"], data["xq"], data["hS"], data["S2"]
    L = len(d)
    nb = L // B
    swv = sliding_window_view(xq, M)
    w = np.zeros(M)
    zi = np.zeros(len(hS) - 1)
    e_all = np.zeros(L)
    shared = X.Shared(S2)
    mode_mf = (kind == "MF")          # for kind SW: False = B0 at start
    Jk = None
    hist = []
    minJ = None
    hold = 0
    gam = 0.0
    nmf = 0
    nsw = 0
    diverged = False
    nb_done = 0
    for i in range(nb):
        s0 = i * B
        Xb = swv[s0 + N - M + 1: s0 + N - M + 1 + B][:, ::-1]
        ys, zi = signal.lfilter(hS, [1.0], Xb @ w, zi=zi)
        e_all[s0:s0 + B] = d[s0:s0 + B] + ys
        E = ML.frame_E(e_all, s0, M)
        s = shared.step(Rall[i], E)
        Srr, Pbar, eps = s["Srr"], s["Pbar"], s["eps"]
        wr = 1.0 if T == 0 else min(1.0, (i * B / FS) / T)
        mu_i = mu * (1 + (m0 - 1) * (1 - wr))
        if kind == "MF" or (kind == "SW" and mode_mf):
            g = mf_dir(Rall[i], E, Srr + eps)
            nmf += 1
        elif kind == "WY":
            g = b0_dir(Rall[i], E, Srr + gam * Pbar + eps)
        else:
            g = b0_dir(Rall[i], E, Srr + eps)
        w = w - 2 * mu_i * B / M * g
        nb_done = i + 1
        if not np.isfinite(w).all():
            e_all[s0 + B:] = np.nan
            diverged = True
            break
        # detectors on the smoothed block error power J (EMA of the block power in dB; decisions apply
        # from the next block). dJ = decrease over the last HOLD blocks, in dB (positive = decreasing).
        Jb = 10 * np.log10(float(np.mean(e_all[s0:s0 + B] ** 2)) + 1e-30)
        Jk = Jb if Jk is None else (1 - J_ALPHA) * Jk + J_ALPHA * Jb
        hist.append(Jk)
        hold += 1
        if len(hist) <= HOLD:
            continue
        dJ = hist[-1 - HOLD] - Jk
        if kind == "SW":
            if not mode_mf:
                if rec is not None:
                    rec.append((dJ, np.nan))
                if hold >= HOLD and dJ < tau:
                    mode_mf, nsw, hold, minJ = True, nsw + 1, 0, Jk
            else:
                minJ = Jk if minJ is None else min(minJ, Jk)
                if rec is not None:
                    rec.append((dJ, Jk - minJ))
                if hold >= HOLD and Jk - minJ > RATIO_DB:
                    mode_mf, nsw, hold, minJ = False, nsw + 1, 0, None
        elif kind == "WY":
            minJ = Jk if minJ is None else min(minJ, Jk)
            if rec is not None:
                rec.append((dJ, Jk - minJ))
            if hold >= HOLD and Jk - minJ > RATIO_DB and gam > 0.0:
                gam, minJ, hold, nsw = 0.0, Jk, 0, nsw + 1
            elif hold >= HOLD and dJ < tau and gam < gmax:
                gam, hold, nsw = min(gmax, gam + DG * gmax), 0, nsw + 1
    return dict(e=e_all, w=w, diverged=diverged, fmf=nmf / max(nb_done, 1), nsw=nsw)


def run_method(name, data, Rall, mu, tau_sw, tau_wy):
    if name == "B0":
        return D.simulate(data, Rall, D.spec_of("B0"), mu, M)
    if name == "W0.5":
        return D.simulate(data, Rall, D.spec_of("W0.5"), mu, M)
    if name == "GS":
        return G.sim_sched(data, Rall, 0.5, 4, 3, mu)
    if name == "SF":
        return SF.sim_sf(data, Rall, 1e-2, 0, 1, mu)
    if name == "SF+ramp":
        return SF.sim_sf(data, Rall, 1e-2, 2, 3, mu)
    if name == "MFBLMS":
        return sim_new(data, Rall, "MF", mu)
    if name == "MFBLMS+ramp":
        return sim_new(data, Rall, "MF", mu, T=2, m0=3)
    if name == "Switch":
        return sim_new(data, Rall, "SW", mu, tau=tau_sw)
    if name == "WY g=1":
        return sim_new(data, Rall, "WY", mu, tau=tau_wy[0], gmax=1.0)
    if name == "WY g=10":
        return sim_new(data, Rall, "WY", mu, tau=tau_wy[1], gmax=10.0)
    raise ValueError(name)


# ----------------------------------------------------------------------------
# Metrics (L1 = oracle ANR, L2 = 10 log10 sum e^2 / sum d^2, same windows; NaN if diverged)
# ----------------------------------------------------------------------------
def score(sim, data):
    e, w, d = sim["e"], sim["w"], data["d"]
    fin = bool(np.isfinite(w).all() and np.isfinite(e).all())
    nanv = dict(L1w=np.nan, L1f5=np.nan, L1l10=np.nan, L2w=np.nan, L2f5=np.nan, L2l10=np.nan)
    if not fin:
        out = dict(nanv, div=1)
    else:
        anr = O.anr_curve(e, d)
        whole, last10 = O.summarize(anr)
        div = (not np.isfinite(last10)) or (not np.isfinite(whole)) or last10 > 0.0
        if div:
            out = dict(nanv, div=1)
        else:
            e2, d2 = e ** 2, d ** 2
            n10 = len(e) // 10
            out = dict(L1w=float(whole), L1f5=float(anr[:N5].mean()), L1l10=float(last10),
                       L2w=float(10 * np.log10(e2.sum() / d2.sum())),
                       L2f5=float(10 * np.log10(e2[:5 * FS].sum() / d2[:5 * FS].sum())),
                       L2l10=float(10 * np.log10(e2[-n10:].sum() / d2[-n10:].sum())), div=0)
    out["fmf"] = float(sim.get("fmf", np.nan))
    out["nsw"] = float(sim.get("nsw", np.nan))
    return out


def score_pc(sim, data):
    """Path-change windows: ANR mean over 15-17 s and 25-30 s; L2 as sums over the same sample windows."""
    e, w, d = sim["e"], sim["w"], data["d"]
    nanv = dict(L1a=np.nan, L1b=np.nan, L2a=np.nan, L2b=np.nan, L1w=np.nan)
    if not (np.isfinite(w).all() and np.isfinite(e).all()):
        return dict(nanv, div=1)
    anr = O.anr_curve(e, d)
    whole, last10 = O.summarize(anr)
    if (not np.isfinite(last10)) or (not np.isfinite(whole)) or last10 > 0.0:
        return dict(nanv, div=1)
    ts = (63 + 64 * np.arange(len(anr))) / FS
    wa, wb = (ts >= 15) & (ts < 17), (ts >= 25) & (ts < 30)
    sa = slice(int(15 * FS), int(17 * FS))
    sb = slice(int(25 * FS), int(30 * FS))
    e2, d2 = e ** 2, d ** 2
    return dict(L1a=float(anr[wa].mean()), L1b=float(anr[wb].mean()),
                L2a=float(10 * np.log10(e2[sa].sum() / d2[sa].sum())),
                L2b=float(10 * np.log10(e2[sb].sum() / d2[sb].sum())), L1w=float(whole), div=0)


def build_pc(x):
    """Path change: primary IR swapped at 15 s for 0.8 x (IR delayed by 8 samples). Secondary path unchanged."""
    data = D.build(x)
    hP = O.pad_or_trim(np.loadtxt(os.path.join(O.ROOT, "impulse_response/primary_ir.dat"), ndmin=1), M)
    hPn = np.zeros(M)
    hPn[PC_DELAY:] = PC_SCALE * hP[:M - PC_DELAY]
    n0 = int(PC_T * FS)
    d_new = signal.lfilter(hPn, [1.0], x)
    data["d"] = np.concatenate([data["d"][:n0], d_new[n0:]])
    data["w_opt"] = O.wiener_opt(data["r"][n0:], data["d"][n0:])   # post-change, first 10 s after the swap
    return data


# ----------------------------------------------------------------------------
# Check: expected-value fixed point of MFBLMS vs the M-tap Wiener, and the mode spread
# ----------------------------------------------------------------------------
def analytic_check():
    hP = O.pad_or_trim(np.loadtxt(os.path.join(O.ROOT, "impulse_response/primary_ir.dat"), ndmin=1), M)
    hS = O.pad_or_trim(np.loadtxt(os.path.join(O.ROOT, "impulse_response/secondary_ir.dat"), ndmin=1), O.C_SEC)

    def xc(a, b, lags):          # sum_i a[i] b[i+lag], white unit-variance input
        full = np.correlate(b, a, "full")
        out = np.zeros(len(lags))
        idx = lags + len(a) - 1
        ok = (idx >= 0) & (idx < len(full))
        out[ok] = full[idx[ok]]
        return out

    lags = np.arange(M)
    rrr = xc(hS, hS, lags)
    p = xc(hS, hP, lags)
    Rrr = linalg.toeplitz(rrr)                      # E[r r^T], filtered reference, M taps
    wM = linalg.solve(Rrr, -p)                      # M-tap Wiener (same as the code's causal solution)
    Sf = np.fft.rfft(hS, N)
    S2n = np.abs(Sf) ** 2
    S2n = S2n / S2n.mean()
    Lam = S2n + 1e-6                                # eps relative 1e-6 as in X.Shared
    C = np.empty((M, M))
    for j in range(M):                              # MFBLMS normalisation as a linear map on the M-tap gradient
        v = np.zeros(M)
        v[j] = 1.0
        C[:, j] = np.fft.irfft(np.fft.rfft(v, N) / Lam, N)[:M]
    # expected update: w <- w - eta C (Rrr w + p). Fixed point solves C (Rrr w + p) = 0.
    fp = linalg.solve(C @ Rrr, -(C @ p))
    bias = 10 * np.log10(np.sum((fp - wM) ** 2) / np.sum(wM ** 2))
    resid = np.linalg.norm(C @ (Rrr @ wM + p)) / np.linalg.norm(C @ p)
    ev = np.sort(np.linalg.eigvals(C @ Rrr).real)
    # the same operator with B0 (pre-constraint, Lam only): modes of Rrr / Lam, for contrast
    out = dict(bias_db=float(bias), resid_rel=float(resid), spread=float(ev[-1] / ev[0]),
               cond_C=float(np.linalg.cond(C)), ev_min=float(ev[0]), ev_max=float(ev[-1]))
    for eps in [1e-4, 1e-2]:
        Ce = np.empty((M, M))
        Le = S2n + eps
        for j in range(M):
            v = np.zeros(M)
            v[j] = 1.0
            Ce[:, j] = np.fft.irfft(np.fft.rfft(v, N) / Le, N)[:M]
        ev2 = np.sort(np.linalg.eigvals(Ce @ Rrr).real)
        out[f"spread_eps{eps:g}"] = float(ev2[-1] / ev2[0])
    return out


def check_refs():
    """Identity checks: the reused simulators reproduce specfac_grid.csv on a145 at shared mu."""
    import csv as _csv
    grid = {}
    p = os.path.join(OUT, "specfac_grid.csv")
    with open(p) as fh:
        for r in _csv.DictReader(fh):
            if r["source"] == "a145":
                grid[(r["label"], float(r["mu"]))] = float(r["whole_db"]) if r["whole_db"] not in ("", "nan") else np.nan
    data = D.load_data(*D.KIND["a145"])
    Rall = ML.prepass(data)["Rall"]
    pairs = {("B0", "B0"): 0.1, ("W0.5", "W0.5"): 0.1, ("GS", "GS (a0.5 T4 m0=3)"): 0.05,
             ("SF", "SF e=0.01 T=0 m0=1"): 0.1, ("SF+ramp", "SF e=0.01 T=2 m0=3"): 0.05}
    out = {}
    for (name, lab), mu in pairs.items():
        m = score(run_method(name, data, Rall, mu, 0.0, (0.0, 0.0)), data)
        out[name] = (m["L1w"], grid.get((lab, mu), np.nan))
    return out


def check():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    a = analytic_check()
    c = check_refs()
    print("[check]", json.dumps(a), flush=True)
    print("[check] vs specfac_grid whole (ours, specfac):", json.dumps({k: [float(x), float(y)] for k, (x, y) in c.items()}), flush=True)
    return a, c


def diag():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    data = D.load_data(*D.KIND["a145"])
    Rall = ML.prepass(data)["Rall"]
    rec = []
    sim_new(data, Rall, "SW", 0.1, tau=-1.0, rec=rec)    # never switches; records the detectors in B0 mode
    rec = np.array(rec)
    rel = rec[:, 0]
    nb = len(rel)
    for a, b in [(0, 40), (40, 400), (400, 1500), (1500, 3000), (3000, nb)]:
        x = rel[a:b]
        print(f"[diag] blocks {a}-{b}: dJ [dB] q05/q50/q95 {np.round(np.quantile(x, [.05, .5, .95]), 3)}, "
              f"frac < 0.1 {np.mean(x < 0.1):.3f}, < 0.3 {np.mean(x < 0.3):.3f}, < 1.0 {np.mean(x < 1.0):.3f}",
              flush=True)
    rec2 = []
    sim_new(data, Rall, "MF", 0.1, rec=rec2)
    rec2 = []
    sim_new(data, Rall, "SW", 0.1, tau=1e9, rec=rec2)   # switches to MF at the first decision
    r2 = np.array([r for r in rec2 if np.isfinite(r[1])])
    if len(r2):
        q2 = np.quantile(r2[:, 1], [0.5, 0.9, 0.99, 1.0])
        print("[diag] MF mode J - min(J) [dB] quantiles 50/90/99/max:", np.round(q2, 2), "frac > 6.02 dB:",
              round(float(np.mean(r2[:, 1] > RATIO_DB)), 4), flush=True)


# ----------------------------------------------------------------------------
# Workers
# ----------------------------------------------------------------------------
def tune_unit(args):
    """a145 only: one (method, tau[, gmax]) over the mu grid."""
    name, tau, gmax = args
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    data = D.load_data(*D.KIND["a145"])
    Rall = ML.prepass(data)["Rall"]
    rows = []
    for mu in MUS:
        if name == "Switch":
            sim = sim_new(data, Rall, "SW", mu, tau=tau)
        else:
            sim = sim_new(data, Rall, "WY", mu, tau=tau, gmax=gmax)
        rows.append(dict(name=name, tau=tau, gmax=gmax, mu=mu, **score(sim, data)))
    print(f"[tune] {name} tau {tau} gmax {gmax} done", flush=True)
    return rows


def unit_A(args):
    tag, tau_sw, tau_wy = args
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    kind, arg = D.KIND[tag]
    data = D.load_data(kind, arg)
    Rall = ML.prepass(data)["Rall"]
    rows = []
    for name in METHODS:
        for mu in MUS:
            rows.append(dict(src=tag, method=name, mu=mu,
                             **score(run_method(name, data, Rall, mu, tau_sw, tau_wy), data)))
        print(f"[cfg] {tag} {name}", flush=True)
    rt = time.time() - t0
    with open(os.path.join(OUT, f"mfblms_src_{tag}.json"), "w") as fh:
        json.dump(dict(rows=rows, runtime=rt), fh, default=float)
    return dict(tag=tag, rows=rows, runtime=rt)


def unit_B(args):
    inp, seed, mu_map, tau_sw, tau_wy = args
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    data = D.build(IC.make_input(inp, seed))
    Rall = ML.prepass(data)["Rall"]
    rows = []
    for name in METHODS:
        rows.append(dict(inp=inp, seed=seed, method=name, mu=mu_map[name],
                         **score(run_method(name, data, Rall, mu_map[name], tau_sw, tau_wy), data)))
    print(f"[B] input {inp} seed {seed} done", flush=True)
    return rows


def unit_C(args):
    tag, mu_map, tau_sw, tau_wy = args
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    kind, arg = D.KIND[tag]
    data = build_pc(D.load_data(kind, arg)["x"])
    Rall = ML.prepass(data)["Rall"]
    rows = []
    for name in METHODS:
        rows.append(dict(src=tag, method=name, mu=mu_map[name],
                         **score_pc(run_method(name, data, Rall, mu_map[name], tau_sw, tau_wy), data)))
    print(f"[C] path change {tag} done", flush=True)
    return rows


# ----------------------------------------------------------------------------
# Timing (closed loop, one process, a145, mu 0.1; cost.py style: several rounds, minimum of medians)
# ----------------------------------------------------------------------------
def timing(tau_sw, tau_wy, rounds=3):
    data = D.load_data(*D.KIND["a145"])
    Rall = ML.prepass(data)["Rall"]
    nb = len(data["d"]) // B
    res = {}
    for name in ["B0", "W0.5", "GS", "SF", "MFBLMS", "MFBLMS+ramp", "Switch", "WY g=10"]:
        ts = []
        for _ in range(rounds):
            t = time.perf_counter()
            run_method(name, data, Rall, 0.1, tau_sw, tau_wy)
            ts.append((time.perf_counter() - t) / nb)
        res[name] = dict(ms_per_block=1e3 * float(np.median(ts)), lo=1e3 * float(np.min(ts)),
                         hi=1e3 * float(np.max(ts)))
        print(f"[timing] {name} {res[name]['ms_per_block']:.3f} ms/block", flush=True)
    return res


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def nm(vals):
    v = np.asarray(vals, dtype=float)
    return float(v.mean()) if np.isfinite(v).all() else float("nan")


def fmt(v, p=2):
    return "nan" if v is None or not np.isfinite(v) else f"{v:.{p}f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    a_chk, c_chk = check()

    # --- tuning on a145 only
    tune_tasks = [("Switch", t, None) for t in TAUS] + \
                 [("WY", t, g) for g in (1.0, 10.0) for t in TAUS]
    with Pool(processes=2) as pool:
        tune_rows = [r for rows in pool.imap_unordered(tune_unit, tune_tasks, chunksize=1) for r in rows]

    def best_tau(name, gmax=None):
        best = None
        for t in TAUS:
            vals = [(r["L1w"], r["mu"]) for r in tune_rows
                    if r["name"] == name and r["tau"] == t and (gmax is None or r["gmax"] == gmax)
                    and np.isfinite(r["L1w"])]
            if vals:
                v, mu = min(vals)
                if best is None or v < best[1]:
                    best = (t, v, mu)
        return best

    b_sw = best_tau("Switch")
    b_wy1 = best_tau("WY", 1.0)
    b_wy10 = best_tau("WY", 10.0)
    tau_sw = b_sw[0]
    tau_wy = (b_wy1[0], b_wy10[0])
    print(f"[tune] Switch tau {tau_sw} (a145 whole {b_sw[1]:.2f} at mu {b_sw[2]}); "
          f"WY1 tau {tau_wy[0]} ({b_wy1[1]:.2f} at {b_wy1[2]}); WY10 tau {tau_wy[1]} ({b_wy10[1]:.2f} at {b_wy10[2]})",
          flush=True)

    # --- condition A
    with Pool(processes=2) as pool:
        resA = list(pool.imap_unordered(unit_A, [(t, tau_sw, tau_wy) for t in ALLTAGS], chunksize=1))
        for rr in resA:
            print(f"[done] A {rr['tag']} {rr['runtime']:.1f} s", flush=True)
    rowsA = [r for rr in resA for r in rr["rows"]]
    cell = {(r["method"], r["mu"], r["src"]): r for r in rowsA}

    def cmean(name, mu, srcs, key):
        return nm([cell[(name, mu, t)][key] for t in srcs])

    def ndiv(name, mu, srcs):
        return int(sum(cell[(name, mu, t)]["div"] for t in srcs))

    def best_mu(name):
        ok = {mu: cmean(name, mu, SIX, "L1w") for mu in MUS}
        ok = {mu: v for mu, v in ok.items() if np.isfinite(v)}
        return min(ok, key=ok.get) if ok else None

    def max_stable(name):
        st = [mu for mu in MUS if ndiv(name, mu, ALLTAGS) == 0]
        return max(st) if st else None

    chosen = {m: best_mu(m) for m in METHODS}
    mu_map = {m: (chosen[m] if chosen[m] is not None else MUS[0]) for m in METHODS}
    print(f"[A] chosen mu {chosen}", flush=True)

    # --- conditions B and C at the condition-A mu
    tasksB = [(inp, s, mu_map, tau_sw, tau_wy) for inp, _ in TAG_B for s in SEEDS_B]
    with Pool(processes=2) as pool:
        rowsB = [r for rr in pool.imap_unordered(unit_B, tasksB, chunksize=1) for r in rr]
        rowsC = [r for rr in pool.imap_unordered(unit_C, [(t, mu_map, tau_sw, tau_wy) for t in PC_SEEDS],
                                                 chunksize=1) for r in rr]
    runtime = time.time() - t0
    print(f"[total] {runtime:.1f} s", flush=True)

    tim = timing(tau_sw, tau_wy)

    # --- outputs
    grid_path = os.path.join(OUT, "mfblms_grid.csv")
    with open(grid_path, "w", newline="") as fh:
        wr_ = csv.writer(fh)
        wr_.writerow(["method", "source", "mu", "L1_whole", "L1_first5", "L1_last10", "L2_whole", "L2_first5",
                      "L2_last10", "diverged", "frac_MF", "switches"])
        for r in sorted(rowsA, key=lambda r: (r["method"], r["src"], r["mu"])):
            wr_.writerow([r["method"], r["src"], r["mu"], r["L1w"], r["L1f5"], r["L1l10"], r["L2w"], r["L2f5"],
                          r["L2l10"], r["div"], r["fmf"], r["nsw"]])
    summ = dict(runtime_s=runtime, tau_sw=tau_sw, tau_wy=tau_wy, tune_best=dict(sw=b_sw, wy1=b_wy1, wy10=b_wy10),
                chosen_mu=chosen, check_analytic=a_chk, check_specfac=c_chk, timing=tim)
    with open(os.path.join(OUT, "mfblms_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1, default=float)
    with open(os.path.join(OUT, "mfblms_condBC.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(B=rowsB, C=rowsC, mu=mu_map), fh, default=float)
    with open(os.path.join(OUT, "mfblms_tune.json"), "w", encoding="utf-8") as fh:
        json.dump(tune_rows, fh, default=float)

    # --- report
    F = fmt
    L = []
    L.append("# MFBLMS, switching and flatness control (atsip/explore/mfblms.py)\n")
    L.append("Numbers only, generated by `atsip/explore/mfblms.py`. Delayless FD-FxNLMS, M = 512, N = 1024, B = 128, "
             "E_M framing. Divergence as specfac (non-finite or end ANR > 0 dB); diverged runs are NaN in means. "
             f"Runtime (2 workers): {runtime:.0f} s. mu grid: {MUS}.\n")
    L.append("Tuned on a145 only (lowest whole L1 over the mu grid): "
             f"Switch TAU = {tau_sw} (a145 whole {F(b_sw[1])} dB at mu {b_sw[2]}); "
             f"WY gamma_max 1: TAU = {tau_wy[0]} ({F(b_wy1[1])} at mu {b_wy1[2]}); "
             f"WY gamma_max 10: TAU = {tau_wy[1]} ({F(b_wy10[1])} at mu {b_wy10[2]}).\n")
    L.append("## Checks\n")
    L.append(f"- Expected-value fixed point of MFBLMS (white input, exact covariances of the M-tap filtered "
             f"reference): the fixed point of w <- w - eta C (Rrr w + p) is the M-tap Wiener solution. "
             f"Analytic bias {F(a_chk['bias_db'], 1)} dB; relative residual of the expected gradient at w_M "
             f"{a_chk['resid_rel']:.1e}.")
    L.append(f"- Mode spread of C Rrr at M = 512, N = 1024: {F(a_chk['spread'], 2)} (min {a_chk['ev_min']:.3f}, "
             f"max {a_chk['ev_max']:.3f}); the reviewer's mf.py gives 7.547 for the same operator at eps 0. "
             f"For contrast, the B0 operator at eps 1e-4 and 1e-2: {F(a_chk['spread_eps0.0001'], 1)} and "
             f"{F(a_chk['spread_eps0.01'], 1)}.")
    L.append(f"- Reused simulators vs specfac_grid.csv (a145, whole L1, dB): " + ", ".join(
        f"{k} {F(x, 3)} vs {F(y, 3)}" for k, (x, y) in c_chk.items()) + ".\n")

    L.append("## A. Ten sources. Means over sources; L1 and L2 in dB\n")
    L.append("White = six alpha 1.45 sets; a1.2 and colored = two sources each. mu chosen per row by the six-source "
             "white whole L1 mean. Max stable mu = largest grid mu with no divergence on any of the ten sources. "
             "Switch and WY columns: fraction of blocks in MFBLMS mode (Switch) and number of mode or gamma "
             "events, white means at the chosen mu.\n")
    L.append("| row | mu | white L1 whole | L1 first5 | L1 last10 | white L2 whole | L2 first5 | L2 last10 | "
             "a1.2 L1 whole | colored L1 whole | #div at mu (of 10) | #div sweep (of 90) | max stable mu | "
             "white frac MF / events |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in METHODS:
        mu = chosen[name]
        if mu is None:
            L.append(f"| {name} | nan | " + " | ".join(["nan"] * 8) + f" | {sum(ndiv(name, m, ALLTAGS) for m in MUS)} | "
                     f"{max_stable(name)} | - |")
            continue
        info = "-"
        if name == "Switch":
            info = (f"{F(cmean(name, mu, SIX, 'fmf'), 2)} / {F(cmean(name, mu, SIX, 'nsw'), 1)}")
        elif name.startswith("WY"):
            info = f"- / {F(cmean(name, mu, SIX, 'nsw'), 1)}"
        L.append(f"| {name} | {mu} | {F(cmean(name, mu, SIX, 'L1w'))} | {F(cmean(name, mu, SIX, 'L1f5'))} | "
                 f"{F(cmean(name, mu, SIX, 'L1l10'))} | {F(cmean(name, mu, SIX, 'L2w'))} | "
                 f"{F(cmean(name, mu, SIX, 'L2f5'))} | {F(cmean(name, mu, SIX, 'L2l10'))} | "
                 f"{F(cmean(name, mu, TWO, 'L1w'))} | {F(cmean(name, mu, COL, 'L1w'))} | "
                 f"{ndiv(name, mu, ALLTAGS)} | {sum(ndiv(name, m, ALLTAGS) for m in MUS)} | "
                 f"{max_stable(name)} | {info} |")
    L.append("")

    L.append("## B. inclass inputs at the condition-A mu, 2 seeds (means; NaN if a seed diverged)\n")
    for inp, label in TAG_B:
        L.append(f"### {label} (inclass {inp})\n")
        L.append("| row | mu | L1 whole | L1 first5 | L1 last10 | L2 whole | L2 first5 | L2 last10 | #div (of 2) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for name in METHODS:
            rr = [r for r in rowsB if r["inp"] == inp and r["method"] == name]
            L.append(f"| {name} | {mu_map[name]} | " + " | ".join(
                F(nm([r[k] for r in rr])) for k in ["L1w", "L1f5", "L1l10", "L2w", "L2f5", "L2l10"]) +
                f" | {sum(r['div'] for r in rr)} |")
        L.append("")

    L.append("## C. Path change at 15 s (primary IR x0.8, delayed 8 samples), three white sources, condition-A mu\n")
    L.append("ANR in dB, means over the three sources (NaN if any diverged). L1 = oracle ANR (windows of 64 samples), "
             "L2 = 10log10 of the energy ratio over the same time window.\n")
    L.append("| row | mu | L1 15-17 s | L1 25-30 s | L2 15-17 s | L2 25-30 s | #div (of 3) |")
    L.append("|---|---|---|---|---|---|---|")
    for name in METHODS:
        rr = [r for r in rowsC if r["method"] == name]
        L.append(f"| {name} | {mu_map[name]} | " + " | ".join(
            F(nm([r[k] for r in rr])) for k in ["L1a", "L1b", "L2a", "L2b"]) + f" | {sum(r['div'] for r in rr)} |")
    L.append("")

    L.append("## D. Per-block cost (closed loop, one process, a145, mu 0.1, 3 rounds)\n")
    L.append("Mean wall time per block over the 30 s run (total / 3750 blocks), median of 3 rounds with min-max. "
             "Not the cost.py protocol (no warm-up, no update-only mode).\n")
    L.append("| method | ms per block | range | ratio to B0 |")
    L.append("|---|---|---|---|")
    base = tim["B0"]["ms_per_block"]
    for name, v in tim.items():
        L.append(f"| {name} | {v['ms_per_block']:.3f} | {v['lo']:.3f}-{v['hi']:.3f} | {v['ms_per_block'] / base:.2f} |")
    L.append("")
    L.append("## Files\n")
    L.append("- `out/mfblms_grid.csv`: every condition-A run (method, source, mu).")
    L.append("- `out/mfblms_condBC.json`: conditions B and C; `out/mfblms_tune.json`: a145 tuning; "
             "`out/mfblms_summary.json`: chosen mu, tau, checks, timing.")
    L.append("")
    with open(os.path.join(HERE, "RESULTS_mfblms.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("[written] RESULTS_mfblms.md", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "check":
        check()
    elif mode == "diag":
        diag()
    else:
        main()
