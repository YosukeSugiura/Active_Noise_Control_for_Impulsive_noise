#!/usr/bin/env python3
"""Reducible-fraction (rho) driven lam for the delayless FD-FxNLMS, E_M framing (main) and E_B (best two).

rho_i = ||h[0:M]||^2 / ||h||^2 with h = irfft(conj(R) E / Lam, N): the causal, in-range lags.
rhob = EMA_a(rho); rho0 = slow running minimum of rhob; lam = clip((rhob - rho0)/(1 - rho0), 0.02, 1).
Candidates P1..P5 plus references B2_M, C5_M, T3_M (from mixlam.py) and B0_M (binwise, E_M).

Reuses sources, prepass, framing, practical lam rules and metrics from mixlam.py / explore.py / genie.py.

Usage:  python3 atsip/explore/projlam.py
Writes  atsip/explore/out/projlam_*.{csv,json,npz,png} and atsip/explore/RESULTS_projlam.md.
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
import explore as X  # noqa: E402
import genie as G    # noqa: E402
import mixlam as ML  # noqa: E402

O = X.O
FS, M, N, B, NB = X.FS, X.M, X.N, X.B, X.NB
OUT = os.path.join(HERE, "out")
MUS = ML.MUS
FRAMES = ML.FRAMES
C2 = ML.C2
TINY = ML.TINY
SIX = ML.SIX
TWO = ML.TWO
LAM_MIN = 0.02
RHO_DRIFT = 1e-4
PCANDS = ["P1", "P2", "P3", "P4", "P5"]
REFBASE = {"B2_M": "B2", "C5_M": "C5", "T3_M": "T3", "B0_M": "B0"}
DESC = {
    "P1": "Lam=(1-lam)Pbar+lam Srr+eps, lam=excess(rho), no shrink (a=0.05)",
    "P2": "P1 with gain x lam (a=0.05)",
    "P3": "P2 with lam=clip(rhob,0.02,1), no floor subtraction (a=0.05)",
    "P4": "P2 with lam'=sqrt(excess) in Lam and as shrink (a=0.05)",
    "P5": "P2 with a=0.2",
    "B2_M": "ref: Lam_mix(0.45), fixed (mixlam B2)",
    "C5_M": "ref: lam=clip(sqrt(gam_prev),0.1,1), no shrink (mixlam C5)",
    "T3_M": "ref: C5 with gain x lam (mixlam T3)",
    "B0_M": "ref: Lam=Srr+eps, binwise",
}
EB_CANDS = None  # set after stage 1


# ----------------------------------------------------------------------------
# rho state (EMA, slow running minimum, excess)
# ----------------------------------------------------------------------------
class RhoState:
    def __init__(self, a):
        self.a = a
        self.rhob = None
        self.rho0 = None

    def update(self, rho):
        if self.rhob is None:                   # block 1 initialises both
            self.rhob = rho
            self.rho0 = rho
        else:
            self.rhob = (1 - self.a) * self.rhob + self.a * rho
            self.rho0 = min(self.rho0 * (1 + RHO_DRIFT), self.rhob)
        return (self.rhob - self.rho0) / max(1 - self.rho0, 1e-9)


def rho_of(h):
    return float(np.sum(h[:M] ** 2) / (np.sum(h ** 2) + TINY))


def pcand_lam(base, ex, rhob):
    """Returns (lam, shrink) for the P candidates; lam scalar, effective (after sqrt for P4)."""
    if base == "P3":
        return float(np.clip(rhob, LAM_MIN, 1.0)), True
    lam = float(np.clip(ex, LAM_MIN, 1.0))
    if base == "P4":
        return float(np.sqrt(lam)), True
    if base == "P1":
        return lam, False
    return lam, True                             # P2, P5


# ----------------------------------------------------------------------------
# Block simulation: w <- w - mu_b * irfft(G)[:M], mu_b = 2 mu B / M
# ----------------------------------------------------------------------------
def simulate(data, pp, base, mu, f="M", track=False):
    Lf = FRAMES[f]
    d, xq, hS, S2, w_opt = (data[k] for k in ("d", "xq", "hS", "S2", "w_opt"))
    L = len(d)
    nb = L // B
    swv = sliding_window_view(xq, M)
    Rall, Psi = pp["Rall"], pp["Psi"][f]
    w = np.zeros(M)
    zi = np.zeros(len(hS) - 1)
    e_all = np.zeros(L)
    mu_b = 2 * mu * B / M
    shared = X.Shared(S2)
    a = 0.2 if base == "P5" else 0.05
    rs = RhoState(a)
    lam_prev = 1.0                               # normaliser lam, start 1.0
    lam_bb_prev = 1.0                            # only used by practical rules, not by refs here
    tr = dict(rho=[], rhob=[], rho0=[], lam=[], lam_true=[]) if track else None
    diverged = False

    for i in range(nb):
        s0 = i * B
        Xb = swv[s0 + N - M + 1: s0 + N - M + 1 + B][:, ::-1]
        y = Xb @ w
        ys, zi = signal.lfilter(hS, [1.0], y, zi=zi)
        e_all[s0:s0 + B] = d[s0:s0 + B] + ys
        R = Rall[i]
        E = ML.frame_E(e_all, s0, Lf)
        s = shared.step(R, E)
        Srr, Pbar, eps, gp = s["Srr"], s["Pbar"], s["eps"], s["gam_prev"]
        Ce = np.conj(R) * E

        if track:
            Pd = G.true_Pd(w, w_opt)
            q = C2[f] * Srr * Pd
            tr["lam_true"].append(float(np.sum(q) / np.sum(q + Psi)))

        if base in PCANDS:
            Lm = (1 - lam_prev) * Pbar + lam_prev * Srr + eps
            h = np.fft.irfft(Ce / Lm, N)              # full irfft; rho from lam_{i-1} normaliser
            rho = rho_of(h)
            ex = rs.update(rho)
            lam, shrink = pcand_lam(base, ex, rs.rhob)
            Lam = (1 - lam) * Pbar + lam * Srr + eps
            Gf = (lam if shrink else 1.0) * Ce / Lam
            lam_prev = lam
            if track:
                tr["rho"].append(rho)
                tr["rhob"].append(rs.rhob)
                tr["rho0"].append(rs.rho0)
                tr["lam"].append(lam)
        elif base == "B0":
            Gf = Ce / (Srr + eps)
        else:
            lam, mult = ML.practical_lam(base, gp, lam_bb_prev)
            Lam = (1 - lam) * Pbar + lam * Srr + eps
            Gf = (lam if mult else 1.0) * Ce / Lam

        g = np.fft.irfft(Gf, N)[:M]
        w = w - mu_b * g
        if not np.isfinite(w).all():
            e_all[s0 + B:] = np.nan
            diverged = True
            break

    track_out = None
    if track:
        track_out = {k: np.asarray(v) for k, v in tr.items()}
    return dict(e=e_all, w=w, diverged=diverged, track=track_out)


def floor_trace(data, pp, a=0.05):
    """rho and lam chain (P2 rule) with w frozen at w_opt: the floor reference."""
    x, d, hS, S2, w_opt = (data[k] for k in ("x", "d", "hS", "S2", "w_opt"))
    L = len(d)
    nb = L // B
    e_opt = d + signal.lfilter(hS, [1.0], np.convolve(x, w_opt)[:L])
    shared = X.Shared(S2)
    rs = RhoState(a)
    lam_prev = 1.0
    out = {k: [] for k in ("rho", "rhob", "rho0", "lam")}
    for i in range(nb):
        s0 = i * B
        R = pp["Rall"][i]
        E = ML.frame_E(e_opt, s0, M)
        s = shared.step(R, E)
        Srr, Pbar, eps = s["Srr"], s["Pbar"], s["eps"]
        Ce = np.conj(R) * E
        Lm = (1 - lam_prev) * Pbar + lam_prev * Srr + eps
        rho = rho_of(np.fft.irfft(Ce / Lm, N))
        ex = rs.update(rho)
        lam = float(np.clip(ex, LAM_MIN, 1.0))
        lam_prev = lam
        out["rho"].append(rho)
        out["rhob"].append(rs.rhob)
        out["rho0"].append(rs.rho0)
        out["lam"].append(lam)
    return {k: np.asarray(v) for k, v in out.items()}


# ----------------------------------------------------------------------------
# Workers
# ----------------------------------------------------------------------------
def run_grid(spec):
    """spec = (tag, kind, arg, [(label, base, f), ...]); runs every label over MUS."""
    tag, kind, arg, labels = spec
    t0 = time.time()
    data = G.load_data(kind, arg)
    d, w_opt = data["d"], data["w_opt"]
    pp = ML.prepass(data)
    rows, anrs = [], {}
    for label, base, f in labels:
        for mu in MUS:
            sim = simulate(data, pp, base, mu, f)
            m = X.metrics(sim["e"], sim["w"], d, w_opt)
            rows.append(dict(cand=label, mu=mu, **m))
            if tag == "a145":
                anrs[f"{label}__{mu}"] = O.anr_curve(sim["e"], d)
    rt = time.time() - t0
    print(f"[grid] {tag} {len(labels)} labels x {len(MUS)} mu runtime={rt:.1f}s", flush=True)
    return dict(tag=tag, rows=rows, anrs=anrs, runtime_s=rt)


def diag_source(spec):
    tag, kind, arg, mu = spec
    t0 = time.time()
    data = G.load_data(kind, arg)
    d, w_opt = data["d"], data["w_opt"]
    pp = ML.prepass(data)
    sim = simulate(data, pp, "P2", mu, "M", track=True)
    tr = sim["track"]
    nb_run = len(tr["lam"])
    fl = floor_trace(data, pp)
    t = np.arange(nb_run) * B / FS
    i = np.arange(nb_run)
    wins = [("0-1 s", t < 1), ("1-5 s", (t >= 1) & (t < 5)),
            ("5-15 s", (t >= 5) & (t < 15)), ("last 10 %", i >= nb_run - int(0.1 * nb_run))]
    keys = ("rho", "rhob", "rho0", "lam", "lam_true")
    wv = {}
    for wn, sel in wins:
        wv[wn] = {k: float(np.mean(tr[k][sel])) for k in keys}
        wv[wn]["rho_floor"] = float(np.mean(fl["rho"][sel]))
        wv[wn]["rhob_floor"] = float(np.mean(fl["rhob"][sel]))
    ii = np.arange(1, nb_run)
    spear = X.spearman(tr["lam"][ii], tr["lam_true"][ii])
    np.savez(os.path.join(OUT, f"projlam_diag_{tag}_P2.npz"),
             **{k: tr[k] for k in keys}, **{f"floor_{k}": fl[k] for k in fl})
    m = X.metrics(sim["e"], sim["w"], d, w_opt)
    print(f"[diag] {tag} runtime={time.time() - t0:.1f}s", flush=True)
    return dict(tag=tag, mu=mu, nb=nb_run, windows=wv, spearman=spear,
                rho0_end=float(tr["rho0"][-1]), rho0_floor_end=float(fl["rho0"][-1]),
                rhob_floor_end=float(fl["rhob"][-1]), whole=m["whole"], diverged=sim["diverged"])


def fmt(v, p=2):
    return G.fmt(v, p)


def main():
    os.makedirs(OUT, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    ref_labels = list(REFBASE.keys())
    labels_M = [(c, c, "M") for c in PCANDS] + [(r, REFBASE[r], "M") for r in ref_labels]
    names_M = [l for l, _, _ in labels_M]
    desc_of = lambda c: DESC[c]

    # ---- Stage 1: E_M, all candidates and references, all sources
    with Pool(processes=2) as pool:
        res1 = pool.map(run_grid, [(tag, kind, arg, labels_M) for tag, kind, arg in G.SOURCES])
    by1 = {r["tag"]: r for r in res1}
    cell = {}
    for r in res1:
        for row in r["rows"]:
            cell.setdefault((row["cand"], row["mu"]), {})[r["tag"]] = row

    def agg(cands, mu, key):
        return G.nan_mean([cell[(cands, mu)][t][key] for t in SIX])

    def ndiv6(c, mu):
        return int(sum(cell[(c, mu)][t]["diverged"] for t in SIX))

    def ndiv_sweep(c):
        return int(sum(ndiv6(c, mu) for mu in MUS))

    def best(c, key):
        ok = {mu: agg(c, mu, key) for mu in MUS if np.isfinite(agg(c, mu, key))}
        return min(ok, key=ok.get) if ok else None

    bw = {c: best(c, "whole") for c in names_M}
    bl = {c: best(c, "last10") for c in names_M}

    # ---- Sanity: references vs mixlam_grid.csv (same code path)
    mix = {}
    with open(os.path.join(OUT, "mixlam_grid.csv")) as fh:
        for row in csv.DictReader(fh):
            mix[(row["cand"], float(row["mu"]), row["source"])] = row
    max_diff, n_cmp = 0.0, 0
    for c in ("B2_M", "C5_M", "T3_M"):
        for mu in MUS:
            for t in SIX + TWO:
                key = (c, float(mu), t)
                if key in mix and t in cell[(c, mu)]:
                    ref = float(mix[key]["whole_db"])
                    mine = cell[(c, mu)][t]["whole"]
                    if np.isfinite(ref) and np.isfinite(mine):
                        max_diff = max(max_diff, abs(mine - ref))
                    n_cmp += 1
    # ---- Stage 2: E_B for the best two new (P) candidates by six-source whole mean
    best_two = sorted(PCANDS, key=lambda c: agg(c, bw[c], "whole") if bw[c] is not None else 1e9)[:2]
    labels_B = [(f"{c}_B", c, "B") for c in best_two]
    names_B = [l for l, _, _ in labels_B]
    with Pool(processes=2) as pool:
        res2 = pool.map(run_grid, [(tag, kind, arg, labels_B) for tag, kind, arg in G.SOURCES])
    for r in res2:
        for row in r["rows"]:
            cell.setdefault((row["cand"], row["mu"]), {})[r["tag"]] = row
    # rows for E_B also needed for alpha=1.2 (included above since G.SOURCES has all 8)
    bwB = {c: best(c, "whole") for c in names_B}
    bwB_last = {c: best(c, "last10") for c in names_B}

    # ---- Best P (E_M) for the figure
    bestP = min(PCANDS, key=lambda c: agg(c, bw[c], "whole") if bw[c] is not None else 1e9)

    # ---- Diagnostics: P2 at its six-source best mu (E_M)
    mu_P2 = bw["P2"]
    kinds = {tag: (kind, arg) for tag, kind, arg in G.SOURCES}
    diag_specs = [("a145",) + kinds["a145"] + (mu_P2,),
                  ("cms145_s0",) + kinds["cms145_s0"] + (mu_P2,)]
    with Pool(processes=2) as pool:
        diag = pool.map(diag_source, diag_specs)

    runtime = time.time() - t0

    # ---- CSV
    with open(os.path.join(OUT, "projlam_grid.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["cand", "mu", "source", "whole_db", "first5s_db", "last10_db", "final_msd_db", "diverged"])
        for r in list(res1) + list(res2):
            for row in r["rows"]:
                wr.writerow([row["cand"], row["mu"], r["tag"], row["whole"], row["first5"],
                             row["last10"], row["msd"], int(row["diverged"])])

    # ---- Figure (alpha=1.45, E_M)
    arr = np.load(os.path.join(OUT, "anr_a145.npz"))
    g_arr = by1["a145"]["anrs"]
    np.savez(os.path.join(OUT, "projlam_anr_a145.npz"), **g_arr)
    fig_list = [("B0_M", bw["B0_M"]), ("B2_M", bw["B2_M"]), ("T3_M", bw["T3_M"]), (bestP, bw[bestP])]
    tgrid = lambda n: (np.arange(n) * 64 + 63) / FS
    fig, ax = plt.subplots(figsize=(10, 5))
    for c, mu in fig_list:
        y = g_arr[f"{c}__{mu}"]
        ax.plot(tgrid(len(y)), y, lw=0.8, label=f"{c} (mu={mu})")
    if "O_band" in arr.files:
        y = arr["O_band"]
        ax.plot(tgrid(len(y)), y, lw=0.8, ls="--", label="O_band")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("ANR [dB]")
    ax.set_title("ANR, alpha=1.45 file, E_M (best mu by six-source whole mean)")
    ax.grid(True, ls=":")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "projlam_anr_a145.png"), dpi=130)
    plt.close(fig)

    # ---- Report
    L = []
    L.append("# rho-driven lam for the delayless FD-FxNLMS (atsip/explore/projlam.py)\n")
    L.append("Numbers only. Generated by `atsip/explore/projlam.py`.\n")
    L.append(f"Runtime (wall clock, 2 worker processes): {runtime:.1f} s. Per-stage runtime is in the stage logs.\n")
    L.append("Six sources = " + ", ".join(SIX) + "; alpha=1.2 (reported separately, per seed) = " + ", ".join(TWO) + ".\n")
    L.append("Framing E_M: last M=512 errors. Framing E_B: last B=128 errors (best two new candidates only). "
             "Means over the six sources; nan if any source is non-finite. "
             "#div over sweep = diverged runs of the six sources over all 7 mu (of 42).\n")
    L.append("## Setup checks\n")
    L.append(f"- Reference rows B2_M, C5_M, T3_M vs mixlam_grid.csv (same sources, same mu): {n_cmp} cells, "
             f"max |diff| of whole-curve ANR = {max_diff:.2e} dB.")
    L.append(f"- Best two new candidates by six-source whole mean (E_M): {best_two[0]}, {best_two[1]}.")
    L.append(f"- Best P candidate by six-source whole mean (E_M), used in the figure: {bestP}.\n")

    def table(bmap, labels, title, key):
        L.append(f"## {title}\n")
        L.append("| cand | description | best mu | whole [dB] | first5s [dB] | last10 [dB] | final MSD [dB] | #div over sweep (of 42) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for c in labels:
            m = bmap[c]
            if m is None:
                L.append(f"| {c} | {desc_of(c)} | nan | nan | nan | nan | nan | {ndiv_sweep(c)} |")
                continue
            L.append(f"| {c} | {desc_of(c)} | {m} | {fmt(agg(c, m, 'whole'))} | {fmt(agg(c, m, 'first5'))} | "
                     f"{fmt(agg(c, m, 'last10'))} | {fmt(agg(c, m, 'msd'))} | {ndiv_sweep(c)} |")
        L.append("")

    table(bw, names_M, "A. E_M, best mu by whole-curve mean (six sources)", "whole")
    table(bl, names_M, "B. E_M, best mu by last-10% mean (six sources)", "last10")

    L.append("## C. alpha=1.2 sources (seeds 0 and 1) at the six-source best mu from table A (E_M)\n")
    L.append("| cand | best mu | whole s0 | whole s1 | first5 s0 | first5 s1 | last10 s0 | last10 s1 | MSD s0 | MSD s1 | div s0 / s1 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in names_M:
        m = bw[c]
        s0, s1 = cell[(c, m)]["a120_s0"], cell[(c, m)]["a120_s1"]
        L.append(f"| {c} | {m} | {fmt(s0['whole'])} | {fmt(s1['whole'])} | {fmt(s0['first5'])} | {fmt(s1['first5'])} | "
                 f"{fmt(s0['last10'])} | {fmt(s1['last10'])} | {fmt(s0['msd'])} | {fmt(s1['msd'])} | "
                 f"{int(s0['diverged'])} / {int(s1['diverged'])} |")
    L.append("")

    L.append("## D. E_B framing, two best new candidates (best mu by six-source whole mean, E_B grid)\n")
    L.append("| cand | framing | best mu | whole [dB] | first5s [dB] | last10 [dB] | final MSD [dB] | #div over sweep (of 42) | E_M best mu | E_M whole [dB] |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in names_B:
        m = bwB[c]
        base = c.rsplit("_", 1)[0]
        if m is None:
            L.append(f"| {c} | B | nan | nan | nan | nan | nan | nan | {bw[base]} | {fmt(agg(base, bw[base], 'whole'))} |")
            continue
        L.append(f"| {c} | B | {m} | {fmt(agg(c, m, 'whole'))} | {fmt(agg(c, m, 'first5'))} | {fmt(agg(c, m, 'last10'))} | "
                 f"{fmt(agg(c, m, 'msd'))} | {ndiv_sweep(c)} | {bw[base]} | {fmt(agg(base, bw[base], 'whole'))} |")
    L.append("")
    L.append("E_B alpha=1.2 per seed at the E_B best mu:\n")
    L.append("| cand | best mu | whole s0 | whole s1 | last10 s0 | last10 s1 |")
    L.append("|---|---|---|---|---|---|")
    for c in names_B:
        m = bwB[c]
        if m is None:
            continue
        s0, s1 = cell[(c, m)]["a120_s0"], cell[(c, m)]["a120_s1"]
        L.append(f"| {c} | {m} | {fmt(s0['whole'])} | {fmt(s1['whole'])} | {fmt(s0['last10'])} | {fmt(s1['last10'])} |")
    L.append("")

    L.append("## E. Diagnostics: P2 at its six-source best mu (E_M)\n")
    L.append(f"P2 mu used: {mu_P2}. Per block: rho (from the full irfft with lam_{{i-1}} normaliser), rhob (EMA a=0.05), "
             "rho0 (slow running minimum), lam (effective, excess clipped to [0.02,1]), scalar lam_true (E_M, as mixlam.py GL1). "
             "rho_floor = rho of the same chain while w is frozen at w_opt (E_opt framing). "
             "Time averages over blocks in each window; windows by block start time; last 10 % by block index.\n")
    L.append("| source | window | rho | rhob | rho0 | lam | lam_true | rho_floor | rhob_floor |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for dg in diag:
        for wn, v in dg["windows"].items():
            L.append(f"| {dg['tag']} | {wn} | {fmt(v['rho'],4)} | {fmt(v['rhob'],4)} | {fmt(v['rho0'],4)} | "
                     f"{fmt(v['lam'],3)} | {fmt(v['lam_true'],3)} | {fmt(v['rho_floor'],4)} | {fmt(v['rhob_floor'],4)} |")
    L.append("")
    L.append("| source | blocks | Spearman(lam, lam_true) | rho0 at end (run) | rho0 at end (floor) | rhob at end (floor) | whole [dB] | diverged |")
    L.append("|---|---|---|---|---|---|---|---|")
    for dg in diag:
        L.append(f"| {dg['tag']} | {dg['nb']} | {fmt(dg['spearman'],3)} | {fmt(dg['rho0_end'],4)} | "
                 f"{fmt(dg['rho0_floor_end'],4)} | {fmt(dg['rhob_floor_end'],4)} | {fmt(dg['whole'])} | {dg['diverged']} |")
    L.append("")

    L.append("## Appendix 1. Full grid, whole-curve mean ANR [dB] (six sources, E_M)\n")
    L.append("| cand | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for c in names_M:
        L.append(f"| {c} | " + " | ".join(fmt(agg(c, m, "whole")) for m in MUS) + " |")
    L.append("")
    L.append("## Appendix 2. Full grid, last-10% mean ANR [dB] (six sources, E_M)\n")
    L.append("| cand | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for c in names_M:
        L.append(f"| {c} | " + " | ".join(fmt(agg(c, m, "last10")) for m in MUS) + " |")
    L.append("")
    L.append("## Appendix 3. Diverged runs (of six) per mu (E_M)\n")
    L.append("| cand | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for c in names_M:
        L.append(f"| {c} | " + " | ".join(str(ndiv6(c, m)) for m in MUS) + " |")
    L.append("")
    L.append("## Appendix 4. alpha=1.2 whole-curve ANR [dB] per seed, every mu (E_M, s0 / s1)\n")
    L.append("| cand | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for c in names_M:
        cells = [f"{fmt(cell[(c,m)]['a120_s0']['whole'])} / {fmt(cell[(c,m)]['a120_s1']['whole'])}" for m in MUS]
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Appendix 5. Full grid, whole-curve mean ANR [dB] (six sources, E_B, best two)\n")
    L.append("| cand | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for c in names_B:
        L.append(f"| {c} | " + " | ".join(fmt(agg(c, m, "whole")) for m in MUS) + " |")
    L.append("")
    L.append("## Figure\n")
    L.append("- `out/projlam_anr_a145.png`: ANR, alpha=1.45 file, E_M, at six-source best mu: "
             + ", ".join(f"{c} (mu={m})" for c, m in fig_list) + ", plus O_band (dashed).\n")
    L.append("## Files\n")
    L.append("- `out/projlam_grid.csv`: every run (candidate, mu, source), E_M and E_B.")
    L.append("- `out/projlam_anr_a145.npz`: ANR curves of the alpha=1.45 runs (E_M grid).")
    L.append("- `out/projlam_diag_<source>_P2.npz`: per-block rho, rhob, rho0, lam, lam_true, and floor traces.")
    L.append("")
    with open(os.path.join(HERE, "RESULTS_projlam.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    summ = dict(runtime_s=runtime, best_whole=bw, best_last10=bl, best_two=best_two, bestP=bestP,
                bestB=bwB, mu_P2=mu_P2, max_diff_mixlam=max_diff, n_cmp=n_cmp,
                diag=[{k: v for k, v in dg.items()} for dg in diag])
    with open(os.path.join(OUT, "projlam_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1, default=float)
    print(f"[total] {runtime:.1f} s", flush=True)


if __name__ == "__main__":
    main()
