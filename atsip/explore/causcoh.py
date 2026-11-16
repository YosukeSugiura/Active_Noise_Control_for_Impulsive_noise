#!/usr/bin/env python3
"""Causal-energy coherence driven lam for the delayless FD-FxNLMS, E_M framing (main).

Per block the smoothed normalised cross-spectrum C(k) = Sre(k)/sqrt(Srr(k) See(k) + eps) is
inverse-transformed to c[n] (length N). With gcaus = sum_{n<M} c^2, gacaus = sum_{n>=M} c^2,
gtot = sum_n c^2:
    lam  = clip((gcaus - gacaus)/(gtot + tiny), 0.02, 1)     (noise-floor subtracted)
    lam2 = clip( gcaus / (gtot + tiny), 0.02, 1)             (no subtraction)
The lam of block i-1 is used inside the update of block i (causal).

Candidates Q1..Q6 (E_M), references B0_M, B2_M, C5_M, T3_M (projlam.simulate, same code path).
Reuses data, framing, prepass and metrics from projlam.py / mixlam.py / explore.py / genie.py.

Usage:  python3 atsip/explore/causcoh.py
Writes  atsip/explore/out/causcoh_*.{csv,json,npz,png} and atsip/explore/RESULTS_causcoh.md.
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
import explore as X    # noqa: E402
import genie as G      # noqa: E402
import mixlam as ML    # noqa: E402
import projlam as PL   # noqa: E402

O = X.O
FS, M, N, B, NB = X.FS, X.M, X.N, X.B, X.NB
OUT = os.path.join(HERE, "out")
MUS = ML.MUS
C2 = ML.C2["M"]
TINY = 1e-30
LAM_MIN = 0.02
SIX = ML.SIX
TWO = ML.TWO
QCANDS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
REFS = ["B0_M", "B2_M", "C5_M", "T3_M"]
A_MAIN = X.ALPHA           # 0.05
A_FAST = 0.2               # Q6, C-spectra only
DESC = {
    "Q1": "Lam=(1-lam)Pbar+lam Srr+eps, lam=clip((gcaus-gacaus)/gtot) of block i-1, no shrink (a=0.05)",
    "Q2": "Q1 with gain x lam (a=0.05)",
    "Q3": "Lam mixed with sqrt(lam), no shrink (a=0.05)",
    "Q4": "Q3 with gain x sqrt(lam) (a=0.05)",
    "Q5": "Q1 with lam2=clip(gcaus/gtot), no subtraction (a=0.05)",
    "Q6": "Q2 with a=0.2 spectra inside C only (normaliser Srr/Pbar at 0.05)",
    "B0_M": "ref: Lam=Srr+eps, binwise (projlam B0)",
    "B2_M": "ref: Lam_mix(0.45), fixed (mixlam B2)",
    "C5_M": "ref: lam=clip(sqrt(gam_prev),0.1,1), no shrink (mixlam C5)",
    "T3_M": "ref: C5 with gain x lam (mixlam T3)",
}


# ----------------------------------------------------------------------------
# Own EMA state for the cross-spectrum in C (a given per candidate)
# ----------------------------------------------------------------------------
class CSpec:
    def __init__(self, a):
        self.a = a
        self.Sre = self.Srr = self.See = None

    def step(self, R, E):
        A = np.abs(R) ** 2
        cre = np.conj(R) * E
        cee = np.abs(E) ** 2
        if self.Srr is None:
            self.Srr, self.Sre, self.See = A.copy(), cre.copy(), cee.copy()
        else:
            a = self.a
            self.Srr = (1 - a) * self.Srr + a * A
            self.Sre = (1 - a) * self.Sre + a * cre
            self.See = (1 - a) * self.See + a * cee
        return self.Sre, self.Srr, self.See


def causal_lam(Sre, Srr, See, eps):
    """Returns (lam, lam2, gtot, gcaus, gacaus) for one block from the smoothed spectra."""
    C = Sre / np.sqrt(Srr * See + eps)
    c = np.fft.irfft(C, N)
    c2 = c * c
    gtot = float(c2.sum())
    gcaus = float(c2[:M].sum())
    gacaus = float(c2[M:].sum())
    lam = float(np.clip((gcaus - gacaus) / (gtot + TINY), LAM_MIN, 1.0))
    lam2 = float(np.clip(gcaus / (gtot + TINY), LAM_MIN, 1.0))
    return lam, lam2, gtot, gcaus, gacaus


# ----------------------------------------------------------------------------
# Block simulation: w <- w - mu_b * irfft(G)[:M], mu_b = 2 mu B / M
# ----------------------------------------------------------------------------
def simulate(data, pp, cand, mu, track=False):
    d, xq, hS, S2, w_opt = (data[k] for k in ("d", "xq", "hS", "S2", "w_opt"))
    L = len(d)
    nb = L // B
    swv = sliding_window_view(xq, M)
    Rall, Psi = pp["Rall"], pp["Psi"]["M"]
    w = np.zeros(M)
    zi = np.zeros(len(hS) - 1)
    e_all = np.zeros(L)
    mu_b = 2 * mu * B / M
    shared = X.Shared(S2)
    cs = CSpec(A_FAST if cand == "Q6" else A_MAIN)
    lam_prev, lam2_prev = 1.0, 1.0
    keys = ("lam_used", "lam_now", "lam2_now", "gtot", "gcaus", "gacaus", "gam", "lam_true")
    tr = {k: [] for k in keys} if track else None
    diverged = False

    for i in range(nb):
        s0 = i * B
        Xb = swv[s0 + N - M + 1: s0 + N - M + 1 + B][:, ::-1]
        y = Xb @ w
        ys, zi = signal.lfilter(hS, [1.0], y, zi=zi)
        e_all[s0:s0 + B] = d[s0:s0 + B] + ys
        R = Rall[i]
        E = ML.frame_E(e_all, s0, M)
        s = shared.step(R, E)
        Srr, Pbar, eps = s["Srr"], s["Pbar"], s["eps"]
        Sre_c, Srr_c, See_c = cs.step(R, E)
        Ce = np.conj(R) * E

        if track:
            Pd = G.true_Pd(w, w_opt)
            q = C2 * Srr * Pd
            tr["lam_true"].append(float(np.sum(q) / np.sum(q + Psi)))
            tr["gam"].append(float(s["gam_prev"].mean()))
            tr["lam_used"].append(lam_prev)

        lam_now, lam2_now, gtot, gcaus, gacaus = causal_lam(Sre_c, Srr_c, See_c, eps)

        if cand in ("Q1", "Q2", "Q6"):
            Lam = (1 - lam_prev) * Pbar + lam_prev * Srr + eps
            Gf = (lam_prev if cand in ("Q2", "Q6") else 1.0) * Ce / Lam
        elif cand in ("Q3", "Q4"):
            l = np.sqrt(lam_prev)
            Lam = (1 - l) * Pbar + l * Srr + eps
            Gf = (l if cand == "Q4" else 1.0) * Ce / Lam
        elif cand == "Q5":
            Lam = (1 - lam2_prev) * Pbar + lam2_prev * Srr + eps
            Gf = Ce / Lam
        elif cand == "B0_M":
            Gf = Ce / (Srr + eps)
        else:
            raise ValueError(cand)

        if track:
            tr["lam_now"].append(lam_now)
            tr["lam2_now"].append(lam2_now)
            tr["gtot"].append(gtot)
            tr["gcaus"].append(gcaus)
            tr["gacaus"].append(gacaus)

        g = np.fft.irfft(Gf, N)[:M]
        w = w - mu_b * g
        if not np.isfinite(w).all():
            e_all[s0 + B:] = np.nan
            diverged = True
            break
        lam_prev, lam2_prev = lam_now, lam2_now

    track_out = None
    if track:
        track_out = {k: np.asarray(v) for k, v in tr.items()}
    return dict(e=e_all, w=w, diverged=diverged, track=track_out)


def floor_trace(data, pp):
    """Same chain for Q2 with w frozen at w_opt (E_M framing): the floor reference."""
    x, d, hS, S2, w_opt = (data[k] for k in ("x", "d", "hS", "S2", "w_opt"))
    L = len(d)
    nb = L // B
    e_opt = d + signal.lfilter(hS, [1.0], np.convolve(x, w_opt)[:L])
    shared = X.Shared(S2)
    cs = CSpec(A_MAIN)
    lam_prev = 1.0
    out = {k: [] for k in ("lam_used", "lam_now", "gtot", "gcaus", "gacaus")}
    for i in range(nb):
        s0 = i * B
        R = pp["Rall"][i]
        E = ML.frame_E(e_opt, s0, M)
        s = shared.step(R, E)
        Sre_c, Srr_c, See_c = cs.step(R, E)
        lam_now, _, gtot, gcaus, gacaus = causal_lam(Sre_c, Srr_c, See_c, s["eps"])
        out["lam_used"].append(lam_prev)
        out["lam_now"].append(lam_now)
        out["gtot"].append(gtot)
        out["gcaus"].append(gcaus)
        out["gacaus"].append(gacaus)
        lam_prev = lam_now
    return {k: np.asarray(v) for k, v in out.items()}


# ----------------------------------------------------------------------------
# Workers
# ----------------------------------------------------------------------------
def run_one(data, pp, label, mu):
    if label in QCANDS:
        return simulate(data, pp, label, mu)
    return PL.simulate(data, pp, label[:-2], mu, "M")   # B0_M -> B0, etc.


def run_grid(spec):
    tag, kind, arg, labels = spec
    t0 = time.time()
    data = G.load_data(kind, arg)
    d, w_opt = data["d"], data["w_opt"]
    pp = ML.prepass(data)
    rows, anrs = [], {}
    for label in labels:
        for mu in MUS:
            sim = run_one(data, pp, label, mu)
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
    sim = simulate(data, pp, "Q2", mu, track=True)
    tr = sim["track"]
    nb_run = len(tr["lam_used"])
    fl = floor_trace(data, pp)
    t = np.arange(nb_run) * B / FS
    i = np.arange(nb_run)
    wins = [("0-1 s", t < 1), ("1-5 s", (t >= 1) & (t < 5)),
            ("5-15 s", (t >= 5) & (t < 15)), ("last 10 %", i >= nb_run - int(0.1 * nb_run))]
    wv = {}
    for wn, sel in wins:
        wv[wn] = {k: float(np.mean(tr[k][sel])) for k in ("lam_used", "lam_now", "gcaus", "gacaus", "gtot", "lam_true")}
        wv[wn].update({f"floor_{k}": float(np.mean(fl[k][sel])) for k in ("lam_used", "gcaus", "gacaus", "gtot")})
    ii = np.arange(1, nb_run)
    spear = dict(lam_used=X.spearman(tr["lam_used"][ii], tr["lam_true"][ii]),
                 lam_now=X.spearman(tr["lam_now"][ii], tr["lam_true"][ii]),
                 gam=X.spearman(tr["gam"][ii], tr["lam_true"][ii]))
    np.savez(os.path.join(OUT, f"causcoh_diag_{tag}_Q2.npz"),
             **{k: tr[k] for k in tr}, **{f"floor_{k}": fl[k] for k in fl})
    m = X.metrics(sim["e"], sim["w"], d, w_opt)
    print(f"[diag] {tag} runtime={time.time() - t0:.1f}s", flush=True)
    return dict(tag=tag, mu=mu, nb=nb_run, windows=wv, spearman=spear,
                floor_lam_end=float(fl["lam_now"][-1]), whole=m["whole"], diverged=sim["diverged"])


def fmt(v, p=2):
    return G.fmt(v, p)


def main():
    os.makedirs(OUT, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    labels = QCANDS + REFS
    names = labels
    kinds = {tag: (kind, arg) for tag, kind, arg in G.SOURCES}

    # ---- Stage 1: all candidates and references, all eight sources, E_M
    with Pool(processes=2) as pool:
        res = pool.map(run_grid, [(tag, kind, arg, labels) for tag, kind, arg in G.SOURCES])
    cell = {}
    for r in res:
        for row in r["rows"]:
            cell.setdefault((row["cand"], row["mu"]), {})[r["tag"]] = row

    def agg(c, mu, key):
        return G.nan_mean([cell[(c, mu)][t][key] for t in SIX])

    def ndiv6(c, mu):
        return int(sum(cell[(c, mu)][t]["diverged"] for t in SIX))

    def ndiv_sweep(c):
        return int(sum(ndiv6(c, mu) for mu in MUS))

    def best(c, key):
        ok = {mu: agg(c, mu, key) for mu in MUS if np.isfinite(agg(c, mu, key))}
        return min(ok, key=ok.get) if ok else None

    bw = {c: best(c, "whole") for c in names}
    bl = {c: best(c, "last10") for c in names}

    # ---- Sanity: references vs projlam_grid.csv and mixlam_grid.csv (same code path)
    max_diff, n_cmp = 0.0, 0
    cmp_rows = {}
    for fn, mapping in (("projlam_grid.csv", {"B0_M": "B0_M", "B2_M": "B2_M", "C5_M": "C5_M", "T3_M": "T3_M"}),
                        ("mixlam_grid.csv", {"B2_M": "B2_M", "C5_M": "C5_M", "T3_M": "T3_M"})):
        with open(os.path.join(OUT, fn)) as fh:
            for row in csv.DictReader(fh):
                c = row["cand"]
                if c not in mapping or row["source"] not in SIX + TWO:
                    continue
                key = (c, float(row["mu"]))
                if key in cell and row["source"] in cell[key]:
                    mine = cell[key][row["source"]]["whole"]
                    ref = float(row["whole_db"])
                    if np.isfinite(mine) and np.isfinite(ref):
                        max_diff = max(max_diff, abs(mine - ref))
                    n_cmp += 1
    # ---- Sanity: alpha=1.45 ANR of the reference candidates vs projlam_anr_a145.npz is done by whole-curve check above

    best_Q = min(QCANDS, key=lambda c: agg(c, bw[c], "whole") if bw[c] is not None else 1e9)

    # ---- CSV
    with open(os.path.join(OUT, "causcoh_grid.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["cand", "mu", "source", "whole_db", "first5s_db", "last10_db", "final_msd_db", "diverged"])
        for r in res:
            for row in r["rows"]:
                wr.writerow([row["cand"], row["mu"], r["tag"], row["whole"], row["first5"],
                             row["last10"], row["msd"], int(row["diverged"])])

    # ---- Diagnostics: Q2 at its six-source best mu (E_M), alpha1.45 and cms145_s0
    mu_Q2 = bw["Q2"]
    diag_specs = [("a145",) + kinds["a145"] + (mu_Q2,), ("cms145_s0",) + kinds["cms145_s0"] + (mu_Q2,)]
    with Pool(processes=2) as pool:
        diag = pool.map(diag_source, diag_specs)

    # ---- Figure (alpha=1.45, E_M): B0_M, B2_M, T3_M, best Q, O_band
    arr = np.load(os.path.join(OUT, "anr_a145.npz"))
    g_arr = {}
    for r in res:
        if r["tag"] == "a145":
            g_arr = r["anrs"]
    np.savez(os.path.join(OUT, "causcoh_anr_a145.npz"), **g_arr)
    fig_list = [("B0_M", bw["B0_M"]), ("B2_M", bw["B2_M"]), ("T3_M", bw["T3_M"]), (best_Q, bw[best_Q])]
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
    fig.savefig(os.path.join(OUT, "causcoh_anr_a145.png"), dpi=130)
    plt.close(fig)

    runtime = time.time() - t0

    # ---- Report
    L = []
    L.append("# Causal-energy coherence lam for the delayless FD-FxNLMS (atsip/explore/causcoh.py)\n")
    L.append("Numbers only. Generated by `atsip/explore/causcoh.py`.\n")
    L.append(f"Runtime (wall clock, 2 worker processes): {runtime:.1f} s. Per-source grid runtime [s]: "
             + ", ".join(f"{r['tag']} {r['runtime_s']:.1f}" for r in res) + ".\n")
    L.append("Six sources = " + ", ".join(SIX) + "; alpha=1.2 (reported separately, per seed) = " + ", ".join(TWO) + ".\n")
    L.append("Framing E_M: last M=512 errors. Means over the six sources; nan if any source is non-finite. "
             "#div over sweep = diverged runs of the six sources over all 7 mu (of 42). "
             "Divergence rule as explore.py: non-finite output or last-10% ANR > -5 dB.\n")
    L.append("## Setup checks\n")
    L.append(f"- References (B0_M, B2_M, C5_M, T3_M) vs projlam_grid.csv / mixlam_grid.csv (same sources, same mu): "
             f"{n_cmp} cells, max |diff| of whole-curve ANR = {max_diff:.2e} dB.")
    L.append(f"- Best Q candidate by six-source whole mean (E_M), used in the figure: {best_Q} (mu={bw[best_Q]}).")
    L.append(f"- Q2 six-source best mu (whole): {mu_Q2}.\n")

    def table(bmap, title):
        L.append(f"## {title}\n")
        L.append("| cand | description | best mu | whole [dB] | first5s [dB] | last10 [dB] | final MSD [dB] | #div over sweep (of 42) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for c in names:
            m = bmap[c]
            if m is None:
                L.append(f"| {c} | {DESC[c]} | nan | nan | nan | nan | nan | {ndiv_sweep(c)} |")
                continue
            L.append(f"| {c} | {DESC[c]} | {m} | {fmt(agg(c, m, 'whole'))} | {fmt(agg(c, m, 'first5'))} | "
                     f"{fmt(agg(c, m, 'last10'))} | {fmt(agg(c, m, 'msd'))} | {ndiv_sweep(c)} |")
        L.append("")

    table(bw, "A. E_M, best mu by whole-curve mean (six sources)")
    table(bl, "B. E_M, best mu by last-10% mean (six sources)")

    L.append("## C. alpha=1.2 sources (seeds 0 and 1) at the six-source best mu from table A (E_M)\n")
    L.append("| cand | best mu | whole s0 | whole s1 | first5 s0 | first5 s1 | last10 s0 | last10 s1 | MSD s0 | MSD s1 | div s0 / s1 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in names:
        m = bw[c]
        s0, s1 = cell[(c, m)]["a120_s0"], cell[(c, m)]["a120_s1"]
        L.append(f"| {c} | {m} | {fmt(s0['whole'])} | {fmt(s1['whole'])} | {fmt(s0['first5'])} | {fmt(s1['first5'])} | "
                 f"{fmt(s0['last10'])} | {fmt(s1['last10'])} | {fmt(s0['msd'])} | {fmt(s1['msd'])} | "
                 f"{int(s0['diverged'])} / {int(s1['diverged'])} |")
    L.append("")

    L.append("## D. Diagnostics: Q2 at its six-source best mu (E_M)\n")
    L.append(f"Q2 mu used: {mu_Q2}. Per block: lam_used = lam of block i-1 (used in the update), lam_now = clip((gcaus-gacaus)/gtot) "
             "from the spectra of block i, gcaus/gacaus/gtot = energies of c (causal half, acausal half, total). "
             "lam_true = scalar lam of mixlam.py GL1 (E_M). floor_* = same chain with w frozen at w_opt. "
             "Time averages over blocks in each window; windows by block start time; last 10 % by block index.\n")
    L.append("| source | window | lam_used | lam_now | gcaus | gacaus | gtot | lam_true | floor lam_used | floor gcaus | floor gacaus | floor gtot |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for dg in diag:
        for wn, v in dg["windows"].items():
            L.append(f"| {dg['tag']} | {wn} | {fmt(v['lam_used'],3)} | {fmt(v['lam_now'],3)} | {fmt(v['gcaus'],4)} | "
                     f"{fmt(v['gacaus'],4)} | {fmt(v['gtot'],4)} | {fmt(v['lam_true'],3)} | {fmt(v['floor_lam_used'],3)} | "
                     f"{fmt(v['floor_gcaus'],4)} | {fmt(v['floor_gacaus'],4)} | {fmt(v['floor_gtot'],4)} |")
    L.append("")
    L.append("| source | blocks | Spearman(lam_used, lam_true) | Spearman(lam_now, lam_true) | Spearman(mean gam_prev, lam_true) | "
             "floor lam_now at end | whole [dB] | diverged |")
    L.append("|---|---|---|---|---|---|---|---|")
    for dg in diag:
        sp = dg["spearman"]
        L.append(f"| {dg['tag']} | {dg['nb']} | {fmt(sp['lam_used'],3)} | {fmt(sp['lam_now'],3)} | {fmt(sp['gam'],3)} | "
                 f"{fmt(dg['floor_lam_end'],3)} | {fmt(dg['whole'])} | {dg['diverged']} |")
    L.append("")

    L.append("## Appendix 1. Full grid, whole-curve mean ANR [dB] (six sources, E_M)\n")
    L.append("| cand | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for c in names:
        L.append(f"| {c} | " + " | ".join(fmt(agg(c, m, "whole")) for m in MUS) + " |")
    L.append("")
    L.append("## Appendix 2. Full grid, last-10% mean ANR [dB] (six sources, E_M)\n")
    L.append("| cand | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for c in names:
        L.append(f"| {c} | " + " | ".join(fmt(agg(c, m, "last10")) for m in MUS) + " |")
    L.append("")
    L.append("## Appendix 3. Diverged runs (of six) per mu (E_M)\n")
    L.append("| cand | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for c in names:
        L.append(f"| {c} | " + " | ".join(str(ndiv6(c, m)) for m in MUS) + " |")
    L.append("")
    L.append("## Appendix 4. alpha=1.2 whole-curve ANR [dB] per seed, every mu (E_M, s0 / s1)\n")
    L.append("| cand | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for c in names:
        cells = [f"{fmt(cell[(c,m)]['a120_s0']['whole'])} / {fmt(cell[(c,m)]['a120_s1']['whole'])}" for m in MUS]
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Figure\n")
    L.append("- `out/causcoh_anr_a145.png`: ANR, alpha=1.45 file, E_M, at six-source best mu: "
             + ", ".join(f"{c} (mu={m})" for c, m in fig_list) + ", plus O_band (dashed).\n")
    L.append("## Files\n")
    L.append("- `out/causcoh_grid.csv`: every run (candidate, mu, source), E_M.")
    L.append("- `out/causcoh_anr_a145.npz`: ANR curves of the alpha=1.45 runs.")
    L.append("- `out/causcoh_diag_<source>_Q2.npz`: per-block lam_used, lam_now, lam2_now, gtot, gcaus, gacaus, gam, lam_true, and floor traces.")
    L.append("")
    with open(os.path.join(HERE, "RESULTS_causcoh.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    summ = dict(runtime_s=runtime, best_whole=bw, best_last10=bl, best_Q=best_Q, mu_Q2=mu_Q2,
                max_diff_refs=max_diff, n_cmp=n_cmp,
                diag=[dict(tag=dg["tag"], mu=dg["mu"], nb=dg["nb"], windows=dg["windows"],
                           spearman=dg["spearman"], floor_lam_end=dg["floor_lam_end"],
                           whole=dg["whole"], diverged=dg["diverged"]) for dg in diag])
    with open(os.path.join(OUT, "causcoh_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1, default=float)
    print(f"[total] {runtime:.1f} s", flush=True)


if __name__ == "__main__":
    main()
