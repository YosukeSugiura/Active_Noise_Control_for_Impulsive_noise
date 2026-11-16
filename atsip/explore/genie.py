#!/usr/bin/env python3
"""Genie and practical frequency-domain step normalizations for the delayless FD-FxNLMS.

Reuses data, ANR and metrics from atsip/oracle/oracle.py and the B0/B2/C5 directions and the
Shared EMA state from atsip/explore/explore.py. Adds the alpha=1.2 CMS sources (seeds 0, 1).

Usage:  python3 atsip/explore/genie.py
Writes  atsip/explore/out/genie_*.{csv,json,npz,png,log} and atsip/explore/RESULTS_genie.md.
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
import explore as X  # noqa: E402  (B0/B2/C5 directions, Shared state, metrics)

O = X.O
FS, M, N, B, NB, N5 = X.FS, X.M, X.N, X.B, X.NB, X.N5
ROOT = O.ROOT
OUT = os.path.join(HERE, "out")
MUS = [0.1, 0.2, 0.36, 0.6, 1.0, 1.5]
REF = ["B0", "B2", "C5", "B2_M", "C5_M"]
GEN = ["G1", "G2", "G3", "G4", "G5"]
KC = ["K1", "K2", "K3"]
ALL = REF + GEN + KC
BASE = {"B0": "B0", "B2": "B2", "C5": "C5", "B2_M": "B2", "C5_M": "C5",
        "K1": "B0", "K2": "B0", "K3": "B0"}   # K candidates use B0 before block 5
DESC = {
    "B0": "Lam = Srr+eps (explore.py)",
    "B2": "Lam = 0.55 Pbar + 0.45 Srr + eps (explore.py)",
    "C5": "Lam = (1-lam) Pbar + lam Srr + eps, lam = clip(sqrt(gam_prev)) (explore.py)",
    "B2_M": "B2, E = last M=512 errors zero-padded",
    "C5_M": "C5, E = last M=512 errors zero-padded",
    "G1": "genie: Lam = |R|^2 + Psi_stat/Pd",
    "G2": "genie: Lam = Srr + Psi_stat/Pd",
    "G3": "genie: Lam = |R|^2 + eta*P_i/Pd",
    "G4": "genie: Lam = Srr + eta*P_i/Pd",
    "G5": "genie: Lam = Srr + eta*Pbar/Pd",
    "K1": "practical: Lam = Srr + Psi_i/Pk, Psi_i = etah*P_i",
    "K2": "practical: Lam = |R|^2 + Psi_i/Pk, Psi_i = etah*P_i",
    "K3": "practical: Lam = Srr + Psi/Pk, stationary Psi",
}
SOURCES = list(O.SOURCES) + [("a120_s0", "cms12", 0), ("a120_s1", "cms12", 1)]
SIX = [s[0] for s in O.SOURCES]
TWO = ["a120_s0", "a120_s1"]
TINY = 1e-30


# ----------------------------------------------------------------------------
# Data (alpha-parametrised CMS; identical to O.load_source for alpha=1.45)
# ----------------------------------------------------------------------------
def load_cms(alpha, seed):
    x = O.cms_symmetric_stable(alpha, O.N_NOISE, seed)
    hP = O.pad_or_trim(np.loadtxt(os.path.join(ROOT, "impulse_response/primary_ir.dat"), ndmin=1), M)
    hS = O.pad_or_trim(np.loadtxt(os.path.join(ROOT, "impulse_response/secondary_ir.dat"), ndmin=1), O.C_SEC)
    d = signal.lfilter(hP, [1.0], x)
    r = signal.lfilter(hS, [1.0], x)
    data = dict(x=x, d=d, r=r, hS=hS, xq=np.concatenate([np.zeros(N), x]), Sf=np.fft.rfft(hS, N))
    data["w_opt"] = O.wiener_opt(r, d)
    p = np.abs(data["Sf"]) ** 2
    data["S2"] = p / p.mean()
    return data


def load_data(kind, arg):
    if kind == "cms12":
        return load_cms(1.2, arg)
    return O.load_source(kind, arg)


# ----------------------------------------------------------------------------
# Pre-pass with w fixed at w_opt: noise statistics and per-frame power
# ----------------------------------------------------------------------------
def prepass(data):
    x, d, xq, hS, Sf, w_opt = (data[k] for k in ("x", "d", "xq", "hS", "Sf", "w_opt"))
    L = len(x)
    nb = L // B
    e_opt = d + signal.lfilter(hS, [1.0], np.convolve(x, w_opt)[:L])
    Rall = np.empty((nb, NB), dtype=np.complex128)
    Eall = np.empty((nb, NB), dtype=np.complex128)
    for i in range(nb):
        s0 = i * B
        Rall[i] = Sf * np.fft.rfft(xq[s0 + B: s0 + B + N])
        Eall[i] = np.fft.rfft(np.concatenate([np.zeros(N - B), e_opt[s0:s0 + B]]))
    P = np.mean(np.abs(Rall) ** 2, axis=1)
    Ep = np.abs(Eall) ** 2
    Psi = Ep.mean(axis=0)                                       # Psi_stat(k)
    eta = (Ep / np.maximum(P, TINY)[:, None]).mean(axis=0)      # eta(k)
    pe = Ep.mean(axis=1)                                        # mean_k |E_opt,i|^2 per frame
    ok = (P > TINY) & (pe > 0) & np.isfinite(P) & np.isfinite(pe)
    lp, le = np.log(P[ok]), np.log(pe[ok])
    reg_r = float(np.corrcoef(lp, le)[0, 1])
    reg_slope = float(np.polyfit(lp, le, 1)[0])
    return dict(Rall=Rall, Psi=Psi, eta=eta, reg_r=reg_r, reg_slope=reg_slope, n_frames=int(ok.sum()))


def true_Pd(w, w_opt):
    Pd = np.abs(np.fft.rfft(w - w_opt, N)) ** 2
    return np.maximum(Pd, 1e-12 * Pd.max() + 1e-300)


# ----------------------------------------------------------------------------
# Block simulation: w <- w - mu_b * irfft(G)[:M], G = conj(R) E / Lam, mu_b = 2 mu B / M
# ----------------------------------------------------------------------------
def simulate(data, pp, cand, mu, track=False):
    d, xq, hS, S2, w_opt = (data[k] for k in ("d", "xq", "hS", "S2", "w_opt"))
    L = len(d)
    nb = L // B
    swv = sliding_window_view(xq, M)
    Rall, Psi_stat, eta = pp["Rall"], pp["Psi"], pp["eta"]
    w = np.zeros(M)
    zi = np.zeros(len(hS) - 1)
    e_all = np.zeros(L)
    mu_b = 2 * mu * B / M
    shared = X.Shared(S2)
    base = BASE.get(cand, cand)
    use_M = cand.endswith("_M")
    is_k = cand in KC
    need_pd = track or cand in GEN
    n5 = int(round(5 * FS / B))             # blocks in the first 5 s
    i_l10 = nb - int(0.1 * nb)              # first block of the last 10 %
    Pk = Pk_floor = etah = Psi_s = None
    tr = dict(n=0, Sx=0.0, Sy=0.0, Sxx=0.0, Syy=0.0, Sxy=0.0, w5=0.0, n5=0, wl=0.0, nl=0) if track else None
    diverged = False

    for i in range(nb):
        s0 = i * B
        Xb = swv[s0 + N - M + 1: s0 + N - M + 1 + B][:, ::-1]
        y = Xb @ w
        ys, zi = signal.lfilter(hS, [1.0], y, zi=zi)
        e = d[s0:s0 + B] + ys
        e_all[s0:s0 + B] = e
        R = Rall[i]
        A = np.abs(R) ** 2
        P = A.mean()
        E = np.fft.rfft(np.concatenate([np.zeros(N - B), e]))
        if use_M:
            ev = np.zeros(M)
            seg = e_all[max(0, s0 + B - M): s0 + B]
            ev[M - len(seg):] = seg
            Ef = np.fft.rfft(np.concatenate([np.zeros(N - M), ev]), N)
        else:
            Ef = E
        s = shared.step(R, Ef)
        Pd = true_Pd(w, w_opt) if need_pd else None

        if cand in GEN:
            Srr = s["Srr"]
            if cand == "G1":
                Lam = A + Psi_stat / Pd
            elif cand == "G2":
                Lam = Srr + Psi_stat / Pd
            elif cand == "G3":
                Lam = A + eta * P / Pd
            elif cand == "G4":
                Lam = Srr + eta * P / Pd
            else:  # G5
                Lam = Srr + eta * s["Pbar"] / Pd
            Gf = np.conj(R) * E / (Lam + TINY)
            Psi_i = None
        elif is_k and i >= 5:
            if Pk is None:                                   # initialise at block 5
                Pk_raw = shared.See / (shared.Srr + TINY)
                Pk_floor = 1e-10 * Pk_raw.mean()
                Pk = np.maximum(Pk_raw, Pk_floor)
                if cand == "K3":
                    Psi_s = 0.01 * shared.See
                else:
                    etah = 0.01 * shared.See / s["Pbar"]
            Q = np.maximum(np.abs(E) ** 2 - Pk * A, 0.0)
            if cand == "K3":
                Psi_s = 0.99 * Psi_s + 0.01 * Q
                Psi_i = Psi_s
            else:
                etah = 0.99 * etah + 0.01 * Q / max(P, TINY)
                Psi_i = etah * P
            num = A if cand == "K2" else s["Srr"]
            Lam = num + Psi_i / Pk + TINY
            Gf = np.conj(R) * E / Lam
            if track:
                lx, ly = np.log(Pk), np.log(Pd)
                tr["n"] += NB
                tr["Sx"] += float(lx.sum())
                tr["Sy"] += float(ly.sum())
                tr["Sxx"] += float((lx * lx).sum())
                tr["Syy"] += float((ly * ly).sum())
                tr["Sxy"] += float((lx * ly).sum())
                dd = float(np.mean(10 * np.log10(Pk / Pd)))
                if i < n5:
                    tr["w5"] += dd
                    tr["n5"] += 1
                if i >= i_l10:
                    tr["wl"] += dd
                    tr["nl"] += 1
        else:
            Gf = X.cand_G(base, R, Ef, s)

        g = np.fft.irfft(Gf, N)[:M]
        w = w - mu_b * g
        if not np.isfinite(w).all():
            e_all[s0 + B:] = np.nan
            diverged = True
            break
        if is_k and i >= 5:
            Pk = Pk * (1 - mu_b * A / Lam) ** 2 + mu_b ** 2 * A * Psi_i / Lam ** 2
            Pk = np.maximum(Pk, Pk_floor)

    track_out = None
    if track:
        n = tr["n"]
        cov = n * tr["Sxy"] - tr["Sx"] * tr["Sy"]
        vx = n * tr["Sxx"] - tr["Sx"] ** 2
        vy = n * tr["Syy"] - tr["Sy"] ** 2
        track_out = dict(corr=float(cov / np.sqrt(vx * vy)),
                         first5_db=tr["w5"] / max(tr["n5"], 1),
                         last10_db=tr["wl"] / max(tr["nl"], 1))
    return dict(e=e_all, w=w, diverged=diverged, track=track_out)


# ----------------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------------
def run_source(spec):
    tag, kind, arg = spec
    t0 = time.time()
    data = load_data(kind, arg)
    d, w_opt = data["d"], data["w_opt"]
    pp = prepass(data)
    keep = tag == "a145"
    rows, trk, anrs = [], {}, {}
    for cand in ALL:
        for mu in MUS:
            sim = simulate(data, pp, cand, mu, track=(keep and cand in KC))
            m = X.metrics(sim["e"], sim["w"], d, w_opt)
            rows.append(dict(cand=cand, mu=mu, **m))
            if sim["track"] is not None:
                trk[f"{cand}__{mu}"] = sim["track"]
            if keep:
                anrs[f"{cand}__{mu}"] = O.anr_curve(sim["e"], d)
    rt = time.time() - t0
    print(f"[done] {tag} runtime={rt:.1f}s reg_r={pp['reg_r']:.3f} slope={pp['reg_slope']:.3f}", flush=True)
    return dict(tag=tag, rows=rows, trk=trk, anrs=anrs, reg_r=pp["reg_r"],
                reg_slope=pp["reg_slope"], n_frames=pp["n_frames"], runtime_s=rt)


# ----------------------------------------------------------------------------
# Report helpers
# ----------------------------------------------------------------------------
def fmt(v, p=2):
    return "nan" if v is None or not np.isfinite(v) else f"{v:.{p}f}"


def nan_mean(vals):
    vals = np.asarray(vals, dtype=float)
    return float(vals.mean()) if np.isfinite(vals).all() else float("nan")


def main():
    os.makedirs(OUT, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()

    # Generator check: alpha=1.45 CMS loader must reproduce O.load_source exactly
    a = load_cms(1.45, 0)
    b = O.load_source("cms", 0)
    gen_ok = bool(np.array_equal(a["x"], b["x"]) and np.array_equal(a["w_opt"], b["w_opt"]))
    if not gen_ok:
        raise SystemExit("CMS loader does not reproduce oracle.py data at alpha=1.45")

    with Pool(processes=2) as pool:
        results = pool.map(run_source, SOURCES)
    by = {r["tag"]: r for r in results}
    runtime = time.time() - t0

    cell = {}
    for r in results:
        for row in r["rows"]:
            cell.setdefault((row["cand"], row["mu"]), {})[r["tag"]] = row
    KEYS = ["whole", "first5", "last10", "msd"]

    def agg(c, mu, key):
        return nan_mean([cell[(c, mu)][t][key] for t in SIX])

    def ndiv(c, mu):
        return int(sum(cell[(c, mu)][t]["diverged"] for t in SIX))

    def best(c, key):
        ok = {mu: agg(c, mu, key) for mu in MUS if np.isfinite(agg(c, mu, key))}
        return min(ok, key=ok.get) if ok else None

    bw = {c: best(c, "whole") for c in ALL}
    bl = {c: best(c, "last10") for c in ALL}

    # Sanity: B0/B2/C5 must match explore.py's grid.csv on the shared mu values
    ex_rows = {}
    with open(os.path.join(OUT, "grid.csv")) as f:
        for row in csv.DictReader(f):
            if row["cand"] in ("B0", "B2", "C5"):
                ex_rows[(row["cand"], float(row["mu"]), row["source"])] = float(row["whole_db"])
    max_diff = 0.0
    n_cmp = 0
    for (c, mu, t), v in ex_rows.items():
        if (c, mu) in cell and t in cell[(c, mu)]:
            mine = cell[(c, mu)][t]["whole"]
            if np.isfinite(mine) and np.isfinite(v):
                max_diff = max(max_diff, abs(mine - v))
            n_cmp += 1

    # ---------------- JSON / CSV ----------------
    with open(os.path.join(OUT, "genie_grid.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["cand", "mu", "source", "whole_db", "first5s_db", "last10_db", "final_msd_db", "diverged"])
        for r in results:
            for row in r["rows"]:
                wr.writerow([row["cand"], row["mu"], r["tag"], row["whole"], row["first5"],
                             row["last10"], row["msd"], int(row["diverged"])])

    # ---------------- Figure ----------------
    arr = np.load(os.path.join(OUT, "anr_a145.npz"))   # explore.py O_band / O_scalar curves
    g_arr = by["a145"]["anrs"]
    tgrid = lambda n: (np.arange(n) * 64 + 63) / FS
    best_g = min(GEN, key=lambda c: agg(c, bw[c], "whole") if bw[c] is not None else 1e9)
    best_k = min(KC, key=lambda c: agg(c, bw[c], "whole") if bw[c] is not None else 1e9)
    fig_list = [("B0", "B0"), ("B2", "B2"), ("C5", "C5"), (best_g, best_g), (best_k, best_k)]
    fig, ax = plt.subplots(figsize=(10, 5))
    for c, lab in fig_list:
        y = g_arr[f"{c}__{bw[c]}"]
        ax.plot(tgrid(len(y)), y, lw=0.8, label=f"{lab} (mu={bw[c]})")
    if "O_band" in arr.files:
        y = arr["O_band"]
        ax.plot(tgrid(len(y)), y, lw=0.8, ls="--", label="O_band")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("ANR [dB]")
    ax.set_title("ANR, alpha=1.45 file (best mu by whole-curve mean over six sources)")
    ax.grid(True, ls=":")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "genie_anr_a145.png"), dpi=130)
    plt.close(fig)

    # ---------------- Report ----------------
    L = []
    L.append("# Genie and practical step normalizations (atsip/explore/genie.py)\n")
    L.append("Numbers only. Generated by `atsip/explore/genie.py`.\n")
    L.append(f"Runtime (wall clock, 2 worker processes): {runtime:.1f} s. "
             "Per-source runtime [s]: " + ", ".join(f"{r['tag']} {r['runtime_s']:.1f}" for r in results) + ".\n")
    L.append("Sources: six = " + ", ".join(SIX) + "; alpha=1.2 (reported separately) = " + ", ".join(TWO) + ".\n")
    L.append("Means over the six sources; nan if any source is non-finite. #div = diverged runs of the six at the given mu.\n")
    L.append("## Setup checks\n")
    L.append(f"- CMS loader at alpha=1.45 reproduces oracle.py data (x and w_opt): {gen_ok}.")
    L.append(f"- B0/B2/C5 whole-curve ANR vs explore.py grid.csv on shared mu and sources: {n_cmp} cells, max |diff| = {max_diff:.2e} dB.\n")

    def table(best, title):
        L.append(f"## {title}\n")
        L.append("| cand | description | best mu | whole [dB] | first5s [dB] | last10 [dB] | final MSD [dB] | #div (of 6) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for c in ALL:
            m = best[c]
            if m is None:
                L.append(f"| {c} | {DESC[c]} | nan | nan | nan | nan | nan | nan |")
                continue
            L.append(f"| {c} | {DESC[c]} | {m} | {fmt(agg(c,m,'whole'))} | {fmt(agg(c,m,'first5'))} | "
                     f"{fmt(agg(c,m,'last10'))} | {fmt(agg(c,m,'msd'))} | {ndiv(c,m)} |")
        L.append("")

    table(bw, "A. Best mu by whole-curve mean (six sources)")
    table(bl, "B. Best mu by last-10% mean (six sources)")

    L.append("## C. alpha=1.2 sources (seeds 0 and 1), at the six-source best mu from table A\n")
    L.append("Reported per seed; not averaged into the six-source means.\n")
    L.append("| cand | best mu | whole s0 | whole s1 | first5 s0 | first5 s1 | last10 s0 | last10 s1 | MSD s0 | MSD s1 | div s0 / s1 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in ALL:
        m = bw[c]
        if m is None:
            L.append(f"| {c} | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |")
            continue
        s0, s1 = cell[(c, m)]["a120_s0"], cell[(c, m)]["a120_s1"]
        L.append(f"| {c} | {m} | {fmt(s0['whole'])} | {fmt(s1['whole'])} | {fmt(s0['first5'])} | {fmt(s1['first5'])} | "
                 f"{fmt(s0['last10'])} | {fmt(s1['last10'])} | {fmt(s0['msd'])} | {fmt(s1['msd'])} | "
                 f"{int(s0['diverged'])} / {int(s1['diverged'])} |")
    L.append("")

    L.append("## D. Residual at the optimum vs input power\n")
    L.append("Per source, over all frames i: Pearson correlation between log P_i and log mean_k |E_opt,i(k)|^2, "
             "and the slope of the log-log least-squares regression (1 = residual proportional to input power).\n")
    L.append("| source | frames | corr(log P, log residual power) | slope |")
    L.append("|---|---|---|---|")
    for t in SIX + TWO:
        r = by[t]
        L.append(f"| {t} | {r['n_frames']} | {fmt(r['reg_r'],3)} | {fmt(r['reg_slope'],3)} |")
    L.append(f"| mean (six) | | {fmt(np.mean([by[t]['reg_r'] for t in SIX]),3)} | {fmt(np.mean([by[t]['reg_slope'] for t in SIX]),3)} |")
    L.append("")

    # Tracking: K candidates on a145 at best mu
    L.append("## E. Pk tracking (alpha=1.45 file)\n")
    L.append("corr = Pearson correlation of log Pk and log Pd over bins x blocks (blocks >= 5). "
             "dB first5 / last10 = mean over bins and blocks of 10 log10(Pk/Pd) in the first 5 s and the last 10 % of blocks. "
             "Pk and Pd are both taken before the update of the block.\n")
    L.append("| cand | best mu (table A) | corr log Pk vs log Pd | first5 mean 10log10(Pk/Pd) [dB] | last10 mean 10log10(Pk/Pd) [dB] |")
    L.append("|---|---|---|---|---|")
    for c in KC:
        m = bw[c]
        if m is None:
            L.append(f"| {c} | nan | nan | nan | nan |")
            continue
        t = by["a145"]["trk"][f"{c}__{m}"]
        L.append(f"| {c} | {m} | {fmt(t['corr'],3)} | {fmt(t['first5_db'],2)} | {fmt(t['last10_db'],2)} |")
    L.append("")

    # Appendix: full grid
    L.append("## Appendix 1. Full grid, whole-curve mean ANR [dB] (six sources)\n")
    L.append("| cand | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for c in ALL:
        L.append(f"| {c} | " + " | ".join(fmt(agg(c, m, 'whole')) for m in MUS) + " |")
    L.append("")
    L.append("## Appendix 2. Full grid, last-10% mean ANR [dB] (six sources)\n")
    L.append("| cand | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for c in ALL:
        L.append(f"| {c} | " + " | ".join(fmt(agg(c, m, 'last10')) for m in MUS) + " |")
    L.append("")
    L.append("## Appendix 3. Diverged runs (of six) per mu\n")
    L.append("| cand | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for c in ALL:
        L.append(f"| {c} | " + " | ".join(str(ndiv(c, m)) for m in MUS) + " |")
    L.append("")
    L.append("## Appendix 4. alpha=1.2 whole-curve ANR [dB] per seed, every mu\n")
    L.append("| cand | " + " | ".join(f"mu={m} s0 / s1" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for c in ALL:
        cells = [f"{fmt(cell[(c,m)]['a120_s0']['whole'])} / {fmt(cell[(c,m)]['a120_s1']['whole'])}" for m in MUS]
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Appendix 5. Pk tracking at every mu (alpha=1.45 file)\n")
    L.append("| cand | mu | corr | first5 dB | last10 dB |")
    L.append("|---|---|---|---|---|")
    for c in KC:
        for m in MUS:
            t = by["a145"]["trk"][f"{c}__{m}"]
            L.append(f"| {c} | {m} | {fmt(t['corr'],3)} | {fmt(t['first5_db'],2)} | {fmt(t['last10_db'],2)} |")
    L.append("")

    L.append("## Figure\n")
    L.append("- `out/genie_anr_a145.png`: ANR, alpha=1.45 file, at six-source best mu: "
             + ", ".join(f"{c} (mu={bw[c]})" for c, _ in fig_list)
             + ", plus O_band (dashed, from out/anr_a145.npz).\n")
    L.append("## Files\n")
    L.append("- `out/genie_grid.csv`: every run (candidate, mu, source).")
    L.append("- `out/genie_summary.json`: best-mu tables, regression and tracking numbers.")
    L.append("- `out/genie_anr_a145.npz`: ANR curves of the alpha=1.45 runs (all candidates and mu).")
    L.append("")
    with open(os.path.join(HERE, "RESULTS_genie.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    summ = dict(runtime_s=runtime,
                best_whole=bw, best_last10=bl,
                agg={c: {str(m): {k: agg(c, m, k) for k in KEYS} | {"ndiv": ndiv(c, m)} for m in MUS} for c in ALL},
                regression={r["tag"]: dict(r=r["reg_r"], slope=r["reg_slope"], n=r["n_frames"]) for r in results},
                tracking={r["tag"]: r["trk"] for r in results if r["tag"] == "a145"},
                max_diff_explore=max_diff)
    with open(os.path.join(OUT, "genie_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1, default=float)
    np.savez(os.path.join(OUT, "genie_anr_a145.npz"), **g_arr)
    print(f"[total] {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
