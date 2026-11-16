#!/usr/bin/env python3
"""Mixed-normalisation step candidates for the delayless FD-FxNLMS under two error framings.

Practical candidates B2, C5, T1..T7 and genie candidates G2c, G1c, GL1..GL3, each run with
framing B (last B=128 errors, scale c = B/N) and framing M (last M=512 errors, c = M/N).
Reuses data, ANR and metrics from oracle.py / explore.py, and sources, prepass and true Pd
from genie.py.

Usage:  python3 atsip/explore/mixlam.py
Writes  atsip/explore/out/mixlam_*.{csv,json,npz,png} and atsip/explore/RESULTS_mixlam.md.
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
import explore as X  # noqa: E402  (Shared EMAs, metrics)
import genie as G    # noqa: E402  (sources, true_Pd, SIX/TWO, fmt helpers)

O = X.O
FS, M, N, B, NB = X.FS, X.M, X.N, X.B, X.NB
OUT = os.path.join(HERE, "out")
MUS = [0.05, 0.1, 0.2, 0.36, 0.6, 1.0, 1.5]
FRAMES = {"B": B, "M": M}                       # framing -> number of error samples
C2 = {f: (L / N) ** 2 for f, L in FRAMES.items()}  # c^2 with c = L/N
PRAC = ["B2", "C5", "T1", "T2", "T3", "T4", "T5", "T6", "T7"]
GENC = ["G2c", "G1c", "GL1", "GL2", "GL3"]
CANDS = PRAC + GENC
NAMES = [f"{c}_{f}" for c in CANDS for f in ("B", "M")]
DESC = {
    "B2": "Lam_mix(0.45) (= 0.55 Pbar + 0.45 Srr + eps), fixed",
    "C5": "lam = clip(sqrt(gam_prev),0.1,1), no shrink",
    "T1": "lam = clip(gam_prev,0.05,1), no shrink",
    "T2": "T1, gain x lam",
    "T3": "C5, gain x lam",
    "T4": "lam = clip(lam_bb_prev,0.05,1) scalar, no shrink",
    "T5": "T4, gain x lam",
    "T6": "lam = clip(sqrt(lam_bb_prev),0.1,1) scalar, no shrink",
    "T7": "T6, gain x lam",
    "G2c": "genie: Lam = Srr + Psi/(c^2 Pd)",
    "G1c": "genie: Lam = |R|^2 + Psi/(c^2 Pd)",
    "GL1": "genie: scalar lam_true, Lam_mix(lam_true), gain x lam_true",
    "GL2": "genie: scalar lam_true, Lam_mix(lam_true), no shrink",
    "GL3": "genie: per-bin lam_true(k), Lam_mix(lam_true(k)), gain x lam_true(k)",
}
TINY = 1e-30
SRC = {t: (k, a) for t, k, a in [(s[0], s[1], s[2]) for s in G.SOURCES]}
SIX = list(G.SIX)
TWO = list(G.TWO)


# ----------------------------------------------------------------------------
# Framing and pre-pass
# ----------------------------------------------------------------------------
def frame_E(e_all, s0, Lf):
    """Error frame: last Lf errors ending at the current block, zero-padded at the start to N."""
    seg = e_all[max(0, s0 + B - Lf): s0 + B]
    ev = np.zeros(Lf)
    ev[Lf - len(seg):] = seg
    return np.fft.rfft(np.concatenate([np.zeros(N - Lf), ev]))


def prepass(data):
    """R_i(k) for all blocks and Psi_stat(k) = mean_i |E_opt,i(k)|^2 for each framing (w = w_opt)."""
    x, d, xq, hS, Sf, w_opt = (data[k] for k in ("x", "d", "xq", "hS", "Sf", "w_opt"))
    L = len(x)
    nb = L // B
    e_opt = d + signal.lfilter(hS, [1.0], np.convolve(x, w_opt)[:L])
    Rall = np.empty((nb, NB), dtype=np.complex128)
    acc = {f: np.zeros(NB) for f in FRAMES}
    for i in range(nb):
        s0 = i * B
        Rall[i] = Sf * np.fft.rfft(xq[s0 + B: s0 + B + N])
        for f, Lf in FRAMES.items():
            acc[f] += np.abs(frame_E(e_opt, s0, Lf)) ** 2
    return dict(Rall=Rall, Psi={f: acc[f] / nb for f in FRAMES})


# ----------------------------------------------------------------------------
# Practical lam(k) / scalar lam and gain multiplier
# ----------------------------------------------------------------------------
def practical_lam(cand, gp, lam_bb_prev):
    """Returns (lam, mult): lam scalar or per-bin array, mult = gain multiplied by lam."""
    if cand == "B2":
        return 0.45, False
    if cand == "C5":
        return np.clip(np.sqrt(gp), 0.1, 1.0), False
    if cand == "T1":
        return np.clip(gp, 0.05, 1.0), False
    if cand == "T2":
        return np.clip(gp, 0.05, 1.0), True
    if cand == "T3":
        return np.clip(np.sqrt(gp), 0.1, 1.0), True
    if cand == "T4":
        return float(np.clip(lam_bb_prev, 0.05, 1.0)), False
    if cand == "T5":
        return float(np.clip(lam_bb_prev, 0.05, 1.0)), True
    if cand == "T6":
        return float(np.clip(np.sqrt(lam_bb_prev), 0.1, 1.0)), False
    if cand == "T7":
        return float(np.clip(np.sqrt(lam_bb_prev), 0.1, 1.0)), True
    raise ValueError(cand)


# ----------------------------------------------------------------------------
# Block simulation: w <- w - mu_b * irfft(G)[:M], mu_b = 2 mu B / M
# ----------------------------------------------------------------------------
def simulate(data, pp, name, mu, track=False):
    cand, f = name.rsplit("_", 1)
    Lf, c2 = FRAMES[f], C2[f]
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
    lam_bb_prev = 1.0
    n_1s = int(round(FS / B))
    need_pd = track or cand in GENC
    tr = dict(lam_bb=[], gam=[], lam_true=[]) if track else None
    Yc, Xc = [], []
    diverged = False

    for i in range(nb):
        s0 = i * B
        Xb = swv[s0 + N - M + 1: s0 + N - M + 1 + B][:, ::-1]
        y = Xb @ w
        ys, zi = signal.lfilter(hS, [1.0], y, zi=zi)
        e_all[s0:s0 + B] = d[s0:s0 + B] + ys
        R = Rall[i]
        A = np.abs(R) ** 2
        E = frame_E(e_all, s0, Lf)
        s = shared.step(R, E)
        Srr, Pbar, eps, gp = s["Srr"], s["Pbar"], s["eps"], s["gam_prev"]
        lam_bb_now = float(np.clip(np.sum(np.abs(shared.Sre) ** 2 / (Srr + eps))
                                   / (np.sum(shared.See) + eps), 0.0, 1.0))
        Ce = np.conj(R) * E

        if need_pd:
            Pd = G.true_Pd(w, w_opt)
            q = c2 * Srr * Pd
            lt = float(np.sum(q) / np.sum(q + Psi))          # scalar lam_true
            if track:
                tr["lam_bb"].append(lam_bb_prev)
                tr["gam"].append(float(gp.mean()))
                tr["lam_true"].append(lt)
                if i < n_1s:                                  # scale model, first 1 s
                    Yc.append(np.abs(E) ** 2)
                    Xc.append(c2 * A * Pd)

        if cand == "G2c":
            Gf = Ce / (Srr + Psi / (c2 * Pd) + TINY)
        elif cand == "G1c":
            Gf = Ce / (A + Psi / (c2 * Pd) + TINY)
        elif cand in ("GL1", "GL2"):
            Lam = (1 - lt) * Pbar + lt * Srr + eps
            Gf = (lt if cand == "GL1" else 1.0) * Ce / Lam
        elif cand == "GL3":
            lk = q / (q + Psi)                                # per-bin lam_true(k)
            Lam = (1 - lk) * Pbar + lk * Srr + eps
            Gf = lk * Ce / Lam
        else:
            lam, mult = practical_lam(cand, gp, lam_bb_prev)
            Lam = (1 - lam) * Pbar + lam * Srr + eps
            Gf = (lam if mult else 1.0) * Ce / Lam

        g = np.fft.irfft(Gf, N)[:M]
        w = w - mu_b * g
        if not np.isfinite(w).all():
            e_all[s0 + B:] = np.nan
            diverged = True
            break
        lam_bb_prev = lam_bb_now

    track_out = None
    if track:
        track_out = {k: np.asarray(v) for k, v in tr.items()}
        Y, Xv = np.concatenate(Yc), np.concatenate(Xc)
        track_out["reg_slope"] = float(np.sum(Y * Xv) / np.sum(Xv * Xv))
        track_out["reg_r"] = float(np.corrcoef(Y, Xv)[0, 1])
        track_out["nb"] = nb
    return dict(e=e_all, w=w, diverged=diverged, track=track_out)


# ----------------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------------
def run_source(spec):
    tag, kind, arg = spec
    t0 = time.time()
    data = G.load_data(kind, arg)
    d, w_opt = data["d"], data["w_opt"]
    pp = prepass(data)
    psi_ref = G.prepass(data)["Psi"]                       # genie B-framing reference
    psi_diff = float(np.max(np.abs(pp["Psi"]["B"] - psi_ref) / (np.abs(psi_ref) + TINY)))
    keep = tag == "a145"
    rows, anrs = [], {}
    for name in NAMES:
        for mu in MUS:
            sim = simulate(data, pp, name, mu)
            m = X.metrics(sim["e"], sim["w"], d, w_opt)
            rows.append(dict(cand=name, mu=mu, **m))
            if keep:
                anrs[f"{name}__{mu}"] = O.anr_curve(sim["e"], d)
    rt = time.time() - t0
    print(f"[done] {tag} runtime={rt:.1f}s psi_B_reldiff_vs_genie={psi_diff:.2e}", flush=True)
    return dict(tag=tag, rows=rows, anrs=anrs, runtime_s=rt, psi_diff=psi_diff)


# ----------------------------------------------------------------------------
# Diagnostics (B2 at its best mu, per framing)
# ----------------------------------------------------------------------------
def diagnostics(best_mu):
    out = {}
    for tag in ("a145", "cms145_s0"):
        kind, arg = SRC[tag]
        data = G.load_data(kind, arg)
        pp = prepass(data)
        for f in ("B", "M"):
            name = f"B2_{f}"
            sim = simulate(data, pp, name, best_mu[name], track=True)
            tr = sim["track"]
            nbf = tr["nb"]
            nb_run = len(tr["lam_bb"])
            t = np.arange(nb_run) * B / FS
            i = np.arange(nb_run)
            wins = [("0-1 s", t < 1), ("1-5 s", (t >= 1) & (t < 5)),
                    ("5-15 s", (t >= 5) & (t < 15)), ("last 10 %", i >= nbf - int(0.1 * nbf))]
            wv = {}
            for wn, sel in wins:
                wv[wn] = {k: (float(np.mean(tr[k][sel])) if sel.any() else float("nan"))
                          for k in ("lam_bb", "gam", "lam_true")}
            ii = np.arange(1, nb_run)                       # skip block 0 (lam_bb_prev is the init 1.0)
            corr = dict(lam_bb_vs_true=X.spearman(tr["lam_bb"][ii], tr["lam_true"][ii]),
                        gam_vs_true=X.spearman(tr["gam"][ii], tr["lam_true"][ii]))
            np.savez(os.path.join(OUT, f"mixlam_diag_{tag}_{f}.npz"), **{k: tr[k] for k in ("lam_bb", "gam", "lam_true")})
            out[(tag, f)] = dict(name=name, mu=best_mu[name], nb=nb_run, windows=wv, spearman=corr,
                                 reg_slope=tr["reg_slope"], reg_r=tr["reg_r"],
                                 whole=X.metrics(sim["e"], sim["w"], data["d"], data["w_opt"])["whole"],
                                 diverged=sim["diverged"])
    return out


def fmt(v, p=2):
    return G.fmt(v, p)


def main():
    os.makedirs(OUT, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()

    a = G.load_cms(1.45, 0)
    b = O.load_source("cms", 0)
    if not (np.array_equal(a["x"], b["x"]) and np.array_equal(a["w_opt"], b["w_opt"])):
        raise SystemExit("CMS loader does not reproduce oracle.py data at alpha=1.45")

    with Pool(processes=2) as pool:
        results = pool.map(run_source, G.SOURCES)
    by = {r["tag"]: r for r in results}
    runtime = time.time() - t0

    cell = {}
    for r in results:
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

    bw = {c: best(c, "whole") for c in NAMES}
    bl = {c: best(c, "last10") for c in NAMES}

    # ---- Sanity: B2_B/C5_B vs genie_grid B-framing rows, B2_M/C5_M vs genie M-framing rows
    mp = {"B2": "B2_B", "C5": "C5_B", "B2_M": "B2_M", "C5_M": "C5_M"}
    max_diff, n_cmp = 0.0, 0
    with open(os.path.join(OUT, "genie_grid.csv")) as fh:
        for row in csv.DictReader(fh):
            if row["cand"] in mp and row["source"] in SIX + TWO:
                key = (mp[row["cand"]], float(row["mu"]))
                if key in cell and row["source"] in cell[key]:
                    mine = cell[key][row["source"]]["whole"]
                    ref = float(row["whole_db"])
                    if np.isfinite(mine) and np.isfinite(ref):
                        max_diff = max(max_diff, abs(mine - ref))
                    n_cmp += 1
    psi_diff = max(r["psi_diff"] for r in results)

    # ---- CSV
    with open(os.path.join(OUT, "mixlam_grid.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["cand", "mu", "source", "whole_db", "first5s_db", "last10_db", "final_msd_db", "diverged"])
        for r in results:
            for row in r["rows"]:
                wr.writerow([row["cand"], row["mu"], r["tag"], row["whole"], row["first5"],
                             row["last10"], row["msd"], int(row["diverged"])])

    # ---- Figure and ANR npz (alpha=1.45)
    arr = np.load(os.path.join(OUT, "anr_a145.npz"))
    g_arr = by["a145"]["anrs"]
    np.savez(os.path.join(OUT, "mixlam_anr_a145.npz"), **g_arr)
    tT = [c for c in NAMES if c.split("_")[0].startswith("T")]
    tG = [c for c in NAMES if c.split("_")[0] in GENC]
    best_T = min(tT, key=lambda c: agg(c, bw[c], "whole") if bw[c] is not None else 1e9)
    best_G = min(tG, key=lambda c: agg(c, bw[c], "whole") if bw[c] is not None else 1e9)
    fig_list = [("B2_M", "B2_M"), ("C5_M", "C5_M"), (best_T, best_T), (best_G, best_G)]
    tgrid = lambda n: (np.arange(n) * 64 + 63) / FS
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
    fig.savefig(os.path.join(OUT, "mixlam_anr_a145.png"), dpi=130)
    plt.close(fig)

    # ---- Diagnostics
    diag = diagnostics(bw)

    # ---- Report
    L = []
    L.append("# Mixed normalisation under two error framings (atsip/explore/mixlam.py)\n")
    L.append("Numbers only. Generated by `atsip/explore/mixlam.py`.\n")
    L.append(f"Runtime (wall clock, 2 worker processes): {runtime:.1f} s. Per-source runtime [s]: "
             + ", ".join(f"{r['tag']} {r['runtime_s']:.1f}" for r in results) + ".\n")
    L.append("Sources: six = " + ", ".join(SIX) + "; alpha=1.2 (reported separately, per seed) = " + ", ".join(TWO) + ".\n")
    L.append("Framing B: last B=128 errors, c = B/N. Framing M: last M=512 errors, c = M/N. "
             "Means over the six sources; nan if any source is non-finite. "
             "#div over sweep = diverged runs of the six sources over all 7 mu (of 42).\n")
    L.append("## Setup checks\n")
    L.append(f"- CMS loader at alpha=1.45 reproduces oracle.py data: True.")
    L.append(f"- Psi_stat (framing B) vs genie.prepass: max relative difference {psi_diff:.2e}.")
    L.append(f"- B2_B, C5_B, B2_M, C5_M whole-curve ANR vs genie_grid.csv (same sources, shared mu): "
             f"{n_cmp} cells, max |diff| = {max_diff:.2e} dB.\n")

    def table(bmap, title, key):
        L.append(f"## {title}\n")
        L.append("| cand | description | best mu | whole [dB] | first5s [dB] | last10 [dB] | final MSD [dB] | #div over sweep (of 42) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for c in NAMES:
            m = bmap[c]
            if m is None:
                L.append(f"| {c} | {DESC[c.rsplit('_', 1)[0]]} | nan | nan | nan | nan | nan | {ndiv_sweep(c)} |")
                continue
            L.append(f"| {c} | {DESC[c.rsplit('_', 1)[0]]} | {m} | {fmt(agg(c,m,'whole'))} | {fmt(agg(c,m,'first5'))} | "
                     f"{fmt(agg(c,m,'last10'))} | {fmt(agg(c,m,'msd'))} | {ndiv_sweep(c)} |")
        L.append("")

    table(bw, "A. Best mu by whole-curve mean (six sources)", "whole")
    table(bl, "B. Best mu by last-10% mean (six sources)", "last10")

    L.append("## C. alpha=1.2 sources (seeds 0 and 1) at the six-source best mu from table A\n")
    L.append("| cand | best mu | whole s0 | whole s1 | first5 s0 | first5 s1 | last10 s0 | last10 s1 | MSD s0 | MSD s1 | div s0 / s1 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in NAMES:
        m = bw[c]
        if m is None:
            L.append(f"| {c} | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |")
            continue
        s0, s1 = cell[(c, m)]["a120_s0"], cell[(c, m)]["a120_s1"]
        L.append(f"| {c} | {m} | {fmt(s0['whole'])} | {fmt(s1['whole'])} | {fmt(s0['first5'])} | {fmt(s1['first5'])} | "
                 f"{fmt(s0['last10'])} | {fmt(s1['last10'])} | {fmt(s0['msd'])} | {fmt(s1['msd'])} | "
                 f"{int(s0['diverged'])} / {int(s1['diverged'])} |")
    L.append("")

    L.append("## D. Diagnostics: B2 at its best mu (table A), per framing\n")
    L.append("Per block: lam_bb_prev, mean_k gam_prev, scalar lam_true (c^2 Srr Pd / sum, with Psi_stat of the framing). "
             "Time-averaged over blocks in each window; window membership by block start time; last 10 % by block index.\n")
    L.append("| source | framing | run | mu | window | lam_bb_prev | mean gam_prev | lam_true |")
    L.append("|---|---|---|---|---|---|---|---|")
    for (tag, f), dg in diag.items():
        for wn, vals in dg["windows"].items():
            L.append(f"| {tag} | {f} | {dg['name']} | {dg['mu']} | {wn} | {fmt(vals['lam_bb'],3)} | "
                     f"{fmt(vals['gam'],3)} | {fmt(vals['lam_true'],3)} |")
    L.append("")
    L.append("Spearman correlation over blocks 1..end (block 0 excluded):\n")
    L.append("| source | framing | rho(lam_bb_prev, lam_true) | rho(mean gam_prev, lam_true) | blocks | diverged |")
    L.append("|---|---|---|---|---|---|")
    for (tag, f), dg in diag.items():
        L.append(f"| {tag} | {f} | {fmt(dg['spearman']['lam_bb_vs_true'],3)} | {fmt(dg['spearman']['gam_vs_true'],3)} | "
                 f"{dg['nb']} | {dg['diverged']} |")
    L.append("")
    L.append("Scale model: least-squares fit of |E(k)|^2 on c^2 |R(k)|^2 Pd(k) through the origin, over all bins and blocks of the first 1 s, "
             "w and Pd taken before the update.\n")
    L.append("| source | framing | fitted ratio | Pearson r |")
    L.append("|---|---|---|---|")
    for (tag, f), dg in diag.items():
        L.append(f"| {tag} | {f} | {fmt(dg['reg_slope'],3)} | {fmt(dg['reg_r'],3)} |")
    L.append("")

    L.append("## Appendix 1. Full grid, whole-curve mean ANR [dB] (six sources)\n")
    L.append("| cand | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for c in NAMES:
        L.append(f"| {c} | " + " | ".join(fmt(agg(c, m, "whole")) for m in MUS) + " |")
    L.append("")
    L.append("## Appendix 2. Full grid, last-10% mean ANR [dB] (six sources)\n")
    L.append("| cand | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for c in NAMES:
        L.append(f"| {c} | " + " | ".join(fmt(agg(c, m, "last10")) for m in MUS) + " |")
    L.append("")
    L.append("## Appendix 3. Diverged runs (of six) per mu\n")
    L.append("| cand | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for c in NAMES:
        L.append(f"| {c} | " + " | ".join(str(ndiv6(c, m)) for m in MUS) + " |")
    L.append("")
    L.append("## Appendix 4. alpha=1.2 whole-curve ANR [dB] per seed, every mu (s0 / s1)\n")
    L.append("| cand | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for c in NAMES:
        cells = [f"{fmt(cell[(c,m)]['a120_s0']['whole'])} / {fmt(cell[(c,m)]['a120_s1']['whole'])}" for m in MUS]
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## Figure\n")
    L.append("- `out/mixlam_anr_a145.png`: ANR, alpha=1.45 file, at six-source best mu: "
             + ", ".join(f"{c} (mu={bw[c]})" for c, _ in fig_list) + ", plus O_band (dashed).\n")
    L.append("## Files\n")
    L.append("- `out/mixlam_grid.csv`: every run (candidate, mu, source).")
    L.append("- `out/mixlam_summary.json`: best-mu tables, diagnostics.")
    L.append("- `out/mixlam_anr_a145.npz`: ANR curves of the alpha=1.45 runs.")
    L.append("- `out/mixlam_diag_<source>_<framing>.npz`: per-block lam_bb_prev, gam_prev mean, lam_true for B2.")
    L.append("")
    with open(os.path.join(HERE, "RESULTS_mixlam.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    summ = dict(runtime_s=runtime, best_whole=bw, best_last10=bl, best_T=best_T, best_G=best_G,
                agg={c: {str(m): {k: agg(c, m, k) for k in ("whole", "first5", "last10", "msd")}
                          | {"ndiv6": ndiv6(c, m)} for m in MUS} for c in NAMES},
                diag={f"{t}__{f}": dict(v, windows=v["windows"]) for (t, f), v in diag.items()},
                max_diff_genie=max_diff, psi_diff=psi_diff)
    with open(os.path.join(OUT, "mixlam_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1, default=float)
    print(f"[total] {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
