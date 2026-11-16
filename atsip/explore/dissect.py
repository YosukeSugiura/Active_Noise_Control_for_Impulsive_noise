#!/usr/bin/env python3
"""Dissection of the lamC(k) = clip(sqrt(gam_prev(k)), 0.1, 1) step normaliser (delayless FD-FxNLMS).

Part 1: controls for lamC: refs B0, B2, C5, T3; freeze-self F1; freeze-cross F2; shuffled lamC
        S1 (dynamic) and S2 (static); power substitute P(b); power-law normaliser W(a).
Part 2: flatness of the irreducible part, rho_irr(k) = time mean of |E_opt(k)|^2 / Srr(k), w = w_opt.
Part 3: error-window length L in {128, 256, 386, 512} for B0, B2, T3.

Reuses data, ANR and metrics from oracle.py / explore.py, sources and load_cms from genie.py,
prepass and frame_E from mixlam.py. Ten sources: six exploration sources, two alpha=1.2 CMS seeds,
and two colored sources (CMS alpha=1.45 seeds 10, 11 through y[n] = x[n] + 1.2 y[n-1] - 0.5 y[n-2],
rescaled to the median |x| of the unfiltered sequence).

Usage:  python3 atsip/explore/dissect.py
Writes  atsip/explore/out/dissect_*.{csv,json,npz,png,log} and atsip/explore/RESULTS_dissect.md.
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
FS, M, N, B, NB, N5 = X.FS, X.M, X.N, X.B, X.NB, X.N5
OUT = os.path.join(HERE, "out")
MUS = [0.05, 0.1, 0.2, 0.36, 0.6, 1.0]
SIX = [t for t, _, _ in O.SOURCES]
TWO = ["a120_s0", "a120_s1"]
COL = ["col145_s10", "col145_s11"]
SRC = [(t, k, a) for t, k, a in O.SOURCES] + [
    ("a120_s0", "cms12", 0), ("a120_s1", "cms12", 1),
    ("col145_s10", "col", 10), ("col145_s11", "col", 11)]
ALLTAGS = [t for t, _, _ in SRC]
KIND = {t: (k, a) for t, k, a in SRC}
LAM_FLOOR = 0.1
PERM = np.random.default_rng(2002).permutation(NB)      # fixed permutation for S2
PART1 = ["B0", "B2", "C5", "T3", "F1", "F2", "S1", "S2", "P0.25", "P0.5", "P1.0", "W0.25", "W0.5", "W0.75"]
PART3_L = [128, 256, 386]
PART3_BASE = ["B0", "B2", "T3"]
DESC = {
    "B0": "lam=1: Lam=Srr+eps, no shrink (ref)",
    "B2": "lam=0.45 fixed, no shrink (ref)",
    "C5": "lam=lamC(k), no shrink (ref)",
    "T3": "lam=lamC(k), gain x lamC (ref)",
    "F1": "lam=lambar(k) frozen from own T3 run (second half), gain x lambar",
    "F2": "lam=lambar(k) frozen, leave-one-out mean of other exploration sources, gain x lambar",
    "S1": "lamC permuted across bins every block (seeded), gain x permuted lam",
    "S2": "lamC permuted by one fixed permutation every block, gain x permuted lam",
    "P0.25": "lam=clip((Srr/Pbar)^0.25,0.1,1), gain x lam",
    "P0.5": "lam=clip((Srr/Pbar)^0.5,0.1,1), gain x lam",
    "P1.0": "lam=clip((Srr/Pbar)^1.0,0.1,1), gain x lam",
    "W0.25": "Lam=Srr^0.25 Pbar^0.75+eps, no lam",
    "W0.5": "Lam=Srr^0.5 Pbar^0.5+eps, no lam",
    "W0.75": "Lam=Srr^0.75 Pbar^0.25+eps, no lam",
}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def colored_x(seed):
    x0 = O.cms_symmetric_stable(1.45, O.N_NOISE, seed)
    y = signal.lfilter([1.0], [1.0, -1.2, 0.5], x0)          # y[n] = x[n] + 1.2 y[n-1] - 0.5 y[n-2]
    y = y * (np.median(np.abs(x0)) / np.median(np.abs(y)))
    return y


def build(x):
    """Same construction as oracle.load_source, for an arbitrary excitation x."""
    hP = O.pad_or_trim(np.loadtxt(os.path.join(O.ROOT, "impulse_response/primary_ir.dat"), ndmin=1), M)
    hS = O.pad_or_trim(np.loadtxt(os.path.join(O.ROOT, "impulse_response/secondary_ir.dat"), ndmin=1), O.C_SEC)
    d = signal.lfilter(hP, [1.0], x)
    r = signal.lfilter(hS, [1.0], x)
    data = dict(x=x, d=d, r=r, hS=hS, xq=np.concatenate([np.zeros(N), x]), Sf=np.fft.rfft(hS, N))
    data["w_opt"] = O.wiener_opt(r, d)
    p = np.abs(data["Sf"]) ** 2
    data["S2"] = p / p.mean()
    return data


def load_data(kind, arg):
    if kind == "col":
        return build(colored_x(arg))
    if kind == "cms12":
        return G.load_cms(1.2, arg)
    return O.load_source(kind, arg)


# ----------------------------------------------------------------------------
# Candidate specs and block simulation
# ----------------------------------------------------------------------------
def spec_of(label):
    if label in ("B0", "B2", "C5", "T3"):
        return dict(kind=label)
    if label == "S1":
        return dict(kind="S1")
    if label == "S2":
        return dict(kind="S2", perm=PERM)
    if label.startswith("P"):
        return dict(kind="P", b=float(label[1:]))
    if label.startswith("W"):
        return dict(kind="W", a=float(label[1:]))
    raise ValueError(label)


def lam_and_mult(spec, Srr, Pbar, lamC, rng):
    k = spec["kind"]
    if k == "B0":
        return 1.0, False
    if k == "B2":
        return 0.45, False
    if k == "C5":
        return lamC, False
    if k == "T3":
        return lamC, True
    if k == "F":
        return spec["lam"], True
    if k == "S1":
        return lamC[rng.permutation(NB)], True
    if k == "S2":
        return lamC[spec["perm"]], True
    if k == "P":
        return np.clip((Srr / Pbar) ** spec["b"], LAM_FLOOR, 1.0), True
    raise ValueError(k)


def simulate(data, Rall, spec, mu, Lf=M, track=False):
    """w <- w - mu_b * irfft(G)[:M], mu_b = 2 mu B / M; E framing = last Lf errors, zero-padded to N."""
    d, xq, hS, S2 = data["d"], data["xq"], data["hS"], data["S2"]
    L = len(d)
    nb = L // B
    swv = sliding_window_view(xq, M)
    w = np.zeros(M)
    zi = np.zeros(len(hS) - 1)
    e_all = np.zeros(L)
    mu_b = 2 * mu * B / M
    shared = X.Shared(S2)
    rng = np.random.default_rng(1001)
    acc = np.zeros(NB)
    cnt = 0
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
        lamC = np.clip(np.sqrt(gp), LAM_FLOOR, 1.0)
        if track and i >= nb // 2:
            acc += lamC
            cnt += 1
        Ce = np.conj(R) * E
        if spec["kind"] == "W":
            Lam = Srr ** spec["a"] * Pbar ** (1 - spec["a"]) + eps
            Gf = Ce / Lam
        else:
            lam, mult = lam_and_mult(spec, Srr, Pbar, lamC, rng)
            Lam = (1 - lam) * Pbar + lam * Srr + eps
            Gf = (lam if mult else 1.0) * Ce / Lam
        g = np.fft.irfft(Gf, N)[:M]
        w = w - mu_b * g
        if not np.isfinite(w).all():
            e_all[s0 + B:] = np.nan
            diverged = True
            break
    out = dict(e=e_all, w=w, diverged=diverged)
    if track:
        out["lambar"] = acc / max(cnt, 1)
    return out


def part2(data, Rall):
    """rho_irr(k), mean |E_opt|^2 (k), and mean Srr/Pbar (k), with w fixed at w_opt, E_M framing."""
    x, d, hS, w_opt = data["x"], data["d"], data["hS"], data["w_opt"]
    L = len(d)
    nb = L // B
    e_opt = d + signal.lfilter(hS, [1.0], np.convolve(x, w_opt)[:L])
    shared = X.Shared(data["S2"])
    rho = np.zeros(NB)
    pw = np.zeros(NB)
    sr = np.zeros(NB)
    for i in range(nb):
        s0 = i * B
        E = ML.frame_E(e_opt, s0, M)
        s = shared.step(Rall[i], E)
        Srr = np.maximum(s["Srr"], 1e-30)
        A = np.abs(E) ** 2
        rho += A / Srr
        pw += A
        sr += Srr / s["Pbar"]
    return rho / nb, pw / nb, sr / nb


def dbspread(v):
    return float(v.max() - v.min())


# ----------------------------------------------------------------------------
# Workers
# ----------------------------------------------------------------------------
def stageA(tag):
    """Part 1 grid (all E_M labels except F1/F2) and Part 3 windows, all mu, one source."""
    t0 = time.time()
    kind, arg = KIND[tag]
    data = load_data(kind, arg)
    d, w_opt = data["d"], data["w_opt"]
    Rall = ML.prepass(data)["Rall"]
    jobs = [(lab, spec_of(lab), M) for lab in PART1 if lab not in ("F1", "F2")]
    jobs += [(f"{b}_L{L}", spec_of(b), L) for L in PART3_L for b in PART3_BASE]
    rows = []
    for lab, spec, Lf in jobs:
        for mu in MUS:
            sim = simulate(data, Rall, spec, mu, Lf)
            m = X.metrics(sim["e"], sim["w"], d, w_opt)
            rows.append(dict(label=lab, Lf=Lf, mu=mu, src=tag, **m))
    return dict(tag=tag, rows=rows, runtime=time.time() - t0)


def stageB(args):
    """T3 run at its best mu -> lambar(k) (second half); F1 at every mu; rho_irr statistics."""
    tag, mu_T3 = args
    t0 = time.time()
    kind, arg = KIND[tag]
    data = load_data(kind, arg)
    d, w_opt = data["d"], data["w_opt"]
    Rall = ML.prepass(data)["Rall"]
    tr = simulate(data, Rall, dict(kind="T3"), mu_T3, M, track=True)
    lam_bar = tr["lambar"]
    t3 = X.metrics(tr["e"], tr["w"], d, w_opt)
    rows = []
    for mu in MUS:
        sim = simulate(data, Rall, dict(kind="F", lam=lam_bar), mu, M)
        rows.append(dict(label="F1", Lf=M, mu=mu, src=tag, **X.metrics(sim["e"], sim["w"], d, w_opt)))
    rho, pw, sr = part2(data, Rall)
    return dict(tag=tag, lam_bar=lam_bar, t3_track=t3, rows=rows, rho=rho, pw=pw, sr=sr,
                S2=data["S2"], runtime=time.time() - t0)


def stageC(args):
    """F2: frozen leave-one-out lambar(k) across mu."""
    tag, lam_fix = args
    t0 = time.time()
    kind, arg = KIND[tag]
    data = load_data(kind, arg)
    d, w_opt = data["d"], data["w_opt"]
    Rall = ML.prepass(data)["Rall"]
    rows = []
    for mu in MUS:
        sim = simulate(data, Rall, dict(kind="F", lam=lam_fix), mu, M)
        rows.append(dict(label="F2", Lf=M, mu=mu, src=tag, **X.metrics(sim["e"], sim["w"], d, w_opt)))
    return dict(tag=tag, rows=rows, runtime=time.time() - t0)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def nmean(vals):
    return G.nan_mean(vals)


def main():
    os.makedirs(OUT, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()

    with Pool(processes=2) as pool:
        resA = pool.map(stageA, ALLTAGS)
    tA = time.time() - t0
    rowsA = [r for res in resA for r in res["rows"]]
    cell = {(r["label"], r["Lf"], r["mu"], r["src"]): r for r in rowsA}

    def m6(lab, Lf, mu, key):
        return nmean([cell[(lab, Lf, mu, t)][key] for t in SIX])

    def best_mu(lab, Lf=M):
        ok = {mu: m6(lab, Lf, mu, "whole") for mu in MUS}
        ok = {mu: v for mu, v in ok.items() if np.isfinite(v)}
        return min(ok, key=ok.get) if ok else None

    mu_T3 = best_mu("T3")

    t1 = time.time()
    with Pool(processes=2) as pool:
        resB = pool.map(stageB, [(t, mu_T3) for t in ALLTAGS])
    tB = time.time() - t1
    byB = {r["tag"]: r for r in resB}
    lam_bar = {t: byB[t]["lam_bar"] for t in ALLTAGS}

    lamfix = {}
    for t in ALLTAGS:
        if t in SIX:
            others = [lam_bar[s] for s in SIX if s != t]
        else:
            others = [lam_bar[s] for s in SIX]
        lamfix[t] = np.mean(np.vstack(others), axis=0)

    t2 = time.time()
    with Pool(processes=2) as pool:
        resC = pool.map(stageC, [(t, lamfix[t]) for t in ALLTAGS])
    tC = time.time() - t2
    for res in list(resB) + list(resC):
        for r in res["rows"]:
            cell[(r["label"], r["Lf"], r["mu"], r["src"])] = r

    runtime = time.time() - t0

    # ---------------- sanity checks
    ref = {}
    with open(os.path.join(OUT, "projlam_grid.csv")) as fh:
        for row in csv.DictReader(fh):
            if row["cand"] in ("B0_M", "B2_M", "C5_M", "T3_M"):
                ref[(row["cand"][:-2], float(row["mu"]), row["source"])] = float(row["whole_db"])
    max_diff, n_cmp = 0.0, 0
    for lab in ("B0", "B2", "C5", "T3"):
        for mu in MUS:
            for t in SIX + TWO:
                key = (lab, float(mu), t)
                if key in ref:
                    mine = cell[(lab, M, mu, t)]["whole"]
                    if np.isfinite(mine) and np.isfinite(ref[key]):
                        max_diff = max(max_diff, abs(mine - ref[key]))
                    n_cmp += 1
    t3_diff = max(abs(byB[t]["t3_track"]["whole"] - cell[("T3", M, mu_T3, t)]["whole"])
                  for t in ALLTAGS if np.isfinite(cell[("T3", M, mu_T3, t)]["whole"]))

    # ---------------- Part 1 table
    best = {lab: best_mu(lab) for lab in PART1}

    def ndiv(lab, mu, srcs, Lf=M):
        return int(sum(cell[(lab, Lf, mu, t)]["diverged"] for t in srcs))

    def ndiv_sweep(lab, Lf=M):
        return int(sum(ndiv(lab, mu, SIX, Lf) for mu in MUS))

    # ---------------- Part 2 (a145 and colored col145_s10 headline; all sources in the appendix)
    p2 = {}
    for t in ALLTAGS:
        rho, pw, sr, S2 = byB[t]["rho"], byB[t]["pw"], byB[t]["sr"], byB[t]["S2"]
        db = 10 * np.log10(rho[1:NB - 1])
        dbp = 10 * np.log10(pw[1:NB - 1])
        p2[t] = dict(
            rho_min=float(db.min()), rho_max=float(db.max()), rho_maxmin=float(db.max() - db.min()),
            rho_p1090=float(np.percentile(db, 90) - np.percentile(db, 10)),
            rho_mean_db=float(db.mean()),
            rho_S2=X.spearman(rho[1:NB - 1], S2[1:NB - 1]),
            pw_maxmin=float(dbp.max() - dbp.min()),
            pw_p1090=float(np.percentile(dbp, 90) - np.percentile(dbp, 10)),
            pw_S2=X.spearman(pw[1:NB - 1], S2[1:NB - 1]))

    # ---------------- Part 1 correlations (lambar from F1, a145)
    lb = lam_bar["a145"]
    corr_a = dict(
        srr_ratio=X.spearman(lb[1:NB - 1], byB["a145"]["sr"][1:NB - 1]),
        S2=X.spearman(lb[1:NB - 1], byB["a145"]["S2"][1:NB - 1]),
        rho_irr=X.spearman(lb[1:NB - 1], byB["a145"]["rho"][1:NB - 1]))

    # ---------------- Figure and saves
    np.savez(os.path.join(OUT, "dissect_rho_irr.npz"),
             **{f"rho_{t}": byB[t]["rho"] for t in ALLTAGS},
             **{f"pw_{t}": byB[t]["pw"] for t in ALLTAGS},
             **{f"srratio_{t}": byB[t]["sr"] for t in ALLTAGS},
             **{f"lambar_{t}": lam_bar[t] for t in ALLTAGS},
             **{f"S2_{t}": byB[t]["S2"] for t in ALLTAGS})
    k = np.arange(NB)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for t in ("a145", "col145_s10"):
        axes[0].plot(k[1:NB - 1], 10 * np.log10(byB[t]["rho"][1:NB - 1]), lw=0.8, label=t)
    axes[0].set_xlabel("bin k")
    axes[0].set_ylabel("rho_irr(k) [dB]")
    axes[0].set_title("rho_irr(k), w = w_opt, E_M framing")
    axes[0].grid(True, ls=":")
    axes[0].legend(fontsize=8)
    axes[1].plot(k[1:NB - 1], lam_bar["a145"][1:NB - 1], lw=0.8, label="lambar(k), a145 (F1)")
    axes[1].set_xlabel("bin k")
    axes[1].set_ylabel("lambar(k)")
    axes[1].set_title("frozen lambar(k) from the T3 run (a145)")
    axes[1].grid(True, ls=":")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dissect_rho_irr.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(OUT, "dissect_grid.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["label", "Lf", "source", "mu", "whole_db", "first5s_db", "last10_db", "final_msd_db", "diverged"])
        for (lab, Lf, mu, t), r in sorted(cell.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], ALLTAGS.index(kv[0][3]))):
            wr.writerow([lab, Lf, t, mu, r["whole"], r["first5"], r["last10"], r["msd"], int(r["diverged"])])

    # ---------------- Report
    L = []
    L.append("# lamC(k) dissection (atsip/explore/dissect.py)\n")
    L.append("Numbers only. Generated by `atsip/explore/dissect.py`.\n")
    L.append(f"Runtime (wall clock, 2 worker processes): total {runtime:.1f} s; "
             f"stage A (grid, 10 sources) {tA:.1f} s; stage B (T3 track, F1, rho_irr) {tB:.1f} s; "
             f"stage C (F2) {tC:.1f} s.\n")
    L.append("Sources: six exploration = " + ", ".join(SIX) + "; alpha=1.2 = " + ", ".join(TWO)
             + "; colored = " + ", ".join(COL) + " (CMS alpha=1.45, IIR y[n]=x[n]+1.2y[n-1]-0.5y[n-2], "
             "median |.| matched to the unfiltered sequence).\n")
    L.append("Framing E_M (last 512 errors, zero-padded to N) unless stated. "
             "Means over the six exploration sources; nan if any source is non-finite. "
             "#div = diverged runs (X.metrics rule), over the six sources at the best mu and over the sweep (of 36).\n")
    L.append("## Setup checks\n")
    L.append(f"- Reference rows B0, B2, C5, T3 (E_M) vs projlam_grid.csv (B0_M, B2_M, C5_M, T3_M): {n_cmp} cells, "
             f"max |diff| of whole-curve ANR = {max_diff:.2e} dB.")
    L.append(f"- T3 at its best mu, tracked run (stage B) vs grid run: max |diff| of whole = {t3_diff:.2e} dB.")
    L.append(f"- T3 best mu (six-source whole mean, E_M): {mu_T3}.\n")

    L.append("## Part 1. Controls for lamC(k), E_M framing, best mu by six-source whole mean\n")
    L.append("| cand | description | best mu | whole [dB] | first5s [dB] | last10 [dB] | final MSD [dB] | #div at best mu (of 6) | #div sweep (of 36) | alpha1.2 whole s0 / s1 [dB] | colored whole s10 / s11 [dB] |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for lab in PART1:
        mu = best[lab]
        if mu is None:
            L.append(f"| {lab} | {DESC[lab]} | nan | nan | nan | nan | nan | nan | {ndiv_sweep(lab)} | nan | nan |")
            continue
        a0, a1 = cell[(lab, M, mu, TWO[0])], cell[(lab, M, mu, TWO[1])]
        c0, c1 = cell[(lab, M, mu, COL[0])], cell[(lab, M, mu, COL[1])]
        L.append(f"| {lab} | {DESC[lab]} | {mu} | {X.fmt(m6(lab, M, mu, 'whole'))} | "
                 f"{X.fmt(m6(lab, M, mu, 'first5'))} | {X.fmt(m6(lab, M, mu, 'last10'))} | "
                 f"{X.fmt(m6(lab, M, mu, 'msd'))} | {ndiv(lab, mu, SIX)} | {ndiv_sweep(lab)} | "
                 f"{X.fmt(a0['whole'])} / {X.fmt(a1['whole'])} | {X.fmt(c0['whole'])} / {X.fmt(c1['whole'])} |")
    L.append("")
    L.append("Note: F1 and F2 are evaluated over the same mu sweep as the other rows, with lambar(k) frozen from the T3 run at mu = "
             f"{mu_T3}.\n")

    L.append("## Part 1. Spearman over bins 1..511, lambar(k) (F1, a145 file)\n")
    L.append("| quantity | Spearman with lambar(k) |")
    L.append("|---|---|")
    L.append(f"| (i) time-averaged Srr(k)/Pbar | {X.fmt(corr_a['srr_ratio'], 3)} |")
    L.append(f"| (ii) |Shat(k)|^2 (= S2 shape) | {X.fmt(corr_a['S2'], 3)} |")
    L.append(f"| (iii) rho_irr(k), a145 | {X.fmt(corr_a['rho_irr'], 3)} |")
    lbm = {t: (float(np.min(lam_bar[t][1:NB - 1])), float(np.mean(lam_bar[t][1:NB - 1])),
               float(np.max(lam_bar[t][1:NB - 1]))) for t in ALLTAGS}
    L.append("")
    L.append("lambar(k) over bins 1..511, min / mean / max per source: "
             + "; ".join(f"{t} {X.fmt(v[0], 3)} / {X.fmt(v[1], 3)} / {X.fmt(v[2], 3)}" for t, v in lbm.items()) + ".\n")

    L.append("## Part 2. Flatness of the irreducible part (w = w_opt, E_M, bins 1..511)\n")
    L.append("| source | rho_irr max-min [dB] | rho_irr p90-p10 [dB] | rho_irr min / mean / max [dB] | Spearman(rho_irr, |Shat|^2) | |E_opt|^2 max-min [dB] | |E_opt|^2 p90-p10 [dB] | Spearman(|E_opt|^2, |Shat|^2) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for t in ("a145", "col145_s10"):
        s = p2[t]
        L.append(f"| {t} | {X.fmt(s['rho_maxmin'])} | {X.fmt(s['rho_p1090'])} | "
                 f"{X.fmt(s['rho_min'])} / {X.fmt(s['rho_mean_db'])} / {X.fmt(s['rho_max'])} | "
                 f"{X.fmt(s['rho_S2'], 3)} | {X.fmt(s['pw_maxmin'])} | {X.fmt(s['pw_p1090'])} | {X.fmt(s['pw_S2'], 3)} |")
    L.append("")
    L.append("All ten sources (same definitions):\n")
    L.append("| source | rho_irr max-min [dB] | rho_irr p90-p10 [dB] | mean rho_irr [dB] | Spearman(rho_irr, |Shat|^2) |")
    L.append("|---|---|---|---|---|")
    for t in ALLTAGS:
        s = p2[t]
        L.append(f"| {t} | {X.fmt(s['rho_maxmin'])} | {X.fmt(s['rho_p1090'])} | {X.fmt(s['rho_mean_db'])} | {X.fmt(s['rho_S2'], 3)} |")
    L.append("")
    L.append("Curves: `out/dissect_rho_irr.npz` (rho_, pw_, srratio_, lambar_, S2_ per source); figure `out/dissect_rho_irr.png`.\n")

    L.append("## Part 3. Error-window length (B0, B2, T3), six-source means\n")
    L.append("| L | cand | best mu | whole [dB] | last10 [dB] | final MSD [dB] | #div at best mu (of 6) | best_mu*L/B |")
    L.append("|---|---|---|---|---|---|---|---|")
    for Lf in [128, 256, 386, 512]:
        for base in PART3_BASE:
            lab = base if Lf == M else f"{base}_L{Lf}"
            mu = best_mu(base, Lf) if Lf == M else best_mu(lab, Lf)
            if mu is None:
                L.append(f"| {Lf} | {base} | nan | nan | nan | nan | nan | nan |")
                continue
            L.append(f"| {Lf} | {base} | {mu} | {X.fmt(m6(lab, Lf, mu, 'whole'))} | {X.fmt(m6(lab, Lf, mu, 'last10'))} | "
                     f"{X.fmt(m6(lab, Lf, mu, 'msd'))} | {ndiv(lab, mu, SIX, Lf)} | {mu * Lf / B:.3f} |")
    L.append("")

    L.append("## Appendix A. Full grid, six-source whole-curve mean [dB] (E_M, Part 1 labels)\n")
    L.append("| cand | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for lab in PART1:
        L.append(f"| {lab} | " + " | ".join(X.fmt(m6(lab, M, mu, "whole")) for mu in MUS) + " |")
    L.append("")
    L.append("## Appendix B. Full grid, six-source last-10% mean [dB]\n")
    L.append("| cand | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for lab in PART1:
        L.append(f"| {lab} | " + " | ".join(X.fmt(m6(lab, M, mu, "last10")) for mu in MUS) + " |")
    L.append("")
    L.append("## Appendix C. Full grid, six-source first-5 s mean [dB]\n")
    L.append("| cand | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for lab in PART1:
        L.append(f"| {lab} | " + " | ".join(X.fmt(m6(lab, M, mu, "first5")) for mu in MUS) + " |")
    L.append("")
    L.append("## Appendix D. Full grid, six-source final MSD [dB]\n")
    L.append("| cand | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for lab in PART1:
        L.append(f"| {lab} | " + " | ".join(X.fmt(m6(lab, M, mu, "msd")) for mu in MUS) + " |")
    L.append("")
    L.append("## Appendix E. Diverged runs (of six) per mu\n")
    L.append("| cand | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for lab in PART1:
        L.append(f"| {lab} | " + " | ".join(str(ndiv(lab, mu, SIX)) for mu in MUS) + " |")
    L.append("")
    L.append("## Appendix F. Whole-curve ANR [dB] per source at every mu (E_M)\n")
    L.append("| cand | source | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|---|" + "---|" * len(MUS))
    for lab in PART1:
        for t in ALLTAGS:
            L.append(f"| {lab} | {t} | " + " | ".join(X.fmt(cell[(lab, M, mu, t)]["whole"]) for mu in MUS) + " |")
    L.append("")
    L.append("## Appendix G. Part 3 whole-curve six-source means [dB] per mu\n")
    L.append("| L | cand | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|---|" + "---|" * len(MUS))
    for Lf in [128, 256, 386, 512]:
        for base in PART3_BASE:
            lab = base if Lf == M else f"{base}_L{Lf}"
            L.append(f"| {Lf} | {base} | " + " | ".join(X.fmt(m6(lab, Lf, mu, "whole")) for mu in MUS) + " |")
    L.append("")
    L.append("## Files\n")
    L.append("- `out/dissect_grid.csv`: every run (label, window length, source, mu).")
    L.append("- `out/dissect_rho_irr.npz`: rho_irr, |E_opt|^2 mean, Srr/Pbar mean, lambar(k), S2 per source.")
    L.append("- `out/dissect_rho_irr.png`: rho_irr(k) in dB (a145, col145_s10) and lambar(k) (a145).")
    L.append("- `out/dissect_summary.json`: best mu, Part 2 and correlation summaries.")
    L.append("")
    with open(os.path.join(HERE, "RESULTS_dissect.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    summ = dict(runtime_s=runtime, mu_T3=mu_T3, best=best, part2=p2, corr_a145=corr_a,
                max_diff_projlam=max_diff, n_cmp=n_cmp, t3_track_diff=t3_diff,
                stage_runtime=dict(A=tA, B=tB, C=tC))
    with open(os.path.join(OUT, "dissect_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1, default=float)
    print(f"[total] {runtime:.1f} s", flush=True)


if __name__ == "__main__":
    main()
