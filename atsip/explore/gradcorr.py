#!/usr/bin/env python3
"""Gradient-correlation estimate of the mixing ratio lam for delayless FD-FxNLMS (gradcorr).

Idea. The expected gradient is (input power) x (misalignment); the unreachable residual and the noise
average to zero in the causal (lags 0..M-1) projection. The correlation between two projected gradients
taken d blocks apart (frames not overlapping) therefore estimates the misalignment part, and normalising by
the gradient power gives lamG = misalignment / (misalignment + noise), an estimate of the optimal mixing ratio.

Indicator, computed every block BEFORE the update from a fixed reference normaliser (Srr + eps, which does
not depend on lam):
    h_i   = irfft( conj(R_i) E_i / (Srr + eps), N )[0:M]         (projected binwise gradient, length M)
    scalar: num = EMA_a(<h_i, h_{i-d}>), den = EMA_a(||h_i||^2), lamG = clip(num/den, lam_min, 1)
    per-bin: Hp_i = rfft(h_i zero-padded to N), num(k) = EMA_a(Re{Hp_i conj(Hp_{i-d})}),
             den(k) = EMA_a(|Hp_i|^2), smoothed over frequency (9-bin moving average), lamG(k) = clip(num/den)
Defaults d = 8, a = 0.05, lam_min = 0.02. Ring buffer of the last d projected gradients.

Update: Lam(k) = (1-lam) Pbar + lam Srr + eps; G = s conj(R) E / Lam, s = lam (shrink) or 1 (no shrink).

Candidates (E_M framing, last 512 errors zero-padded to N):
    R1 scalar lamG, shrink          R5 R3 with d = 4
    R2 scalar lamG, no shrink       R6 R3 with a = 0.02
    R3 per-bin lamG(k), shrink      R7 R3 with a = 0.2
    R4 per-bin lamG(k), no shrink   R8 per-bin lamG(k) in Lam, scalar lamG on the gain
    Refs (dissect.py): B0, B2, T3, W0.5.

Usage:  python3 atsip/explore/gradcorr.py
Writes  atsip/explore/out/gradcorr_*.{csv,json,npz,png,log} and atsip/explore/RESULTS_gradcorr.md.
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
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explore as X    # noqa: E402
import genie as G      # noqa: E402
import mixlam as ML    # noqa: E402
import dissect as D   # noqa: E402

O = X.O
FS, M, N, B, NB = X.FS, X.M, X.N, X.B, X.NB
OUT = os.path.join(HERE, "out")
MUS = [0.05, 0.1, 0.15, 0.2, 0.28, 0.36, 0.6, 1.0]
SIX, TWO, COL, ALLTAGS = D.SIX, D.TWO, D.COL, D.ALLTAGS
TINY = 1e-30
LAM_MIN = 0.02
SMOOTH = 9
REFS = ["B0", "B2", "T3", "W0.5"]
CANDS = {
    "R1": dict(mix="scalar", gain="scalar", d=8, a=0.05),
    "R2": dict(mix="scalar", gain="none", d=8, a=0.05),
    "R3": dict(mix="bin", gain="bin", d=8, a=0.05),
    "R4": dict(mix="bin", gain="none", d=8, a=0.05),
    "R5": dict(mix="bin", gain="bin", d=4, a=0.05),
    "R6": dict(mix="bin", gain="bin", d=8, a=0.02),
    "R7": dict(mix="bin", gain="bin", d=8, a=0.2),
    "R8": dict(mix="bin", gain="scalar", d=8, a=0.05),
}
DESC = {
    "R1": "scalar lamG, mix and gain x lamG",
    "R2": "scalar lamG, mix, no shrink",
    "R3": "per-bin lamG(k), mix and gain x lamG(k)",
    "R4": "per-bin lamG(k), mix, no shrink",
    "R5": "R3 with d = 4",
    "R6": "R3 with a = 0.02",
    "R7": "R3 with a = 0.2",
    "R8": "per-bin lamG(k) in Lam, gain x scalar lamG",
    "B0": "lam=1: Lam=Srr+eps, no shrink (ref)",
    "B2": "lam=0.45 fixed, no shrink (ref)",
    "T3": "lam=lamC(k), gain x lamC (ref)",
    "W0.5": "Lam=Srr^0.5 Pbar^0.5+eps, no lam (ref)",
}
LABS = REFS + list(CANDS)
WIN = [("0-1 s", 0, FS // B), ("1-5 s", FS // B, 5 * FS // B), ("5-15 s", 5 * FS // B, 15 * FS // B), ("last10", None, None)]


# ----------------------------------------------------------------------------
# Simulation with the gradient-correlation indicator
# ----------------------------------------------------------------------------
def simulate_gc(data, pp, cfg, mu, w0=None, track=False):
    """w <- w - mu_b irfft(G)[:M], mu_b = 2 mu B / M; E framing = last M errors, zero-padded to N.

    w0 and mu = 0 give a fixed-w run (indicator only). track=True records per-block scalar lamG,
    mean_k lamG(k) and the GL1 scalar lam_true (genie definition, E_M, pre-update w)."""
    d_, xq, hS, S2, w_opt = (data[k] for k in ("d", "xq", "hS", "S2", "w_opt"))
    Rall, Psi = pp["Rall"], pp["Psi"]["M"]
    c2 = ML.C2["M"]
    L = len(d_)
    nb = L // B
    lag, a = cfg["d"], cfg["a"]
    swv = sliding_window_view(xq, M)
    w = np.zeros(M) if w0 is None else np.array(w0, dtype=float).copy()
    zi = np.zeros(len(hS) - 1)
    e_all = np.zeros(L)
    mu_b = 2 * mu * B / M
    shared = X.Shared(S2)
    bufH = np.zeros((lag, M))
    bufP = np.zeros((lag, NB), dtype=np.complex128)
    snum = sden = 0.0
    knum = np.zeros(NB)
    kden = np.zeros(NB)
    rec = dict(lamS=[], lamK=[], lt=[]) if track else None
    diverged = False
    for i in range(nb):
        s0 = i * B
        Xb = swv[s0 + N - M + 1: s0 + N - M + 1 + B][:, ::-1]
        y = Xb @ w
        ys, zi = signal.lfilter(hS, [1.0], y, zi=zi)
        e_all[s0:s0 + B] = d_[s0:s0 + B] + ys
        R = Rall[i]
        E = ML.frame_E(e_all, s0, M)
        s = shared.step(R, E)
        Srr, Pbar, eps = s["Srr"], s["Pbar"], s["eps"]
        Ce = np.conj(R) * E

        # ---- indicator (uses blocks up to i, before the update of block i)
        h = np.fft.irfft(Ce / (Srr + eps), N)[:M]
        slot = i % lag
        prev_h, prev_P = bufH[slot].copy(), bufP[slot].copy()     # zeros while i < lag
        Hp = np.fft.rfft(h, N)
        bufH[slot] = h
        bufP[slot] = Hp
        snum = (1 - a) * snum + a * float(h @ prev_h)
        sden = (1 - a) * sden + a * float(h @ h)
        knum = (1 - a) * knum + a * np.real(Hp * np.conj(prev_P))
        kden = (1 - a) * kden + a * np.abs(Hp) ** 2
        if i >= lag:
            lamS = float(np.clip(snum / max(sden, TINY), LAM_MIN, 1.0))
            lamK = np.clip(uniform_filter1d(knum, SMOOTH, mode="nearest")
                           / np.maximum(uniform_filter1d(kden, SMOOTH, mode="nearest"), TINY), LAM_MIN, 1.0)
        else:
            lamS, lamK = 1.0, np.ones(NB)                          # no information yet

        if track:
            Pd = G.true_Pd(w, w_opt)
            q = c2 * Srr * Pd
            rec["lamS"].append(lamS)
            rec["lamK"].append(float(lamK.mean()))
            rec["lt"].append(float(np.sum(q) / np.sum(q + Psi)))

        # ---- update
        if cfg["mix"] == "scalar":
            Lam = (1 - lamS) * Pbar + lamS * Srr + eps
        else:
            Lam = (1 - lamK) * Pbar + lamK * Srr + eps
        if cfg["gain"] == "scalar":
            gmul = lamS
        elif cfg["gain"] == "bin":
            gmul = lamK
        else:
            gmul = 1.0
        g = np.fft.irfft(gmul * Ce / Lam, N)[:M]
        w = w - mu_b * g
        if not np.isfinite(w).all():
            e_all[s0 + B:] = np.nan
            diverged = True
            break
    out = dict(e=e_all, w=w, diverged=diverged)
    if track:
        out["rec"] = {k: np.asarray(v) for k, v in rec.items()}
    return out


def run_sim(data, pp, lab, mu):
    """Reference rows via dissect.simulate, candidate rows via simulate_gc (both E_M)."""
    if lab in CANDS:
        return simulate_gc(data, pp, CANDS[lab], mu)
    return D.simulate(data, pp["Rall"], D.spec_of(lab), mu, M)


def score(sim, data):
    """X.metrics, then NaN ANR metrics (whole, first5, last10) for every run flagged diverged."""
    m = X.metrics(sim["e"], sim["w"], data["d"], data["w_opt"])
    if m["diverged"]:
        m["whole"] = m["first5"] = m["last10"] = float("nan")
    return m


# ----------------------------------------------------------------------------
# Workers
# ----------------------------------------------------------------------------
def stageA(tag):
    """Full grid for one source: all labels x MUS; ANR curves kept for alpha=1.45 only."""
    t0 = time.time()
    kind, arg = D.KIND[tag]
    data = D.load_data(kind, arg)
    pp = ML.prepass(data)
    rows, anr = [], {}
    for mu in MUS:
        for lab in LABS:
            sim = run_sim(data, pp, lab, mu)
            m = score(sim, data)
            rows.append(dict(label=lab, mu=mu, src=tag, **m))
            if tag == "a145":
                anr[f"{lab}__{mu}"] = O.anr_curve(sim["e"], data["d"]).astype(np.float32)
    return dict(tag=tag, rows=rows, anr=anr, runtime=time.time() - t0)


def stageB(job):
    """Diagnostics for one (source, candidate) at its best mu: track run, and fixed-w runs at w_opt and 0."""
    tag, lab, mu = job
    t0 = time.time()
    kind, arg = D.KIND[tag]
    data = D.load_data(kind, arg)
    pp = ML.prepass(data)
    cfg = CANDS[lab]
    tr = simulate_gc(data, pp, cfg, mu, track=True)
    fw_opt = simulate_gc(data, pp, cfg, 0.0, w0=data["w_opt"], track=True)
    fw_zero = simulate_gc(data, pp, cfg, 0.0, w0=np.zeros(M), track=True)
    nbt = len(data["d"]) // B
    res = dict(tag=tag, lab=lab, mu=mu, nb=nbt, track=tr["rec"], track_score=score(tr, data),
               w_opt={k: float(v.mean()) for k, v in fw_opt["rec"].items()},
               w_zero={k: float(v.mean()) for k, v in fw_zero["rec"].items()},
               rho_spearman=dict(
                   lamS_lt=X.spearman(tr["rec"]["lamS"], tr["rec"]["lt"]),
                   lamK_lt=X.spearman(tr["rec"]["lamK"], tr["rec"]["lt"])),
               runtime=time.time() - t0)
    return res


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def nanmean6(vals):
    vals = np.asarray(vals, dtype=float)
    return float(vals.mean()) if np.isfinite(vals).all() else float("nan")


def main():
    os.makedirs(OUT, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()

    with Pool(processes=2) as pool:
        resA = pool.map(stageA, ALLTAGS)
    tA = time.time() - t0
    cell = {}
    anr_a145 = {}
    for res in resA:
        for r in res["rows"]:
            cell[(r["label"], r["mu"], r["src"])] = r
        if res["tag"] == "a145":
            anr_a145 = res["anr"]

    def m6(lab, mu, key):
        return nanmean6([cell[(lab, mu, t)][key] for t in SIX])

    def best_mu(lab, key="whole"):
        ok = {mu: m6(lab, mu, key) for mu in MUS}
        ok = {mu: v for mu, v in ok.items() if np.isfinite(v)}
        return min(ok, key=ok.get) if ok else None

    def ndiv(lab, mu, srcs=SIX):
        return int(sum(cell[(lab, mu, t)]["diverged"] for t in srcs))

    def ndiv_sweep(lab):
        return int(sum(ndiv(lab, mu) for mu in MUS))

    best_w = {lab: best_mu(lab, "whole") for lab in LABS}
    best_f = {lab: best_mu(lab, "first5") for lab in LABS}

    # best R by six-source whole at its own best mu
    rkeys = [lab for lab in LABS if lab.startswith("R")]
    best_R = min((lab for lab in rkeys if best_w[lab] is not None),
                 key=lambda lab: m6(lab, best_w[lab], "whole"))

    # ---- stage B: diagnostics at best mu (R1, R3 on a145 and cms145_s0)
    t1 = time.time()
    jobs = [(t, lab, best_w[lab]) for t in ("a145", "cms145_s0") for lab in ("R1", "R3")
            if best_w[lab] is not None]
    with Pool(processes=2) as pool:
        resB = pool.map(stageB, jobs)
    tB = time.time() - t1
    diag = {(r["tag"], r["lab"]): r for r in resB}

    # ---- sanity checks
    dgrid = {}
    with open(os.path.join(OUT, "dissect_grid.csv")) as fh:
        for row in csv.DictReader(fh):
            if row["label"] in ("B0", "T3") and row["Lf"] == str(M):
                dgrid[(row["label"], round(float(row["mu"]), 6), row["source"])] = float(row["whole_db"])
    max_d_dissect, n_d = 0.0, 0
    for lab in ("B0", "T3"):
        for mu in MUS:
            for t in SIX + TWO:
                key = (lab, round(mu, 6), t)
                if key in dgrid and np.isfinite(cell[(lab, mu, t)]["whole"]):
                    max_d_dissect = max(max_d_dissect, abs(cell[(lab, mu, t)]["whole"] - dgrid[key]))
                    n_d += 1
    # anr_a145.npz was written by explore.py, whose B0 uses E = last B errors (E_B framing);
    # reproduce those B0 curves with dissect.simulate(Lf=B) and compare.
    data_a = D.load_data(*D.KIND["a145"])
    pp_a = ML.prepass(data_a)
    max_d_anr, n_a = 0.0, 0
    with np.load(os.path.join(OUT, "anr_a145.npz")) as za:
        for mu in MUS:
            k = f"B0__{mu}"
            if k in za.files:
                s_b = D.simulate(data_a, pp_a["Rall"], D.spec_of("B0"), mu, B)
                cur = O.anr_curve(s_b["e"], data_a["d"]).astype(np.float32)
                if np.isfinite(cur).all():
                    max_d_anr = max(max_d_anr, float(np.max(np.abs(cur - za[k]))))
                    n_a += 1
        anr_O_band = za["O_band"].astype(np.float32)

    # ---- save curves, grid, summary
    np.savez(os.path.join(OUT, "gradcorr_anr_a145.npz"), **anr_a145, O_band=anr_O_band)
    with open(os.path.join(OUT, "gradcorr_grid.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["label", "mu", "source", "whole_db", "first5s_db", "last10_db", "final_msd_db", "diverged"])
        for (lab, mu, t), r in sorted(cell.items(), key=lambda kv: (LABS.index(kv[0][0]), kv[0][1], ALLTAGS.index(kv[0][2]))):
            wr.writerow([lab, mu, t, r["whole"], r["first5"], r["last10"], r["msd"], int(r["diverged"])])

    # ---- figure: alpha=1.45 ANR, B0 / T3 / W0.5 / best R at its best mu, O_band
    fig, ax = plt.subplots(figsize=(9, 4.2))
    tt = np.arange(len(anr_O_band)) * 64 / FS
    curves = [("B0", best_w["B0"]), ("T3", best_w["T3"]), ("W0.5", best_w["W0.5"]), (best_R, best_w[best_R])]
    for lab, mu in curves:
        if mu is None:
            continue
        ax.plot(tt, anr_a145[f"{lab}__{mu}"], lw=0.8, label=f"{lab} (mu={mu})")
    ax.plot(tt, anr_O_band, lw=0.8, color="k", ls="--", label="O_band")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("ANR [dB]")
    ax.set_title("alpha=1.45 ANR, E_M framing")
    ax.grid(True, ls=":")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gradcorr_anr_a145.png"), dpi=130)
    plt.close(fig)

    runtime = time.time() - t0

    # ---- report
    L = []
    L.append("# gradcorr: gradient-correlation mixing ratio (atsip/explore/gradcorr.py)\n")
    L.append("Numbers only. Generated by `atsip/explore/gradcorr.py`. E_M framing (last 512 errors, zero-padded to N) throughout.\n")
    L.append(f"Runtime (wall clock, 2 worker processes): total {runtime:.1f} s; grid (10 sources x {len(LABS)} labels x {len(MUS)} mu) "
             f"{tA:.1f} s; diagnostics {tB:.1f} s.\n")
    L.append("Sources: six exploration = " + ", ".join(SIX) + "; alpha=1.2 = " + ", ".join(TWO)
             + "; colored = " + ", ".join(COL) + ".\n")
    L.append("Means are over the six exploration sources and are NaN if any of the six is NaN (diverged). "
             "Diverged runs (X.metrics rule, plus the rule last10 > -5 dB) have NaN whole, first5 and last10. "
             "#div counts runs flagged diverged over the six sources; sweep = 6 x 8 = 48 runs per label. "
             "best mu = minimiser of the six-source mean over the eight mu values.\n")
    L.append(f"Candidate descriptions: " + "; ".join(f"{k}: {v}" for k, v in DESC.items()) + ".\n")
    L.append("## Setup checks\n")
    L.append(f"- B0 and T3 (E_M) vs dissect_grid.csv (Lf=512, whole-curve dB, mu in both grids): {n_d} cells, "
             f"max |diff| = {max_d_dissect:.2e} dB.")
    L.append(f"- B0 ANR curves vs anr_a145.npz (B0__mu, written by explore.py with E = last B errors): {n_a} curves, "
             f"max |diff| = {max_d_anr:.2e} dB. The comparison uses E_B framing (dissect.simulate, Lf = B). "
             "B0 depends on the framing through conj(R) E, so E_M curves are not comparable with that file.")
    L.append("- Indicator: scalar and per-bin lamG use blocks 0..i; for i < d the indicator is undefined and lamG = 1 is used.\n")

    L.append("## Table 1. Best mu by six-source whole mean\n")
    L.append("| label | description | best mu | whole [dB] | first5 [dB] | last10 [dB] | final MSD [dB] | #div at best mu (of 6) | #div sweep (of 48) | alpha1.2 whole s0 / s1 [dB] | colored whole s10 / s11 [dB] |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for lab in LABS:
        mu = best_w[lab]
        if mu is None:
            L.append(f"| {lab} | {DESC[lab]} | nan | nan | nan | nan | nan | nan | {ndiv_sweep(lab)} | nan | nan |")
            continue
        a0, a1 = cell[(lab, mu, TWO[0])], cell[(lab, mu, TWO[1])]
        c0, c1 = cell[(lab, mu, COL[0])], cell[(lab, mu, COL[1])]
        L.append(f"| {lab} | {DESC[lab]} | {mu} | {X.fmt(m6(lab, mu, 'whole'))} | {X.fmt(m6(lab, mu, 'first5'))} | "
                 f"{X.fmt(m6(lab, mu, 'last10'))} | {X.fmt(m6(lab, mu, 'msd'))} | {ndiv(lab, mu)} | {ndiv_sweep(lab)} | "
                 f"{X.fmt(a0['whole'])} / {X.fmt(a1['whole'])} | {X.fmt(c0['whole'])} / {X.fmt(c1['whole'])} |")
    L.append("")

    L.append("## Table 2. Best mu by six-source first-5 s mean\n")
    L.append("| label | best mu | first5 [dB] | whole at that mu [dB] | last10 at that mu [dB] | #div at that mu (of 6) |")
    L.append("|---|---|---|---|---|---|")
    for lab in LABS:
        mu = best_f[lab]
        if mu is None:
            L.append(f"| {lab} | nan | nan | nan | nan | nan |")
            continue
        L.append(f"| {lab} | {mu} | {X.fmt(m6(lab, mu, 'first5'))} | {X.fmt(m6(lab, mu, 'whole'))} | "
                 f"{X.fmt(m6(lab, mu, 'last10'))} | {ndiv(lab, mu)} |")
    L.append("")

    L.append(f"Best R by six-source whole: {best_R} at mu = {best_w[best_R]}.\n")

    L.append("## Diagnostics (R1 and R3 at their best mu)\n")
    L.append("Windows: 0-1 s, 1-5 s, 5-15 s, last 10% of blocks. lamS = scalar lamG; meanK = mean over k of lamG(k); "
             "lt = scalar lam_true (GL1 definition, E_M, pre-update w). Entries are window means, written as "
             "0-1 s / 1-5 s / 5-15 s / last10.\n")
    L.append("| label | source | mu | lamS | meanK | lam_true | Spearman(lamS, lam_true) | Spearman(meanK, lam_true) | run diverged |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for (tag, lab), r in sorted(diag.items(), key=lambda kv: (kv[0][1], ALLTAGS.index(kv[0][0]))):
        rec, nb = r["track"], r["nb"]
        def wins(v):
            out = []
            for name, lo, hi in WIN:
                if lo is None:
                    lo, hi = int(0.9 * nb), nb
                seg = v[lo:min(hi, len(v))]
                out.append(X.fmt(float(seg.mean()) if len(seg) else float("nan"), 3))
            return " / ".join(out)
        L.append(f"| {lab} | {tag} | {r['mu']} | {wins(rec['lamS'])} | {wins(rec['lamK'])} | {wins(rec['lt'])} | "
                 f"{X.fmt(r['rho_spearman']['lamS_lt'], 3)} | {X.fmt(r['rho_spearman']['lamK_lt'], 3)} | "
                 f"{int(r['track_score']['diverged'])} |")
    L.append("")
    L.append("Fixed w (no adaptation), time-averaged over all blocks (scalar lamG; mean_k lamG(k)); "
             "w = w_opt should sit near lam_min = 0.02, w = 0 should sit near 1.\n")
    L.append("| label | source | w fixed at | lamS mean | meanK mean |")
    L.append("|---|---|---|---|---|")
    for (tag, lab), r in sorted(diag.items(), key=lambda kv: (kv[0][1], ALLTAGS.index(kv[0][0]))):
        L.append(f"| {lab} | {tag} | w_opt | {X.fmt(r['w_opt']['lamS'], 3)} | {X.fmt(r['w_opt']['lamK'], 3)} |")
        L.append(f"| {lab} | {tag} | 0 | {X.fmt(r['w_zero']['lamS'], 3)} | {X.fmt(r['w_zero']['lamK'], 3)} |")
    L.append("")

    L.append("## ANR figure\n")
    L.append("`out/gradcorr_anr_a145.png`: alpha=1.45, B0, T3, W0.5 and the best R at their best mu (E_M), with O_band "
             "taken as stored in `out/anr_a145.npz` (written by explore.py, not re-simulated here). "
             "Curves in `out/gradcorr_anr_a145.npz`.\n")

    def grid_table(key, title):
        L.append(f"## Appendix: {title}\n")
        L.append("| label | " + " | ".join(f"mu={m}" for m in MUS) + " |")
        L.append("|---|" + "---|" * len(MUS))
        for lab in LABS:
            L.append(f"| {lab} | " + " | ".join(X.fmt(m6(lab, mu, key)) for mu in MUS) + " |")
        L.append("")

    grid_table("whole", "six-source whole-curve mean [dB]")
    grid_table("first5", "six-source first-5 s mean [dB]")
    grid_table("last10", "six-source last-10% mean [dB]")
    grid_table("msd", "six-source final MSD [dB]")

    L.append("## Appendix: diverged runs (of six) per mu\n")
    L.append("| label | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for lab in LABS:
        L.append(f"| {lab} | " + " | ".join(str(ndiv(lab, mu)) for mu in MUS) + " |")
    L.append("")

    L.append("## Appendix: whole-curve ANR [dB] per source at every mu\n")
    L.append("| label | source | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|---|" + "---|" * len(MUS))
    for lab in LABS:
        for t in ALLTAGS:
            L.append(f"| {lab} | {t} | " + " | ".join(X.fmt(cell[(lab, mu, t)]["whole"]) for mu in MUS) + " |")
    L.append("")

    L.append("## Files\n")
    L.append("- `out/gradcorr_grid.csv`: every run (label, mu, source).")
    L.append("- `out/gradcorr_anr_a145.npz`: alpha=1.45 ANR curves per label and mu, plus O_band.")
    L.append("- `out/gradcorr_anr_a145.png`: ANR figure.")
    L.append("- `out/gradcorr_summary.json`: best mu, diagnostics, runtimes.")
    L.append("")
    with open(os.path.join(HERE, "RESULTS_gradcorr.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    summ = dict(runtime_s=runtime, best_mu_whole=best_w, best_mu_first5=best_f, best_R=best_R,
                max_diff_dissect=max_d_dissect, n_dissect=n_d, max_diff_anr=max_d_anr, n_anr=n_a,
                stage_runtime=dict(A=tA, B=tB))
    with open(os.path.join(OUT, "gradcorr_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1, default=float)
    print(f"[total] {runtime:.1f} s", flush=True)


if __name__ == "__main__":
    main()
