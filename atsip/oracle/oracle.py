#!/usr/bin/env python3
"""Oracle experiment: which frequency-domain normalization gives the best update direction
for the delayless FD-FxNLMS under impulsive reference input.

Usage:  python3 atsip/oracle/oracle.py
Writes  atsip/oracle/out/*.csv, *.json, *.png and atsip/oracle/RESULTS.md.
"""
import os
import time
import json
import warnings
from multiprocessing import Pool

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal, linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS, M, N, B = 16000, 512, 1024, 128   # sample rate, control taps, FFT size, block hop
C_SEC = 128                           # secondary path taps (trimmed)
ALPHA = 0.05                          # EMA weight on the newest frame
MU_DIAG = 0.36                        # step of the plain N0 driver (Experiment A)
MUS = [0.05, 0.1, 0.2, 0.36, 0.6, 1.0]
NORMS = ["N0", "N1", "N2", "N3", "N4"]
NORM_DESC = {
    "N0": "binwise EMA",
    "N1": "block scalar mean",
    "N2": "mixed 0.55*N1+0.45*N0",
    "N3": "instantaneous",
    "N4": "two-component",
}
N_NOISE = 480000
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "out")
SOURCES = [
    ("a145", "file", "sounds/impulsive_noise_alpha1.45.dat"),
    ("a165", "file", "sounds/impulsive_noise_alpha1.65.dat"),
    ("a185", "file", "sounds/impulsive_noise_alpha1.85.dat"),
    ("cms145_s0", "cms", 0),
    ("cms145_s1", "cms", 1),
    ("cms145_s2", "cms", 2),
]


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def cms_symmetric_stable(alpha, n, seed):
    """Chambers-Mallows-Stuck sample of symmetric alpha-stable noise (beta=0, scale c=1)."""
    rng = np.random.default_rng(seed)
    U = rng.uniform(-np.pi / 2, np.pi / 2, n)
    W = rng.exponential(1.0, n)
    return (np.sin(alpha * U) / np.cos(U) ** (1 / alpha)
            * (np.cos((1 - alpha) * U) / W) ** ((1 - alpha) / alpha))


def pad_or_trim(h, L):
    h = np.asarray(h, dtype=np.float64)
    out = np.zeros(L)
    k = min(L, h.size)
    out[:k] = h[:k]
    return out


def wiener_opt(r, d, T=10 * FS):
    """LS solution of min ||d + R w||^2 (R[n,k] = r[n-k]) from the first T samples."""
    r, d = r[:T], d[:T]
    T = len(r)
    phi = np.array([np.dot(r[j:], r[:T - j]) for j in range(M)])
    c = np.array([np.dot(r[:T - j], d[j:]) for j in range(M)])
    phi[0] += 1e-8 * phi[0]              # trace-scaled ridge (trace = M*phi[0])
    return linalg.solve_toeplitz(phi, -c)


def load_source(kind, arg):
    if kind == "file":
        x = np.loadtxt(os.path.join(ROOT, arg), ndmin=1).astype(np.float64)
    else:
        x = cms_symmetric_stable(1.45, N_NOISE, seed=arg)
    hP = pad_or_trim(np.loadtxt(os.path.join(ROOT, "impulse_response/primary_ir.dat"), ndmin=1), M)
    hS = pad_or_trim(np.loadtxt(os.path.join(ROOT, "impulse_response/secondary_ir.dat"), ndmin=1), C_SEC)
    d = signal.lfilter(hP, [1.0], x)     # primary noise at error mic
    r = signal.lfilter(hS, [1.0], x)     # filtered reference s*x
    data = dict(x=x, d=d, r=r, hS=hS,
                xq=np.concatenate([np.zeros(N), x]),   # xq[m] = x[m-N]
                Sf=np.fft.rfft(hS, N))
    data["w_opt"] = wiener_opt(r, d)
    return data


# ----------------------------------------------------------------------------
# Normalizations Lam(k)
# ----------------------------------------------------------------------------
class Normalizers:
    """State for the five Lam(k) choices. update() takes the current frame R_i."""

    def __init__(self):
        self.srr = None       # EMA of |R|^2 (base of N0, N1, N2)
        self.logPbg = None    # EMA of log P_i (N4)
        self.shape = None     # EMA of |R|^2 / P_i (N4)
        self.psum = 0.0
        self.count = 0

    def update(self, R):
        A = np.abs(R) ** 2
        tiny = 1e-30
        P = A.mean()
        self.psum += P
        self.count += 1
        eps = 1e-6 * self.psum / self.count

        # N0 base: EMA updated with the current frame, then used
        self.srr = A.copy() if self.srr is None else (1 - ALPHA) * self.srr + ALPHA * A

        # N4 background: used BEFORE its update (initialised with the first frame)
        if self.logPbg is None:
            self.logPbg = np.log(P + tiny)
        Pbg = np.exp(self.logPbg)
        z = P / (Pbg + tiny)
        self.logPbg = (1 - ALPHA) * self.logPbg + ALPHA * np.log(P + tiny)
        s_new = A / (P + tiny)
        self.shape = s_new.copy() if self.shape is None else (1 - ALPHA) * self.shape + ALPHA * s_new

        lam0 = self.srr + eps
        lam1 = np.full_like(A, self.srr.mean() + eps)
        lam2 = (1 - 0.45) * lam1 + 0.45 * lam0
        lam3 = A + eps
        lam4 = Pbg * self.shape + np.maximum(P - Pbg, 0.0) + eps
        return [lam0, lam1, lam2, lam3, lam4], P, z


def direction(R, E, lam):
    """Causal part of the block gradient: g = irfft(conj(R) E / Lam)[0:M]."""
    return np.fft.irfft(np.conj(R) * E / lam, N)[:M]


# ----------------------------------------------------------------------------
# Block simulation (delayless, w held fixed within each block)
# rule = 'fixed'  : w <- w - mu_b * g_norm,  mu_b = 2*mu*B/M
# rule = 'oracle' : w <- w - clip(mu_star_norm, 0, cap) * g_norm
# Diagnostics for all five directions are computed at every block before the update.
# ----------------------------------------------------------------------------
def simulate(data, rule, norm=None, mu=None):
    x, d, xq, hS, Sf, w_opt = (data[k] for k in ("x", "d", "xq", "hS", "Sf", "w_opt"))
    L = len(x)
    nb = L // B
    swv = sliding_window_view(xq, M)
    w = np.zeros(M)
    zi = np.zeros(len(hS) - 1)
    e_all = np.zeros(L)
    nan_rec = lambda: np.full(nb, np.nan)
    msd_db, z_rec, gam_rec, P_rec, mu_used = nan_rec(), nan_rec(), nan_rec(), nan_rec(), nan_rec()
    mu_star = np.full((5, nb), np.nan)
    cos2 = np.full((5, nb), np.nan)
    mu_b = 2 * mu * B / M if rule == "fixed" else None
    wo2 = w_opt @ w_opt
    norms = Normalizers()
    Sre = Srr_c = See = None
    pos_hist = []
    diverged = False

    for i in range(nb):
        s0 = i * B
        # output y = w^T x over the block (rows: x[n], x[n-1], ..., x[n-M+1])
        Xb = swv[s0 + N - M + 1: s0 + N - M + 1 + B][:, ::-1]
        y = Xb @ w
        ys, zi = signal.lfilter(hS, [1.0], y, zi=zi)
        e = d[s0:s0 + B] + ys
        e_all[s0:s0 + B] = e

        # frame spectra: X from the last N samples of x, E from the last B samples of e
        X = np.fft.rfft(xq[s0 + B: s0 + B + N])
        R = Sf * X
        E = np.fft.rfft(np.concatenate([np.zeros(N - B), e]))

        lams, P, z = norms.update(R)
        cRE, cRR, cEE = R * np.conj(E), np.abs(R) ** 2, np.abs(E) ** 2
        if Sre is None:
            Sre, Srr_c, See = cRE, cRR, cEE
        else:
            Sre = (1 - ALPHA) * Sre + ALPHA * cRE
            Srr_c = (1 - ALPHA) * Srr_c + ALPHA * cRR
            See = (1 - ALPHA) * See + ALPHA * cEE
        gam = np.mean(np.abs(Sre) ** 2 / (Srr_c * See + 1e-30))

        gs = [direction(R, E, lam) for lam in lams]
        Delta = w - w_opt
        dd = Delta @ Delta
        msd_db[i] = 10 * np.log10(dd / wo2 + 1e-30)
        for n, g in enumerate(gs):
            gg = g @ g + 1e-30
            gD = g @ Delta
            mu_star[n, i] = gD / gg
            cos2[n, i] = gD ** 2 / (gg * dd + 1e-30)
        z_rec[i], gam_rec[i], P_rec[i] = z, gam, P

        g = gs[norm]
        if rule == "fixed":
            step = mu_b
        else:
            ms = mu_star[norm, i]
            cap = 1.0 if (i < 20 or not pos_hist) else 4 * np.median(pos_hist)
            step = np.clip(ms, 0.0, cap)
            if ms > 0:
                pos_hist.append(ms)
        mu_used[i] = step
        w = w - step * g
        if not np.isfinite(w).all():
            e_all[s0 + B:] = np.nan
            diverged = True
            break

    return dict(e=e_all, w=w, msd_db=msd_db, z=z_rec, gamma=gam_rec, P=P_rec,
                mu_star=mu_star, cos2=cos2, mu_used=mu_used, diverged=diverged)


# ----------------------------------------------------------------------------
# ANR (paper definition, sampled every 64 samples)
# ----------------------------------------------------------------------------
def anr_curve(e, d):
    Ae = signal.lfilter([0.001], [1.0, -0.999], np.abs(e))
    Ad = signal.lfilter([0.001], [1.0, -0.999], np.abs(d))
    idx = np.arange(63, len(e), 64)
    return 20 * np.log10((Ae[idx] + 1e-12) / (Ad[idx] + 1e-12))


def summarize(anr):
    """(whole-curve mean, last-10% mean); NaN if the run diverged."""
    if not np.isfinite(anr).all():
        return float("nan"), float("nan")
    n10 = max(1, int(0.1 * len(anr)))
    return float(anr.mean()), float(anr[-n10:].mean())


def final_msd_db(w, w_opt):
    return float(10 * np.log10((w - w_opt) @ (w - w_opt) / (w_opt @ w_opt) + 1e-30))


# ----------------------------------------------------------------------------
# Per-source worker
# ----------------------------------------------------------------------------
def write_diag_csv(tag, res):
    nb = len(res["msd_db"])
    cols = np.column_stack([np.arange(nb), np.arange(nb) * B / FS, res["P"], res["z"],
                            res["gamma"], res["msd_db"], res["mu_star"].T, res["cos2"].T])
    header = ("block,time_s,P,z,gamma_mean,msd_db," + ",".join(f"mu_star_{n}" for n in NORMS)
              + "," + ",".join(f"cos2_{n}" for n in NORMS))
    np.savetxt(os.path.join(OUT, f"diag_{tag}.csv"), cols, delimiter=",", header=header,
               comments="", fmt="%.8g")


def run_source(spec):
    tag, kind, arg = spec
    t0 = time.time()
    data = load_source(kind, arg)
    d, x, w_opt = data["d"], data["x"], data["w_opt"]
    e_opt = d + signal.lfilter(data["hS"], [1.0], np.convolve(x, w_opt)[:len(x)])
    res = {"tag": tag, "opt_anr": summarize(anr_curve(e_opt, d)),
           "oracle": {}, "fixed": {}}

    diag = simulate(data, "fixed", norm=0, mu=MU_DIAG)
    write_diag_csv(tag, diag)
    res["diag_N0_mu036"] = summarize(anr_curve(diag["e"], d))
    res["diag_N0_mu036_msd_end"] = final_msd_db(diag["w"], w_opt)

    for n, nm in enumerate(NORMS):
        o = simulate(data, "oracle", norm=n)
        res["oracle"][nm] = dict(zip(("whole", "last10"), summarize(anr_curve(o["e"], d))),
                                 msd_end=final_msd_db(o["w"], w_opt), diverged=o["diverged"])
        if tag == "a145":
            cols = np.column_stack([np.arange(len(o["msd_db"])) * B / FS, o["msd_db"]])
            np.savetxt(os.path.join(OUT, f"oracle_msd_{nm}_a145.csv"), cols, delimiter=",",
                       header="time_s,msd_db", comments="", fmt="%.8g")

    for n, nm in enumerate(NORMS):
        res["fixed"][nm] = {}
        for mu in MUS:
            f = simulate(data, "fixed", norm=n, mu=mu)
            res["fixed"][nm][str(mu)] = dict(zip(("whole", "last10"), summarize(anr_curve(f["e"], d))),
                                              msd_end=final_msd_db(f["w"], w_opt),
                                              diverged=f["diverged"])
    res["runtime_s"] = time.time() - t0
    print(f"[done] {tag}  N0@0.36 whole={res['diag_N0_mu036'][0]:.2f} dB  "
          f"runtime={res['runtime_s']:.1f}s", flush=True)
    return res


# ----------------------------------------------------------------------------
# Analysis and report
# ----------------------------------------------------------------------------
def diag_metrics(tag):
    a = np.loadtxt(os.path.join(OUT, f"diag_{tag}.csv"), delimiter=",", skiprows=1)
    z, gam = a[:, 3], a[:, 4]
    mus, cos2 = a[:, 6:11].T, a[:, 11:16].T
    nb = len(z)
    start = int(2 * FS / B)
    imp, non = z > 10, z < 2
    out = {}
    for n, nm in enumerate(NORMS):
        c, m = cos2[n], mus[n]
        lm = np.log(m[(np.arange(nb) >= start) & (m > 0)])
        out[nm] = dict(
            cos2=float(c.mean()),
            cos2_imp=float(c[imp].mean()) if imp.any() else float("nan"),
            cos2_non=float(c[non].mean()) if non.any() else float("nan"),
            cv_log=float(lm.std() / abs(lm.mean())),
            std_log=float(lm.std()),
            med_imp=float(np.median(m[imp])) if imp.any() else float("nan"),
            med_non=float(np.median(m[non])) if non.any() else float("nan"),
        )
    out["_meta"] = dict(n_imp=int(imp.sum()), n_non=int(non.sum()), n_blocks=nb,
                        gamma_mean=float(gam.mean()), frac_z_gt10=float(imp.mean()))
    return out


def _avg(rows, key):
    vals = [r[key] for r in rows]
    return float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")


def fmt(v, p=3):
    return "nan" if v is None or not np.isfinite(v) else f"{v:.{p}f}"


def make_figures(tags, metrics_unused=None):
    data = {t: np.loadtxt(os.path.join(OUT, f"diag_{t}.csv"), delimiter=",", skiprows=1) for t in tags}
    # Fig 1: log mu_star vs log z, N0, N2, N4 (pooled over sources, every 4th block)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ax, n in zip(axs, [0, 2, 4]):
        xs, ys = [], []
        for t in tags:
            a = data[t][::4]
            m, z = a[:, 6 + n], a[:, 3]
            k = (m > 0) & (z > 0)
            xs.append(np.log10(z[k]))
            ys.append(np.log10(m[k]))
        ax.scatter(np.concatenate(xs), np.concatenate(ys), s=3, alpha=0.3)
        ax.set_title(f"{NORMS[n]} ({NORM_DESC[NORMS[n]]})")
        ax.set_xlabel("log10 z_i (P_i / P_bg)")
        ax.set_ylabel("log10 mu_star")
        ax.grid(True, ls=":")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_logmustar_vs_logz.png"), dpi=130)
    plt.close(fig)

    # Fig 2: cos2 vs gamma_mean for all five directions (pooled, every 4th block)
    fig, axs = plt.subplots(1, 5, figsize=(18, 3.6), sharey=True)
    for n, ax in enumerate(axs):
        xs, ys = [], []
        for t in tags:
            a = data[t][::4]
            xs.append(a[:, 4])
            ys.append(a[:, 11 + n])
        ax.scatter(np.concatenate(xs), np.concatenate(ys), s=2, alpha=0.25)
        ax.set_title(NORMS[n])
        ax.set_xlabel("gamma_mean")
        ax.grid(True, ls=":")
    axs[0].set_ylabel("cos2")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_cos2_vs_gamma.png"), dpi=130)
    plt.close(fig)

    # Fig 3: MSD of oracle trajectories, alpha=1.45 file
    fig, ax = plt.subplots(figsize=(7, 4))
    for nm in NORMS:
        a = np.loadtxt(os.path.join(OUT, f"oracle_msd_{nm}_a145.csv"), delimiter=",", skiprows=1)
        ax.plot(a[:, 0], a[:, 1], lw=0.8, label=f"{nm} oracle")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("MSD [dB]")
    ax.set_title("Oracle-step trajectories, alpha=1.45")
    ax.grid(True, ls=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_oracle_msd_a145.png"), dpi=130)
    plt.close(fig)


def write_report(results, metrics, runtime):
    tags = [r["tag"] for r in results]
    by_tag = {r["tag"]: r for r in results}
    L = []
    L.append("# Oracle experiment: normalization of the delayless FD-FxNLMS direction (atsip/oracle)\n")
    L.append("Generated by `atsip/oracle/oracle.py`. Numbers only; see the spec in the commit message for setup.\n")
    L.append(f"Runtime (wall clock, 2 worker processes): {runtime:.1f} s.\n")

    L.append("## 1. ANR floor of w_opt and sanity check\n")
    L.append("| source | w_opt floor whole [dB] | w_opt floor last10 [dB] | N0 mu=0.36 whole [dB] | N0 mu=0.36 last10 [dB] | N0 final MSD [dB] |")
    L.append("|---|---|---|---|---|---|")
    for t in tags:
        r = by_tag[t]
        L.append(f"| {t} | {fmt(r['opt_anr'][0], 2)} | {fmt(r['opt_anr'][1], 2)} | "
                 f"{fmt(r['diag_N0_mu036'][0], 2)} | {fmt(r['diag_N0_mu036'][1], 2)} | "
                 f"{fmt(r['diag_N0_mu036_msd_end'], 2)} |")
    L.append("")
    L.append("Repro reference (repro/RESULTS.md, alpha 1.45, whole-curve): FxNLMS -24.46, "
             "FreqFxLMS -25.67, Prop -27.03, paper FxNLMS -24.32. No repro row is N0 with delayless "
             "FD normalization, so the comparison is only indicative.\n")

    L.append("## 2. Direction diagnostics (Experiment A, driver = N0 at mu=0.36)\n")
    L.append("Averages over the six sources. Impulsive = z_i > 10, non-impulsive = z_i < 2. "
             "CV = std(log mu*)/|mean(log mu*)| over blocks after 2 s with mu*>0; std_log = std of the same values.\n")
    L.append("| norm | description | mean cos2 | cos2 impulsive | cos2 non-imp. | CV log mu* | std log mu* | median mu* impulsive | median mu* non-imp. |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for nm in NORMS:
        rows = [metrics[t][nm] for t in tags]
        L.append(f"| {nm} | {NORM_DESC[nm]} | {fmt(_avg(rows,'cos2'))} | {fmt(_avg(rows,'cos2_imp'))} | "
                 f"{fmt(_avg(rows,'cos2_non'))} | {fmt(_avg(rows,'cv_log'))} | {fmt(_avg(rows,'std_log'))} | "
                 f"{fmt(_avg(rows,'med_imp'),4)} | {fmt(_avg(rows,'med_non'),4)} |")
    L.append("")
    meta = [metrics[t]["_meta"] for t in tags]
    L.append(f"Mean over sources: blocks per run {meta[0]['n_blocks']}, impulsive blocks "
             f"{np.mean([m['n_imp'] for m in meta]):.0f}, non-impulsive blocks {np.mean([m['n_non'] for m in meta]):.0f}, "
             f"mean gamma_mean {np.mean([m['gamma_mean'] for m in meta]):.4f}.\n")

    L.append("### Appendix A: per-source diagnostics\n")
    for t in tags:
        L.append(f"**{t}** (impulsive blocks {metrics[t]['_meta']['n_imp']}, gamma_mean {metrics[t]['_meta']['gamma_mean']:.4f})\n")
        L.append("| norm | cos2 | cos2 imp. | cos2 non-imp. | CV log mu* | median mu* imp. | median mu* non-imp. |")
        L.append("|---|---|---|---|---|---|---|")
        for nm in NORMS:
            m = metrics[t][nm]
            L.append(f"| {nm} | {fmt(m['cos2'])} | {fmt(m['cos2_imp'])} | {fmt(m['cos2_non'])} | "
                     f"{fmt(m['cv_log'])} | {fmt(m['med_imp'],4)} | {fmt(m['med_non'],4)} |")
        L.append("")

    L.append("## 3. Oracle-step and best fixed-step ANR (Experiment B)\n")
    L.append("Oracle: w <- w - clip(mu_star, 0, cap) g, cap = 1.0 for the first 20 blocks, then 4 x median of positive mu_star so far. "
             "Fixed: best mu from {0.05, 0.1, 0.2, 0.36, 0.6, 1.0} chosen by the mean whole-curve ANR over the six sources "
             "(mu_b = 2 mu B/M). Values are means over the six sources.\n")
    L.append("| norm | oracle whole [dB] | oracle last10 [dB] | best mu | fixed whole [dB] | fixed last10 [dB] | oracle final MSD [dB] | fixed final MSD [dB] |")
    L.append("|---|---|---|---|---|---|---|---|")
    best = {}
    for nm in NORMS:
        scores = {}
        for mu in MUS:
            key = str(mu)
            w = [by_tag[t]["fixed"][nm][key]["whole"] for t in tags]
            l10 = [by_tag[t]["fixed"][nm][key]["last10"] for t in tags]
            if np.all(np.isfinite(w)):
                scores[mu] = (float(np.mean(w)), float(np.mean(l10)))
        bmu = min(scores, key=lambda m: scores[m][0]) if scores else None
        bw = scores[bmu] if bmu is not None else (float("nan"), float("nan"))
        best[nm] = (bmu, bw)
        o_w = np.mean([by_tag[t]["oracle"][nm]["whole"] for t in tags])
        o_l = np.mean([by_tag[t]["oracle"][nm]["last10"] for t in tags])
        o_m = np.mean([by_tag[t]["oracle"][nm]["msd_end"] for t in tags])
        f_m = np.mean([by_tag[t]["fixed"][nm][str(bmu)]["msd_end"] for t in tags]) if bmu is not None else float("nan")
        L.append(f"| {nm} | {fmt(o_w,2)} | {fmt(o_l,2)} | {bmu} | {fmt(bw[0],2)} | {fmt(bw[1],2)} | "
                 f"{fmt(o_m,2)} | {fmt(f_m,2)} |")
    L.append("")
    L.append("Fixed-step mean whole-curve ANR [dB] by mu (mean over six sources; nan = diverged in some source):\n")
    L.append("| norm | " + " | ".join(f"mu={m}" for m in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for nm in NORMS:
        cells = []
        for mu in MUS:
            w = [by_tag[t]["fixed"][nm][str(mu)]["whole"] for t in tags]
            cells.append(fmt(float(np.mean(w)), 2) if np.all(np.isfinite(w)) else "nan")
        L.append(f"| {nm} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("### Appendix B: per-source oracle and best fixed step (whole / last10 [dB])\n")
    for t in tags:
        L.append(f"**{t}**\n")
        L.append("| norm | oracle whole | oracle last10 | best mu (this source) | fixed whole | fixed last10 |")
        L.append("|---|---|---|---|---|---|")
        for nm in NORMS:
            o = by_tag[t]["oracle"][nm]
            sc = {mu: by_tag[t]["fixed"][nm][str(mu)]["whole"] for mu in MUS}
            ok = {m: v for m, v in sc.items() if np.isfinite(v)}
            bmu = min(ok, key=ok.get) if ok else None
            fw = by_tag[t]["fixed"][nm][str(bmu)] if bmu is not None else {"whole": float("nan"), "last10": float("nan")}
            L.append(f"| {nm} | {fmt(o['whole'],2)} | {fmt(o['last10'],2)} | {bmu} | "
                     f"{fmt(fw['whole'],2)} | {fmt(fw['last10'],2)} |")
        L.append("")

    L.append("## 4. Figures (atsip/oracle/out/)\n")
    L.append("- `fig_logmustar_vs_logz.png`: log10 mu_star_n vs log10 z_i for N0, N2, N4 (pooled over the six sources, every 4th block, mu_star>0 only).")
    L.append("- `fig_cos2_vs_gamma.png`: cos2 vs gamma_mean for N0 to N4 (pooled, every 4th block).")
    L.append("- `fig_oracle_msd_a145.png`: MSD of the oracle-step trajectories, alpha=1.45 file.")
    L.append("")
    L.append("## 5. Notes on choices\n")
    L.append("- N0, N1, N2 and the shape EMA of N4 include the current frame before use. The N4 background P_bg is used before its update, as specified.")
    L.append("- eps = 1e-6 x cumulative mean of P_i (up to the current frame), used for all normalizations.")
    L.append("- The N3 instantaneous Lam uses the same eps (not eps times a frame mean).")
    L.append("- The CMS seeds (0, 1, 2) are generated in memory with rng = numpy default_rng(seed) and are not saved.")
    L.append("- If a fixed or oracle run diverges (non-finite w), its ANR is reported as nan and it is excluded from best-mu selection.")
    with open(os.path.join(HERE, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    with Pool(processes=2) as pool:
        results = pool.map(run_source, SOURCES)
    tags = [s[0] for s in SOURCES]
    metrics = {t: diag_metrics(t) for t in tags}
    make_figures(tags)
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"results": results, "diag_metrics": metrics}, f, indent=1, default=float)
    write_report(results, metrics, time.time() - t0)
    print(f"[total] {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
