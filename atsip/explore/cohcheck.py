#!/usr/bin/env python3
"""Coherence check: why the estimated coherence between the filtered reference R and the error E
stays high after convergence (atsip/explore).

Part 1: ideal two-sided LS Wiener filter with taps at lags -1024..4095 (w_long) vs the 512-tap
        causal optimum (w_opt), first 10 s.
Part 2: time-averaged coherence mean_k gamma(k) (blocks after 2 s) of the error signals
        (a) e_opt, (b) e_long, (c) Gaussian noise with the PSD of e_opt, (d) e_opt circularly shifted,
        under framings B (last 128 errors) and M (last 512 errors), at EMA alpha 0.05 and 0.01.
        Gamma uses the same block framing and the same EMA recursion as explore.Shared (incl. current frame).

Usage:  python3 atsip/explore/cohcheck.py
Writes  atsip/explore/RESULTS_cohcheck.md and atsip/explore/out/cohcheck_*.{png,json}.
"""
import os
import sys
import json
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy import signal, linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explore as X  # noqa: E402  (Shared, metrics, constants)
import genie as G    # noqa: E402  (data loaders)

O = X.O
FS, M, N, B, NB = X.FS, X.M, X.N, X.B, X.NB
EPS_REL = X.EPS_REL
OUT = os.path.join(HERE, "out")
T_FIT = 10 * FS
LAG_MIN, LAG_MAX = -1024, 4095
LM = LAG_MAX - LAG_MIN + 1                    # 5120 taps
LAGS = np.arange(LAG_MIN, LAG_MAX + 1)
ALPHAS = (0.05, 0.01)
FRAMES = {"B": B, "M": M}
FRAME_LABEL = {"B": "last 128 errors", "M": "last 512 errors", "N": "last 1024 errors (extra diagnostic)"}
SRC = [("a145", "file", "sounds/impulsive_noise_alpha1.45.dat"), ("cms145_s0", "cms", 0)]
SIG = ["a", "b", "c", "d"]
BANDS = [(0, 1000), (1000, 4000), (4000, 8000)]
PSD_BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
I_START = int(round(2.0 * FS / B))           # first block at or after 2 s
FK = np.arange(NB) * FS / N                  # bin frequencies [Hz]
SHIFT_MIN = 3 * FS                           # random circular shift lower bound (3 s)
DESC = {"a": "e_opt (512-tap optimum)", "b": "e_long (two-sided long optimum)",
        "c": "v_ind (Gaussian, PSD of e_opt)", "d": "v_ind_imp (e_opt circularly shifted)"}


# ----------------------------------------------------------------------------
# Part 1: two-sided long Wiener filter
# ----------------------------------------------------------------------------
def long_wiener(r, d, T=T_FIT):
    """min ||d + s*(w*x)||, tap at lag j uses regressor r[n-j], j = LAG_MIN..LAG_MAX.
    Same Toeplitz autocorrelation convention and ridge as O.wiener_opt (diagonal += 1e-8 * phi[0])."""
    r, d = r[:T], d[:T]
    T = len(r)
    a = np.array([np.dot(r[t:], r[:T - t]) for t in range(LM)])
    a[0] += 1e-8 * a[0]
    c = np.array([np.dot(r[:T - j], d[j:]) if j >= 0 else np.dot(r[-j:], d[:T + j]) for j in LAGS])
    return linalg.solve_toeplitz(a, -c)


def apply_long(x, w_long):
    """y[n] = sum_j w[j] x[n-j] for j in LAG_MIN..LAG_MAX (non-causal taps use future samples of x)."""
    L = len(x)
    z = signal.fftconvolve(x, w_long)            # z[m] = sum_k h[k] x[m-k], k = j - LAG_MIN
    return z[-LAG_MIN:-LAG_MIN + L]              # y[n] = z[n - LAG_MIN]


def lag_energy(w):
    tot = np.sum(w ** 2)
    return dict(neg=float(np.sum(w[LAGS < 0] ** 2) / tot),
                mid=float(np.sum(w[(LAGS >= 0) & (LAGS <= 511)] ** 2) / tot),
                far=float(np.sum(w[LAGS >= 512] ** 2) / tot))


# ----------------------------------------------------------------------------
# Error signals (open loop, w fixed)
# ----------------------------------------------------------------------------
def build_signals(data, w_long, seed):
    x, d, hS, w_opt = data["x"], data["d"], data["hS"], data["w_opt"]
    L = len(x)
    e_opt = d + signal.lfilter(hS, [1.0], np.convolve(x, w_opt)[:L])
    e_long = d + signal.lfilter(hS, [1.0], apply_long(x, w_long))
    # (c) Gaussian noise with the Welch PSD of e_opt (random phase / Gaussian amplitude in the full-length FFT)
    f, Pxx = signal.welch(e_opt, fs=FS, nperseg=N)
    Pf = np.interp(np.fft.rfftfreq(L, 1.0 / FS), f, Pxx)
    rng = np.random.default_rng(seed)
    v = np.fft.irfft(np.fft.rfft(rng.standard_normal(L)) * np.sqrt(Pf), L)
    v *= np.sqrt(np.sum(e_opt ** 2) / np.sum(v ** 2))
    # (d) e_opt circularly shifted by a random offset in [3 s, L - 3 s]
    off = int(rng.integers(SHIFT_MIN, L - SHIFT_MIN))
    v_imp = np.roll(e_opt, off)
    return dict(a=e_opt, b=e_long, c=v, d=v_imp), off


# ----------------------------------------------------------------------------
# Coherence gamma with the explore.Shared recursion (alpha generic)
# ----------------------------------------------------------------------------
def frame_E(e, s0, Lf):
    """Last Lf errors ending at the current block end, zero-padded at the front to N (as mixlam.frame_E)."""
    seg = e[max(0, s0 + B - Lf): s0 + B]
    ev = np.zeros(Lf)
    ev[Lf - len(seg):] = seg
    return np.fft.rfft(np.concatenate([np.zeros(N - Lf), ev]))


def frame_matrix(e, Lf):
    nb = len(e) // B
    return np.stack([frame_E(e, i * B, Lf) for i in range(nb)])


def gamma_tracks(Rall, E, alpha):
    A = np.abs(Rall) ** 2
    cRE = Rall * np.conj(E)
    cEE = np.abs(E) ** 2
    nb = Rall.shape[0]
    gam = np.empty((nb, NB))
    Srr, Sre, See = A[0].copy(), cRE[0].copy(), cEE[0].copy()
    psum = 0.0
    for i in range(nb):
        if i > 0:
            Srr = (1 - alpha) * Srr + alpha * A[i]
            Sre = (1 - alpha) * Sre + alpha * cRE[i]
            See = (1 - alpha) * See + alpha * cEE[i]
        psum += A[i].mean()
        eps = EPS_REL * psum / (i + 1)
        gam[i] = np.clip(np.abs(Sre) ** 2 / (Srr * See + eps), 0.0, 1.0)
    return gam


def gamma_summary(gam):
    per_block = gam[I_START:].mean(axis=1)
    per_k = gam[I_START:].mean(axis=0)
    bands = [float(per_k[(FK >= lo) & (FK < hi)].mean()) for lo, hi in BANDS]
    return dict(mean=float(per_block.mean()), per_k=per_k, bands=bands)


# ----------------------------------------------------------------------------
# Per-source worker
# ----------------------------------------------------------------------------
def run_source(tag, kind, arg, seed):
    t0 = time.time()
    data = G.load_data(kind, arg)
    x, d, xq, Sf, S2, w_opt = (data[k] for k in ("x", "d", "xq", "Sf", "S2", "w_opt"))
    L = len(x)
    nb = L // B
    out = dict(tag=tag)

    # ---- Part 1
    r = data["r"]
    w_long = long_wiener(r, d)
    # self-check of the lag convention: a single tap at lag 3 must give x[n-3]
    tst = np.zeros(LM)
    tst[3 - LAG_MIN] = 1.0
    chk = apply_long(x, tst)
    assert np.allclose(chk[3:], x[:-3]) and np.allclose(chk[:3], 0.0)
    ee_opt = d + signal.lfilter(data["hS"], [1.0], np.convolve(x, w_opt)[:L])
    sig, off = build_signals(data, w_long, seed)
    e_long = sig["b"]
    t_fit = T_FIT
    p1 = dict(
        energy_long=lag_energy(w_long),
        energy_opt=dict(neg=0.0, mid=1.0, far=0.0),
        anr_long=O.summarize(O.anr_curve(e_long, d)),
        anr_long_oos=O.summarize(O.anr_curve(e_long[t_fit:], d[t_fit:])),
        anr_opt=O.summarize(O.anr_curve(ee_opt, d)),
        zeros_out=int(np.sum(np.abs(np.roots(np.trim_zeros(data["hS"], "b"))) > 1.0)),
        zeros_max=float(np.max(np.abs(np.roots(np.trim_zeros(data["hS"], "b"))))),
        pow_opt=float(np.sum(ee_opt ** 2) / np.sum(d ** 2)),
        pow_long=float(np.sum(e_long ** 2) / np.sum(d ** 2)),
        w_long_causal_rel_diff=float(np.max(np.abs(w_long[-LAG_MIN:-LAG_MIN + 512] - w_opt))
                                     / np.max(np.abs(w_opt))),
    )
    out["part1"] = p1

    # ---- Part 2: R frames and E frames
    Rall = np.stack([Sf * np.fft.rfft(xq[i * B + B: i * B + B + N]) for i in range(nb)])
    out["shift_samples"] = off
    gam = {}
    frames = dict(FRAMES, N=N)
    for fn, Lf in frames.items():
        for s in SIG:
            E = frame_matrix(sig[s], Lf)
            for al in ALPHAS:
                gam[(fn, s, al)] = gamma_tracks(Rall, E, al)
    # sanity: gamma from explore.Shared for the first 300 blocks of e_opt, both framings
    chk_diff = {}
    for fn, Lf in FRAMES.items():
        E = frame_matrix(sig["a"], Lf)
        sh = X.Shared(S2)
        ref = []
        for i in range(300):
            sh.step(Rall[i], E[i])
            ref.append(sh.gam_prev.copy())
        chk_diff[fn] = float(np.max(np.abs(np.array(ref) - gam[(fn, "a", 0.05)][:300])))
    out["shared_check_maxdiff"] = chk_diff

    res = {}
    for fn in frames:
        for s in SIG:
            for al in ALPHAS:
                res[(fn, s, al)] = gamma_summary(gam[(fn, s, al)])
    out["gamma"] = res

    # ---- PSD of e_opt vs e_long (full signal, Welch nperseg = N)
    f, Pa = signal.welch(ee_opt, fs=FS, nperseg=N)
    _, Pb = signal.welch(e_long, fs=FS, nperseg=N)
    psd = {}
    for lo, hi in PSD_BANDS:
        sel = (f >= lo) & (f < hi)
        psd[f"{lo}-{hi}"] = dict(opt_db=float(10 * np.log10(Pa[sel].mean())),
                                 long_db=float(10 * np.log10(Pb[sel].mean())))
    out["psd"] = psd
    out["psd_curves"] = dict(f=f, opt=Pa, long=Pb)
    out["runtime_s"] = time.time() - t0
    print(f"[done] {tag} runtime={out['runtime_s']:.1f}s shift={off}", flush=True)
    return out


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def make_figures(results):
    for fn in FRAMES:
        fig, axs = plt.subplots(1, len(results), figsize=(12, 4.2), sharey=True)
        # figure per framing: B or M (N is in the table only)
        for ax, r in zip(np.atleast_1d(axs), results):
            for s, col in zip(("a", "b", "c"), ("C0", "C1", "C2")):
                for al, ls in zip(ALPHAS, ("-", "--")):
                    ax.plot(FK / 1000, r["gamma"][(fn, s, al)]["per_k"], color=col, ls=ls, lw=0.9,
                            label=f"({s}) alpha={al}")
            ax.set_title(f"{r['tag']}, framing {fn}")
            ax.set_xlabel("frequency [kHz]")
            ax.grid(True, ls=":")
        np.atleast_1d(axs)[0].set_ylabel("time-averaged gamma(k)")
        np.atleast_1d(axs)[0].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"cohcheck_gamma_E_{fn}.png"), dpi=130)
        plt.close(fig)

    fig, axs = plt.subplots(1, len(results), figsize=(12, 4), sharey=True)
    for ax, r in zip(np.atleast_1d(axs), results):
        ax.semilogy(r["psd_curves"]["f"] / 1000, r["psd_curves"]["opt"], lw=0.8, label="e_opt")
        ax.semilogy(r["psd_curves"]["f"] / 1000, r["psd_curves"]["long"], lw=0.8, label="e_long")
        ax.set_title(r["tag"])
        ax.set_xlabel("frequency [kHz]")
        ax.grid(True, ls=":")
    np.atleast_1d(axs)[0].set_ylabel("PSD (Welch)")
    np.atleast_1d(axs)[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cohcheck_psd.png"), dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def frames_all():
    return list(FRAMES) + ["N"]


def fmt(v, p=3):
    return "nan" if v is None or not np.isfinite(v) else f"{v:.{p}f}"


def write_report(results, runtime):
    L = []
    L.append("# Coherence check: gamma(R, E) at convergence (atsip/explore/cohcheck.py)\n")
    L.append("Numbers only. Generated by `atsip/explore/cohcheck.py`.\n")
    L.append(f"Runtime (wall clock, sequential): {runtime:.1f} s. Per source [s]: "
             + ", ".join(f"{r['tag']} {r['runtime_s']:.1f}" for r in results) + ".\n")
    L.append("Setup: fs 16000, M=512, N=1024, B=128; hP padded to 512, hS trimmed to 128, Shat = S. "
             "gamma: explore.Shared recursion (EMA alpha incl. current frame, eps = 1e-6 x cumulative mean power), "
             "clipped to [0,1]. Time average over blocks starting at or after 2 s. "
             "Band gamma = mean of time-averaged gamma(k) over bins in the band.\n")
    L.append("## Setup checks\n")
    for r in results:
        L.append(f"- {r['tag']}: gamma(alpha 0.05) vs explore.Shared, first 300 blocks, max |diff| "
                 f"framing B {r['shared_check_maxdiff']['B']:.2e}, framing M {r['shared_check_maxdiff']['M']:.2e}. "
                 f"v_ind_imp shift = {r['shift_samples']} samples ({r['shift_samples'] / FS:.2f} s).")
    for r in results:
        p = r["part1"]
        L.append(f"- {r['tag']}: secondary path hS has {p['zeros_out']} zero(s) outside the unit circle "
                 f"(largest zero magnitude {p['zeros_max']:.3f}); out-of-sample ANR of w_long on 10-30 s: "
                 f"whole {fmt(p['anr_long_oos'][0],2)} dB, last10 {fmt(p['anr_long_oos'][1],2)} dB.")
    L.append("- Long filter: ridge diagonal += 1e-8 x phi[0] (same as O.wiener_opt). Lag convention verified with a unit tap at lag 3.\n")

    L.append("## Part 1. Ideal filter outside the modelable range (first 10 s fit)\n")
    L.append("Energy fraction of w_long by lag region; ANR floor [dB] of w_long vs w_opt (whole curve / last 10 %); "
             "residual power ratio ||e||^2/||d||^2 over the full signal.\n")
    L.append("| source | E frac lag<0 | E frac 0..511 | E frac 512..4095 | ANR whole w_long | ANR whole w_opt | "
             "ANR last10 w_long | ANR last10 w_opt | ||e_long||^2/||d||^2 | ||e_opt||^2/||d||^2 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        p = r["part1"]
        e = p["energy_long"]
        L.append(f"| {r['tag']} | {fmt(e['neg'],4)} | {fmt(e['mid'],4)} | {fmt(e['far'],4)} | "
                 f"{fmt(p['anr_long'][0],2)} | {fmt(p['anr_opt'][0],2)} | {fmt(p['anr_long'][1],2)} | "
                 f"{fmt(p['anr_opt'][1],2)} | {p['pow_long']:.3e} | {p['pow_opt']:.3e} |")
    L.append("")
    L.append("Relative max difference of w_long on lags 0..511 vs w_opt: "
             + ", ".join(f"{r['tag']} {r['part1']['w_long_causal_rel_diff']:.3f}" for r in results) + ".\n")

    L.append("## Part 2. Time-averaged mean_k gamma(k) (blocks after 2 s)\n")
    L.append("Columns: alpha 0.05 and 0.01 (mean over k); band means at alpha 0.05 over 0-1 kHz, 1-4 kHz, 4-8 kHz.\n")
    for fn in frames_all():
        L.append(f"### Framing {fn} ({FRAME_LABEL[fn]})\n")
        L.append("| source | signal | mean_k gamma, alpha 0.05 | mean_k gamma, alpha 0.01 | band 0-1k | band 1-4k | band 4-8k |")
        L.append("|---|---|---|---|---|---|---|")
        for r in results:
            for s in SIG:
                g5 = r["gamma"][(fn, s, 0.05)]
                g1 = r["gamma"][(fn, s, 0.01)]
                L.append(f"| {r['tag']} | {s}: {DESC[s]} | {fmt(g5['mean'])} | {fmt(g1['mean'])} | "
                         + " | ".join(fmt(b) for b in g5["bands"]) + " |")
        L.append("")

    L.append("## Part 2. Error power spectrum (Welch, nperseg = N), band means in dB\n")
    L.append("| source | band [Hz] | e_opt [dB] | e_long [dB] | e_long - e_opt [dB] |")
    L.append("|---|---|---|---|---|")
    for r in results:
        for bnd, v in r["psd"].items():
            L.append(f"| {r['tag']} | {bnd} | {fmt(v['opt_db'],2)} | {fmt(v['long_db'],2)} | "
                     f"{fmt(v['long_db'] - v['opt_db'],2)} |")
    L.append("")

    L.append("## Figures (atsip/explore/out/)\n")
    L.append("- `cohcheck_gamma_E_B.png`, `cohcheck_gamma_E_M.png`: time-averaged gamma(k) for (a), (b), (c), "
             "alpha 0.05 solid and 0.01 dashed, one panel per source.")
    L.append("- `cohcheck_psd.png`: Welch PSD of e_opt and e_long.")
    L.append("- `cohcheck_summary.json`: all numbers above.\n")
    with open(os.path.join(HERE, "RESULTS_cohcheck.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    results = [run_source(tag, kind, arg, seed) for seed, (tag, kind, arg) in enumerate(SRC)]
    runtime = time.time() - t0

    make_figures(results)
    summ = {r["tag"]: dict(part1=r["part1"], shift_samples=r["shift_samples"],
                           shared_check_maxdiff=r["shared_check_maxdiff"], psd=r["psd"],
                           gamma={f"{k[0]}_{k[1]}_{k[2]}": dict(mean=v["mean"], bands=v["bands"])
                                  for k, v in r["gamma"].items()},
                           runtime_s=r["runtime_s"]) for r in results}
    summ["runtime_s"] = runtime
    with open(os.path.join(OUT, "cohcheck_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1, default=float)
    write_report(results, runtime)
    print(f"[total] {runtime:.1f} s", flush=True)


if __name__ == "__main__":
    main()
