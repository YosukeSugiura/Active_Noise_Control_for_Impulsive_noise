#!/usr/bin/env python3
"""Exploration of step normalizations for the delayless FD-FxNLMS under impulsive reference.

Reuses the data, block simulation, ANR and N0..N6 normalizers from atsip/oracle/oracle.py.
Candidates: B0, B2, B4, PAPER (port of Freq_FxNLMS_CohVSS), C1..C8.
Oracles: O_scalar (oracle.simulate, norm N0), O_band (16-band NNLS oracle on the B0 direction).

Usage:  python3 atsip/explore/explore.py
Writes  atsip/explore/out/*.csv, *.json, *.npz, *.png and atsip/explore/RESULTS.md.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
import json
import time
import warnings
from multiprocessing import Pool

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal, optimize, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "oracle"))
import oracle as O  # noqa: E402

FS, M, N, B = O.FS, O.M, O.N, O.B
ALPHA = O.ALPHA
NB = N // 2 + 1
EPS_REL = 1e-6
MUS = [0.1, 0.2, 0.36, 0.6, 1.0]
SOURCES = O.SOURCES
OUT = os.path.join(HERE, "out")
N5 = int(round(5 * FS / 64))           # ANR samples in the first 5 s (ANR sampled every 64)
CANDS = ["B0", "B2", "B4", "PAPER", "PAPER_oE", "C1", "C1s", "C2", "C2b", "C3", "C4", "C5", "C6", "C7", "C8"]
CAND_DESC = {
    "B0": "Lam = Srr+eps",
    "B2": "Lam = 0.55 Pbar + 0.45 Srr + eps",
    "B4": "N4 (oracle.py)",
    "PAPER": "Freq_FxNLMS_CohVSS port (paper params), E = last M errors as in class",
    "PAPER_oE": "PAPER port with E = last B errors as in oracle.py",
    "C1": "B0 * gam_prev",
    "C1s": "B0 * sqrt(gam_prev)",
    "C2": "Lam = abs(R)^2 + Srr(1-g)/g + eps, g=clip(gam_prev)",
    "C2b": "C2 with bias-corrected g",
    "C3": "Lam = max(Srr, abs(R)^2) + eps",
    "C4": "Lam = Srr + 1.22 P_i + eps",
    "C5": "Lam = (1-lam) Pbar + lam Srr + eps, lam=clip(sqrt(gam_prev))",
    "C6": "B0 * min(1, 2/z_i)",
    "C7": "B2 * min(1, 2/z_i)",
    "C8": "Lam = abs(R)^2 + (0.55 Pbar+0.45 Srr)(1-g)/g + eps",
}
PAPER_P = dict(coh_pivot=0.35, mu_scale_min=0.6, mu_scale_max=1.3, boost_power=0.6,
               suppress_power=1.6, coh_gain=1.5, coh_weight_mix=0.60, coh_weight_power=1.6,
               coh_weight_min=0.55, grad_norm_mix=0.45, rg_alpha=0.05, eps=1e-6)


# ----------------------------------------------------------------------------
# Shared per-block quantities
# ----------------------------------------------------------------------------
class Shared:
    """Srr/See/Sre EMAs (first-frame init as in oracle.py), N4 background via O.Normalizers."""

    def __init__(self, S2):
        self.norms = O.Normalizers(S2)
        self.Srr = self.Sre = self.See = None
        self.gam_prev = np.full(NB, 0.5)

    def step(self, R, E):
        lams, P, z = self.norms.update(R)
        A = np.abs(R) ** 2
        cRE = R * np.conj(E)
        cEE = np.abs(E) ** 2
        if self.Srr is None:
            self.Srr, self.Sre, self.See = A.copy(), cRE.copy(), cEE.copy()
        else:
            self.Srr = (1 - ALPHA) * self.Srr + ALPHA * A
            self.Sre = (1 - ALPHA) * self.Sre + ALPHA * cRE
            self.See = (1 - ALPHA) * self.See + ALPHA * cEE
        eps = EPS_REL * self.norms.psum / self.norms.count
        gam_now = np.clip(np.abs(self.Sre) ** 2 / (self.Srr * self.See + eps), 0.0, 1.0)
        out = dict(lams=lams, P=P, z=z, A=A, eps=eps, Srr=self.Srr,
                   Pbar=float(np.mean(self.Srr)), gam_prev=self.gam_prev)
        self.gam_prev = gam_now
        return out


def cand_G(cand, R, E, s):
    """Frequency-domain direction G(k) for the fixed-step candidates (before q-scalars are applied)."""
    Ce = np.conj(R) * E
    A, Srr, P, z, eps = s["A"], s["Srr"], s["P"], s["z"], s["eps"]
    gp, Pbar, lams = s["gam_prev"], s["Pbar"], s["lams"]
    if cand == "B0":
        return Ce / (Srr + eps)
    if cand == "B2":
        return Ce / (0.55 * Pbar + 0.45 * Srr + eps)
    if cand == "B4":
        return Ce / lams[4]
    if cand == "C1":
        return Ce / (Srr + eps) * gp
    if cand == "C1s":
        return Ce / (Srr + eps) * np.sqrt(gp)
    if cand == "C2":
        g = np.clip(gp, 0.02, 0.98)
        return Ce / (A + Srr * (1 - g) / g + eps)
    if cand == "C2b":
        g = np.clip((gp - 0.0256) / (1 - 0.0256), 0.02, 0.98)
        return Ce / (A + Srr * (1 - g) / g + eps)
    if cand == "C3":
        return Ce / (np.maximum(Srr, A) + eps)
    if cand == "C4":
        return Ce / (Srr + 1.22 * P + eps)
    if cand == "C5":
        lam = np.clip(np.sqrt(gp), 0.1, 1.0)
        return Ce / ((1 - lam) * Pbar + lam * Srr + eps)
    if cand == "C6":
        return Ce / (Srr + eps) * min(1.0, 2.0 / z)
    if cand == "C7":
        return Ce / (0.55 * Pbar + 0.45 * Srr + eps) * min(1.0, 2.0 / z)
    if cand == "C8":
        g = np.clip(gp, 0.02, 0.98)
        return Ce / (A + (0.55 * Pbar + 0.45 * Srr) * (1 - g) / g + eps)
    raise ValueError(cand)


# ----------------------------------------------------------------------------
# Port of Freq_FxNLMS_CohVSS.update (apsipa2026/algorithms_20260815.py), gradient part
# ----------------------------------------------------------------------------
def paper_init():
    return dict(Sre=np.zeros(NB, dtype=np.complex128), Srr=np.zeros(NB), See=np.zeros(NB))


def paper_gradient(st, R, Ef, p=PAPER_P):
    a, eps = p["rg_alpha"], p["eps"]
    st["Sre"] = (1 - a) * st["Sre"] + a * (R * np.conj(Ef))
    st["Srr"] = (1 - a) * st["Srr"] + a * np.abs(R) ** 2
    st["See"] = (1 - a) * st["See"] + a * np.abs(Ef) ** 2
    coh = np.clip(np.abs(st["Sre"]) ** 2 / (st["Srr"] * st["See"] + eps), 0.0, 1.0)
    pivot = np.clip(p["coh_pivot"], eps, 1.0 - eps)
    low_rel = np.clip((pivot - coh) / pivot, 0.0, 1.0)
    high_rel = np.clip((coh - pivot) / (1.0 - pivot), 0.0, 1.0)
    mu_low = 1.0 + (p["mu_scale_max"] - 1.0) * np.power(low_rel, p["boost_power"])
    mu_high = 1.0 - (1.0 - p["mu_scale_min"]) * (
        1.0 - np.exp(-p["coh_gain"] * np.power(high_rel, p["suppress_power"])))
    msv = np.clip(np.where(coh <= pivot, mu_low, mu_high), p["mu_scale_min"], p["mu_scale_max"])
    srr_scalar = float(np.mean(st["Srr"]))
    gm = p["grad_norm_mix"]
    grad_norm = (1.0 - gm) * (srr_scalar + eps) + gm * (st["Srr"] + eps)
    excess = np.clip((coh - pivot) / (1.0 - pivot), 0.0, 1.0)
    cw = np.clip(1.0 - p["coh_weight_mix"] * np.power(excess, p["coh_weight_power"]),
                 p["coh_weight_min"], 1.0)
    return (np.conj(R) * Ef / grad_norm) * cw * msv


# ----------------------------------------------------------------------------
# Block simulations
# ----------------------------------------------------------------------------
def sim_fixed(data, cand, mu):
    """Fixed-step run: w <- w - 2 mu B/M * irfft(G)[:M].
    PAPER uses E = last M errors (as in the class); PAPER_oE uses oracle.py E (last B errors)."""
    x, d, xq, hS, Sf, S2 = (data[k] for k in ("x", "d", "xq", "hS", "Sf", "S2"))
    L = len(x)
    nb = L // B
    swv = sliding_window_view(xq, M)
    w = np.zeros(M)
    zi = np.zeros(len(hS) - 1)
    e_all = np.zeros(L)
    mu_b = 2 * mu * B / M
    is_paper = cand.startswith("PAPER")
    shared = None if is_paper else Shared(S2)
    pst = paper_init() if is_paper else None
    diverged = False
    for i in range(nb):
        s0 = i * B
        Xb = swv[s0 + N - M + 1: s0 + N - M + 1 + B][:, ::-1]
        y = Xb @ w
        ys, zi = signal.lfilter(hS, [1.0], y, zi=zi)
        e = d[s0:s0 + B] + ys
        e_all[s0:s0 + B] = e
        X = np.fft.rfft(xq[s0 + B: s0 + B + N])
        R = Sf * X
        E = np.fft.rfft(np.concatenate([np.zeros(N - B), e]))
        if is_paper:
            if cand == "PAPER":
                ev = np.zeros(M)
                seg = e_all[max(0, s0 + B - M): s0 + B]
                ev[M - len(seg):] = seg
                Ef = np.fft.rfft(np.concatenate([np.zeros(M), ev]), N)
            else:
                Ef = E
            GradF = paper_gradient(pst, R, Ef)
        else:
            s = shared.step(R, E)
            GradF = cand_G(cand, R, E, s)
        g = np.fft.irfft(GradF, N)[:M]
        w = w - mu_b * g
        if not np.isfinite(w).all():
            e_all[s0 + B:] = np.nan
            diverged = True
            break
    return e_all, w, diverged


def sim_oband(data, J=16, cap=4.0):
    """O_band on the B0 direction: NNLS on J band-restricted directions, mu_j capped at cap."""
    x, d, xq, hS, Sf, S2, w_opt = (data[k] for k in ("x", "d", "xq", "hS", "Sf", "S2", "w_opt"))
    L = len(x)
    nb = L // B
    swv = sliding_window_view(xq, M)
    w = np.zeros(M)
    zi = np.zeros(len(hS) - 1)
    e_all = np.zeros(L)
    edges = np.linspace(0, NB, J + 1).round().astype(int)
    mu_log = np.full((nb, J), np.nan)
    gam_log = np.full((nb, J), np.nan)
    mis_log = np.full((nb, J), np.nan)
    z_log = np.full(nb, np.nan)
    init_pw = np.abs(np.fft.rfft(-w_opt, N)) ** 2
    init_band = np.array([init_pw[edges[j]:edges[j + 1]].mean() for j in range(J)])
    shared = Shared(S2)
    diverged = False
    for i in range(nb):
        s0 = i * B
        Xb = swv[s0 + N - M + 1: s0 + N - M + 1 + B][:, ::-1]
        y = Xb @ w
        ys, zi = signal.lfilter(hS, [1.0], y, zi=zi)
        e = d[s0:s0 + B] + ys
        e_all[s0:s0 + B] = e
        X = np.fft.rfft(xq[s0 + B: s0 + B + N])
        R = Sf * X
        E = np.fft.rfft(np.concatenate([np.zeros(N - B), e]))
        s = shared.step(R, E)
        G = cand_G("B0", R, E, s)
        A_mat = np.empty((M, J))
        for j in range(J):
            a, b = edges[j], edges[j + 1]
            Gm = np.zeros_like(G)
            Gm[a:b] = G[a:b]
            A_mat[:, j] = np.fft.irfft(Gm, N)[:M]
        Delta = w - w_opt
        mu_j, _ = optimize.nnls(A_mat, Delta)
        mu_j = np.minimum(mu_j, cap)
        pw = np.abs(np.fft.rfft(Delta, N)) ** 2
        band_pw = np.array([pw[edges[j]:edges[j + 1]].mean() for j in range(J)])
        mis_log[i] = 10 * np.log10(band_pw / init_band + 1e-30)
        gp = s["gam_prev"]
        gam_log[i] = [gp[edges[j]:edges[j + 1]].mean() for j in range(J)]
        z_log[i] = s["z"]
        mu_log[i] = mu_j
        w = w - A_mat @ mu_j
        if not np.isfinite(w).all():
            e_all[s0 + B:] = np.nan
            diverged = True
            break
    return dict(e=e_all, w=w, diverged=diverged, mu=mu_log, gam=gam_log, z=z_log, mis=mis_log)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def metrics(e, w, d, w_opt):
    anr = O.anr_curve(e, d)
    fin = bool(np.isfinite(anr).all())
    whole, last10 = O.summarize(anr)
    first5 = float(anr[:N5].mean()) if fin else float("nan")
    msd = O.final_msd_db(w, w_opt)
    wfin = bool(np.isfinite(w).all())
    diverged = (not fin) or (not wfin) or (not np.isfinite(last10)) or last10 > -5.0
    return dict(whole=whole, first5=first5, last10=last10, msd=msd, diverged=bool(diverged))


def run_source(spec):
    tag, kind, arg = spec
    t0 = time.time()
    data = O.load_source(kind, arg)
    d, x, w_opt = data["d"], data["x"], data["w_opt"]
    res = {"tag": tag, "rows": [], "oscalar": None, "oband": None, "sanity": {}}
    anrs = {}
    keep = tag == "a145"

    for cand in CANDS:
        for mu in MUS:
            e, w, _ = sim_fixed(data, cand, mu)
            res["rows"].append(dict(cand=cand, mu=mu, **metrics(e, w, d, w_opt)))
            if keep:
                anrs[f"{cand}__{mu}"] = O.anr_curve(e, d)

    o = O.simulate(data, "oracle", norm=0)
    res["oscalar"] = metrics(o["e"], o["w"], d, w_opt)
    if keep:
        anrs["O_scalar"] = O.anr_curve(o["e"], d)

    ob = sim_oband(data)
    res["oband"] = metrics(ob["e"], ob["w"], d, w_opt)
    np.savez(os.path.join(OUT, f"oband_{tag}.npz"), mu=ob["mu"], gam=ob["gam"],
             z=ob["z"], mis=ob["mis"])
    if keep:
        anrs["O_band"] = O.anr_curve(ob["e"], d)
        np.savez(os.path.join(OUT, "anr_a145.npz"), **anrs)

    res["runtime_s"] = time.time() - t0
    print(f"[done] {tag} runtime={res['runtime_s']:.1f}s", flush=True)
    return res


# ----------------------------------------------------------------------------
# Post-processing
# ----------------------------------------------------------------------------
def _nanmean_or_nan(vals):
    vals = np.asarray(vals, dtype=float)
    return float(vals.mean()) if np.isfinite(vals).all() else float("nan")


def fmt(v, p=2):
    return "nan" if v is None or not np.isfinite(v) else f"{v:.{p}f}"


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def main():
    os.makedirs(OUT, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    with Pool(processes=2) as pool:
        results = pool.map(run_source, SOURCES)
    tags = [s[0] for s in SOURCES]
    by_tag = {r["tag"]: r for r in results}
    runtime = time.time() - t0

    # Grid: grid[cand][mu] -> per-source list
    grid = {c: {mu: {} for mu in MUS} for c in CANDS}
    flat = []
    for r in results:
        for row in r["rows"]:
            grid[row["cand"]][row["mu"]][r["tag"]] = row
            flat.append((row["cand"], row["mu"], r["tag"], row))
    with open(os.path.join(OUT, "grid.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["cand", "mu", "source", "whole_db", "first5s_db", "last10_db", "final_msd_db", "diverged"])
        for c, mu, t, rw in flat:
            wr.writerow([c, mu, t, rw["whole"], rw["first5"], rw["last10"], rw["msd"], int(rw["diverged"])])

    def agg(c, mu):
        rs = [grid[c][mu][t] for t in tags]
        return dict(whole=_nanmean_or_nan([q["whole"] for q in rs]),
                    first5=_nanmean_or_nan([q["first5"] for q in rs]),
                    last10=_nanmean_or_nan([q["last10"] for q in rs]),
                    msd=_nanmean_or_nan([q["msd"] for q in rs]),
                    ndiv=int(sum(q["diverged"] for q in rs)))

    A = {c: {mu: agg(c, mu) for mu in MUS} for c in CANDS}

    def best(c, key):
        ok = {mu: A[c][mu][key] for mu in MUS if np.isfinite(A[c][mu][key])}
        return min(ok, key=ok.get) if ok else None

    best_whole = {c: best(c, "whole") for c in CANDS}
    best_last = {c: best(c, "last10") for c in CANDS}

    # Sanity values
    a145 = by_tag["a145"]
    def row36(c):
        return [r for r in a145["rows"] if r["cand"] == c and r["mu"] == 0.36][0]["whole"]
    sanity = dict(PAPER_mu036_a145_whole=row36("PAPER"),
                  PAPER_oE_mu036_a145_whole=row36("PAPER_oE"),
                  B0_mu036_a145_whole=row36("B0"))

    # Oracle tables
    oscal = {t: by_tag[t]["oscalar"] for t in tags}
    oband = {t: by_tag[t]["oband"] for t in tags}

    def oavg(d, key):
        return _nanmean_or_nan([d[t][key] for t in tags])

    # O_band correlations and tables
    pooled = {"mu": [], "gam": [], "z": [], "mis": [], "blk": []}
    for t in tags:
        z_ = np.load(os.path.join(OUT, f"oband_{t}.npz"))
        mu, gam, z, mis = z_["mu"], z_["gam"], z_["z"], z_["mis"]
        nb = mu.shape[0]
        blk = np.arange(nb)
        keep = (blk >= int(FS / B)) & np.isfinite(mu).all(axis=1)
        pooled["mu"].append(mu[keep])
        pooled["gam"].append(gam[keep])
        pooled["mis"].append(mis[keep])
        pooled["z"].append(np.repeat(z[keep][:, None], mu.shape[1], axis=1))
    mu_all = np.concatenate([p.ravel() for p in pooled["mu"]])
    gam_all = np.concatenate([p.ravel() for p in pooled["gam"]])
    z_all = np.concatenate([p.ravel() for p in pooled["z"]])
    mis_all = np.concatenate([p.ravel() for p in pooled["mis"]])
    frac_zero = float(np.mean(mu_all == 0))
    pos = mu_all > 0
    lm = np.log(mu_all[pos])
    gclip = np.clip(gam_all[pos], 1e-6, 1 - 1e-6)
    logit_g = np.log(gclip / (1 - gclip))
    corr = dict(
        logit_gam=spearman(lm, logit_g),
        log_z=spearman(lm, np.log(np.maximum(z_all[pos], 1e-300))),
        mis_db=spearman(lm, mis_all[pos]),
        n_pos=int(pos.sum()), n_total=int(mu_all.size), frac_zero=frac_zero,
    )
    edges_dec = np.quantile(gam_all[pos], np.linspace(0, 1, 11))
    dec_rows = []
    for k in range(10):
        lo, hi = edges_dec[k], edges_dec[k + 1]
        if k < 9:
            sel = (gam_all[pos] >= lo) & (gam_all[pos] < hi)
        else:
            sel = (gam_all[pos] >= lo) & (gam_all[pos] <= hi)
        mv = mu_all[pos][sel]
        dec_rows.append(dict(decile=k + 1, lo=float(lo), hi=float(hi), n=int(sel.sum()),
                             median_mu=float(np.median(mv)) if mv.size else float("nan")))
    zbins = [("<1", 0, 1), ("1-2", 1, 2), ("2-10", 2, 10), (">10", 10, np.inf)]
    zz = z_all[pos]
    zb_rows = []
    for name, lo, hi in zbins:
        sel = (zz >= lo) & (zz < hi) if hi != np.inf else (zz >= lo)
        mv = mu_all[pos][sel]
        zb_rows.append(dict(bin=name, n=int(sel.sum()),
                            median_mu=float(np.median(mv)) if mv.size else float("nan")))

    # Figure: a145 ANR curves
    arr = np.load(os.path.join(OUT, "anr_a145.npz"))
    tgrid = lambda n: (np.arange(n) * 64 + 63) / FS
    base = ["B0", "B2", "PAPER"]
    others = sorted([c for c in CANDS if c not in base],
                    key=lambda c: A[c][best_whole[c]]["whole"] if best_whole[c] is not None else 0.0)
    top3 = others[:3]
    fig_list = base + top3 + ["O_scalar", "O_band"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for nm in fig_list:
        if nm in ("O_scalar", "O_band"):
            key = nm
            lbl = nm
        else:
            key = f"{nm}__{best_whole[nm]}"
            lbl = f"{nm} (mu={best_whole[nm]})"
        y = arr[key]
        ax.plot(tgrid(len(y)), y, lw=0.8, label=lbl)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("ANR [dB]")
    ax.set_title("ANR, alpha=1.45 file (best mu by whole-curve mean over six sources)")
    ax.grid(True, ls=":")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_anr_a145.png"), dpi=130)
    plt.close(fig)

    # JSON and report
    summary = dict(runtime_s=runtime, sanity=sanity,
                   best_by_whole={c: best_whole[c] for c in CANDS},
                   best_by_last10={c: best_last[c] for c in CANDS},
                   agg={c: {str(mu): A[c][mu] for mu in MUS} for c in CANDS},
                   oscalar=oscal, oband=oband, oband_corr=corr,
                   oband_deciles=dec_rows, oband_zbins=zb_rows, fig_curves=fig_list,
                   per_source={r["tag"]: dict(runtime_s=r["runtime_s"]) for r in results})
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=float)

    write_report(results, A, best_whole, best_last, sanity, oscal, oband, oavg, corr,
                 dec_rows, zb_rows, top3, runtime)
    print(f"[total] {runtime:.1f} s", flush=True)


def write_report(results, A, best_whole, best_last, sanity, oscal, oband, oavg, corr,
                 dec_rows, zb_rows, top3, runtime):
    tags = [r["tag"] for r in results]
    L = []
    L.append("# Exploration: step normalization for delayless FD-FxNLMS (atsip/explore)\n")
    L.append("Generated by `atsip/explore/explore.py`. Numbers only. Sources: " + ", ".join(tags) + ".\n")
    L.append(f"Runtime (wall clock, 2 worker processes): {runtime:.1f} s.\n")

    L.append("## 1. PAPER sanity (alpha 1.45, mu = 0.36)\n")
    L.append("| run | whole-curve ANR [dB] |")
    L.append("|---|---|")
    L.append(f"| PAPER port, E = last M errors as in the class (candidate PAPER) | {fmt(sanity['PAPER_mu036_a145_whole'],2)} |")
    L.append(f"| PAPER port, E as in oracle.py (candidate PAPER_oE) | {fmt(sanity['PAPER_oE_mu036_a145_whole'],2)} |")
    L.append(f"| B0 at mu = 0.36 (cross-check against oracle.py N0 fixed, -26.98) | {fmt(sanity['B0_mu036_a145_whole'],2)} |")
    L.append("Reference from repro/RESULTS.md: Prop -27.03 dB.\n")

    def best_table(best, title):
        L.append(f"## {title}\n")
        L.append("Means over the six sources. #diverged = diverged runs out of 30 (5 mu x 6 sources).\n")
        L.append("| candidate | description | best mu | whole [dB] | first5s [dB] | last10 [dB] | final MSD [dB] | #diverged (of 30) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for c in CANDS:
            m = best[c]
            if m is None:
                L.append(f"| {c} | {CAND_DESC[c]} | nan | nan | nan | nan | nan | {sum(A[c][mu]['ndiv'] for mu in MUS)} |")
                continue
            a = A[c][m]
            nd = sum(A[c][mu]["ndiv"] for mu in MUS)
            L.append(f"| {c} | {CAND_DESC[c]} | {m} | {fmt(a['whole'])} | {fmt(a['first5'])} | {fmt(a['last10'])} | {fmt(a['msd'])} | {nd} |")
        L.append("")

    best_table(best_whole, "2a. Per candidate at best mu (chosen by mean whole-curve ANR)")
    best_table(best_last, "2b. Per candidate at best mu (chosen by mean last-10% ANR)")

    L.append("## 3. Appendix: full grid (mean over six sources)\n")
    L.append("Whole-curve mean ANR [dB]; cells with any non-finite source are shown as nan.\n")
    L.append("| candidate | " + " | ".join(f"mu={mu}" for mu in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for c in CANDS:
        L.append(f"| {c} | " + " | ".join(fmt(A[c][mu]['whole']) for mu in MUS) + " |")
    L.append("")
    L.append("Last-10% mean ANR [dB]:\n")
    L.append("| candidate | " + " | ".join(f"mu={mu}" for mu in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for c in CANDS:
        L.append(f"| {c} | " + " | ".join(fmt(A[c][mu]['last10']) for mu in MUS) + " |")
    L.append("")
    L.append("Diverged runs (of 6 sources) per mu:\n")
    L.append("| candidate | " + " | ".join(f"mu={mu}" for mu in MUS) + " |")
    L.append("|---|" + "---|" * len(MUS))
    for c in CANDS:
        L.append(f"| {c} | " + " | ".join(str(A[c][mu]['ndiv']) for mu in MUS) + " |")
    L.append("")

    L.append("## 4. Oracles (driver = own trajectory)\n")
    L.append("Means over six sources. O_scalar = oracle.py rule on the B0 direction; O_band = 16-band NNLS (cap 4.0).\n")
    L.append("| oracle | whole [dB] | first5s [dB] | last10 [dB] | final MSD [dB] | #diverged (of 6) |")
    L.append("|---|---|---|---|---|---|")
    for nm, dct in (("O_scalar (B0)", oscal), ("O_band (B0, J=16)", oband)):
        nd = sum(int(dct[t]["diverged"]) for t in tags)
        L.append(f"| {nm} | {fmt(oavg(dct,'whole'))} | {fmt(oavg(dct,'first5'))} | {fmt(oavg(dct,'last10'))} | {fmt(oavg(dct,'msd'))} | {nd} |")
    L.append("")
    L.append("Spearman correlation of log mu_j with each feature, pooled over six sources, blocks after 1 s, mu_j > 0:\n")
    L.append("| feature | Spearman rho |")
    L.append("|---|---|")
    L.append(f"| (a) logit(band mean gam_prev) | {fmt(corr['logit_gam'],3)} |")
    L.append(f"| (b) log z_i | {fmt(corr['log_z'],3)} |")
    L.append(f"| (c) band misalignment [dB] | {fmt(corr['mis_db'],3)} |")
    L.append("")
    L.append(f"Pooled band-blocks after 1 s: {corr['n_total']}; with mu_j > 0: {corr['n_pos']}. "
             f"Fraction of mu_j equal to 0: {corr['frac_zero']:.4f}.\n")
    L.append("Median mu_j (mu_j > 0) by decile of band mean gam_prev (pooled, after 1 s):\n")
    L.append("| decile | gam_prev range | n (mu_j>0) | median mu_j |")
    L.append("|---|---|---|---|")
    for r in dec_rows:
        L.append(f"| {r['decile']} | {r['lo']:.4f} - {r['hi']:.4f} | {r['n']} | {fmt(r['median_mu'],4)} |")
    L.append("")
    L.append("Median mu_j (mu_j > 0) by z_i bin (pooled, after 1 s):\n")
    L.append("| z_i bin | n (mu_j>0) | median mu_j |")
    L.append("|---|---|---|")
    for r in zb_rows:
        L.append(f"| {r['bin']} | {r['n']} | {fmt(r['median_mu'],4)} |")
    L.append("")
    L.append("Per-source O_band and O_scalar (whole / last10 [dB], final MSD [dB]):\n")
    L.append("| source | O_scalar whole | O_scalar last10 | O_scalar MSD | O_band whole | O_band last10 | O_band MSD |")
    L.append("|---|---|---|---|---|---|---|")
    for t in tags:
        a, b = oscal[t], oband[t]
        L.append(f"| {t} | {fmt(a['whole'])} | {fmt(a['last10'])} | {fmt(a['msd'])} | {fmt(b['whole'])} | {fmt(b['last10'])} | {fmt(b['msd'])} |")
    L.append("")

    L.append("## 5. Figure\n")
    L.append("- `out/fig_anr_a145.png`: ANR curves, alpha=1.45 file, at best mu (by whole-curve mean): "
             "B0, B2, PAPER, the three best of the remaining candidates (" + ", ".join(top3) + "), O_scalar, O_band.\n")
    L.append("## 6. Files\n")
    L.append("- `out/grid.csv`: every run (candidate, mu, source).")
    L.append("- `out/summary.json`: aggregated tables and oracle results.")
    L.append("- `out/oband_<source>.npz`: per-block mu_j, band gam_prev, z_i, band misalignment for O_band.")
    L.append("- `out/anr_a145.npz`: ANR curves of the alpha=1.45 runs used in the figure.")
    L.append("")
    with open(os.path.join(HERE, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
