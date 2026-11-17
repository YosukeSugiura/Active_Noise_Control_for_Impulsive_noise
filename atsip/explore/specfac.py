#!/usr/bin/env python3
"""Spectral-factor (SF) normaliser, with and without a manual step control.

SF, per block. Srr(k) is the EMA of the reference spectrum (as in geosched). Its minimum-phase factor F
comes from the real cepstrum of 0.5 log(Srr + eps_shared + eps_reg Pbar), so |F|^2 = Srr.
  X(k) = conj(R(k)) E(k) / conj(F(k));  x = irfft(X);  keep lags 0..TRUNC-1 (TRUNC = N/2);
  G = rfft(x) / F;  w <- w - mu_i * irfft(G)[0:M]   (mu_i = 2 mu B / M as in geosched).
With constant Srr (= Pbar), SF reduces to the a=0 normaliser Lam = Pbar (checked in `check`).

Rows: SF with eps_reg in {0, 1e-3, 1e-2} x {no ramp, (m0, T) in {2,3} x {2,4}}, step control as in geosched.
References (same sources, same mu grid): B0, W0.5, and GS = geosched's best manual schedule
(a=0.5, T=4 s, m0=3). The mu grid is geosched's dissect grid plus 0.3 and 0.4.

Usage:
  python3 atsip/explore/specfac.py check   constant-Srr reduction + prototype (/tmp/chk/fp.py) reproduction on a145
  python3 atsip/explore/specfac.py smoke   one source (a145), all rows, prints the rows at mu = 0.1
  python3 atsip/explore/specfac.py         full run, ten sources, Pool(2), writes RESULTS_specfac.md

Writes atsip/explore/out/specfac_*.{csv,json} (small) and atsip/explore/RESULTS_specfac.md.
Divergence: as geosched (non-finite output or end ANR > 0 dB); diverged runs are NaN.
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dissect as D  # noqa: E402
import explore as X  # noqa: E402
import mixlam as ML  # noqa: E402
import geosched as G  # noqa: E402

FS, M, N, B, NB = D.FS, D.M, D.N, D.B, D.NB
OUT = D.OUT
MUS = sorted(set(D.MUS) | {0.3, 0.4})          # 0.2 already in the dissect grid
SIX, TWO, COL, ALLTAGS = D.SIX, D.TWO, D.COL, D.ALLTAGS
TRUNC = N // 2
EPS_REG = [0.0, 1e-3, 1e-2]
RAMP = [(2, 2), (2, 4), (3, 2), (3, 4)]        # (m0, T)
TOL_DB = 1.0


def sf_dir(R, E, Srr, Pbar, s_eps, eps_reg, trunc=TRUNC, ret_y=False):
    """Spectral-factor direction, returns irfft(G)[:M].
    ret_y=True: returns (direction, x, F) with x the length-N intermediate vector after the causal truncation and
    before the inverse factor (x = {R* E / F*}_+, lags >= trunc zeroed) and F the minimum-phase factor (rfft bins),
    so that direction == irfft(rfft(x) / F)[:M].  The direction is computed by the same code either way."""
    Lam = np.maximum(Srr + s_eps + eps_reg * Pbar, 1e-300)
    full = np.concatenate([Lam, Lam[-2:0:-1]])              # length N, symmetric
    c = np.fft.ifft(0.5 * np.log(full)).real                 # real cepstrum
    c[1:N // 2] *= 2.0                                       # fold to causal cepstrum
    c[N // 2 + 1:] = 0.0
    F = np.exp(np.fft.rfft(c))                               # minimum-phase factor, |F|^2 = Lam
    x = np.fft.irfft(np.conj(R) * E / np.conj(F), N)
    x[trunc:] = 0.0                                          # keep causal lags
    G_ = np.fft.rfft(x) / F
    if ret_y:
        return np.fft.irfft(G_, N)[:M], x, F
    return np.fft.irfft(G_, N)[:M]


def sim_sf(data, Rall, eps_reg, T, m0, mu, trunc=TRUNC):
    """Same loop as geosched.sim_sched, with the SF direction in place of the W(a) normaliser."""
    d, xq, hS, S2 = data["d"], data["xq"], data["hS"], data["S2"]
    L = len(d)
    nb = L // B
    swv = sliding_window_view(xq, M)
    w = np.zeros(M)
    zi = np.zeros(len(hS) - 1)
    e_all = np.zeros(L)
    shared = X.Shared(S2)
    diverged = False
    for i in range(nb):
        s0 = i * B
        Xb = swv[s0 + N - M + 1: s0 + N - M + 1 + B][:, ::-1]
        y = Xb @ w
        ys, zi = signal.lfilter(hS, [1.0], y, zi=zi)
        e_all[s0:s0 + B] = d[s0:s0 + B] + ys
        E = ML.frame_E(e_all, s0, M)
        s = shared.step(Rall[i], E)
        wr = 1.0 if T == 0 else min(1.0, (i * B / FS) / T)
        mu_i = mu * (1 + (m0 - 1) * (1 - wr))
        g = sf_dir(Rall[i], E, s["Srr"], s["Pbar"], s["eps"], eps_reg, trunc)
        w = w - 2 * mu_i * B / M * g
        if not np.isfinite(w).all():
            e_all[s0 + B:] = np.nan
            diverged = True
            break
    return dict(e=e_all, w=w, diverged=diverged)


# ----------------------------------------------------------------------------
# Rows
# ----------------------------------------------------------------------------
CFGS = [("SF", e, 0, 1) for e in EPS_REG] + \
       [("SF", e, T, m0) for e in EPS_REG for (m0, T) in RAMP] + \
       [("D", "B0"), ("D", "W0.5"), ("S", 0.5, 4, 3)]


def label_of(cfg):
    if cfg[0] == "SF":
        _, e, T, m0 = cfg
        return f"SF e={e:g} T={T} m0={m0}"
    if cfg[0] == "D":
        return cfg[1]
    return "GS (a0.5 T4 m0=3)"


def run_cfg(data, Rall, cfg, mu):
    if cfg[0] == "SF":
        _, e, T, m0 = cfg
        return sim_sf(data, Rall, e, T, m0, mu)
    if cfg[0] == "D":
        return D.simulate(data, Rall, D.spec_of(cfg[1]), mu, M)
    _, a, T, m0 = cfg
    return G.sim_sched(data, Rall, a, T, m0, mu)


def row_from_sim(lab, sim, data, mu):
    d, w_opt = data["d"], data["w_opt"]
    m = X.metrics(sim["e"], sim["w"], d, w_opt)
    div_task = (not bool(np.isfinite(sim["w"]).all())) or (not np.isfinite(m["last10"])) \
        or (not np.isfinite(m["whole"])) or m["last10"] > 0.0
    vals = {k: (float("nan") if div_task else float(m[k])) for k in ["whole", "first5", "last10", "msd"]}
    return dict(label=lab, mu=mu, **vals, diverged=int(div_task), xrule_div=int(m["diverged"]),
                raw_whole=float(m["whole"]) if np.isfinite(m["whole"]) else float("nan"))


def unit(tag):
    t0 = time.time()
    kind, arg = D.KIND[tag]
    data = D.load_data(kind, arg)
    Rall = ML.prepass(data)["Rall"]
    rows = []
    for cfg in CFGS:
        lab = label_of(cfg)
        for mu in MUS:
            rows.append(dict(src=tag, **row_from_sim(lab, run_cfg(data, Rall, cfg, mu), data, mu)))
        print(f"[cfg] {tag} {lab} ", flush=True)
    return dict(tag=tag, rows=rows, runtime=time.time() - t0)


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def proto_er(data, Rall, mu):
    """Verbatim logic of /tmp/chk/fp.py mode ER (truncation at M, eps = shared eps only)."""
    from scipy import signal as sg
    def minphase(S):
        full = np.concatenate([S, S[-2:0:-1]])
        c = np.fft.ifft(0.5 * np.log(full)).real
        c[1:N // 2] *= 2
        c[N // 2 + 1:] = 0
        return np.exp(np.fft.fft(c))[:NB]
    d, xq, hS = data["d"], data["xq"], data["hS"]
    L = len(d)
    nb = L // B
    swv = sliding_window_view(xq, M)
    w = np.zeros(M)
    zi = np.zeros(len(hS) - 1)
    e_all = np.zeros(L)
    shared = X.Shared(data["S2"])
    for i in range(nb):
        s0 = i * B
        Xb = swv[s0 + N - M + 1:s0 + N - M + 1 + B][:, ::-1]
        ys, zi = sg.lfilter(hS, [1.0], Xb @ w, zi=zi)
        e_all[s0:s0 + B] = d[s0:s0 + B] + ys
        E = ML.frame_E(e_all, s0, M)
        s = shared.step(Rall[i], E)
        Srr, eps = s["Srr"], s["eps"]
        F = minphase(Srr + eps)
        q = np.fft.irfft(np.conj(Rall[i]) * E / np.conj(F), N)
        q[M:] = 0
        g = np.fft.irfft(np.fft.rfft(q) / F, N)[:M]
        w = w - 2 * mu * B / M * g
        if not np.isfinite(w).all():
            return None
    return X.metrics(e_all, w, d, data["w_opt"])


def check(verbose=True):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    out = {}
    rng = np.random.default_rng(0)
    R = np.fft.rfft(rng.standard_normal(N))
    E = np.fft.rfft(rng.standard_normal(N))
    for c in [0.37, 2.7]:
        Srr = np.full(NB, c)
        g_sf = sf_dir(R, E, Srr, c, 0.0, 0.0)
        g_w0 = np.fft.irfft(np.conj(R) * E / (Srr ** 0 * c ** 1 + 0.0), N)[:M]
        out[f"const_Srr_{c}_maxabs"] = float(np.max(np.abs(g_sf - g_w0)))
    data = D.load_data(*D.KIND["a145"])
    Rall = ML.prepass(data)["Rall"]
    p_m = proto_er(data, Rall, 0.1)
    sim_m = sim_sf(data, Rall, 0.0, 0, 1, 0.1, trunc=M)
    m_m = X.metrics(sim_m["e"], sim_m["w"], data["d"], data["w_opt"])
    sim_n = sim_sf(data, Rall, 0.0, 0, 1, 0.1, trunc=TRUNC)
    m_n = X.metrics(sim_n["e"], sim_n["w"], data["d"], data["w_opt"])
    out["proto_a145_mu0.1"] = None if p_m is None else {k: float(p_m[k]) for k in ["whole", "first5", "last10"]}
    out["sf_trunc_M_a145_mu0.1"] = {k: float(m_m[k]) for k in ["whole", "first5", "last10"]}
    out["sf_trunc_N2_a145_mu0.1"] = {k: float(m_n[k]) for k in ["whole", "first5", "last10"]}
    out["M_N_NB"] = [M, N, NB]
    if verbose:
        print("[check]", json.dumps(out), flush=True)
    return out


def smoke():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    rr = unit("a145")
    for r in rr["rows"]:
        if r["mu"] == 0.1:
            print(f"[smoke] {r['label']:<24} whole {r['whole']} first5 {r['first5']} last10 {r['last10']} "
                  f"div {r['diverged']}", flush=True)
    print(f"[smoke] runtime {rr['runtime']:.1f} s for {len(CFGS)} rows x {len(MUS)} mu", flush=True)


def timing():
    """Per-block wall time of SF (eps 0) and W0.5 on a145 at mu=0.1, one process."""
    data = D.load_data(*D.KIND["a145"])
    Rall = ML.prepass(data)["Rall"]
    nb = len(data["d"]) // B
    t = time.time()
    D.simulate(data, Rall, D.spec_of("W0.5"), 0.1, M)
    tw = (time.time() - t) / nb
    t = time.time()
    sim_sf(data, Rall, 0.0, 0, 1, 0.1)
    ts = (time.time() - t) / nb
    return dict(w05_ms_per_block=1e3 * tw, sf_ms_per_block=1e3 * ts, ratio=ts / tw)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    chk = check(verbose=False)
    t0 = time.time()
    res = []
    with Pool(processes=2) as pool:
        for rr in pool.imap_unordered(unit, ALLTAGS, chunksize=1):
            res.append(rr)
            print(f"[done] {rr['tag']} {rr['runtime']:.1f} s ({len(res)}/{len(ALLTAGS)})", flush=True)
            with open(os.path.join(OUT, f"specfac_src_{rr['tag']}.json"), "w") as fh:
                json.dump(dict(rows=rr["rows"]), fh, default=float)
    runtime = time.time() - t0
    tim = timing()
    print(f"[timing] {tim}", flush=True)

    rows = [r for rr in res for r in rr["rows"]]
    labels = [label_of(c) for c in CFGS]
    cell = {(r["label"], r["mu"], r["src"]): r for r in rows}

    def cmean(lab, mu, srcs, key):
        return D.nmean([cell[(lab, mu, t)][key] for t in srcs])

    def ndiv(lab, mu, srcs):
        return int(sum(cell[(lab, mu, t)]["diverged"] for t in srcs))

    def best_mu(lab):
        ok = {mu: cmean(lab, mu, SIX, "whole") for mu in MUS}
        ok = {mu: v for mu, v in ok.items() if np.isfinite(v)}
        return min(ok, key=ok.get) if ok else None

    def max_stable(lab):
        st = [mu for mu in MUS if ndiv(lab, mu, ALLTAGS) == 0]
        return max(st) if st else None

    chosen = {lab: best_mu(lab) for lab in labels}

    grid_path = os.path.join(OUT, "specfac_grid.csv")
    with open(grid_path, "w", newline="") as fh:
        wr_ = csv.writer(fh)
        wr_.writerow(["label", "source", "mu", "whole_db", "first5_db", "last10_db", "final_msd_db",
                      "diverged_task", "diverged_xrule(-5dB)"])
        for r in sorted(rows, key=lambda r: (r["label"], r["src"], r["mu"])):
            wr_.writerow([r["label"], r["src"], r["mu"], r["whole"], r["first5"], r["last10"], r["msd"],
                          r["diverged"], r["xrule_div"]])
    with open(os.path.join(OUT, "specfac_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(runtime_s=runtime, chosen_mu=chosen, check=chk, timing=tim), fh, indent=1, default=float)

    F = X.fmt
    L = []
    L.append("# Spectral-factor normalisation (atsip/explore/specfac.py)\n")
    L.append("Numbers only. Generated by `atsip/explore/specfac.py`. Ten sources as in dissect.py, E_M framing, "
             "divergence as geosched (task rule, NaN when diverged). Runtime (2 workers): "
             f"{runtime:.1f} s. mu grid: {MUS}.\n")
    L.append("Checks:\n")
    L.append(f"- Constant Srr (= Pbar), eps 0: max |SF - W(a=0)| direction = "
             f"{chk['const_Srr_0.37_maxabs']:.2e} and {chk['const_Srr_2.7_maxabs']:.2e}.")
    p = chk["proto_a145_mu0.1"]
    s_m = chk["sf_trunc_M_a145_mu0.1"]
    s_n = chk["sf_trunc_N2_a145_mu0.1"]
    L.append(f"- Prototype (fp.py, ER, truncation at M) on a145, mu 0.1, eps 0: whole/first5/last10 = "
             f"{F(p['whole'])}/{F(p['first5'])}/{F(p['last10'])} (expected -29.00/-28.08/-29.24). "
             f"Same logic in SF code: {F(s_m['whole'])}/{F(s_m['first5'])}/{F(s_m['last10'])}.")
    L.append(f"- Spec truncation at N/2 (used below): {F(s_n['whole'])}/{F(s_n['first5'])}/{F(s_n['last10'])}.")
    L.append(f"- Timing per block (a145, mu 0.1, one process): W0.5 {tim['w05_ms_per_block']:.3f} ms, "
             f"SF {tim['sf_ms_per_block']:.3f} ms, ratio SF/W0.5 {tim['ratio']:.2f}.\n")

    L.append("## Table. Means over sources; whole/first5/last10 in dB\n")
    L.append("White = six alpha=1.45 sources; a1.2 and colored = two sources each. mu chosen per row by the "
             "six-source white whole mean. Max stable mu = largest grid mu with no divergence on any of the ten "
             "sources.\n")
    L.append("| row | mu | white whole | white first5 | white last10 | a1.2 whole | a1.2 first5 | a1.2 last10 | "
             "colored whole | colored first5 | colored last10 | #div at mu (of 10) | #div sweep (of 80) | max stable mu |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lab in labels:
        mu = chosen[lab]
        if mu is None:
            L.append(f"| {lab} | nan | " + " | ".join(["nan"] * 9) + f" | {sum(ndiv(lab, m, ALLTAGS) for m in MUS)} | "
                     f"{max_stable(lab)} |")
            continue
        L.append(f"| {lab} | {mu} | {F(cmean(lab, mu, SIX, 'whole'))} | {F(cmean(lab, mu, SIX, 'first5'))} | "
                 f"{F(cmean(lab, mu, SIX, 'last10'))} | {F(cmean(lab, mu, TWO, 'whole'))} | "
                 f"{F(cmean(lab, mu, TWO, 'first5'))} | {F(cmean(lab, mu, TWO, 'last10'))} | "
                 f"{F(cmean(lab, mu, COL, 'whole'))} | {F(cmean(lab, mu, COL, 'first5'))} | "
                 f"{F(cmean(lab, mu, COL, 'last10'))} | {ndiv(lab, mu, ALLTAGS)} | "
                 f"{sum(ndiv(lab, m, ALLTAGS) for m in MUS)} | {max_stable(lab)} |")
    L.append("")
    L.append("## Files\n")
    L.append("- `out/specfac_grid.csv`: every run (label, source, mu), both divergence rules.")
    L.append("- `out/specfac_summary.json`: chosen mu, check values, timing.")
    L.append("")
    with open(os.path.join(HERE, "RESULTS_specfac.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[total] {runtime:.1f} s", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "check":
        print(json.dumps(check(verbose=False), indent=1, default=float))
    elif mode == "smoke":
        smoke()
    else:
        main()
