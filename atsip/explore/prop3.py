#!/usr/bin/env python3
"""Analytic (expected-value) check of Proposition 3 for the spectral-factor (SF) form.

No sample noise: all second-order statistics are the expected values for white x,
computed from the measured IRs (impulse_response/primary_ir.dat, secondary_ir.dat,
secondary trimmed to 128 taps as in the reviewer code under /tmp/chk4).

For each M (N = 2M, the frame model's weight formula assumes N = 2M):
  * w_M, J_o(M): M-tap Wiener solution (expected cost).
  * w_SF = w_M + v, where v solves the linear mean fixed point A v + b = 0 of the
    frame model (circular expected correlations, operator = SF / geometric a).
  * [w_inf]_M: first M taps of the causal infinite-length Wiener solution (G = 2^17 grid).
  * Excesses J(w) - J_o(M) = d' Phi d with Phi the expected M x M autocorrelation of the
    filtered reference (exact for white x, no cancellation for small d).
  * Geometric a in {0, 0.5, 1}: frame-model fixed point at the same M, and the closed form
    J_b(a) = || f^a [f^-a q_-]_+ ||^2 (M -> inf), as in /tmp/chk4/main2.py.

Usage: python3 atsip/explore/prop3.py
Writes atsip/explore/out/prop3_summary.json.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import numpy as np
from scipy import linalg

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "out")
G = 1 << 17
C_SEC = 128
EPS_REL = 1e-6
MS = [128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
PRED = {128: -25.6, 192: -18.8, 256: -16.5, 384: -11.5, 512: -17.1, 768: -44.0, 1024: -64.1}
GEO_A = [0.0, 0.5, 1.0]
N_CAUSAL = 16384                      # taps of w_inf used for the norm
FIT_MS = [512, 768, 1024, 1536, 2048]  # fit window (M >= 512)


def dB(x):
    return 10.0 * np.log10(np.asarray(x, dtype=float) + 1e-300)


def pad(h, n):
    o = np.zeros(n)
    m = min(n, len(h))
    o[:m] = h[:m]
    return o


hP = np.loadtxt(os.path.join(ROOT, "impulse_response/primary_ir.dat"), ndmin=1).astype(float)
hS = pad(np.loadtxt(os.path.join(ROOT, "impulse_response/secondary_ir.dat"), ndmin=1), C_SEC)
Jd = hP @ hP


def xc(a, b, lags):
    """Expected sum_k a(k) b(k + tau) for white x (lags array)."""
    lags = np.asarray(lags)
    full = np.correlate(b, a, "full")
    out = np.zeros(len(lags))
    idx = lags + len(a) - 1
    ok = (idx >= 0) & (idx < len(full))
    out[ok] = full[idx[ok]]
    return out


def wiener(M):
    lags = np.arange(M)
    rrr = xc(hS, hS, lags)
    prd = xc(hS, hP, lags)
    w = linalg.solve_toeplitz(rrr, -prd)
    g = np.zeros(max(len(hP), len(hS) + M - 1))
    g[:len(hP)] += hP
    g[:len(hS) + M - 1] += np.convolve(hS, w)
    return w, g


def minphase(S, Gn):
    c = np.fft.irfft(0.5 * np.log(S), Gn)
    c[1:Gn // 2] *= 2
    c[Gn // 2 + 1:] = 0
    return np.exp(np.fft.rfft(c))


def frame_fixed_point(M, op):
    """Mean fixed point v (deviation from w_M) of the frame model with N = 2M.

    op = 'sf' : SF operator  q = irfft(Y/conj F), keep lags < M, g = irfft(rfft(q)/F)[:M]
    op = a    : geometric operator  g = irfft(Y * Lam^-a)[:M]  (Lam normalised to mean 1)
    The expected circular correlations C (N x (M+1)) follow /tmp/chk4/sf.py impl().
    """
    N = 2 * M
    w, g = wiener(M)
    Jo = g @ g
    tau = np.arange(N)
    rr = lambda la: xc(hS, hS, np.asarray(la))
    big = np.arange(-N - M, N + 1)
    rrb = rr(big)
    pb = xc(hS, g, np.arange(-N, N + 1))

    def ctil(rho):
        c = np.zeros(N)
        c[:M + 1] = rho(tau[:M + 1])
        t = tau[M + 1:]
        c[M + 1:] = ((N - t) * rho(t) + (t - M) * rho(t - N)) / M
        return c

    C = np.zeros((N, M + 1))
    for j in range(M):
        C[:, j] = ctil(lambda la: rrb[np.asarray(la) - j + N + M])
    C[:, M] = ctil(lambda la: pb[np.asarray(la) + N])

    Sn = np.abs(np.fft.rfft(hS, N)) ** 2
    Lam = Sn + EPS_REL * Sn.mean()
    Y = np.fft.rfft(C, axis=0)
    if op == "sf":
        Fn = minphase(Lam, N)
        qq = np.fft.irfft(Y / np.conj(Fn)[:, None], N, axis=0)
        qq[M:] = 0
        gg = np.fft.irfft(np.fft.rfft(qq, axis=0) / Fn[:, None], N, axis=0)[:M]
    else:
        Ln = Lam / Lam.mean()
        gg = np.fft.irfft(Y * (Ln ** (-float(op)))[:, None], N, axis=0)[:M]
    A = gg[:, :M]
    b = gg[:, M]
    v = -np.linalg.solve(A, b)
    resid = np.linalg.norm(A @ v + b) / np.linalg.norm(b)
    Phi = linalg.toeplitz(rr(np.arange(M)))
    return dict(w=w, Jo=Jo, Phi=Phi, v=v, resid=float(resid))


def main():
    # ---- spectral factor and ideal causal Wiener (G-point grid)
    Sw = np.fft.rfft(hS, G)
    Pw = np.fft.rfft(hP, G)
    S2 = np.abs(Sw) ** 2
    F = minphase(S2, G)
    q = np.fft.irfft(np.conj(Sw) * Pw / np.conj(F), G)
    qp = q.copy()
    qp[G // 2:] = 0
    winf_full = np.fft.irfft(-np.fft.rfft(qp) / F, G)
    winf = winf_full[:N_CAUSAL]
    tail_energy_frac = float((winf_full[N_CAUSAL:G // 2] ** 2).sum() / (winf_full[:G // 2] ** 2).sum())
    Wn2 = float(winf @ winf)

    # ---- rho: reflected-zero radius of the secondary path (as /tmp/chk4/base.py)
    nzS = np.max(np.nonzero(hS)[0]) + 1
    z = np.roots(hS[:nzS])
    az = np.abs(z)
    refl = np.where(az > 1, 1 / az, az)
    rho = float(refl.max())

    # ---- closed form J_b(a), M -> inf (as /tmp/chk4/main2.py)
    Pb = xc(hS, hS, np.array([0]))[0]
    f = F / np.sqrt(Pb)
    qm = q.copy()
    qm[:G // 2] = 0
    Qm = np.fft.rfft(qm)
    closed = {}
    for a in GEO_A:
        y = np.fft.irfft(f ** (-a) * Qm, G)
        y[G // 2:] = 0
        closed[a] = float((np.abs(np.fft.irfft(f ** a * np.fft.rfft(y), G)) ** 2).sum())

    rows = []
    for M in MS:
        fp_sf = frame_fixed_point(M, "sf")
        Jo = fp_sf["Jo"]
        Phi, w_M, v_sf = fp_sf["Phi"], fp_sf["w"], fp_sf["v"]
        d = winf[:M] - w_M                       # [w_inf]_M - w_M
        exc_ideal = float(d @ Phi @ d / Jo)
        exc_sf = float(v_sf @ Phi @ v_sf / Jo)
        dist_sf_ideal = float((v_sf - d) @ (v_sf - d) / Wn2)
        geo = {}
        for a in GEO_A:
            fp = frame_fixed_point(M, a)
            geo[a] = dict(exc=float(fp["v"] @ Phi @ fp["v"] / Jo), resid=fp["resid"])
        rows.append(dict(
            M=M, N=2 * M, Jo=float(Jo), Jo_over_Jd_dB=float(dB(Jo / Jd)),
            exc_SF_dB=float(dB(exc_sf)), exc_ideal_dB=float(dB(exc_ideal)),
            pred_dB=PRED.get(M), dist_SF_vs_ideal_dB=float(dB(dist_sf_ideal)),
            w_SF_minus_wM_norm2=float(v_sf @ v_sf), sf_resid=fp_sf["resid"],
            rel_dev_SF_vs_ideal_dB=float(dB(((v_sf - d) @ (v_sf - d)) / (d @ d))),
            geo_exc_dB={str(a): float(dB(geo[a]["exc"])) for a in GEO_A},
            geo_resid={str(a): geo[a]["resid"] for a in GEO_A},
            closed_geo_exc_dB={str(a): float(dB(closed[a] / Jo)) for a in GEO_A},
        ))
        print("M=%5d Jo=%.4g | SF frame %7.2f | ideal [winf]_M %7.2f | pred %s | dist(SF,ideal) %7.2f dB | "
              "geo0 %7.2f geo0.5 %7.2f geo1 %7.2f | closed a0.5 %7.2f a1 %7.2f | resid %.1e"
              % (M, Jo, dB(exc_sf), dB(exc_ideal), PRED.get(M), dB(dist_sf_ideal),
                 dB(geo[0.0]["exc"]), dB(geo[0.5]["exc"]), dB(geo[1.0]["exc"]),
                 dB(closed[0.5] / Jo), dB(closed[1.0] / Jo), fp_sf["resid"]), flush=True)

    Ms = np.array(FIT_MS, dtype=float)
    ideal =np.array([r["exc_ideal_dB"] for r in rows if r["M"] in FIT_MS])
    sfv = np.array([r["exc_SF_dB"] for r in rows if r["M"] in FIT_MS])
    slope_ideal = float(np.polyfit(Ms, ideal, 1)[0])
    slope_sf = float(np.polyfit(Ms, sfv, 1)[0])

    def local_slope(vals, grid):
        return [float((vals[i + 1] - vals[i]) / (grid[i + 1] - grid[i])) for i in range(len(grid) - 1)]

    grid_hi = [1024, 1536, 2048]
    ideal_hi = [r["exc_ideal_dB"] for r in rows if r["M"] in grid_hi]
    sf_hi = [r["exc_SF_dB"] for r in rows if r["M"] in grid_hi]
    tails = [10 * np.log10((winf_full[M:N_CAUSAL] ** 2).sum()) for M in grid_hi]
    summary = dict(
        rho=rho, rho_dB_per_tap=float(20 * np.log10(rho)), tail_energy_frac_beyond_16384=tail_energy_frac,
        local_slope_ideal_1024_2048=local_slope(ideal_hi, grid_hi),
        local_slope_SF_1024_2048=local_slope(sf_hi, grid_hi),
        local_slope_w_inf_tail_energy_1024_2048=local_slope(tails, grid_hi),
        w_inf_norm2=Wn2, closed_form_abs={str(a): closed[a] for a in GEO_A},
        Jd=float(Jd), rows=rows,
        fit_M=FIT_MS, slope_ideal_dB_per_tap=slope_ideal, slope_SF_dB_per_tap=slope_sf,
        ref_slope_20log10rho=float(20 * np.log10(rho)),
    )
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "prop3_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("rho=%.6f  20log10(rho)=%.4f dB/tap" % (rho, 20 * np.log10(rho)))
    print("local slopes 1024-1536-2048: ideal", np.round(summary["local_slope_ideal_1024_2048"], 4),
          "SF", np.round(summary["local_slope_SF_1024_2048"], 4),
          "tail energy of w_inf", np.round(summary["local_slope_w_inf_tail_energy_1024_2048"], 4))
    print("slope (M=512..2048) ideal %.4f dB/tap, SF frame %.4f dB/tap" % (slope_ideal, slope_sf))
    print("closed form J_b(a) rel Jo(M) at M->inf: " + ", ".join(
        "a=%g: %s dB(abs) " % (a, np.round(dB(closed[a]), 2)) for a in GEO_A))


if __name__ == "__main__":
    main()
