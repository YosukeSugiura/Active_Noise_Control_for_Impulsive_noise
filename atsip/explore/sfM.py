#!/usr/bin/env python3
"""Finite-length test of the spectral-factor (SF) fixed-point bias.

Hypothesis: the SF fixed-point condition trunc_M[f^-1 * {f^-* * r_re}_+] = 0 is equivalent to
sum_j f^-1(j) r_re(tau+j) = 0 for tau < M, and r_re(tau >= M) != 0 for the M-tap Wiener solution.
So the SF bias should come from the tail lags tau >= M and shrink as M grows.

Per source and M (N = 2M, B = M/4 unless --B is given):
  * M-tap Wiener w_opt and floor J_o(M) (primary IR kept at full length, causal).
  * Linearised mean fixed point (fp.py conventions: sampled blocks i >= 16, every 2nd block)
    for SF (reg 0 and 1e-2) and geometric a in {0, 0.5, 1}: excess over J_o(M) and L1 ANR.
  * Tail energy ratio ||r_re(tau>=M)||^2 / ||r_re(tau)||^2 at w_opt(M), with
    r_re(tau) = (1/L) sum_n r(n) e_opt(n+tau), tau in [0, N).
  * Stationary prediction: the SF operator applied to the lag sequence c(tau) (frame-window
    weighted, see below), tail lags only vs all lags. A scalar kappa calibrates the stationary
    forcing to the frame forcing (fitted on the full-lag SF forcing), reported with the cosine.
  * Control (M = 512): primary IR truncated to K taps, to shorten the unconstrained filter.

Usage: python3 atsip/explore/sfM.py run SRC M [B]     (one source, one M, one process)
       python3 atsip/explore/sfM.py control M K        (control: M-tap run with primary IR of K taps)
       python3 atsip/explore/sfM.py report             (writes RESULTS_sfM.md from out/sfM_*.json)
Writes atsip/explore/out/sfM_*.json and atsip/explore/RESULTS_sfM.md (report).
"""
import os
import sys
import json
import time
import glob
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy import signal, linalg

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "oracle"))
import oracle as O  # noqa: E402

FS = O.FS
C_SEC = O.C_SEC
ALPHA = O.ALPHA            # EMA weight, as in oracle.Normalizers / explore.Shared
EPS_REL = 1e-6
I0, STEP = 16, 2           # fixed-point sampling, as in fp.py / sfbias_lib
OUT = os.path.join(HERE, "out")
SRCS = ["a145", "a185", "col145_s10"]


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_x(src):
    if src == "gauss":                                     # control: Gaussian excitation (no impulses)
        return np.random.default_rng(0).standard_normal(O.N_NOISE)
    if src.startswith("col145_s"):                           # as dissect.colored_x
        seed = int(src.split("_s")[1])
        x0 = O.cms_symmetric_stable(1.45, O.N_NOISE, seed)
        y = signal.lfilter([1.0], [1.0, -1.2, 0.5], x0)
        return y * (np.median(np.abs(x0)) / np.median(np.abs(y)))
    for tag, kind, arg in O.SOURCES:
        if tag == src:
            if kind == "file":
                return np.loadtxt(os.path.join(O.ROOT, arg), ndmin=1).astype(np.float64)
            return O.cms_symmetric_stable(1.45, O.N_NOISE, seed=arg)
    raise KeyError(src)


def load_irs(hp_taps=None):
    hP = np.loadtxt(os.path.join(O.ROOT, "impulse_response/primary_ir.dat"), ndmin=1).astype(float)
    hS = O.pad_or_trim(np.loadtxt(os.path.join(O.ROOT, "impulse_response/secondary_ir.dat"), ndmin=1), C_SEC)
    if hp_taps is not None:
        hP = hP[:hp_taps]
    return hP, hS


def wiener(r, d, M, T):
    """LS M-tap solution of min ||d + R w||^2 from the first T samples (as oracle.wiener_opt)."""
    r, d = r[:T], d[:T]
    T = len(r)
    phi = np.array([np.dot(r[j:], r[:T - j]) for j in range(M)])
    c = np.array([np.dot(r[:T - j], d[j:]) for j in range(M)])
    phi[0] += 1e-8 * phi[0]
    return linalg.solve_toeplitz(phi, -c)


# ----------------------------------------------------------------------------
# Operators (N-grid)
# ----------------------------------------------------------------------------
def minphase(Lam, N):
    """Minimum-phase factor F on the rfft grid, |F|^2 = Lam (same cepstral formula as specfac.sf_dir)."""
    full = np.concatenate([Lam, Lam[-2:0:-1]])
    c = np.fft.ifft(0.5 * np.log(np.maximum(full, 1e-300))).real
    c[1:N // 2] *= 2.0
    c[N // 2 + 1:] = 0.0
    return np.exp(np.fft.rfft(c))


def op_sf(Y, F, N, M):
    """SF direction: q = irfft(Y/conj F), keep lags < M, g = irfft(rfft(q)/F)[:M]. Y: (NB,) or (NB, C)."""
    if Y.ndim == 1:
        q = np.fft.irfft(Y / np.conj(F), N)
        q[M:] = 0.0
        return np.fft.irfft(np.fft.rfft(q) / F, N)[:M]
    q = np.fft.irfft(Y / np.conj(F)[:, None], N, axis=0)
    q[M:] = 0.0
    return np.fft.irfft(np.fft.rfft(q, axis=0) / F[:, None], N, axis=0)[:M]


def op_geo(Y, Lam, N, M):
    """Geometric direction: g = irfft(Y/Lam)[:M] (no causal split)."""
    if Y.ndim == 1:
        return np.fft.irfft(Y / Lam, N)[:M]
    return np.fft.irfft(Y / Lam[:, None], N, axis=0)[:M]


VARIANTS = [
    ("SF_reg0", "sf", 0.0, None),
    ("SF_reg1e-2", "sf", 1e-2, None),
    ("geo_a0", "geo", None, 0.0),
    ("geo_a0.5", "geo", None, 0.5),
    ("geo_a1", "geo", None, 1.0),
]


# ----------------------------------------------------------------------------
# One run
# ----------------------------------------------------------------------------
def anr1(e, d):
    L = len(d)
    return float(20 * np.log10(np.abs(e[L // 2:]).mean() / np.abs(d[L // 2:]).mean()))


def excess_db(ex, e_opt):
    L = len(e_opt)
    return float(10 * np.log10(np.mean(ex[L // 2:] ** 2) / np.mean(e_opt[L // 2:] ** 2)))


def run(src, M, B=None, hp_taps=None, log=print, dump=None):
    from scipy import signal as sg
    t0 = time.time()
    N = 2 * M
    B = B or M // 4
    NB = N // 2 + 1
    x = load_x(src)
    L = len(x)
    nb = L // B
    hP, hS = load_irs(hp_taps)
    d = sg.lfilter(hP, [1.0], x)                  # primary path, full length (causal)
    r = sg.lfilter(hS, [1.0], x)
    w = wiener(r, d, M, 10 * FS)
    e = d + sg.lfilter(hS, [1.0], np.convolve(x, w)[:L])
    Jo = float(np.mean(e[L // 2:] ** 2))
    xq = np.concatenate([np.zeros(N), x])
    rq = np.concatenate([np.zeros(N), r])
    Sf = np.fft.rfft(hS, N)

    # Shared per-block state (explore.Shared, Srr/eps/Pbar only)
    R = np.empty((nb, NB), dtype=np.complex128)
    Srr = np.empty((nb, NB))
    eps = np.empty(nb)
    Pbar = np.empty(nb)
    srr, psum, cnt = None, 0.0, 0
    for i in range(nb):
        s0 = i * B
        R[i] = Sf * np.fft.rfft(xq[s0 + B: s0 + B + N])
        A_ = np.abs(R[i]) ** 2
        psum += A_.mean()
        cnt += 1
        eps[i] = EPS_REL * psum / cnt
        srr = A_.copy() if srr is None else (1 - ALPHA) * srr + ALPHA * A_
        Srr[i] = srr
        Pbar[i] = srr.mean()
    sampled = list(range(I0, nb, STEP))
    log(f"[{src} M={M} B={B}] nb={nb} sampled={len(sampled)} Jo={10*np.log10(Jo):.2f} dB  prep {time.time()-t0:.1f}s")

    # Fixed-point accumulation
    Ab = [np.zeros((M, M)) for _ in VARIANTS]
    bb = [np.zeros(M) for _ in VARIANTS]
    lidx = np.arange(M)
    for n_done, i in enumerate(sampled):
        s0 = i * B
        t = np.arange(s0 + B - M, s0 + B)
        Xi = rq[t[:, None] + N - lidx[None, :]]
        eo = e[s0 + B - M: s0 + B]
        Z = np.zeros((N, M + 1))
        Z[N - M:, :M] = Xi
        Z[N - M:, M] = eo
        Y = np.fft.rfft(Z, axis=0) * np.conj(R[i])[:, None]
        del Z
        for k, (nm, kind, reg, a) in enumerate(VARIANTS):
            if kind == "sf":
                F = minphase(np.maximum(Srr[i] + eps[i] + reg * Pbar[i], 1e-300), N)
                g = op_sf(Y, F, N, M)
            else:
                Lam = Srr[i] ** a * Pbar[i] ** (1 - a) + eps[i]
                g = op_geo(Y, Lam, N, M)
            Ab[k] += g[:, :M]
            bb[k] += g[:, M]
        if n_done % 100 == 0:
            log(f"  block {n_done}/{len(sampled)} {time.time()-t0:.0f}s")
    used = len(sampled)
    res = dict(src=src, M=M, N=N, B=B, L=L, hp_taps=hp_taps, used=used,
               Jo_db=10 * np.log10(Jo), L1_floor_db=anr1(e, d), rows={})

    # Fixed points
    ex_by = {}
    for k, (nm, kind, reg, a) in enumerate(VARIANTS):
        Am, bm = Ab[k] / used, bb[k] / used
        v = -np.linalg.solve(Am, bm)
        ex = sg.lfilter(v, [1.0], r)
        ex_by[nm] = (Am, bm, v, ex)
        res["rows"][nm] = dict(excess_db=excess_db(ex, e), L1_db=anr1(e + ex, d))

    # Cross-correlation r_re(tau) = (1/L) sum_n r(n) e_opt(n+tau) for all linear lags (FFT, zero-padded)
    P2 = 1 << int(np.ceil(np.log2(L + N)))
    call = np.fft.irfft(np.conj(np.fft.rfft(r, P2)) * np.fft.rfft(e, P2), P2) / L   # index l mod P2 = lag l
    cfull = call[:N]
    tail = cfull.copy()
    tail[:M] = 0.0
    res["tail_ratio"] = float(np.sum(tail ** 2) / np.sum(cfull ** 2))
    res["causal_ratio"] = float(np.sum(cfull[:M] ** 2) / np.sum(cfull ** 2))
    nrm = np.mean(r ** 2) * np.mean(e ** 2)          # normalised energies, not forced to 1 by orthogonality
    res["tail_norm"] = float(np.sum(tail ** 2) / nrm)
    res["causal_norm"] = float(np.sum(cfull[:M] ** 2) / nrm)

    # Stationary prediction for SF(reg 0). Circular N-grid kernel: a block frame contributes linear lag m
    # with weight min(M, N-m) and lag m-N with weight max(0, M-(N-m)) (frame/window overlap), both / M.
    m = np.arange(N)
    kpos = cfull * np.clip((N - m) / M, 0.0, 1.0)
    kneg = call[P2 + (m - N)] * np.clip((M - (N - m)) / M, 0.0, 1.0)
    kfull = kpos + kneg
    ktail = kpos.copy()
    ktail[:M] = 0.0                                   # tail lags tau >= M (positive lags)
    epsbar = eps[sampled].mean()
    Fst = minphase(np.maximum(Srr[sampled].mean(0) + epsbar, 1e-300), N)
    bfull = op_sf(np.fft.rfft(kfull), Fst, N, M)
    btail = op_sf(np.fft.rfft(ktail), Fst, N, M)
    Am, bm, _, _ = ex_by["SF_reg0"]
    kappa = float(bm @ bfull / (bfull @ bfull))
    cos = float(bm @ bfull / (np.linalg.norm(bm) * np.linalg.norm(bfull)))
    ex_full = sg.lfilter(-np.linalg.solve(Am, kappa * bfull), [1.0], r)
    ex_tail = sg.lfilter(-np.linalg.solve(Am, kappa * btail), [1.0], r)
    res["pred"] = dict(kappa=kappa, cos_actual_vs_stat_full=cos,
                       stat_full_excess_db=excess_db(ex_full, e),
                       stat_tail_excess_db=excess_db(ex_tail, e),
                       actual_SF_excess_db=res["rows"]["SF_reg0"]["excess_db"],
                       norm_bfull=float(np.linalg.norm(bfull)), norm_btail=float(np.linalg.norm(btail)),
                       norm_bact=float(np.linalg.norm(bm)))
    if dump:
        np.savez(dump, Am=Am, bm=bm, cfull=cfull, Fst=Fst, bfull=bfull, btail=btail, r=r, e=e)
    res["w_opt_tail_energy"] = {int(k): float(np.sum(w[k:] ** 2) / np.sum(w ** 2)) for k in (M // 4, M // 2)}
    res["time_s"] = time.time() - t0
    log(f"  done {res['time_s']:.0f}s")
    return res


def save(res, name):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


def control_length(M=1024):
    """Energy of the M-tap Wiener beyond lag M/4 and M/2 for the full and truncated primary IR."""
    out = {}
    for tag, taps in (("full", None), ("hp128", 128), ("hp64", 64)):
        x = load_x("a145")
        hP, hS = load_irs(taps)
        L = len(x)
        d = signal.lfilter(hP, [1.0], x)
        r = signal.lfilter(hS, [1.0], x)
        w = wiener(r, d, M, 10 * FS)
        out[tag] = {int(k): float(np.sum(w[k:] ** 2) / np.sum(w ** 2)) for k in (M // 8, M // 4, M // 2)}
        out[tag]["Jo_db"] = float(10 * np.log10(np.mean((d + signal.lfilter(hS, [1.0], np.convolve(x, w)[:L]))[L // 2:] ** 2)))
    return out


def report():
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "sfM_*.json"))):
        with open(f, encoding="utf-8") as fh:
            rows.append(json.load(fh))
    lines = ["# SF fixed-point bias vs finite filter length (atsip/explore/sfM.py)", "",
             "Linearised mean fixed point, fp.py conventions (sampled blocks i >= 16, every 2nd block).",
             "Excess = 10 log10(mean |ex|^2 / mean |e_opt|^2) over the second half, relative to the M-tap floor J_o(M).",
             "More negative is better. L1 = fixed-point ANR (dB). Primary IR kept at full length (422 taps).",
             "tail = ||r_re(tau>=M)||^2 / ||r_re||^2 over tau in [0,2M) at w_opt(M) (unweighted, r_re = (1/L) sum r(n) e_opt(n+tau)).",
             "tail_norm = sum_{tau>=M} r_re^2 / (mean r^2 * mean e^2) (normalised, so not forced to 1 by the Wiener orthogonality).",
             "pred = stationary SF operator on the circular lag kernel of the frame window, scaled by kappa fitted on the full-lag forcing (tail-only uses tau>=M).",
             ""]
    srcs = sorted(set(r["src"] for r in rows), key=lambda s: SRCS.index(s) if s in SRCS else 99)
    for s in srcs:
        lines += [f"## {s}", "",
                  "| M | B | J_o (dB) | SF r0 | SF r1e-2 | geo a0 | geo a0.5 | geo a1 | tail ratio | causal ratio | pred tail-only | pred full | actual SF r0 | kappa | cos(act,pred) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted([q for q in rows if q["src"] == s and q["hp_taps"] is None],
                        key=lambda q: (q["M"], q["B"])):
            rw = r["rows"]
            p = r["pred"]
            lines.append("| {M} | {B} | {jo:.2f} | {a:.2f} | {b:.2f} | {c:.2f} | {d:.2f} | {e:.2f} | {t:.3f} | {cr:.3f} | {pt:.2f} | {pf:.2f} | {ac:.2f} | {k:.3g} | {co:.2f} |".format(
                M=r["M"], B=r["B"], jo=r["Jo_db"], a=rw["SF_reg0"]["excess_db"], b=rw["SF_reg1e-2"]["excess_db"],
                c=rw["geo_a0"]["excess_db"], d=rw["geo_a0.5"]["excess_db"], e=rw["geo_a1"]["excess_db"],
                t=r["tail_ratio"], cr=r["causal_ratio"], pt=p["stat_tail_excess_db"], pf=p["stat_full_excess_db"],
                ac=p["actual_SF_excess_db"], k=p["kappa"], co=p["cos_actual_vs_stat_full"]))
        lines.append("")
    with open(os.path.join(HERE, "RESULTS_sfM.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cmd = sys.argv[1]
    if cmd == "run":
        src, M = sys.argv[2], int(sys.argv[3])
        B = int(sys.argv[4]) if len(sys.argv) > 4 else None
        res = run(src, M, B=B)
        tag = f"{src}_M{M}_B{res['B']}"
        save(res, f"sfM_{tag}.json")
    elif cmd == "control":
        M, K = int(sys.argv[2]), int(sys.argv[3])
        res = run("a145", M, hp_taps=K)
        save(res, f"sfM_a145_M{M}_B{res['B']}_hp{K}.json")
    elif cmd == "ctrl_len":
        out = control_length(int(sys.argv[2]) if len(sys.argv) > 2 else 1024)
        print(json.dumps(out, indent=1))
        save(dict(control_length=out), "ctrl_len.json")
    elif cmd == "report":
        report()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
