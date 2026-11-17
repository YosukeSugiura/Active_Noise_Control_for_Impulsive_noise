#!/usr/bin/env python3
"""Library for the SF-bias study (sfbias.py).

Normaliser variants for the spectral-factor (SF) direction, all sharing one operator:
  Y_j  = FFT_G(Z_j) * conj(R)            (Z_j: N-frame, last M samples = regressor / error column)
  q    = irfft_G(Y / conj(F)); q[trunc:] = 0          (causal split, lags < trunc)
  g    = irfft_G(rfft(q) / F)[:M]        (tail "trunc", as specfac / fp.py)
  g    = causal conv of q[:M] with irfft_G(1/F)[:M], output [:M]   (tail "causalM")
Variant axes (spec dict keys):
  src    "Rall": R = Rall[i] (circular N-grid product, as in fp.py / specfac), Lam = Srr_shared + eps + reg*Pbar
         "zp":   R = FFT_G of the N-sample reference frame zero-padded to G (linear, no circular wrap);
                 Lam = EMA of |FFT_GL(zero-padded frame)|^2 + eps + reg*Pbar
  G      operator grid (N, 2N, 4N); for src "Rall" G = N
  GL     grid on which F is computed (GL >= G); F is sampled back to G bins as F[::GL//G]
  reg    eps_reg (coefficient of Pbar)
  frozen src "Rall" only: Lam from the long-term mean of |Rall|^2 instead of the EMA
  tail   "trunc" | "causalM"
"""
import os
import sys

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

FS, M, N, B, NB = D.FS, D.M, D.N, D.B, D.NB
ALPHA = X.ALPHA
TRUNC_ADAPT = N // 2          # specfac.TRUNC
TINY = 1e-300


def minphase_bins(Lam):
    """Minimum-phase factor on the rfft grid of length G = 2(len-1): |F|^2 = Lam (same formula as fp.py/specfac)."""
    G = 2 * (len(Lam) - 1)
    full = np.concatenate([Lam, Lam[-2:0:-1]])
    c = np.fft.ifft(0.5 * np.log(np.maximum(full, TINY))).real
    c[1:G // 2] *= 2.0
    c[G // 2 + 1:] = 0.0
    return np.exp(np.fft.rfft(c))


def zp(frame, G):
    out = np.zeros(G)
    out[:len(frame)] = frame
    return out


def direction(G, Rg, Fg, Zf, trunc, tail):
    """Zf: rfft_G of the Z columns, shape (G/2+1, C). Returns g with shape (M, C)."""
    Y = Zf * np.conj(Rg)[:, None]
    q = np.fft.irfft(Y / np.conj(Fg)[:, None], G, axis=0)
    q[trunc:] = 0.0
    if tail == "trunc":
        return np.fft.irfft(np.fft.rfft(q, axis=0) / Fg[:, None], G, axis=0)[:M]
    if tail == "causalM":
        h = np.fft.irfft(1.0 / Fg, G)[:M]               # causal 1/F, truncated to M taps
        L2 = 2 * G
        Hq = np.fft.rfft(q[:trunc], L2, axis=0)
        Hh = np.fft.rfft(h, L2)
        return np.fft.irfft(Hq * Hh[:, None], L2, axis=0)[:M]
    raise ValueError(tail)


class Norm:
    """One normaliser variant; call step() once per block (it keeps its own EMA state)."""

    def __init__(self, spec, Lfz=None):
        self.s = dict(spec)
        self.s.setdefault("reg", 0.0)
        self.s.setdefault("tail", "trunc")
        self.s.setdefault("frozen", False)
        self.Srr = None
        self.Lfz = Lfz

    def step(self, rN, Rall_i, sh):
        """rN: N-sample reference frame ending at block end. sh: Shared outputs. Returns (G, Rg, Fg)."""
        s = self.s
        if s["src"] == "W":                       # geometric a=0 reference: Lam = Pbar + eps, no causal split
            return N, Rall_i, sh["Pbar"] + sh["eps"]
        if s["src"] == "Rall":
            G = N
            if s["frozen"]:
                Lsrc = self.Lfz
                Pb = float(np.mean(self.Lfz))
            else:
                Lsrc = sh["Srr"]
                Pb = sh["Pbar"]
            if s.get("flat"):                          # Lam = Pbar: flat F, the a=0 shape
                Lsrc = np.full(NB, Pb)
            elif s.get("a") is not None:               # Lam = Lfz^a Pbar^(1-a)
                Lsrc = Lsrc ** s["a"] * Pb ** (1 - s["a"])
            Lam = Lsrc + sh["eps"] + s["reg"] * Pb
            return G, Rall_i, minphase_bins(Lam)
        G, GL = s["G"], s["GL"]
        Pw = np.abs(np.fft.rfft(zp(rN, GL))) ** 2
        self.Srr = Pw if self.Srr is None else (1 - ALPHA) * self.Srr + ALPHA * Pw
        Pb = float(np.mean(self.Srr))
        Lam = self.Srr + sh["eps"] + s["reg"] * Pb
        FL = minphase_bins(Lam)
        Fg = FL[:: GL // G][: G // 2 + 1]
        Rg = np.fft.rfft(zp(rN, G))
        return G, Rg, Fg


def W_direction(Rg, Zf, lam):
    return np.fft.irfft(Zf * np.conj(Rg)[:, None] / lam, N, axis=0)[:M]


def dir_any(nm, G, Rg, Fg, Zf, trunc):
    if nm.s["src"] == "W":
        return W_direction(Rg, Zf, Fg)
    return direction(G, Rg, Fg, Zf, trunc, nm.s["tail"])


def Zcolumns(G, cols):
    """Place the M-sample columns at frame positions N-M..N-1 of a G-length frame (same convention as fp.py)."""
    Z = np.zeros((G, cols.shape[1]))
    Z[N - M:N] = cols
    return Z


def fixed_point_pass(data, specs, step=2, i0=16):
    """Linearised mean fixed point for each spec, one shared pass over the blocks (fp.py conventions:
    error frame = e_opt, sampled blocks i >= i0 with (i - i0) % step == 0)."""
    x, d, hS, w_opt = data["x"], data["d"], data["hS"], data["w_opt"]
    L = len(d)
    nb = L // B
    r = signal.lfilter(hS, [1.0], x)
    e_opt = d + signal.lfilter(hS, [1.0], np.convolve(x, w_opt)[:L])
    rq = np.concatenate([np.zeros(N), r])
    Rall = ML.prepass(data)["Rall"]
    Lfz = np.mean(np.abs(Rall) ** 2, axis=0)
    shared = X.Shared(data["S2"])
    norms = [Norm(s, Lfz) for s in specs]
    A = [np.zeros((M, M)) for _ in specs]
    bv = [np.zeros(M) for _ in specs]
    used = 0
    lidx = np.arange(M)
    for i in range(nb):
        s0 = i * B
        E = ML.frame_E(e_opt, s0, M)
        sh = shared.step(Rall[i], E)
        rN = rq[s0 + B: s0 + B + N]
        sampled = i >= i0 and (i - i0) % step == 0
        if sampled:
            t = np.arange(s0 + B - M, s0 + B)
            Xi = rq[t[:, None] + N - lidx[None, :]]
            eo = e_opt[s0 + B - M: s0 + B]
            cols = np.concatenate([Xi, eo[:, None]], axis=1)   # (M, M+1)
            used += 1
        for k, nm in enumerate(norms):
            G, Rg, Fg = nm.step(rN, Rall[i], sh)
            if sampled:
                Zf = np.fft.rfft(Zcolumns(G, cols), axis=0)
                g = dir_any(nm, G, Rg, Fg, Zf, M)
                A[k] += g[:, :M]
                bv[k] += g[:, M]
    out = []
    anr1_floor = anr1(e_opt, d)
    for k, spec in enumerate(specs):
        Am, bm = A[k] / used, bv[k] / used
        v = -np.linalg.solve(Am, bm)
        ex = signal.lfilter(v, [1.0], r)
        e = e_opt + ex
        out.append(dict(name=spec["name"],
                        excess_db=10 * np.log10(np.mean(ex[L // 2:] ** 2) / np.mean(e_opt[L // 2:] ** 2)),
                        L1_db=anr1(e, d), L1_floor_db=anr1_floor))
    return out


def anr1(e, d):
    L = len(d)
    return float(20 * np.log10(np.abs(e[L // 2:]).mean() / np.abs(d[L // 2:]).mean()))


def sim_variant(data, Rall, spec, mu, trunc=TRUNC_ADAPT):
    """Adaptive SF loop (specfac.sim_sf conventions, no ramp). spec as in Norm. Returns e, w, diverged."""
    d, xq, hS, S2, x = data["d"], data["xq"], data["hS"], data["S2"], data["x"]
    r = signal.lfilter(hS, [1.0], x)
    rq = np.concatenate([np.zeros(N), r])
    L = len(d)
    nb = L // B
    swv = sliding_window_view(xq, M)
    w = np.zeros(M)
    zi = np.zeros(len(hS) - 1)
    e_all = np.zeros(L)
    shared = X.Shared(S2)
    Lfz = np.mean(np.abs(Rall) ** 2, axis=0)
    nm = Norm(spec, Lfz)
    diverged = False
    for i in range(nb):
        s0 = i * B
        Xb = swv[s0 + N - M + 1: s0 + N - M + 1 + B][:, ::-1]
        y = Xb @ w
        ys, zi = signal.lfilter(hS, [1.0], y, zi=zi)
        e_all[s0:s0 + B] = d[s0:s0 + B] + ys
        E = ML.frame_E(e_all, s0, M)
        sh = shared.step(Rall[i], E)
        rN = rq[s0 + B: s0 + B + N]
        G, Rg, Fg = nm.step(rN, Rall[i], sh)
        Zf = np.fft.rfft(Zcolumns(G, e_all[s0 + B - M: s0 + B][:, None]), axis=0)
        g = dir_any(nm, G, Rg, Fg, Zf, trunc)[:, 0]
        w = w - 2 * mu * B / M * g
        if not np.isfinite(w).all():
            e_all[s0 + B:] = np.nan
            diverged = True
            break
    return dict(e=e_all, w=w, diverged=diverged)


# ----------------------------------------------------------------------------
# Standard variant sets
# ----------------------------------------------------------------------------
def fp_specs():
    return [
        dict(name="geometric a=0 (Lam=Pbar), reference", src="W"),
        dict(name="base Rall eps_shared (= fp.py ER)", src="Rall"),
        dict(name="frozen F (long-term mean |Rall|^2)", src="Rall", frozen=True),
        dict(name="eps_reg 1e-3 Pbar", src="Rall", reg=1e-3),
        dict(name="eps_reg 1e-2 Pbar", src="Rall", reg=1e-2),
        dict(name="tail causalM (Rall, N grid)", src="Rall", tail="causalM"),
        dict(name="diag: flat F (Lam=Pbar), same causal op", src="Rall", flat=True),
        dict(name="diag: frozen a=0.5 (Lfz^0.5 Pbar^0.5)", src="Rall", frozen=True, a=0.5),
        dict(name="control: N grid, zp N-frame R (no padding)", src="zp", G=N, GL=N),
        dict(name="F from 4N grid, sampled to N", src="zp", G=N, GL=4 * N),
        dict(name="F from 8N grid, sampled to N", src="zp", G=N, GL=8 * N),
        dict(name="causal split on 2N grid", src="zp", G=2 * N, GL=2 * N),
        dict(name="causal split on 4N grid", src="zp", G=4 * N, GL=4 * N),
        dict(name="2N grid + causalM tail", src="zp", G=2 * N, GL=2 * N, tail="causalM"),
        dict(name="4N grid + causalM tail", src="zp", G=4 * N, GL=4 * N, tail="causalM"),
    ]
