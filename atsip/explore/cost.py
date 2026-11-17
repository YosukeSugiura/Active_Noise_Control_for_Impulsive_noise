#!/usr/bin/env python3
"""Per-block cost of the frequency-domain adaptation and the delayless time-domain filtering.

Setting: M = 512, N = 2M = 1024 (overlap-save FFT length), B = 128, fs = 16 kHz.
Methods (same NumPy/scipy.fft style, one block update each):
  FD0      plain FD-FxNLMS, fullband (scalar normalisation, a = 0)
  B0       binwise normalisation by the EMA of |R|^2 (a = 1)
  W05      geometric a = 0.5 (divide by sqrt of the EMA)
  GS       geometric with a ramp: a(b) = min(1, 0.5 + 0.5 b / 1000), power computed every block
  SF       spectral factor F from the EMA via cepstrum every block, SF operator
  SF8      as SF, but F recomputed every 8 blocks
  APSIPA   Freq_FxNLMS_CohVSS per-block update (kernel form of anc-fast fastsim/kernels_fd.py,
           whose update() is pure NumPy; numba is only used for the time-domain fill_block)
Filtering: delayless time-domain filter, 512 MACs per sample, one (B x M) matrix-vector per block.

Two timed modes per method:
  update          FFT update only, errors taken from a precomputed stream
  update+filter   filter the block (y), e = d - y, then the FFT update
plus a filter-only row. Each (method, mode) run has NWARM warm-up blocks and NMEAS timed blocks,
repeated ROUNDS times (rounds are interleaved over methods). Per round: median and p90 of the
per-block time; reported values are medians over rounds.

Usage: python3 atsip/explore/cost.py     (writes atsip/explore/out/cost_bench.json)
"""
import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import platform
import time
import numpy as np
import scipy
import scipy.fft as sfft
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "out")

M, N, B, FS = 512, 1024, 128, 16000
K = N // 2 + 1
EPS = 1e-6
ALPHA = 0.05          # EMA weight of the spectral estimates (as rg_alpha in the class)
MU = 0.1              # step size (timing only; the value does not change the work per block)
RAMP = 1000           # GS ramp length in blocks
NWARM = 300
NMEAS = 4000
ROUNDS = 3
MODES = ("update", "update+filter")
METHODS = ("FD0", "B0", "W05", "GS", "SF", "SF8", "APSIPA")
P = N                 # buffer offset so that every window exists at the first block
MU_BLK = 2.0 * MU * B / M

# Theoretical per-block counts (excluding the delayless filter), N-point FFTs and K = N/2+1 bin ops.
# FFT: count of N-point rfft/irfft. cmul: complex multiplies/divides on K bins. tr: real transcendental
# calls on K-vectors (log/exp/sqrt/pow). extra_cep: FFTs only needed to build the spectral factor.
THEORY = {
    "FD0":    dict(fft=3.00, cmul=2, tr="none",                       extra_cep=0.0),
    "B0":     dict(fft=3.00, cmul=2, tr="none (EMA + divide)",        extra_cep=0.0),
    "W05":    dict(fft=3.00, cmul=2, tr="sqrt x1",                    extra_cep=0.0),
    "GS":     dict(fft=3.00, cmul=2, tr="pow x1 (float exponent)",    extra_cep=0.0),
    "SF":     dict(fft=7.00, cmul=4, tr="log x1, exp x1",             extra_cep=2.0),
    "SF8":    dict(fft=5.25, cmul=4, tr="log x1/8, exp x1/8 (avg)",   extra_cep=0.25),
    "APSIPA": dict(fft=3.00, cmul=3, tr="pow x3, exp x1",             extra_cep=0.0),
}


def load_hS():
    h = np.loadtxt(os.path.join(ROOT, "impulse_response/secondary_ir.dat"), ndmin=1).astype(float)
    out = np.zeros(128)
    m = min(128, len(h))
    out[:m] = h[:m]
    return out


HS = load_hS()
HC = sfft.rfft(HS, n=N)


# ----------------------------------------------------------------------------
# Per-block updates. Each gets the N-sample x window xw, the M-sample error window ew,
# the block index b, and updates w (length M) in place.
# ----------------------------------------------------------------------------
def init_state(name):
    st = dict(ep=np.zeros(N), Srr=np.zeros(K), Sre=np.zeros(K, complex), See=np.zeros(K), F=None)
    if name == "APSIPA":
        st["Srr"] = np.zeros(K)
        st["See"] = np.zeros(K)
    return st


def spec_factor(Lam):
    c = sfft.irfft(0.5 * np.log(Lam), N)          # real cepstrum of the log-magnitude
    c[1:N // 2] *= 2.0
    c[N // 2 + 1:] = 0.0
    return np.exp(sfft.rfft(c))                   # minimum-phase factor, K bins


def upd_fd0(st, w, xw, ew, b):
    Rf = HC * sfft.rfft(xw)
    st["ep"][M:] = ew
    Ef = sfft.rfft(st["ep"])
    p = np.mean(np.abs(Rf) ** 2) + EPS
    G = np.conj(Rf) * Ef / p
    w -= MU_BLK * sfft.irfft(G, N)[:M]


def upd_b0(st, w, xw, ew, b):
    Rf = HC * sfft.rfft(xw)
    st["ep"][M:] = ew
    Ef = sfft.rfft(st["ep"])
    st["Srr"] = (1 - ALPHA) * st["Srr"] + ALPHA * np.abs(Rf) ** 2
    G = np.conj(Rf) * Ef / (st["Srr"] + EPS)
    w -= MU_BLK * sfft.irfft(G, N)[:M]


def upd_w05(st, w, xw, ew, b):
    Rf = HC * sfft.rfft(xw)
    st["ep"][M:] = ew
    Ef = sfft.rfft(st["ep"])
    st["Srr"] = (1 - ALPHA) * st["Srr"] + ALPHA * np.abs(Rf) ** 2
    G = np.conj(Rf) * Ef / np.sqrt(st["Srr"] + EPS)
    w -= MU_BLK * sfft.irfft(G, N)[:M]


def upd_gs(st, w, xw, ew, b):
    Rf = HC * sfft.rfft(xw)
    st["ep"][M:] = ew
    Ef = sfft.rfft(st["ep"])
    st["Srr"] = (1 - ALPHA) * st["Srr"] + ALPHA * np.abs(Rf) ** 2
    a = min(1.0, 0.5 + 0.5 * b / RAMP)
    G = np.conj(Rf) * Ef * np.power(st["Srr"] + EPS, -a)
    w -= MU_BLK * sfft.irfft(G, N)[:M]


def _upd_sf(st, w, xw, ew, b, every):
    Rf = HC * sfft.rfft(xw)
    st["ep"][M:] = ew
    Ef = sfft.rfft(st["ep"])
    st["Srr"] = (1 - ALPHA) * st["Srr"] + ALPHA * np.abs(Rf) ** 2
    if b % every == 0 or st["F"] is None:
        st["F"] = spec_factor(st["Srr"] + EPS)
    F = st["F"]
    Ce = np.conj(Rf) * Ef
    q = sfft.irfft(Ce / np.conj(F), N)
    q[M:] = 0.0
    G = sfft.irfft(sfft.rfft(q) / F, N)[:M]
    w -= MU_BLK * G


def upd_sf(st, w, xw, ew, b):
    _upd_sf(st, w, xw, ew, b, 1)


def upd_sf8(st, w, xw, ew, b):
    _upd_sf(st, w, xw, ew, b, 8)


def upd_apsipa(st, w, xw, ew, b):
    # kernel form of _FreqFxNLMS_CohVSS.update (anc-fast fastsim/kernels_fd.py), default parameters
    a = 0.05
    Rf = HC * sfft.rfft(xw)
    st["ep"][M:] = ew
    Ef = sfft.rfft(st["ep"])
    st["Sre"] = (1 - a) * st["Sre"] + a * (Rf * np.conj(Ef))
    st["Srr"] = (1 - a) * st["Srr"] + a * np.abs(Rf) ** 2
    st["See"] = (1 - a) * st["See"] + a * np.abs(Ef) ** 2
    coh = np.clip(np.abs(st["Sre"]) ** 2 / (st["Srr"] * st["See"] + EPS), 0.0, 1.0)
    pivot = 0.5
    low_rel = np.clip((pivot - coh) / pivot, 0.0, 1.0)
    high_rel = np.clip((coh - pivot) / (1.0 - pivot), 0.0, 1.0)
    mu_low = 1.0 + 0.6 * np.power(low_rel, 0.6)
    mu_high = 1.0 - 0.4 * (1.0 - np.exp(-1.5 * np.power(high_rel, 1.6)))
    msv = np.clip(np.where(coh <= pivot, mu_low, mu_high), 0.6, 1.6)
    gn = 0.55 * (np.mean(st["Srr"]) + EPS) + 0.45 * (st["Srr"] + EPS)
    exc = np.clip((coh - pivot) / (1.0 - pivot), 0.0, 1.0)
    cw = np.clip(1.0 - 0.6 * np.power(exc, 1.6), 0.55, 1.0)
    G = np.conj(Rf) * Ef / gn * cw * msv
    w -= MU_BLK * sfft.irfft(G, N)[:M]


UPD = dict(FD0=upd_fd0, B0=upd_b0, W05=upd_w05, GS=upd_gs, SF=upd_sf, SF8=upd_sf8, APSIPA=upd_apsipa)


# ----------------------------------------------------------------------------
# Benchmark driver
# ----------------------------------------------------------------------------
def make_signals(nblk, seed):
    from scipy.signal import lfilter
    hP = np.loadtxt(os.path.join(ROOT, "impulse_response/primary_ir.dat"), ndmin=1).astype(float)
    rng = np.random.default_rng(seed)
    L = P + (nblk + NWARM) * B + N
    X = np.zeros(L)
    X[P:] = rng.standard_normal(L - P)
    D = np.zeros(L)
    D[P:] = lfilter(hP, [1.0], X[P:])          # disturbance = primary path applied to x (well-posed loop)
    E_pre = np.zeros(L)
    E_pre[P:] = rng.standard_normal(L - P)
    return X, D, E_pre


def run_once(method, mode, X, D, E_pre, SW, nblk):
    """Timed per-block loop. Timed: the filter (update+filter) and the FFT update. Untimed: the
    plant (secondary path applied to y to form the error e = d - s*y) which only simulates the
    physical error microphone; it is outside the timed region in every mode."""
    E_all = E_pre.copy() if mode == "update" else np.zeros_like(X)
    Y_all = np.zeros_like(X)
    SWY = sliding_window_view(Y_all, 128)
    w = np.zeros(M)
    w[:4] = 0.01
    st = init_state(method) if method != "filter" else None
    upd = UPD.get(method)
    times = np.empty(nblk)
    for b in range(NWARM + nblk):
        s0 = P + b * B
        end = s0 + B
        tot = 0
        if mode == "filter":
            t0 = time.perf_counter_ns()
            _ = SW[s0 - M + 1: end - M + 1] @ w[::-1]
            tot = time.perf_counter_ns() - t0
        else:
            if mode == "update+filter":
                t0 = time.perf_counter_ns()
                Y_all[s0:end] = SW[s0 - M + 1: end - M + 1] @ w[::-1]
                tot += time.perf_counter_ns() - t0
                E_all[s0:end] = D[s0:end] + SWY[s0 - 127: end - 127] @ HS[::-1]   # untimed plant, repo sign: e = d + s*y
            t0 = time.perf_counter_ns()
            upd(st, w, X[end - N:end], E_all[end - M:end], b)
            tot += time.perf_counter_ns() - t0
        if b >= NWARM:
            times[b - NWARM] = tot / 1e3
    return times, w


def cpu_model():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main():
    X, D, E_pre = make_signals(NMEAS, seed=1)
    SW = sliding_window_view(X, M)
    res = {}
    for rnd in range(ROUNDS):
        for method in METHODS:
            for mode in MODES:
                t, w = run_once(method, mode, X, D, E_pre, SW, NMEAS)
                assert np.all(np.isfinite(w))
                res.setdefault((method, mode), []).append(t)
        t, _ = run_once("filter", "filter", X, D, E_pre, SW, NMEAS)
        res.setdefault(("filter", "filter"), []).append(t)
        print("round %d done" % (rnd + 1), flush=True)

    rows = {}
    for key, rounds in res.items():
        med = [float(np.median(t)) for t in rounds]
        p90 = [float(np.percentile(t, 90)) for t in rounds]
        rows[key] = dict(median_us=float(np.median(med)), median_us_min=float(min(med)),
                         median_us_max=float(max(med)), p90_us=float(np.median(p90)),
                         p90_us_min=float(min(p90)), p90_us_max=float(max(p90)),
                         n_blocks=int(len(rounds[0])), rounds=ROUNDS)
    blk_ms = B / FS * 1e6
    out = dict(
        env=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                 cpu=cpu_model(), threads=dict(OMP=os.environ.get("OMP_NUM_THREADS"),
                                               OPENBLAS=os.environ.get("OPENBLAS_NUM_THREADS"),
                                               MKL=os.environ.get("MKL_NUM_THREADS")),
                 fft="scipy.fft (pocketfft), workers=1", nproc=os.cpu_count()),
        config=dict(M=M, N=N, B=B, fs=FS, block_period_us=blk_ms, nwarm=NWARM, nmeas=NMEAS, rounds=ROUNDS),
        results={f"{k[0]}|{k[1]}": v for k, v in rows.items()},
        theory=THEORY,
    )
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "cost_bench.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    print("block period B/fs = %.0f us" % blk_ms)
    for mode in MODES + ("filter",):
        print("== mode:", mode)
        for method in METHODS + ("filter",):
            key = (method, mode) if method != "filter" else ("filter", "filter")
            if key not in rows:
                continue
            r = rows[key]
            print("%-7s median %8.1f us (range %.1f-%.1f)  p90 %8.1f us  RTF %.4f" % (
                method, r["median_us"], r["median_us_min"], r["median_us_max"], r["p90_us"],
                r["median_us"] / blk_ms))


if __name__ == "__main__":
    main()
