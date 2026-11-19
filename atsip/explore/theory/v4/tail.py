"""Check of the review comment on Lemma F: with ideal whitening the rebuilt-frame SF update is
    v+ = (I - a G_n) v,   G_n = F_M^{-1} (U_n^T X_n) / Me,
U = regressors of the whitened reference u, X = regressors of the filtered reference r = F u.
G_n = F_M^{-1} A_u F_M + (tail term from the part of F v beyond lag M), A_u = U^T U / Me.
Compare the recursion with G_n (actual secondary path) against the pure sample-covariance model.
Run: python3 tail.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "closedform"))
import numpy as np
from scipy import linalg, signal
from numpy.lib.stride_tricks import sliding_window_view
from base import minphase, S2, G as GG
M, B, Me = 512, 128, 512
rng = np.random.default_rng(3)
f = np.fft.irfft(minphase(S2, GG), GG)[:GG // 2]; f = f / np.sqrt(np.sum(f ** 2))
FM = linalg.toeplitz(f[:M], np.zeros(M))
tail_energy = 1 - np.cumsum(f ** 2)[M - 1]
print(f"F: energy beyond lag M-1 = {tail_energy:.2e}; |f| peak at lag {np.argmax(np.abs(f))}; energy in lags 0..15: {np.sum(f[:16]**2):.3f}")
def regs(x, end, L):
    return sliding_window_view(x, M)[end - L - (M - 1):end - (M - 1)][:, ::-1]
nb = 64
for a in (0.25, 0.375, 0.5, 0.55, 0.625, 0.75):
    res = {"sample-cov": [], "SF (U^T X, v coords)": []}
    for run in range(3):
        u = rng.standard_normal(nb * B + Me + 2 * M + len(f)); r = signal.lfilter(f, [1.0], u)
        off = len(f)
        v1 = np.ones(M) / np.sqrt(M); v2 = v1.copy(); t1, t2 = [], []
        Phi_w = lambda v: np.sum((FM @ v) ** 2)
        for n in range(nb):
            end = off + Me + 2 * M + n * B - M
            U, X = regs(u, end, Me), regs(r, end, Me)
            y = FM @ v1; y = y - a * (U.T @ (U @ y)) / Me; v1 = linalg.solve_triangular(FM, y, lower=True)
            v2 = v2 - a * linalg.solve_triangular(FM, U.T @ (X @ v2), lower=True) / Me
            t1.append(10 * np.log10(Phi_w(v1) + 1e-300)); t2.append(10 * np.log10(Phi_w(v2) + 1e-300))
        res["sample-cov"].append([t1[15], t1[31], t1[63]]); res["SF (U^T X, v coords)"].append([t2[15], t2[31], t2[63]])
    print(f"a={a:5.3f}: " + " | ".join(f"{k}: " + " ".join(f"{x:7.1f}" for x in np.mean(v, axis=0)) for k, v in res.items()))
