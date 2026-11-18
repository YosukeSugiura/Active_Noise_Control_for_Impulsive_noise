"""Lemma F check (THEORY_v4.md): rebuilt error frame, ideal whitening.
Update v+ = (I - a A_n) v,  A_n = (1/Me) sum_{t in window n} r_t r_t^T,  r_t = (x_t, ..., x_{t-M+1}), x white, E A_n = I,
window = last Me samples, hop B.  Claims:
  (1) spectrum of A_n: lambda_max ~ (1 + sqrt(M/Me))^2, E lambda^2 = 1 + M/Me (per unit direction: E|A v|^2/|v|^2);
  (2) stability of the recursion: a < 2 / lambda_max (overlapping windows do not change the bound);
  (3) split coherence of two disjoint windows of L samples: mismatch -> 1/(1 + M/L); white error noise -> 0 +- 1/sqrt(M).
Run: python3 mp.py   (about a minute)"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
rng = np.random.default_rng(1)
M, B = 512, 128

def regs(x, end, L):
    """Rows r_t for t = end-L .. end-1 (x has M-1 samples of history before index 0 of the window)."""
    return sliding_window_view(x, M)[end - L - (M - 1):end - (M - 1)][:, ::-1]

print("(1) spectrum of A_n")
for Me in (128, 256, 512, 1024, 2048):
    lm, m2 = [], []
    for _ in range(6):
        x = rng.standard_normal(Me + 2 * M); X = regs(x, len(x), Me)
        ev = np.linalg.eigvalsh(X.T @ X / Me); lm.append(ev[-1]); m2.append(np.mean(ev ** 2))
    c = M / Me
    print(f"  Me={Me:5d} c={c:5.2f}: lambda_max {np.mean(lm):.3f} (MP (1+sqrt c)^2 = {(1 + c ** .5) ** 2:.3f}), "
          f"mean lambda^2 {np.mean(m2):.3f} (1 + c = {1 + c:.3f}), bound a < {2 / np.mean(lm):.3f} (MP {2 / (1 + c ** .5) ** 2:.3f})")

print("(2) recursion with sliding windows (hop B): excess after 16/32/64 blocks (dB re initial), 3 runs each")
for Me in (512, 128):
    grid = (0.1, 0.2, 0.3, 0.375, 0.45, 0.5, 0.55, 0.625, 0.75) if Me == 512 else (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)
    for a in grid:
        out = []
        for run in range(3):
            nb = 64; x = rng.standard_normal(nb * B + Me + 2 * M); v = np.ones(M) / np.sqrt(M); tr = []
            for n in range(nb):
                X = regs(x, Me + 2 * M + n * B, Me)
                v = v - a * (X.T @ (X @ v)) / Me; tr.append(10 * np.log10(v @ v + 1e-300))
            out.append([tr[15], tr[31], tr[63]])
        o = np.mean(out, axis=0)
        print(f"  Me={Me} a={a:5.3f}: {o[0]:8.1f} {o[1]:8.1f} {o[2]:8.1f}")

print("(3) split coherence, windows of L samples")
for L in (128, 256, 512):
    cm, cn = [], []
    for _ in range(300):
        x = rng.standard_normal(2 * L + 2 * M); v = rng.standard_normal(M)
        Xa, Xb = regs(x, len(x) - L, L), regs(x, len(x), L)
        da, db = Xa.T @ (Xa @ v), Xb.T @ (Xb @ v)
        cm.append(da @ db / np.sqrt((da @ da) * (db @ db)))
        na, nb_ = Xa.T @ rng.standard_normal(L), Xb.T @ rng.standard_normal(L)
        cn.append(na @ nb_ / np.sqrt((na @ na) * (nb_ @ nb_)))
    print(f"  L={L}: mismatch coh mean {np.mean(cm):.3f} sd {np.std(cm):.3f} (1/(1+M/L) = {1 / (1 + M / L):.3f}); "
          f"noise coh mean {np.mean(cn):+.3f} sd {np.std(cn):.3f} (1/sqrt(M) = {M ** -.5:.3f})")
