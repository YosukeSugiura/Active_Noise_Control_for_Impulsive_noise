"""Decay/growth boundary of the SF recursion v+ = v - a*F_M^{-1} U^T X v / Me for different inputs.
A: white Gaussian u; B: AR(1)-colored f; C: alpha-stable u (C1 plain step, C2 step / max(g_n, 0.55),
C3 step / (Pbar_n max(g_n, 0.55)), where Pbar_n, the running average of the frame power, stands for the level of the
spectral factor that the algorithm estimates from the same average).
tail_inputs20.py: 20 runs, cases A, B, C3 only. Run: python3 tail_inputs20.py > tail_inputs20.out"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "closedform"))
import numpy as np
from scipy import linalg, signal, stats
from numpy.lib.stride_tricks import sliding_window_view
from base import minphase, S2, G as GG
M, B, Me, nb, NR, WU = 512, 128, 512, 64, 20, 64
AS = (0.125, 0.25, 0.375, 0.455, 0.5, 0.55, 0.625, 0.75)
w = np.linspace(0, np.pi, GG // 2 + 1)
def make_f(p=None):
    S = S2 if p is None else S2 / np.abs(1 - p * np.exp(-1j * w)) ** 2
    f = np.fft.irfft(minphase(S, GG), GG)[:GG // 2]; return f / np.sqrt(np.sum(f ** 2))
def regs(x, end, L): return sliding_window_view(x, M)[end - L - (M - 1):end - (M - 1)][:, ::-1]
def run(f, FM, u, a, norm):
    """u has WU warm-up blocks in front; returns Phi_w(v_n)/Phi_w(v_0) in dB, n = 1..nb (inf if not finite)."""
    r = signal.lfilter(f, [1.0], u); off = len(f) + Me + 2 * M - M
    ends = [off + (n + WU) * B for n in range(-WU, nb)]  # block n=-WU..nb-1 -> end index
    P = np.array([np.mean(r[e - Me:e] ** 2) for e in ends]); Pb = np.median(P[:WU])
    v = np.ones(M) / np.sqrt(M); ref = np.sum((FM @ v) ** 2); out = []
    with np.errstate(all="ignore"):
        for n in range(nb):
            e = ends[WU + n]; g = P[WU + n] / Pb; step = a / (Pb * max(g, 0.55)) if norm == 2 else a / max(g, 0.55) if norm else a
            if np.all(np.isfinite(v)):
                v = v - step * linalg.solve_triangular(FM, regs(u, e, Me).T @ (regs(r, e, Me) @ v), lower=True, check_finite=False) / Me
            Pb = 0.9 * Pb + 0.1 * P[WU + n]
            phi = np.sum((FM @ v) ** 2) if np.all(np.isfinite(v)) else np.inf
            out.append(10 * np.log10(phi / ref) if np.isfinite(phi) and phi > 0 else (np.inf if not np.isfinite(phi) else -300.0))
    return np.array(out)
white = lambda rng, L: rng.standard_normal(L)
stable = lambda al: (lambda rng, L: stats.levy_stable.rvs(al, 0.0, size=L, random_state=rng))
f0 = make_f(); seeds = list(range(100, 100 + NR))
CASES = [("A. white Gaussian, original f", None, "white", 0, False),
         ("B. white Gaussian u, f = minphase(S2/|1-p e^-jw|^2), p=0.9", 0.9, "white", 0, False),
         ("B. white Gaussian u, f = minphase(S2/|1-p e^-jw|^2), p=0.98", 0.98, "white", 0, False)]
for al in (1.45, 1.20): CASES.append((f"C3. alpha-stable alpha={al}, step / (Pbar_n max(g_n, 0.55))", None, "stable", al, 2))
_cache = {}
def task(args):
    ci, seed = args; name, p, kind, al, norm = CASES[ci]
    if ci not in _cache:
        f = make_f(p); _cache[ci] = (f, linalg.toeplitz(f[:M], np.zeros(M)))
    f, FM = _cache[ci]
    u = (white if kind == "white" else stable(al))(np.random.default_rng(seed), (nb + WU) * B + Me + 2 * M + len(f))
    return ci, seed, np.array([run(f, FM, u, a, norm)[[15, 31, 63]] for a in AS])  # (len(AS), 3)
if __name__ == "__main__":
    from multiprocessing import Pool
    for lab, f in (("original", f0), ("p=0.9", make_f(0.9)), ("p=0.98", make_f(0.98))):
        print(f"f ({lab}): energy beyond lag M-1 = {1 - np.cumsum(f ** 2)[M - 1]:.2e}")
    jobs = [(ci, s) for ci in range(len(CASES)) for s in seeds]
    with Pool(os.cpu_count()) as pool: res = pool.map(task, jobs, chunksize=1)
    for ci, (name, *_r) in enumerate(CASES):
        R = np.array([r[2] for r in sorted(res, key=lambda r: r[1]) if r[0] == ci])  # (NR, len(AS), 3)
        print(f"\n== {name}  (seeds {seeds[0]}..{seeds[-1]}; mean over {NR} runs of Phi_w [dB] rel. block 0)")
        print("    a     blk16   blk32   blk64   grow  decay | per-run blk64 (seeds in order)")
        dec20, gro_any = [], []
        for k, a in enumerate(AS):
            x = R[:, k, :]; mu = x.mean(axis=0)
            grow = int(np.sum(~(x[:, 2] <= 0))); decay = int(np.sum((x[:, 2] < x[:, 0]) & (x[:, 2] < 0)))
            if decay == NR: dec20.append(a)
            if grow >= 1: gro_any.append(a)
            print(f"{a:6.3f} " + " ".join(f"{v:7.1f}" for v in mu) + f"  {grow:2d}/{NR}  {decay:2d}/{NR} | " + " ".join(f"{v:.1f}" for v in x[:, 2]), flush=True)
        print(f"largest a with {NR}/{NR} decay: {max(dec20) if dec20 else None};  smallest a with >=1 growing run: {min(gro_any) if gro_any else None}")
