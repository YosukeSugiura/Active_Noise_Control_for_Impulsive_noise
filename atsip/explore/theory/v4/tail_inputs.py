"""Decay/growth boundary of the SF recursion v+ = v - a*F_M^{-1} U^T X v / Me for different inputs.
A: white Gaussian u; B: AR(1)-colored f; C: alpha-stable u (C1 plain step, C2 step / max(g_n, 0.55),
C3 step / (Pbar_n max(g_n, 0.55)), where Pbar_n, the running average of the frame power, stands for the level of the
spectral factor that the algorithm estimates from the same average).
Run: python3 tail_inputs.py > tail_inputs.out   (python3 tail_inputs.py c3 runs case C3 only)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "closedform"))
import numpy as np
from scipy import linalg, signal, stats
from numpy.lib.stride_tricks import sliding_window_view
from base import minphase, S2, G as GG
M, B, Me, nb, NR, WU = 512, 128, 512, 64, 5, 64
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
def case(name, f, gen, seeds, norm=False, per_run=False):
    FM = linalg.toeplitz(f[:M], np.zeros(M)); L = (nb + WU) * B + Me + 2 * M + len(f)
    us = [gen(np.random.default_rng(s), L) for s in seeds]
    print(f"\n== {name}  (seeds {list(seeds)}; mean over {len(seeds)} runs of Phi_w [dB] rel. block 0)")
    print("    a     blk16   blk32   blk64" + ("   per-run blk64 ... | #growth/nonfinite" if per_run else ""))
    m64, m16 = {}, {}
    for a in AS:
        R = np.array([run(f, FM, u, a, norm)[[15, 31, 63]] for u in us]); mu = R.mean(axis=0)
        m16[a], m64[a] = mu[0], mu[2]
        s = f"{a:6.3f} " + " ".join(f"{x:7.1f}" for x in mu)
        if per_run: s += "   " + " ".join(f"{x:7.1f}" for x in R[:, 2]) + f" | {int(np.sum(~(R[:, 2] <= 0)))}/{len(us)}"
        print(s, flush=True)
    dec = [a for a in AS if m64[a] < m16[a] and m64[a] < 0]; gro = [a for a in AS if not m64[a] <= 0]
    print(f"largest a that decays: {max(dec) if dec else None};  smallest a that grows: {min(gro) if gro else None}")
white = lambda rng, L: rng.standard_normal(L)
stable = lambda al: (lambda rng, L: stats.levy_stable.rvs(al, 0.0, size=L, random_state=rng))
f0 = make_f(); seeds = range(100, 100 + NR)
def c3():
    for al in (1.45, 1.20): case(f"C3. alpha-stable alpha={al}, step / (Pbar_n max(g_n, 0.55))", f0, stable(al), seeds, 2, True)
if sys.argv[1:] == ["c3"]: c3(); sys.exit()
for lab, f in (("original", f0), ("p=0.9", make_f(0.9)), ("p=0.98", make_f(0.98))):
    print(f"f ({lab}): energy beyond lag M-1 = {1 - np.cumsum(f ** 2)[M - 1]:.2e}")
case("A. white Gaussian, original f", f0, white, seeds)
for p in (0.9, 0.98): case(f"B. white Gaussian u, f = minphase(S2/|1-p e^-jw|^2), p={p}", make_f(p), white, seeds)
for al in (1.45, 1.20):
    case(f"C1. alpha-stable alpha={al}, plain step", f0, stable(al), seeds, False, True)
    case(f"C2. alpha-stable alpha={al}, step / max(g_n, 0.55)", f0, stable(al), seeds, True, True)
c3()
