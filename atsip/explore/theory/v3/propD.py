"""Proposition D / E checks: delayed mean recursion v+ = v - a sum_k c_k v_{-k}, structural constants.
Run: python3 propD.py"""
import numpy as np, itertools
def comp(a, c):
    K = len(c); A = np.zeros((K, K)); A[0, :] = -a * np.asarray(c); A[0, 0] += 1.0
    A[1:, :-1] += np.eye(K - 1); return A
rho = lambda a, c: np.abs(np.linalg.eigvals(comp(a, c))).max()
def astar(c):
    g = np.arange(1e-4, 2.2, 1e-4); r = np.array([rho(a, c) for a in g]); k = r.argmin()
    lim = g[np.argmax(r >= 1)] if (r >= 1).any() else np.nan
    return g[k], r[k], lim
print("uniform c_k = 1/m (m = Me/B): a*, rho*, mean stability limit, 1/e blocks")
for m in (1, 2, 3, 4, 6, 8):
    a, r, lim = astar(np.ones(m) / m)
    print(f"  m={m}: a*={a:.4f} rho*={r:.4f} limit={lim:.3f} blocks/e={(-1/np.log(r) if r>0 else 0):.2f}  a*m={a*m:.3f}")
print("measured / alternative delay kernels")
for nm, c in (("measured Phi i", [0.238, 0.247, 0.247, 0.247, 0.010]), ("measured E iii-like", [0.206, 0.219, 0.219, 0.219, 0.013]),
              ("pure delay 1", [0, 1]), ("pure delay 2", [0, 0, 1]), ("pure delay 3", [0, 0, 0, 1]),
              ("mean delay 2.6 (0.4@2, 0.6@3)", [0, 0, 0.4, 0.6]), ("uniform 6 (mean 2.5)", np.ones(6) / 6)):
    c = np.asarray(c, float) / np.sum(c); a, r, lim = astar(c)
    print(f"  {nm}: mean delay {np.arange(len(c)) @ c:.2f} a*={a:.4f} rho*={r:.4f} limit={lim:.3f} | rho at a=0.161: {rho(0.161, c):.4f}")

# second-moment model of protoC.dopt_curve (N = 0): mean with delay + p_{t+1} = rho^2 p + mu^2 fK h^2 x
def contraction(a, c, fK, T=8):
    K = len(c); m = np.ones(K); p = 0.0; r2 = rho(a, c) ** 2
    for _ in range(T):
        x = m[0] ** 2 + p; mn = m[0] - a * (m @ c); p = r2 * p + a * a * fK * x; m = np.r_[mn, m[:-1]]
    return (m[0] ** 2 + p) ** (1.0 / T)
c4 = np.ones(4) / 4; g = np.arange(0.01, 0.6, 0.001)
for fK, nm in ((8.0, "full: 1+(2tau-1)f = 2M/B = 8"), (7.0, "excl1: (2tau-1)f = 7"), (0.0, "mean only")):
    for T in (8, 16, 32):
        J = np.array([contraction(a, c4, fK, T) for a in g]); k = J.argmin()
        print(f"Me=M model {nm}, T={T}: argmin a={g[k]:.3f} (mu={g[k]/0.2375:.3f} at h_eff 0.2375, {g[k]/0.25:.3f} at h 0.25), per-block factor of x {J[k]:.3f}; at a=0.161: {contraction(0.161, c4, fK, T):.3f}")
# Me = B: no delay, tau = 1, f = M/Me: x+ = x (1 - 2a + (1+f) a^2)
for f in (4.0,):
    a = 1 / (1 + f); print(f"Me=B (no delay), f={f}: a*={a:.3f}, per-block factor {1 - 2*a + (1+f)*a*a:.3f}")
# E: switching gains a_n in {0, a, kap*a}: worst periodic product (length <= Lmax), numerical instability threshold
def worst(a, c, kaps, Lmax=9):
    Ms = [comp(k * a, c) for k in kaps]; w = 0.0
    for L in range(1, Lmax + 1):
        for seq in itertools.product(range(len(Ms)), repeat=L):
            P = np.eye(len(c))
            for i in seq: P = Ms[i] @ P
            w = max(w, np.abs(np.linalg.eigvals(P)).max() ** (1.0 / L))
    return w
for kaps in ((1.0,), (0.0, 1.0, 2.0), (0.5, 1.0, 2.0)):
    lo, hi = 0.05, 2.0
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if worst(mid, c4, kaps, Lmax=8) <= 1 + 1e-9: lo = mid
        else: hi = mid
    print(f"gain multipliers {kaps}: no growing periodic product (length <= 8) for a <= {lo:.3f}; worst rate at a=0.161: {worst(0.161, c4, kaps, 8):.3f}, at a=0.375: {worst(0.375, c4, kaps, 8):.3f}")
