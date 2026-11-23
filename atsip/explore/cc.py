"""Convex combination filtered-x algorithm (comparison method "C_FXLMS").

C-FXLMS of Ferrer, Gonzalez, de Diego and Pinero, "Convex combination filtered-X algorithms for active noise control
systems," IEEE TASLP 21(1), 2013: Algorithm 1 (conventional filtered-x structure), eqs. (1)-(8), beta = 0.9 and
mu_a = 1 as in their Section V, a(n) limited to [-4, 4].  The paper uses LMS component filters (norm = 0); norm = 1
divides both steps by r^T r + eps (normalized components).  Sign convention of the paper and of this repository:
e = d + h*y.

  1-4   y_i = w_i^T x,  y = lam(n-1) y1 + (1-lam(n-1)) y2               (both filters drive the loudspeaker)
  6-7   yf_i = y_i-history^T h_hat                                       (filtered component outputs)
  8     p = beta p + (1-beta) (yf2 - yf1)^2
  9-10  a += mu_a/p e (yf2 - yf1) lam(n-1)(1-lam(n-1)),  |a| <= 4,  lam(n) = 1/(1+exp(-a))
  13    w1 -= mu1 v {e + (1-lam(n)) (yf1 - yf2)}
  14    w2 -= mu2 v {e + lam(n) (yf2 - yf1)},   v = h_hat * x,  mu2 = ratio * mu1

Returned coefficients are lam*w1 + (1-lam)*w2.  No clipping and no coefficient transfer between the filters.

  python cc.py smoke        short run (G, source switch far, seed 10) and check against FxNLMS with lam fixed at 1
"""
import math
import sys
import time

import numpy as np

import fastbase
from fastbase import njit

NAME = "C_FXLMS"
BASE = (0.3, {"ratio": 1 / 8, "mu_a": 1.0, "beta": 0.9, "a_max": 4.0, "norm": 1.0})   # fast step = FxNLMS published step
RATIOS = (1 / 4, 1 / 16)                                                    # slow step / fast step
NORMS = (1.0, 0.0)                                                          # normalized components / LMS as in the paper
LMS_SCALE = 1e-3                                                            # LMS step grid = this x the NLMS grid


def pack(p):
    g = lambda k, v: float(p.get(k, v))
    return [g("eps", 1e-3), g("ratio", 1 / 8), g("mu_a", 1.0), g("beta", 0.9), g("a_max", 4.0), g("eps_a", 1e-12),
            g("a0", 0.0), g("freeze_a", 0.0), g("norm", 1.0)]


@njit(cache=True)
def _loop(par, mu, d, xr, rr, sw, hrevs, lengths, hm, M, B, pref, div_gain):
    T_ = d.shape[0]; L = hrevs.shape[1]; Lm = hm.shape[0]
    w1 = np.zeros(M); w2 = np.zeros(M); e = np.zeros(T_); lam_log = np.zeros(T_)
    y2 = np.zeros(2 * L); u1 = np.zeros(2 * Lm); u2 = np.zeros(2 * Lm)
    head = 0; hm_ = 0; seg = 0
    div = False; n_done = T_; fail_n = -1
    eps = par[0]; ratio = par[1]; mu_a = par[2]; beta = par[3]; a_max = par[4]; eps_a = par[5]
    a = par[6]; freeze = par[7] != 0.0; norm = par[8] != 0.0
    p = 0.0
    lam = 1.0 / (1.0 + math.exp(-a))
    for n in range(T_):
        if seg + 1 < sw.shape[0] and n == sw[seg + 1]:
            seg += 1
        i = T_ - 1 - n
        xv = xr[i:i + M]; rv = rr[i:i + M]
        c1 = np.dot(w1, xv); c2 = np.dot(w2, xv)
        y = lam * c1 + (1.0 - lam) * c2
        y2[head] = y; y2[head + L] = y
        head = head + 1 if head + 1 < L else 0
        u1[hm_] = c1; u1[hm_ + Lm] = c1; u2[hm_] = c2; u2[hm_ + Lm] = c2
        hm_ = hm_ + 1 if hm_ + 1 < Lm else 0
        lh = lengths[seg]
        en = d[n] + np.dot(y2[head + L - lh:head + L], hrevs[seg, :lh])
        e[n] = en; lam_log[n] = lam
        finite = math.isfinite(en)
        if finite:
            yf1 = np.dot(u1[hm_:hm_ + Lm], hm); yf2 = np.dot(u2[hm_:hm_ + Lm], hm)
            dv = yf2 - yf1
            if not freeze:
                p = beta * p + (1.0 - beta) * dv * dv
                a += (mu_a / (p + eps_a)) * en * dv * lam * (1.0 - lam)
                if a > a_max:
                    a = a_max
                elif a < -a_max:
                    a = -a_max
                lam = 1.0 / (1.0 + math.exp(-a))
            g = mu / (np.dot(rv, rv) + eps) if norm else mu
            s1 = g * (en - (1.0 - lam) * dv); s2 = g * ratio * (en + lam * dv)
            for k in range(M):
                w1[k] -= s1 * rv[k]
                w2[k] -= s2 * rv[k]
        bad = not finite
        if not bad and (n + 1) % B == 0:
            pw = 0.0
            for k in range(n + 1 - B, n + 1):
                pw += e[k] * e[k]
            pw /= B
            if not math.isfinite(pw) or pw > div_gain * pref:
                bad = True
            else:
                for k in range(M):
                    if not (math.isfinite(w1[k]) and math.isfinite(w2[k])):
                        bad = True
                        break
        if bad:
            div = True; n_done = (n // B) * B; fail_n = n
            break
    return e, lam * w1 + (1.0 - lam) * w2, div, n_done, fail_n, lam_log


def run_baseline(F, mu, params, dd, n_tot=None):
    """Same return value as final.run_baseline (plus lam: mixing parameter per sample)."""
    M, B = F.M, F.B
    d, x, r = dd["d"], dd["x"], dd["r"]
    T_ = len(d) if n_tot is None else min(n_tot, len(d))
    plant = dd["hS_seq"]
    lengths = np.array([len(h) for _, h in plant], dtype=np.int64)
    hrevs = np.zeros((len(plant), int(lengths.max())))
    for i, (_, h) in enumerate(plant):
        hrevs[i, :lengths[i]] = h[::-1]
    sw = np.array([b0 * B for b0, _ in plant], dtype=np.int64)
    xr = np.concatenate([x[:T_][::-1], np.zeros(M - 1)])
    rr = np.concatenate([r[:T_][::-1], np.zeros(M - 1)])
    pref = dd.get("primary_power", float(np.mean(d ** 2)))
    hm = np.ascontiguousarray(dd["hSm_seq"][0][1][::-1], dtype=np.float64)          # secondary path model (fixed here)
    t0 = time.perf_counter()
    e, w, div, n_done, fail_n, lam = _loop(np.array(pack(params)), float(mu), np.ascontiguousarray(d[:T_], dtype=np.float64),
                                           xr, rr, sw, hrevs, lengths, hm, M, B, float(pref), float(F.DIV_GAIN))
    failure = None
    if div:
        failure = dict(F.error_peak(e, n_done), sample=int(fail_n))
        e[n_done:] = np.nan
    return dict(e=e, w=w, div=bool(div), n_done=int(n_done), failure=failure, seconds=time.perf_counter() - t0, lam=lam)


def install(F):
    """Make final.py treat CC_FxNLMS as one more external method (call in the parent and in every worker)."""
    F.BASE[NAME] = BASE
    if getattr(F.run_baseline, "_cc", False):
        return
    orig = F.run_baseline

    def run(name, mu, params, dd, n_tot=None):
        return run_baseline(F, mu, params, dd, n_tot) if name == NAME else orig(name, mu, params, dd, n_tot)
    run._cc = True
    F.run_baseline = run


def tune_grid(F):
    mu0, par = BASE
    return [(f"{'n' if nm else 'lms'}x{k:g}|r{q:g}", mu0 * k * (1.0 if nm else LMS_SCALE), dict(par, ratio=q, norm=nm))
            for nm in NORMS for q in RATIOS for k in F.TUNE_FACTORS]


if __name__ == "__main__":
    import final as F
    install(F)
    dd, _, _ = F.build("G", "RS-far", 10, F.LS_FINAL)
    n = 2400 * F.B
    ref = F.run_baseline("FxNLMS", 0.3 / 16, {}, dd, n)
    one = F.run_baseline(NAME, 0.3 / 16, dict(BASE[1], a0=50.0, freeze_a=1.0, norm=1.0), dd, n)       # lam = 1: fast filter only
    print("lam fixed at 1 vs FxNLMS: max |e diff| =", float(np.nanmax(np.abs(one["e"] - ref["e"]))))
    for k in (1 / 16, 1 / 4, 1.0):
        s = F.run_baseline(NAME, 0.3 * k, dict(BASE[1]), dd, n)
        pw = lambda a, b: 10 * np.log10(np.mean(s["e"][a * F.B:b * F.B] ** 2) / np.mean(dd["d"][a * F.B:b * F.B] ** 2))
        print(f"x{k:g}: div={s['div']} NR before change {pw(1600, 1875):.2f} dB, 1-2 s after {pw(2000, 2125):.2f} dB, "
              f"lam mean before {s['lam'][1600 * F.B:1875 * F.B].mean():.2f}, 0-1 s after {s['lam'][1875 * F.B:2000 * F.B].mean():.2f}, {s['seconds']:.1f} s")
