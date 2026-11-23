"""Convex combination of two filtered-x NLMS filters (comparison method "CC_FxNLMS").

Structure after Ferrer et al., IEEE TASLP 21(1), 2013 (convex combination filtered-x), with the usual normalised
update of the mixing parameter (Arenas-Garcia et al. 2006; Azpicueta-Ruiz et al. 2008).  Sign convention of this
repository: e = d + s*y, coefficients move against e*r.

  y(n)   = lam*y1(n) + (1-lam)*y2(n),  y_i = w_i^T x(n)                (both filters drive the loudspeaker)
  v_i(n) = w_i^T r(n),  r = s_hat * x                                   (filtered outputs)
  e_i(n) = e(n) - [lam*v1 + (1-lam)*v2] + v_i                           (error each filter would give; uses s_hat)
  w_1   -= mu        /(r^T r + eps) * e_1 * r                           (fast filter)
  w_2   -= mu*ratio  /(r^T r + eps) * e_2 * r                           (slow filter)
  q      = beta*q + (1-beta)*(v1-v2)^2
  a     -= mu_a/(q + eps_a) * e * (v1-v2) * lam*(1-lam),  |a| <= a_max,  lam = 1/(1+exp(-a))

Returned coefficients are lam*w_1 + (1-lam)*w_2.  No clipping and no coefficient transfer between the filters.

  python cc.py smoke        short run (G, source switch far, seed 10) and comparison with FxNLMS at lam fixed
"""
import math
import sys
import time

import numpy as np

import fastbase
from fastbase import njit

NAME = "CC_FxNLMS"
BASE = (0.3, {"ratio": 1 / 8, "mu_a": 1.0, "beta": 0.9, "a_max": 4.0})     # fast step = FxNLMS published step
RATIOS = (1 / 4, 1 / 16)                                                    # slow step / fast step, tuned with the step


def pack(p):
    g = lambda k, v: float(p.get(k, v))
    return [g("eps", 1e-3), g("ratio", 1 / 8), g("mu_a", 1.0), g("beta", 0.9), g("a_max", 4.0), g("eps_a", 1e-12),
            g("a0", 0.0), g("freeze_a", 0.0)]


@njit(cache=True)
def _loop(par, mu, d, xr, rr, sw, hrevs, lengths, M, B, pref, div_gain):
    T_ = d.shape[0]; L = hrevs.shape[1]
    w1 = np.zeros(M); w2 = np.zeros(M); e = np.zeros(T_); y2 = np.zeros(2 * L); lam_log = np.zeros(T_)
    head = 0; seg = 0
    div = False; n_done = T_; fail_n = -1
    eps = par[0]; mu2 = mu * par[1]; mu_a = par[2]; beta = par[3]; a_max = par[4]; eps_a = par[5]
    a = par[6]; freeze = par[7] != 0.0
    q = 0.0
    lam = 1.0 / (1.0 + math.exp(-a))
    for n in range(T_):
        if seg + 1 < sw.shape[0] and n == sw[seg + 1]:
            seg += 1
        i = T_ - 1 - n
        xv = xr[i:i + M]; rv = rr[i:i + M]
        y = lam * np.dot(w1, xv) + (1.0 - lam) * np.dot(w2, xv)
        y2[head] = y; y2[head + L] = y
        head = head + 1 if head + 1 < L else 0
        lh = lengths[seg]
        en = d[n] + np.dot(y2[head + L - lh:head + L], hrevs[seg, :lh])
        e[n] = en; lam_log[n] = lam
        finite = math.isfinite(en)
        if finite:
            v1 = np.dot(w1, rv); v2 = np.dot(w2, rv)
            v = lam * v1 + (1.0 - lam) * v2
            nr = np.dot(rv, rv) + eps
            s1 = (mu / nr) * (en - v + v1); s2 = (mu2 / nr) * (en - v + v2)
            for k in range(M):
                w1[k] -= s1 * rv[k]
                w2[k] -= s2 * rv[k]
            if not freeze:
                dv = v1 - v2
                q = beta * q + (1.0 - beta) * dv * dv
                a -= (mu_a / (q + eps_a)) * en * dv * lam * (1.0 - lam)
                if a > a_max:
                    a = a_max
                elif a < -a_max:
                    a = -a_max
                lam = 1.0 / (1.0 + math.exp(-a))
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
    t0 = time.perf_counter()
    e, w, div, n_done, fail_n, lam = _loop(np.array(pack(params)), float(mu), np.ascontiguousarray(d[:T_], dtype=np.float64),
                                           xr, rr, sw, hrevs, lengths, M, B, float(pref), float(F.DIV_GAIN))
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
    return [(f"x{k:g}|r{q:g}", mu0 * k, dict(par, ratio=q)) for q in RATIOS for k in F.TUNE_FACTORS]


if __name__ == "__main__":
    import final as F
    install(F)
    dd, _, _ = F.build("G", "RS-far", 10, F.LS_FINAL)
    n = 2400 * F.B
    ref = F.run_baseline("FxNLMS", 0.3 / 16, {}, dd, n)
    one = F.run_baseline(NAME, 0.3 / 16, dict(BASE[1], a0=50.0, freeze_a=1.0), dd, n)       # lam = 1: fast filter only
    print("lam fixed at 1 vs FxNLMS: max |e diff| =", float(np.nanmax(np.abs(one["e"] - ref["e"]))))
    for k in (1 / 16, 1 / 4, 1.0):
        s = F.run_baseline(NAME, 0.3 * k, dict(BASE[1]), dd, n)
        pw = lambda a, b: 10 * np.log10(np.mean(s["e"][a * F.B:b * F.B] ** 2) / np.mean(dd["d"][a * F.B:b * F.B] ** 2))
        print(f"x{k:g}: div={s['div']} NR before change {pw(1600, 1875):.2f} dB, 1-2 s after {pw(2000, 2125):.2f} dB, "
              f"lam mean before {s['lam'][1600 * F.B:1875 * F.B].mean():.2f}, 0-1 s after {s['lam'][1875 * F.B:2000 * F.B].mean():.2f}, {s['seconds']:.1f} s")
