"""Compiled per-sample loop for the time-domain external methods (same recursion as final.run_baseline with the
classes of algorithms/algorithms.py; results agree to rounding).  Used by final.run_baseline when numba is
installed and ANC_NO_JIT is not set; otherwise the Python loop runs.

  python fastbase.py check     compare with the Python loop for the six methods (with and without a path switch)
"""
import math
import os
import sys
import time

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:                                    # numba missing: final.run_baseline falls back to the Python loop
    HAVE_NUMBA = False

    def njit(*a, **k):
        return lambda f: f

KIND = {"FxNLMS": 0, "Th_FxNLMS": 1, "FxlogNLMS": 2, "MFxLMM": 3, "FxNFLOC": 4, "EVSS_FxeLMS": 5}


def pack(name, p):
    """Parameters in the order the kernel reads them; defaults as in the classes."""
    g = lambda k, v: float(p.get(k, v))
    if name == "FxNLMS":
        return [g("eps", 1e-3)]
    if name == "Th_FxNLMS":
        return [g("c1", -200.0), g("c2", 200.0), g("eps", 1e-3)]
    if name == "FxlogNLMS":
        return [g("G_l", 1e4), g("eps", 1e-3)]
    if name == "MFxLMM":
        return [g("gzai_l", 5e-2), g("lmd_l", 0.97), g("eps_l", 1e-3), g("sigma_l", 1.0)]
    if name == "FxNFLOC":
        return [g("p", 1.0), g("eps", 1e-8)]
    if name == "EVSS_FxeLMS":
        return [g("k", 1.0), g("alpha", 1.0), g("lmd_e", 0.99), g("gamma", 1e-3), g("Ee_init", 0.0)]
    raise KeyError(name)


@njit(cache=True)
def _loop(kind, par, mu, d, xr, rr, sw, hrevs, M, B, pref, div_gain):
    T_ = d.shape[0]; L = hrevs.shape[1]
    w = np.zeros(M); e = np.zeros(T_); y2 = np.zeros(2 * L)
    head = 0; seg = 0; hrev = hrevs[0]
    div = False; n_done = T_; fail_n = -1
    state = 0.0
    if kind == 3:
        state = par[3]
    elif kind == 5:
        state = par[4]
    two_over_sqrt_pi = 4.0 / math.sqrt(math.pi)
    for n in range(T_):
        if seg + 1 < sw.shape[0] and n == sw[seg + 1]:
            seg += 1; hrev = hrevs[seg]
        a = T_ - 1 - n
        xv = xr[a:a + M]; rv = rr[a:a + M]
        y = np.dot(w, xv)
        y2[head] = y; y2[head + L] = y
        head = head + 1 if head + 1 < L else 0
        en = d[n] + np.dot(y2[head:head + L], hrev)
        e[n] = en
        finite = math.isfinite(en)
        if finite:
            if kind == 0:
                s = (mu / (np.dot(rv, rv) + par[0])) * en
                for k in range(M):
                    w[k] -= s * rv[k]
            elif kind == 1:
                ec = par[0] if en < par[0] else (par[1] if en > par[1] else en)
                s = (mu / (np.dot(rv, rv) + par[2])) * ec
                for k in range(M):
                    w[k] -= s * rv[k]
            elif kind == 2:
                aa = par[0] * abs(en)
                if aa <= 0.0:
                    ec = 0.0
                else:
                    el = math.log(aa)
                    if el < 0.0:
                        el = 0.0
                    ec = math.copysign(el / aa, en)
                s = (mu / (np.dot(rv, rv) + par[1])) * ec
                for k in range(M):
                    w[k] -= s * rv[k]
            elif kind == 3:
                gz = par[0]; d1 = 3.0 * gz; d2 = 4.0 * gz
                ae = abs(en); sgn = 1.0 if en >= 0.0 else -1.0
                if ae < gz:
                    psi = en
                elif ae < d1:
                    psi = gz * sgn
                elif ae < d2:
                    psi = gz * sgn * (d2 - ae) / (d2 - d1)
                else:
                    psi = 0.0
                state = par[1] * state + (1.0 - par[1]) * (ae * ae)
                s = (mu / (np.dot(rv, rv) / M + state + par[2])) * psi
                for k in range(M):
                    w[k] -= s * rv[k]
            elif kind == 4:
                p = par[0]; pf = 0.5 * p; pi_ = 2.0 / max(p, 1e-12)
                ae = abs(en)
                ep = (ae ** pf) * (1.0 if en >= 0.0 else -1.0) if ae > 0.0 else 0.0
                den = par[1]
                if p == 1.0:
                    for k in range(M):
                        den += abs(rv[k])
                    s = mu * ep / den
                    for k in range(M):
                        wk = w[k]; rk = rv[k]
                        wp = math.copysign(math.sqrt(abs(wk)), wk) if wk != 0.0 else 0.0
                        rp = math.copysign(math.sqrt(abs(rk)), rk) if rk != 0.0 else 0.0
                        wp -= s * rp
                        w[k] = math.copysign(wp * wp, wp) if wp != 0.0 else 0.0
                else:
                    for k in range(M):
                        den += abs(rv[k]) ** p
                    s = mu * ep / den
                    for k in range(M):
                        wk = w[k]; rk = rv[k]
                        wp = math.copysign(abs(wk) ** pf, wk) if wk != 0.0 else 0.0
                        rp = math.copysign(abs(rk) ** pf, rk) if rk != 0.0 else 0.0
                        wp -= s * rp
                        w[k] = math.copysign(abs(wp) ** pi_, wp) if wp != 0.0 else 0.0
            else:
                state = par[2] * state + (1.0 - par[2]) * (en * en)
                norm = np.dot(rv, rv) + state + par[3]
                z = en / par[0]
                phi = two_over_sqrt_pi * math.erf(z) * math.exp(-(z * z))
                s = (mu / norm) * (math.sin(par[1] * abs(en) - math.pi / 2.0) + 1.5) * phi
                for k in range(M):
                    w[k] -= s * rv[k]
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
                    if not math.isfinite(w[k]):
                        bad = True
                        break
        if bad:
            div = True; n_done = (n // B) * B; fail_n = n
            break
    return e, w, div, n_done, fail_n


def run(name, mu, params, dd, n_tot, M, B, pref, div_gain):
    """Arrays for the kernel from the data dictionary of final.build; returns (e, w, div, n_done, fail_n, seconds)."""
    x, d, r = dd["x"], dd["d"], dd["r"]
    T_ = len(d) if n_tot is None else min(n_tot, len(d))
    plant = dd["hS_seq"]
    L = len(plant[0][1])
    assert all(len(h) == L for _, h in plant)
    hrevs = np.ascontiguousarray(np.stack([h[::-1] for _, h in plant]), dtype=np.float64)
    sw = np.array([b0 * B for b0, _ in plant], dtype=np.int64)
    xr = np.concatenate([x[:T_][::-1], np.zeros(M - 1)])
    rr = np.concatenate([r[:T_][::-1], np.zeros(M - 1)])
    t0 = time.perf_counter()
    e, w, div, n_done, fail_n = _loop(KIND[name], np.array(pack(name, params), dtype=np.float64), float(mu),
                                      np.ascontiguousarray(d[:T_], dtype=np.float64), xr, rr, sw, hrevs, M, B, float(pref), float(div_gain))
    return e, w, bool(div), int(n_done), int(fail_n), time.perf_counter() - t0


def check():
    os.environ["ANC_NO_JIT"] = "1"
    import json
    import final as F
    assert HAVE_NUMBA, "numba is not installed"
    tune = json.loads((F.HERE / "TUNE.json").read_text())["chosen"]
    n_tot = 17 * F.BPS * F.B
    worst = 0.0
    for cls, scen in (("a145", "RS-far"), ("G", "RM-10"), ("II300", "RM-10s")):
        dd, _, _ = F.build(cls, scen, 10, F.LS_FINAL)
        for name in list(KIND) + list(F.BLOCK_BASE):
            for mu, par in ((tune[name]["mu"], tune[name]["params"]), F.BASE[name]):
                os.environ["ANC_NO_JIT"] = "1"
                ref = F.run_baseline(name, mu, par, dd, n_tot)
                del os.environ["ANC_NO_JIT"]
                new = F.run_baseline(name, mu, par, dd, n_tot)
                same_div = ref["div"] == new["div"]
                if ref["div"] or new["div"]:
                    ok = same_div and abs(ref["n_done"] - new["n_done"]) <= F.B
                    rel = float("nan")
                else:
                    rel = float(np.max(np.abs(ref["e"] - new["e"])) / np.max(np.abs(ref["e"])))
                    ok = rel < 1e-6; worst = max(worst, rel)
                print(f"{cls:6s} {scen:7s} {name:12s} mu={mu:<9.6g} div {int(ref['div'])}/{int(new['div'])} rel diff {rel:.1e} "
                      f"time {ref['seconds']:.1f} -> {new['seconds']:.2f} s {'ok' if ok else 'MISMATCH'}", flush=True)
                assert ok, (cls, scen, name, mu)
    print(f"fastbase check ok: largest relative difference {worst:.1e}")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "check":
        check()
    else:
        raise SystemExit(__doc__)
