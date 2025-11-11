# -*- coding: utf-8 -*-
"""
ANC algorithms with unified helpers (no-alloc design)
- Common helpers: adv, den_l2, sub_scaled_inplace
- TD LMS family uses per-class preallocated tmp buffers sized by params['order_control']
- FD variants avoid concat-based allocs and support double-ring buffering
- FxLMLS avoids unnecessary copy; returns new array

API per algo:
  update(w: np.ndarray, e: float, r_buf: np.ndarray, x_buf: np.ndarray, mu: float) -> (w, state|None)
  state() -> dict

Use:
  from algorithms import make_algo
  algo = make_algo("fxnlms", {"eps":1e-3, "order_control":512})
"""
from __future__ import annotations
import math
import numpy as np
from typing import Dict, Any, Type

# ---- profiler shim ----
try:
    from prof_hooks import PROFILE
except Exception:
    class _NoProf:
        def tic(self, *_a, **_k): return None
        def toc(self, *_a, **_k): return None
    PROFILE = _NoProf()

# ---- helpers ----
def adv(head: int, L: int) -> int:
    head += 1
    return 0 if head == L else head

def den_l2(x: np.ndarray, eps: float) -> float:
    return float(np.dot(x, x)) + eps

def sub_scaled_inplace(w: np.ndarray, r: np.ndarray, scale: float, tmp: np.ndarray) -> None:
    # w -= scale * r  (no allocation)
    np.multiply(r, scale, out=tmp)
    np.subtract(w, tmp, out=w)

# ---- registry ----
REG: Dict[str, Type] = {}
class AlgoBase:
    __aliases__: tuple[str, ...] = ()
    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        REG[cls.__name__.lower()] = cls
        for a in getattr(cls, "__aliases__", ()): REG[a.lower()] = cls
    def state(self) -> Dict[str, Any]: return {}

def make_algo(name: str, params: Dict[str, Any] | None = None):
    cls = REG.get(name.lower())
    if cls is None:
        raise KeyError(f"Unknown algorithm: {name}. Known: {sorted(REG.keys())[:32]} ...")
    return cls({} if params is None else params)

EMPTY: Dict[str, Any] = {}

# -------------------------
# Time-domain family (preallocated tmp buffers)
# -------------------------
class FxNLMS(AlgoBase):
    __slots__ = ("eps", "_tmp")
    __aliases__ = ("nlms", "fxnlms")

    def __init__(self, params=None):
        p = {} if params is None else params
        self.eps = float(p.get("eps", 1e-3))
        W = int(p.get("order_control", 512))
        self._tmp = np.empty(W, dtype=np.float64)

    def update(self, w, e, r_buf, _x_buf, mu):
        PROFILE.tic("FxNLMS:den"); den = den_l2(r_buf, self.eps); PROFILE.toc()
        PROFILE.tic("FxNLMS:update")
        sub_scaled_inplace(w, r_buf, (mu / den) * e, self._tmp)
        PROFILE.toc()
        return w, EMPTY

class BlockFxNLMS(AlgoBase):
    __slots__ = ("eps","BlockSize","cnt","M","grad")
    __aliases__ = ("blockfxnlms",)
    def __init__(self, params=None):
        p = {} if params is None else params
        self.eps = float(p.get("eps", 1e-3))
        self.BlockSize = int(p.get("BlockSize", 64))
        self.cnt = 0
        self.M = int(p.get("M", int(p.get("order_control", 512))))
        self.grad = np.zeros(self.M, dtype=np.float64)
    def update(self, w, e, r_buf, _x_buf, mu):
        PROFILE.tic("BlockFxNLMS:accum")
        self.grad += e * r_buf / den_l2(r_buf, self.eps)
        self.cnt += 1
        PROFILE.toc()
        if self.cnt < self.BlockSize:
            return w, EMPTY
        PROFILE.tic("BlockFxNLMS:apply")
        np.subtract(w, mu * self.grad, out=w)
        self.grad.fill(0.0); self.cnt = 0
        PROFILE.toc()
        return w, EMPTY

class Th_FxNLMS(AlgoBase):
    __slots__ = ("c1","c2","_tmp","_eps")
    __aliases__ = ("th_fxnlms",)
    def __init__(self, params=None):
        p = {} if params is None else params
        self.c1 = float(p.get("c1", -200.0))
        self.c2 = float(p.get("c2", 200.0))
        W = int(p.get("order_control", 512))
        self._tmp = np.empty(W, dtype=np.float64)
        self._eps = float(p.get("eps", 1e-3))
    def update(self, w, e, r_buf, _x_buf, mu):
        e_c = self.c1 if e < self.c1 else self.c2 if e > self.c2 else e
        den = den_l2(r_buf, self._eps)
        sub_scaled_inplace(w, r_buf, (mu/den) * e_c, self._tmp)
        return w, EMPTY

class FxlogNLMS(AlgoBase):
    __slots__ = ("G_l","_tmp","_eps")
    __aliases__ = ("fxlognlms",)
    def __init__(self, params=None):
        p = {} if params is None else params
        self.G_l = float(p.get("G_l", 1e4))
        W = int(p.get("order_control", 512))
        self._tmp = np.empty(W, dtype=np.float64)
        self._eps = float(p.get("eps", 1e-3))
    def update(self, w, e, r_buf, _x_buf, mu):
        a = self.G_l * abs(e)
        if a <= 0.0:
            e_c = 0.0
        else:
            el = math.log(a)
            if el < 0.0: el = 0.0
            e_c = math.copysign(el / a, e)
        den = den_l2(r_buf, self._eps)
        sub_scaled_inplace(w, r_buf, (mu/den) * e_c, self._tmp)
        return w, EMPTY

class FxlogNLMS_plus(AlgoBase):
    __slots__ = ("G_ml","_tmp","_eps")
    __aliases__ = ("fxlognlms_plus", "fxloglms+")
    def __init__(self, params=None):
        p = {} if params is None else params
        self.G_ml = float(p.get("G_ml", 5.0))
        W = int(p.get("order_control", 512))
        self._tmp = np.empty(W, dtype=np.float64)
        self._eps = float(p.get("eps", 1e-3))
    def update(self, w, e, r_buf, _x_buf, mu):
        a = self.G_ml * abs(e) + 1.0
        e_c = math.copysign(math.log(a) / a, e)
        den = den_l2(r_buf, self._eps)
        sub_scaled_inplace(w, r_buf, (mu/den) * e_c, self._tmp)
        return w, EMPTY

class NSS_FxlogLMS_plus(AlgoBase):
    __slots__ = ("G_ml","lmd_e","E_e","_tmp")
    __aliases__ = ("nss_fxloglms_plus",)
    def __init__(self, params=None):
        p = {} if params is None else params
        self.G_ml = float(p.get("G_ml", 5.0))
        self.lmd_e = float(p.get("lmd_e", 0.9999))
        self.E_e = float(p.get("E_e", 0.0))
        W = int(p.get("order_control", 512))
        self._tmp = np.empty(W, dtype=np.float64)
    def update(self, w, e, r_buf, _x_buf, mu):
        ae = abs(e)
        a = self.G_ml * ae + 1.0
        e_c = math.copysign(math.log(a) / a, e)
        self.E_e = self.lmd_e * self.E_e + (1.0 - self.lmd_e) * (ae * ae)
        den = float(np.dot(r_buf, r_buf)) + self.E_e + 1e-3
        sub_scaled_inplace(w, r_buf, (mu/den) * e_c, self._tmp)
        return w, {"E_e": self.E_e}

class NS_FxlogLMS(AlgoBase):
    __slots__ = ("G_l","NS_t","_tmp","_eps")
    __aliases__ = ("ns_fxloglms",)
    def __init__(self, params=None):
        p = {} if params is None else params
        self.G_l = float(p.get("G_l", 1e4))
        self.NS_t = float(p.get("NS_t", 1.0))
        W = int(p.get("order_control", 512))
        self._tmp = np.empty(W, dtype=np.float64)
        self._eps = float(p.get("eps", 1e-3))
    def update(self, w, e, r_buf, _x_buf, mu):
        ae = abs(e)
        if ae > self.NS_t:
            a = self.G_l * ae
            e_c = 0.0 if a <= 0.0 else math.copysign(max(0.0, math.log(a)) / a, e)
        else:
            e_c = e
        den = den_l2(r_buf, self._eps)
        sub_scaled_inplace(w, r_buf, (mu/den) * e_c, self._tmp)
        return w, EMPTY

class FxgsnNLMS(AlgoBase):
    __slots__ = ("sigma_e","_eps","_tmp")
    __aliases__ = ("fxgsnlms",)
    def __init__(self, params=None):
        p = {} if params is None else params
        self.sigma_e = float(p.get("sigma_e", 1.0))
        self._eps = float(p.get("eps", 1e-3))
        W = int(p.get("order_control", 512))
        self._tmp = np.empty(W, dtype=np.float64)
    def update(self, w, e, r_buf, x_buf, mu):
        sig = self.sigma_e
        x0 = float(x_buf[0])
        mu_ = mu * math.exp(-(x0 * x0) / (2.0 * sig * sig)) / (math.sqrt(2.0 * math.pi) * sig)
        mu_ = mu_ / den_l2(r_buf, self._eps)
        sub_scaled_inplace(w, r_buf, mu_ * e, self._tmp)
        return w, EMPTY

class FxLMM(AlgoBase):
    __slots__ = ("gzai","d1","d2","_tmp")
    __aliases__ = ("fxlmm",)
    def __init__(self, params=None):
        p = {} if params is None else params
        self.gzai = float(p.get("gzai", 5e-2))
        self.d1 = 3.0 * self.gzai; self.d2 = 4.0 * self.gzai
        W = int(p.get("order_control", 512))
        self._tmp = np.empty(W, dtype=np.float64)
    def update(self, w, e, r, _x_buf, mu):
        ae = abs(e); sgn = 1.0 if e >= 0.0 else -1.0
        if ae < self.gzai:
            psi = e
        elif ae < self.d1:
            psi = self.gzai * sgn
        elif ae < self.d2:
            psi = self.gzai * sgn * (self.d2 - ae) / (self.d1 - self.gzai)
        else:
            psi = 0.0
        sub_scaled_inplace(w, r, mu * psi, self._tmp)
        return w, EMPTY

class MFxLMM(AlgoBase):
    __slots__ = ("gzai_l","d1","d2","lmd_l","eps_l","sigma_l","_tmp")
    __aliases__ = ("mfxlmm",)
    def __init__(self, params=None):
        p = {} if params is None else params
        self.gzai_l = float(p.get("gzai_l", 5e-2))
        self.d1 = 3.0 * self.gzai_l; self.d2 = 4.0 * self.gzai_l
        self.lmd_l = float(p.get("lmd_l", 0.97))
        self.eps_l = float(p.get("eps_l", 1e-3))
        self.sigma_l = float(p.get("sigma_l", 1.0))
        W = int(p.get("order_control", 512))
        self._tmp = np.empty(W, dtype=np.float64)
    def update(self, w, e, r_buf, _x_buf, mu):
        ae = abs(e); sgn = 1.0 if e >= 0.0 else -1.0
        if ae < self.gzai_l:
            psi = e
        elif ae < self.d1:
            psi = self.gzai_l * sgn
        elif ae < self.d2:
            psi = self.gzai_l * sgn * (self.d2 - ae) / (self.d2 - self.d1)
        else:
            psi = 0.0
        self.sigma_l = self.lmd_l * self.sigma_l + (1.0 - self.lmd_l) * (ae * ae)
        den = float(np.mean(r_buf * r_buf)) + self.sigma_l + self.eps_l
        sub_scaled_inplace(w, r_buf, (mu/den) * psi, self._tmp)
        return w, {"sigma_l": self.sigma_l}

class MFxLCH(AlgoBase):
    __slots__ = ("rho","lmd_m","dlt2","Ke","_tmp")
    __aliases__ = ("mfxlch","fxlch")
    def __init__(self, params=None):
        p = {} if params is None else params
        self.rho = float(p.get("rho", 10.0))
        self.lmd_m = float(p.get("lmd_m", 0.9999))
        self.dlt2 = float(p.get("dlt2", 0.01))
        self.Ke = float(p.get("Ke", 0.0))
        W = int(p.get("order_control", 512))
        self._tmp = np.empty(W, dtype=np.float64)
    def update(self, w, e, r_buf, _x_buf, mu):
        ae = abs(e)
        if ae > 1.0 / self.rho:
            e_c = 1.0 if e >= 0.0 else -1.0
        else:
            e_c = -e * ae * (self.rho * self.rho) + 2.0 * self.rho * e
        self.Ke = self.lmd_m * self.Ke + (1.0 - self.lmd_m) * (e * e)
        den = den_l2(r_buf, 0.0) + self.Ke + self.dlt2
        sub_scaled_inplace(w, r_buf, (mu/den) * e_c, self._tmp)
        return w, {"Ke": self.Ke}

class MFxRNLMAT(AlgoBase):
    __slots__ = ("lmd_m","dlt2","beta2","Ke","_tmp")
    __aliases__ = ("mfxrnlmat",)
    def __init__(self, params=None):
        p = {} if params is None else params
        self.lmd_m = float(p.get("lmd_m", 0.9999))
        self.dlt2 = float(p.get("dlt2", 0.01))
        self.beta2 = float(p.get("beta2", 5000.0))
        self.Ke = float(p.get("Ke", 0.0))
        W = int(p.get("order_control", 512))
        self._tmp = np.empty(W, dtype=np.float64)
    def update(self, w, e, r_buf, _x_buf, mu):
        ae = abs(e)
        self.Ke = self.lmd_m * self.Ke + (1.0 - self.lmd_m) * (ae * ae)
        den = den_l2(r_buf, 0.0) + self.Ke + self.dlt2
        mu_m = mu / den
        mu_ = mu_m / (1.0 + self.beta2 * (ae * ae * ae))
        scale = mu_ * (ae * ae) * (1.0 if e >= 0.0 else -1.0)
        sub_scaled_inplace(w, r_buf, scale, self._tmp)
        return w, {"Ke": self.Ke}

class Fair(AlgoBase):
    __slots__ = ("M","gain","_e_hist","_tmp")
    __aliases__ = ("fair",)
    def __init__(self, params=None):
        p = {} if params is None else params
        self.M = int(p.get("M", 0))
        self.gain = float(p.get("gain", 3.0))
        self._e_hist = None
        W = int(p.get("order_control", 512))
        self._tmp = np.empty(W, dtype=np.float64)
    def update(self, w, e, r_buf, _x_buf, mu):
        if self._e_hist is None:
            M = len(w) if self.M <= 0 else self.M
            self._e_hist = np.zeros(M, dtype=np.float64)
        hist = self._e_hist
        hist[1:] = hist[:-1]; hist[0] = abs(e)
        c = self.gain * float(hist.mean()) + 1e-12
        e_c = e / (1.0 + abs(e) / c)
        sub_scaled_inplace(w, r_buf, mu * e_c, self._tmp)
        return w, {"mean_abs_e": float(hist.mean())}

class LSN_FxlogLMS_plus(AlgoBase):
    __slots__ = ("lmd_x","sigma2_x","eps_b","G_ml","_tmp")
    __aliases__ = ("lsn_fxloglms_plus",)
    def __init__(self, params=None):
        p = {} if params is None else params
        self.lmd_x = float(p.get("lmd_x", 0.999))
        self.sigma2_x = float(p.get("sigma2_x", 1.0))
        self.eps_b = float(p.get("eps_b", 1e-12))
        self.G_ml = float(p.get("G_ml", 5.0))
        W = int(p.get("order_control", 512))
        self._tmp = np.empty(W, dtype=np.float64)
    def update(self, w, e, r_buf, x_buf, mu):
        x = float(x_buf[0])
        b = self.sigma2_x = self.lmd_x*self.sigma2_x + (1.0-self.lmd_x)*(x*x)
        mu_hat = mu * math.exp(-abs(x) / (b + self.eps_b)) / (2.0 * b + self.eps_b)
        a = self.G_ml * abs(e) + 1.0
        psi = math.copysign(math.log(a) / a, e)
        sub_scaled_inplace(w, r_buf, mu_hat * psi, self._tmp)
        return w, {"sigma2_x": self.sigma2_x}

class FxNMVC(AlgoBase):
    __slots__ = ("p","use_adaptive_tau","tau_lmd","b_factor","mv_scale","has_tau","tau_fixed","b_fixed","_tmp")
    __aliases__ = ("fxnmvc",)
    def __init__(self, params=None):
        p = {} if params is None else params
        self.p = float(p.get("p", 4.0))
        self.use_adaptive_tau = bool(p.get("use_adaptive_tau", False))
        self.tau_lmd = float(p.get("tau_lmd", 0.99))
        self.b_factor = float(p.get("b_factor", 1.0))
        self.mv_scale = float(p.get("mv_scale", 0.0))
        self.has_tau = "tau" in p
        self.tau_fixed = float(p.get("tau", 1e-2))
        self.b_fixed = float(p.get("b", 1.0))
        self.eps = float(p.get("eps", 1e-6))
        W = int(p.get("order_control", 512))
        self._tmp = np.empty(W, dtype=np.float64)
    def _tau(self, z: float) -> float:
        p = self.p
        if self.use_adaptive_tau:
            self.mv_scale = self.tau_lmd*self.mv_scale + (1.0-self.tau_lmd)*z
            b_hat = max(self.b_factor*self.mv_scale, self.eps)
            return (1.0/(2.0*b_hat))**p
        if self.has_tau: return self.tau_fixed
        b = max(self.b_fixed, self.eps); return (1.0/(2.0*b))**p
    def update(self, w, e, r_buf, _x_buf, mu):
        r2 = float(np.dot(r_buf, r_buf)) + self.eps
        z = abs(float(e)) / r2
        tau = self._tau(z); p = self.p
        z_pm1 = z**(p-1.0)
        denom = r2 * (1.0 + tau*(z**p))**2 + self.eps
        phi = (tau * p * z_pm1 * (1.0 if e>=0.0 else -1.0)) / denom
        sub_scaled_inplace(w, r_buf, mu * phi, self._tmp)
        return w, {"tau": tau, "z": z}

class NFXtanhLMS(AlgoBase):
    __slots__ = ("rho","lmd_e","k","delta","eps","Pe","_tmp")
    __aliases__ = ("nfxtanhlms", "nfx_tanh_lms")

    def __init__(self, params=None):
        p = {} if params is None else params
        self.rho   = float(p.get("rho", 1.0))
        self.lmd_e = float(p.get("lmd_e", 0.999))
        self.k     = float(p.get("k", 0.5))
        self.delta = float(p.get("delta", 1e-3))
        self.eps   = float(p.get("eps", 1e-12))
        self.Pe    = float(p.get("Pe_init", 0.0))
        W = int(p.get("order_control", 512))
        self._tmp = np.empty(W, dtype=np.float64)

    def update(self, w, e, r_buf, _x_buf, mu):
        e = float(e)
        self.Pe = self.lmd_e * self.Pe + (1.0 - self.lmd_e) * (e * e)
        b = self.rho / (1.0 + math.exp(-self.Pe))
        te = math.tanh(b * e)
        psi = te * (1.0 - te * te)
        xpow = float(np.dot(r_buf, r_buf))
        mu_ad = mu / (self.k * xpow + (1.0 - self.k) * self.Pe + self.delta + self.eps)
        sub_scaled_inplace(w, r_buf, mu_ad * psi, self._tmp)
        return w, {"Pe": self.Pe, "b": b, "mu_ad": mu_ad}

class EFxatanLMS(AlgoBase):
    __slots__ = ("k", "_a", "_c1", "_eps", "_tmp")
    __aliases__ = ("efxatanlms",)

    def __init__(self, params=None):
        p = {} if params is None else params
        self.k = float(p.get("k", 1.0))
        self._a = math.pi * self.k * 0.5
        self._c1 = 4.0 / (self.k * (math.pi**2))
        self._eps = float(p.get("eps", 1e-12))
        W = int(p.get("order_control", 512))
        self._tmp = np.empty(W, dtype=np.float64)

    def _f(self, e: float) -> float:
        return self._c1 * math.atan(self._a * e)

    def _uopt(self, e: float) -> float:
        den = 1.0 + (self._a * e) * (self._a * e)
        df = (2.0 / math.pi) / den
        return 2.0 * self._f(e) * df

    def update(self, w, e, r_buf, _x_buf, mu):
        u = self._uopt(float(e))
        sub_scaled_inplace(w, r_buf, -mu * u, self._tmp)
        return w, EMPTY

class NSS_EFxatanLMS(AlgoBase):
    __slots__ = ("k","omega","Nx","Ne","eps","_x2","_e2","_ix","_ie","_sumx","_sume","_tmp")
    __aliases__ = ("nss_efxatanlms", "nss_fxatanlms", "nss-efxatanlms")

    def __init__(self, params=None):
        p = {} if params is None else params
        self.k     = float(p.get("k", 1.0))
        self.omega = float(p.get("omega", 0.5))
        self.Nx    = int(p.get("Nx", int(p.get("N", 64))))
        self.Ne    = int(p.get("Ne", int(p.get("L", 64))))
        self.eps   = float(p.get("eps", 1e-12))
        self._x2 = np.zeros(self.Nx, dtype=np.float64)
        self._e2 = np.zeros(self.Ne, dtype=np.float64)
        self._ix = 0; self._ie = 0
        self._sumx = 0.0; self._sume = 0.0
        W = int(p.get("order_control", 512))
        self._tmp = np.empty(W, dtype=np.float64)

    @staticmethod
    def _phi_e(e, k):
        pi = math.pi
        a  = (k * pi * float(e)) / 2.0
        num = 1.5 * math.atan(a)
        den = (pi**4)*(k**2)*(float(e)**2) + 4.0*(pi**2)
        return num / den

    def _update_power_means(self, x_f_scalar, e_scalar):
        x2_new = x_f_scalar * x_f_scalar
        e2_new = e_scalar   * e_scalar
        self._sumx += x2_new - self._x2[self._ix]
        self._x2[self._ix] = x2_new
        self._ix += 1
        if self._ix == self.Nx: self._ix = 0
        Px = self._sumx / float(self.Nx)
        self._sume += e2_new - self._e2[self._ie]
        self._e2[self._ie] = e2_new
        self._ie += 1
        if self._ie == self.Ne: self._ie = 0
        Pe = self._sume / float(self.Ne)
        return Px, Pe

    def update(self, w, e, r_buf, _x_buf, mu):
        x_f_n = float(r_buf[0])
        Px, Pe = self._update_power_means(x_f_n, float(e))
        denom_ref = (1.0 - self.omega) * (self.Nx * Px)
        denom_err = self.omega / (1.0 + math.exp(-(self.Ne * Pe)))
        mu_ad = mu / (denom_ref + denom_err + self.eps)
        pe = self._phi_e(float(e), self.k)
        sub_scaled_inplace(w, r_buf, mu_ad * pe, self._tmp)
        return w, {"Px": Px, "Pe": Pe, "mu_ad": mu_ad}

class FxNFLOC(AlgoBase):
    __slots__ = ("p", "eps", "_pow_fwd", "_pow_inv")
    __aliases__ = ("fxnfloc",)

    def __init__(self, params=None):
        p = {} if params is None else params
        self.p = float(p.get("p", 1.0))                   # 0<p<=2
        self.eps = float(p.get("eps", 1e-8))              # denominator for normalization
        self._pow_fwd = 0.5 * self.p                       # p/2
        self._pow_inv = 2.0 / max(self.p, 1e-12)           # 2/p

    @staticmethod
    def _proj_power(x: np.ndarray, pw: float) -> np.ndarray:
        out = np.empty_like(x, dtype=np.float64)
        np.abs(x, out=out)
        np.power(out, pw, out=out, where=(out != 0.0))
        sgn = np.sign(x, dtype=np.float64)
        np.multiply(out, sgn, out=out)
        out[out == -0.0] = 0.0
        return out

    def update(self, w, e, r_buf, _x_buf, mu):
        w_p2 = self._proj_power(w, self._pow_fwd)
        ae = abs(float(e))
        e_p2 = (ae ** self._pow_fwd) * (1.0 if e >= 0.0 else -1.0) if ae > 0.0 else 0.0
        r_p2 = self._proj_power(r_buf, self._pow_fwd)
        den = float(np.sum(np.abs(r_buf) ** self.p)) + self.eps
        w_p2 -= (mu * e_p2 / den) * r_p2
        w_abs = np.abs(w_p2)
        np.power(w_abs, self._pow_inv, out=w_abs, where=(w_abs != 0.0))
        np.multiply(w_abs, np.sign(w_p2, dtype=np.float64), out=w)
        w[w == -0.0] = 0.0
        return w, EMPTY

class FxLMLS(AlgoBase):
    __slots__ = ("_phi_max")
    __aliases__ = ("fxlmls",)

    def __init__(self, params=None):
        p = {} if params is None else params
        # Guard for exp input to avoid overflow: log(max_float) with headroom
        self._phi_max = math.log(np.finfo(np.float64).max) - 2.0

    @staticmethod
    def _phi(x: np.ndarray) -> np.ndarray:
        ax = np.abs(x)
        y = np.log(ax + 1.0)
        return np.sign(x, dtype=np.float64) * y

    @staticmethod
    def _phi_scalar(z: float) -> float:
        return math.copysign(math.log(abs(z) + 1.0), z)

    def _phi_inv(self, y: np.ndarray) -> np.ndarray:
        ay = np.clip(np.abs(y), 0.0, self._phi_max)
        x = np.expm1(ay)  # better precision for small magnitudes
        return np.sign(y, dtype=np.float64) * x

    def update(self, w, e, r_buf, _x_buf, mu):
        w_phi = self._phi(w)
        e_phi = self._phi_scalar(float(e))
        r_phi = self._phi(r_buf)
        np.subtract(w_phi, mu * e_phi * r_phi, out=w_phi)
        np.clip(w_phi, -self._phi_max, self._phi_max, out=w_phi)
        w_new = self._phi_inv(w_phi)
        return w_new, EMPTY

class EVSS_FxeLMS(AlgoBase):
    __slots__ = ("k","alpha","lmd_e","gamma","Ee","eps","_tmp")
    __aliases__ = ("evss_fxelms","evssfxelms","fxerf_vss")

    def __init__(self, params=None):
        p = {} if params is None else params
        self.k      = float(p.get("k", 1.0))        # erf scale
        self.alpha  = float(p.get("alpha", 1.0))    # sine VSS coefficient
        self.lmd_e  = float(p.get("lmd_e", 0.99))   # error-energy smoothing lambda
        self.gamma  = float(p.get("gamma", 1e-3))   # normalization offset
        self.Ee     = float(p.get("Ee_init", 0.0))  # initial error energy
        self.eps    = 1e-12
        W = int(p.get("order_control", 512))
        self._tmp = np.empty(W, dtype=np.float64)

    @staticmethod
    def _phi(e, k):
        z = float(e) / k
        return (4.0 / math.sqrt(math.pi)) * math.erf(z) * math.exp(-(z*z))

    def update(self, w, e, r_buf, _x_buf, mu):
        e = float(e)
        self.Ee = self.lmd_e * self.Ee + (1.0 - self.lmd_e) * (e * e)
        x2 = float(np.dot(r_buf, r_buf))
        norm_term = x2 + self.Ee + self.gamma
        sine_term = math.sin(self.alpha * abs(e) - math.pi/2.0) + 1.5
        mu_ad = (mu / norm_term) * sine_term
        sub_scaled_inplace(w, r_buf, mu_ad * self._phi(e, self.k), self._tmp)
        return w, {"Ee": self.Ee, "mu_ad": mu_ad}

class CSS_FxAPCMAF(AlgoBase):
    __slots__ = ("gamma","rho","kappa","mu1","mu2","mu_z","C","zeta","zeta_lim","eps","_qprev")
    __aliases__ = ("css_fxapcmaf",)

    def __init__(self, params=None):
        p = {} if params is None else params
        self.gamma = float(p.get("gamma", 100.0))
        self.rho   = float(p.get("rho",   40.0))
        self.kappa = float(p.get("kappa",  0.1))
        self.mu1   = float(p.get("mu1",   0.08))
        self.mu2   = float(p.get("mu2",   0.001))
        self.mu_z  = float(p.get("mu_z",   0.6))   # zeta learning rate
        self.C     = float(p.get("C",        5.0)) # eta scaling
        self.zeta  = float(p.get("zeta0",    0.1))
        self.eps   = float(p.get("eps",   1e-12))
        self.zeta_lim = math.log((self.C + 1.0) / max(self.C - 1.0, 1e-12))
        self._qprev = None

    @staticmethod
    def _phi(gamma, rho, kappa, e):
        ke = kappa * e
        return gamma * kappa * math.sinh(ke) / ((rho + math.cosh(ke))**2)

    def _eta_from_zeta(self):
        z = self.zeta
        if z < -self.zeta_lim: z = -self.zeta_lim
        elif z > self.zeta_lim: z =  self.zeta_lim
        self.zeta = z
        return self.C / (1.0 + math.exp(-z)) - (self.C*0.5 - 0.5)

    def update(self, w, e, r_buf, _x_buf, mu):
        e = float(e)
        phi_e = self._phi(self.gamma, self.rho, self.kappa, e)
        g = phi_e * r_buf
        gnorm = float(np.linalg.norm(g)) + self.eps
        q = g / gnorm
        eta = self._eta_from_zeta()
        mu_eff = eta * self.mu1 + (1.0 - eta) * self.mu2
        np.subtract(w, mu_eff * q, out=w)
        if self._qprev is not None:
            s = 1.0 if e >= 0.0 else -1.0
            xt_qprev = float(np.dot(r_buf, self._qprev))
            self.zeta += self.mu_z * (self.mu1 - self.mu2) * (eta * (1.0 - eta)) * xt_qprev * s
        if self._qprev is None or self._qprev.shape != w.shape:
            self._qprev = np.zeros_like(w, dtype=np.float64)
        np.copyto(self._qprev, q)
        return w, {"eta": eta, "mu_eff": mu_eff, "zeta": self.zeta}

class SFxLMS(AlgoBase):
    __slots__ = ("lam", "beta", "_tmp")
    __aliases__ = ("sfxlms", "softsign_fxlms")

    def __init__(self, params=None):
        p = {} if params is None else params
        # λ: control parameter for error compression (0 < λ < 1)
        # β: shape parameter controlling softsign curvature (> 0)
        self.lmd  = float(p.get("lambda", p.get("lambda", 0.08)))
        self.beta = float(p.get("beta", 1.0))
        W = int(p.get("order_control", 512))
        self._tmp = np.empty(W, dtype=np.float64)

    def update(self, w, e, r_buf, _x_buf, mu):
        # Softsign function f(e) = e / [1 + λ * |e|^(2β)]^((1+β)/β)
        ae2 = float(e) * float(e)            # |e|^2
        t   = 1.0 + self.lmd * (ae2 ** self.beta)
        denom = t ** ((1.0 + self.beta) / self.beta)
        f = float(e) / denom                 # non-linear compressed error term
        # Weight update: w ← w + μ * f(e) * r_buf
        # Since helper performs w -= scale * r_buf, set scale = -μ * f(e)
        sub_scaled_inplace(w, r_buf, mu * f, self._tmp)
        return w, EMPTY

class VSFxLMS(AlgoBase):
    __slots__ = ("beta", "chi", "delta", "sigma",
                 "lam_min", "lam_max", "lam",
                 "_g", "_tmp")
    __aliases__ = ("vsfxlms", "vs_softsign_fxlms")

    def __init__(self, params=None):
        p = {} if params is None else params
        # Softsign shape parameter β > 0
        self.beta   = float(p.get("beta", 1.0))
        # Gradient smoothing χ ~ 0.95–0.995
        self.chi    = float(p.get("chi", 0.99))
        # δ > 0 controls sensitivity of λ(k) to ||g(k)||
        self.delta  = float(p.get("delta", 1e-3))
        # σ > 0 for γ(k) = exp(-1/(σ ||x_s(k)||))
        self.sigma  = float(p.get("sigma", 1.0))
        # Bounds for λ(k)
        self.lam_min = float(p.get("lam_min", 0.02))
        self.lam_max = float(p.get("lam_max", 0.20))
        # Initialize λ(0) (any value in [lam_min, lam_max] is valid)
        self.lam     = float(p.get("lambda_init", 0.08))

        # Buffers
        W = int(p.get("order_control", 512))
        self._g   = np.zeros(W, dtype=np.float64)   # smoothed gradient g(k)
        self._tmp = np.empty(W, dtype=np.float64)

    @staticmethod
    def _norm2(x: np.ndarray) -> float:
        # l2 norm with tiny floor to avoid division by zero
        return float(np.sqrt(np.dot(x, x)) + 1e-12)

    def _denom_power(self, e: float, lam: float) -> float:
        # [1 + λ |e|^{2β}]^{(1+β)/β}   (paper Eq. (9) definition of softsign term)
        ae2 = e * e
        t   = 1.0 + lam * ((ae2) ** self.beta)
        return t ** ((1.0 + self.beta) / self.beta)

    def update(self, w, e, r_buf, _x_buf, mu):
        # 1) Smoothed gradient g(k) (paper Eq. (10))
        #    g(k) = χ g(k-1) + (1-χ) * e(k) * x_s(k) * [1 + λ |e^2(k)|^{β}]^{(1+β)/β}
        e = float(e)
        D_prev = self._denom_power(e, self.lam)  # uses previous λ
        np.multiply(r_buf, (1.0 - self.chi) * e * D_prev, out=self._tmp)
        np.add(self._g, self._tmp, out=self._g)
        self._g *= self.chi

        # 2) γ(k) and λ(k) update (paper Eq. (11)(12) + clamping Eq. (13))
        xs_norm = self._norm2(r_buf)
        gamma_k = math.exp(-1.0 / (self.sigma * xs_norm))
        g_norm  = self._norm2(self._g)
        lam_k   = gamma_k * math.exp(-1.0 / (self.delta * g_norm))
        # Clamp to [lam_min, lam_max]
        lam_k = self.lam_max if lam_k > self.lam_max else self.lam_min if lam_k < self.lam_min else lam_k
        self.lam = lam_k  # store λ(k) for the next iteration

        # 3) Weight update (paper Eq. (9) with λ(k)):
        #    φ(k+1) = φ(k) + η * e(k) * x_s(k) / [1 + λ(k) |e^2(k)|^{β}]^{(1+β)/β}
        D = self._denom_power(e, lam_k)
        scale = mu * (e / D)                # helper does w -= scale * r_buf
        sub_scaled_inplace(w, r_buf, scale, self._tmp)

        return w, {"lambda": self.lam, "gamma": gamma_k, "g_norm": g_norm}



# -------------------------
# Frequency-domain family (double-ring friendly)
# -------------------------
class FreqFxLMS(AlgoBase):
    __slots__ = ("M","N","BlockSize","eps","Hc","x2","x_head","e2","e_head","e_pad",
                 "count","rg_alpha","lmd","ratio","ratio_ema")
    __aliases__ = ("freqfxlms",)
    def __init__(self, params):
        M = int(params.get("M", 1024)); self.M = M; self.N = 2*M
        self.BlockSize = int(params.get("BlockSize", M))
        self.eps = float(params.get("eps", 1e-3))
        c = np.asarray(params.get("c_impulse", np.zeros(M)), dtype=np.float64)
        self.Hc = np.fft.rfft(c, n=self.N)
        self.x2 = np.zeros(2*self.N, dtype=np.float64); self.x_head = 0
        self.e2 = np.zeros(2*self.M, dtype=np.float64); self.e_head = 0
        self.e_pad = np.zeros(self.N, dtype=np.float64)
        self.count = 0
        self.rg_alpha = float(params.get("rg_alpha", 0.05))
        self.lmd = float(params.get("lmd", 0.9))
        self.ratio = 1.0; self.ratio_ema = 1.0
    def update(self, w, e, r_buf, x_buf, mu):
        M=self.M; N=self.N
        x = float(x_buf[0])
        # double-ring
        self.x2[self.x_head] = x; self.x2[self.x_head+N] = x; self.x_head = adv(self.x_head, N)
        self.e2[self.e_head] = float(e); self.e2[self.e_head+M] = float(e); self.e_head = adv(self.e_head, M)
        self.count += 1
        if self.count < self.BlockSize: return w, None
        xv = self.x2[self.x_head : self.x_head + N]
        Xf = np.fft.rfft(xv, n=N)
        Rf = self.Hc * Xf
        ev = self.e2[self.e_head : self.e_head + M]
        ep = self.e_pad; ep[:M] = 0.0; ep[M:] = ev
        Ef = np.fft.rfft(ep, n=N)
        Rf2_mean = float(np.vdot(Rf, Rf).real) / Rf.size + self.eps
        GradF = Rf.conj() * Ef / Rf2_mean
        g = np.fft.irfft(GradF, n=N)
        Sxy = np.vdot(Rf, Ef) / Rf.size
        Sxx = float(np.vdot(Rf, Rf).real) / Rf.size
        Syy = float(np.vdot(Ef, Ef).real) / Rf.size
        gamma2 = float((Sxy.conjugate()*Sxy).real / (Sxx * Syy + self.eps))
        self.ratio = float(np.clip(gamma2, 0.0, 1.0))
        self.ratio_ema = (1 - self.rg_alpha) * self.ratio_ema + self.rg_alpha * self.ratio
        mu_ = mu * self.BlockSize / M
        Lw = min(M, len(w))
        g[:Lw] *= mu_
        np.subtract(w[:Lw], g[:Lw], out=w[:Lw])
        self.count = 0
        return w, None
    def state(self):
        return {"ratio": self.ratio, "ratio_ema": self.ratio_ema}
