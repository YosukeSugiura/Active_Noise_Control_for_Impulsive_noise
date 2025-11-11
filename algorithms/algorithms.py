# -*- coding: utf-8 -*-
"""
ANC algorithms with auto-registration via __init_subclass__.
- No manual _REG edits
- Minimal allocations, uses __slots__
- TD LMS family uses preallocated tmp buffers sized by params['order_control']
- FD variants support double-ring buffering

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

# ---- optional profiler shim ----
try:
    from prof_hooks import PROFILE  # 外部が --profile で有効化
except Exception:  # プロファイラ無しでも動作
    class _NoProf:
        def tic(self, *_a, **_k):
            return None
        def toc(self, *_a, **_k):
            return None
    PROFILE = _NoProf()

# -------------------------
# Auto registry
# -------------------------
REG: Dict[str, Type] = {}

class AlgoBase:
    __aliases__: tuple[str, ...] = ()  # optional extra keys
    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        REG[cls.__name__.lower()] = cls
        for a in getattr(cls, "__aliases__", ()):  # allow short names
            REG[a.lower()] = cls

    def state(self) -> Dict[str, Any]:
        return {}

def make_algo(name: str, params: Dict[str, Any] | None = None):
    cls = REG.get(name.lower())
    if cls is None:
        raise KeyError(f"Unknown algorithm: {name}. Known: {sorted(REG.keys())[:32]} ...")
    return cls({} if params is None else params)

EMPTY: Dict[str, Any] = {}

def _nlms_den(pwr: float, eps: float = 1e-3) -> float:
    return pwr + eps

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
        PROFILE.tic("FxNLMS:den")
        den = float(np.dot(r_buf, r_buf)) + self.eps
        PROFILE.toc()
        PROFILE.tic("FxNLMS:update")
        scale = (mu / den) * e
        np.multiply(r_buf, scale, out=self._tmp)
        np.subtract(w, self._tmp, out=w)
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
        self.grad += e * r_buf / (float(np.dot(r_buf, r_buf)) + self.eps)
        self.cnt += 1
        PROFILE.toc()
        if self.cnt < self.BlockSize:
            return w, EMPTY
        PROFILE.tic("BlockFxNLMS:apply")
        # w -= mu * self.grad  (in-place)
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
        den = float(np.dot(r_buf, r_buf)) + self._eps
        scale = (mu/den) * e_c
        np.multiply(r_buf, scale, out=self._tmp)
        np.subtract(w, self._tmp, out=w)
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
        den = float(np.dot(r_buf, r_buf)) + self._eps
        scale = (mu/den) * e_c
        np.multiply(r_buf, scale, out=self._tmp)
        np.subtract(w, self._tmp, out=w)
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
        den = float(np.dot(r_buf, r_buf)) + self._eps
        scale = (mu/den) * e_c
        np.multiply(r_buf, scale, out=self._tmp)
        np.subtract(w, self._tmp, out=w)
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
        scale = (mu/den) * e_c
        np.multiply(r_buf, scale, out=self._tmp)
        np.subtract(w, self._tmp, out=w)
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
        den = float(np.dot(r_buf, r_buf)) + self._eps
        scale = (mu/den) * e_c
        np.multiply(r_buf, scale, out=self._tmp)
        np.subtract(w, self._tmp, out=w)
        return w, EMPTY

class FxgsnLMS(AlgoBase):
    __slots__ = ("sigma_e",)
    __aliases__ = ("fxgsnlms",)
    def __init__(self, params=None):
        p = {} if params is None else params
        self.sigma_e = float(p.get("sigma_e", 1.0))
    def update(self, w, e, r_buf, x_buf, mu):
        sig = self.sigma_e
        x0 = float(x_buf[0])
        mu_ = mu * math.exp(-(x0 * x0) / (2.0 * sig * sig)) / (math.sqrt(2.0 * math.pi) * sig)
        denom = float(np.dot(x_buf, x_buf))
        if denom != 0.0 and mu_ >= 1.0 / denom:
            mu_ = 1.0 / denom
        return w - mu_ * e * r_buf, EMPTY

class FxLMM(AlgoBase):
    __slots__ = ("gzai","d1","d2")
    __aliases__ = ("fxlmm",)
    def __init__(self, params=None):
        p = {} if params is None else params
        self.gzai = float(p.get("gzai", 5e-2))
        self.d1 = 3.0 * self.gzai; self.d2 = 4.0 * self.gzai
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
        return w - mu * psi * r, EMPTY

class MFxLMM(AlgoBase):
    __slots__ = ("gzai_l","d1","d2","lmd_l","eps_l","sigma_l")
    __aliases__ = ("mfxlmm",)
    def __init__(self, params=None):
        p = {} if params is None else params
        self.gzai_l = float(p.get("gzai_l", 5e-2))
        self.d1 = 3.0 * self.gzai_l; self.d2 = 4.0 * self.gzai_l
        self.lmd_l = float(p.get("lmd_l", 0.97))
        self.eps_l = float(p.get("eps_l", 1e-3))
        self.sigma_l = float(p.get("sigma_l", 1.0))
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
        scale = (mu/den) * psi
        # ここは e_c=psi がすでに符号を含む
        return w - scale * r_buf, {"sigma_l": self.sigma_l}

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
        den = float(np.dot(r_buf, r_buf)) + self.Ke + self.dlt2
        scale = (mu/den) * e_c
        np.multiply(r_buf, scale, out=self._tmp)
        np.subtract(w, self._tmp, out=w)
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
        den = float(np.dot(r_buf, r_buf)) + self.Ke + self.dlt2
        mu_m = mu / den
        mu_ = mu_m / (1.0 + self.beta2 * (ae * ae * ae))
        scale = mu_ * (ae * ae) * (1.0 if e >= 0.0 else -1.0)
        np.multiply(r_buf, scale, out=self._tmp)
        np.subtract(w, self._tmp, out=w)
        return w, {"Ke": self.Ke}

class Fair(AlgoBase):
    __slots__ = ("M","gain","_e_hist")
    __aliases__ = ("fair",)
    def __init__(self, params=None):
        p = {} if params is None else params
        self.M = int(p.get("M", 0))
        self.gain = float(p.get("gain", 3.0))
        self._e_hist = None
    def update(self, w, e, r_buf, _x_buf, mu):
        if self._e_hist is None:
            M = len(w) if self.M <= 0 else self.M
            self._e_hist = np.zeros(M, dtype=np.float64)
        hist = self._e_hist
        hist[1:] = hist[:-1]; hist[0] = abs(e)
        c = self.gain * float(hist.mean()) + 1e-12
        e_c = e / (1.0 + abs(e) / c)
        return w - mu * e_c * r_buf, {"mean_abs_e": float(hist.mean())}

class LSN_FxlogLMS_plus(AlgoBase):
    __slots__ = ("lmd_x","sigma2_x","eps_b","G_ml")
    __aliases__ = ("lsn_fxloglms_plus",)
    def __init__(self, params=None):
        p = {} if params is None else params
        self.lmd_x = float(p.get("lmd_x", 0.999))
        self.sigma2_x = float(p.get("sigma2_x", 1.0))
        self.eps_b = float(p.get("eps_b", 1e-12))
        self.G_ml = float(p.get("G_ml", 5.0))
    def update(self, w, e, r_buf, x_buf, mu):
        x = float(x_buf[0]); b = self.sigma2_x = self.lmd_x*self.sigma2_x + (1.0-self.lmd_x)*(x*x)
        mu_hat = mu * math.exp(-abs(x) / (b + self.eps_b)) / (2.0 * b + self.eps_b)
        a = self.G_ml * abs(e) + 1.0
        psi = math.copysign(math.log(a) / a, e)
        return w - mu_hat * psi * r_buf, {"sigma2_x": self.sigma2_x}

class FxNMVC(AlgoBase):
    __slots__ = ("pwr","use_adaptive_tau","tau_lmd","b_factor","mv_scale","has_tau","tau_fixed","b_fixed")
    __aliases__ = ("fxnmvc",)
    def __init__(self, params=None):
        p = {} if params is None else params
        self.pwr = float(p.get("p", 4.0))
        self.use_adaptive_tau = bool(p.get("use_adaptive_tau", False))
        self.tau_lmd = float(p.get("tau_lmd", 0.99))
        self.b_factor = float(p.get("b_factor", 1.0))
        self.mv_scale = float(p.get("mv_scale", 0.0))
        self.has_tau = "tau" in p
        self.tau_fixed = float(p.get("tau", 1e-2))
        self.b_fixed = float(p.get("b", 1.0))
    def _tau(self, z: float) -> float:
        p = self.pwr; eps = 1e-12
        if self.use_adaptive_tau:
            self.mv_scale = self.tau_lmd*self.mv_scale + (1.0-self.tau_lmd)*z
            b_hat = max(self.b_factor*self.mv_scale, eps)
            return (1.0/(2.0*b_hat))**p
        if self.has_tau: return self.tau_fixed
        b = max(self.b_fixed, eps); return (1.0/(2.0*b))**p
    def update(self, w, e, r_buf, _x_buf, mu):
        eps = 1e-12; r2 = float(np.dot(r_buf, r_buf)) + eps
        z = abs(float(e)) / r2
        tau = self._tau(z); p = self.pwr
        z_pm1 = z**(p-1.0) if z>0.0 else 0.0
        denom = r2 * (1.0 + tau*(z**p))**2 + eps
        phi = (tau * p * z_pm1 * (1.0 if e>=0.0 else -1.0)) / denom
        return w - mu * phi * r_buf, {"tau": tau, "z": z}

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
    @staticmethod
    def _adv(h, L):
        h += 1
        return 0 if h == L else h
    def update(self, w, e, r_buf, x_buf, mu):
        M=self.M; N=self.N
        x = float(x_buf[0])
        self.x2[self.x_head] = x; self.x2[self.x_head+N] = x; self.x_head = self._adv(self.x_head, N)
        self.e2[self.e_head] = float(e); self.e2[self.e_head+M] = float(e); self.e_head = self._adv(self.e_head, M)
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
        w[:Lw] -= mu_ * g[:Lw]
        self.count = 0
        return w, None
    def state(self):
        return {"ratio": self.ratio, "ratio_ema": self.ratio_ema}

class FreqFxLMS_VSS(AlgoBase):
    __slots__ = ("M","N","BlockSize","eps","Hc","x2","x_head","e2","e_head","e_pad",
                 "count","rg_alpha","ratio","ratio_ema")
    __aliases__ = ("freqfxlms_vss", "vss")
    def __init__(self, params):
        M = int(params.get("M", 1024)); self.M = M; self.N = 2*M
        self.BlockSize = int(params.get("BlockSize", M))
        self.eps = float(params.get("eps", 1e-6))
        c = np.asarray(params.get("c_impulse", np.zeros(M)), dtype=np.float64)
        self.Hc = np.fft.rfft(c, n=self.N)
        self.x2 = np.zeros(2*self.N, dtype=np.float64); self.x_head = 0
        self.e2 = np.zeros(2*self.M, dtype=np.float64); self.e_head = 0
        self.e_pad = np.zeros(self.N, dtype=np.float64)
        self.count = 0
        self.rg_alpha = float(params.get("rg_alpha", 0.99))
        self.ratio = 1.0; self.ratio_ema = 0.0
    @staticmethod
    def _adv(h, L):
        h += 1
        return 0 if h == L else h
    def update(self, w, e, r_buf, x_buf, mu):
        M=self.M; N=self.N
        x = float(x_buf[0])
        self.x2[self.x_head] = x; self.x2[self.x_head+N] = x; self.x_head = self._adv(self.x_head, N)
        self.e2[self.e_head] = float(e); self.e2[self.e_head+M] = float(e); self.e_head = self._adv(self.e_head, M)
        self.count += 1
        if self.count < self.BlockSize: return w, None
        xv = self.x2[self.x_head : self.x_head + N]
        Xf = np.fft.rfft(xv, n=N)
        Rf = self.Hc * Xf
        ev = self.e2[self.e_head : self.e_head + M]
        ep = self.e_pad; ep[:M] = 0.0; ep[M:] = ev
        Ef = np.fft.rfft(ep, n=N)
        Sxy = np.vdot(Rf, Ef) / Rf.size
        Sxx = float(np.vdot(Rf, Rf).real) / Rf.size
        Syy = float(np.vdot(Ef, Ef).real) / Rf.size
        gamma2 = float((Sxy.conjugate()*Sxy).real / (Sxx * Syy + self.eps))
        self.ratio = float(np.clip(gamma2, 0.0, 1.0))
        self.ratio_ema = (1 - self.rg_alpha) * self.ratio_ema + self.rg_alpha * self.ratio
        self.count = 0
        return w, None
    def state(self):
        return {"ratio": self.ratio, "ratio_ema": self.ratio_ema}

class FreqFxlogLMS_debug(AlgoBase):
    __slots__ = ("M","G","eps","den_floor","BlockSize","Hc","x2","x_head","e2","e_head","count")
    __aliases__ = ("freqfxloglms_debug",)
    def __init__(self, params):
        p = {} if params is None else params
        self.M = int(p.get("M", 1024))
        self.G = float(p.get("G", 1e4))
        self.eps = float(p.get("eps", 1e-12))
        self.den_floor = float(p.get("den_floor", 1e-3))
        self.BlockSize = int(p.get("BlockSize", 64))
        c = np.asarray(p.get("c_impulse", np.zeros(self.M)), np.float64)
        self.Hc = np.fft.rfft(c, n=2*self.M)
        self.x2 = np.zeros(4*self.M)  # 2N = 4M
        self.x_head = 0
        self.e2 = np.zeros(2*self.M)
        self.e_head = 0
        self.count = 0
    @staticmethod
    def _adv(h, L):
        h += 1
        return 0 if h == L else h
    def update(self, w, e, r_buf, x_buf, mu):
        x = float(x_buf[0]); M=self.M; N=2*M
        self.x2[self.x_head] = x; self.x2[self.x_head+N] = x; self.x2[self.x_head+2*N] = x; self.x2[self.x_head+3*N] = x
        self.x_head = self._adv(self.x_head, 2*N)
        self.e2[self.e_head] = e; self.e2[self.e_head+M] = e; self.e_head = self._adv(self.e_head, M)
        self.count += 1
        if self.count < self.BlockSize: return w, None
        xv = self.x2[self.x_head : self.x_head + 2*M]
        Xf = np.fft.rfft(xv)
        Rf = self.Hc * Xf
        ev = self.e2[self.e_head : self.e_head + M]
        Ef = np.fft.rfft(np.concatenate([np.zeros(M), ev]))
        GradF = Ef * np.conj(Rf) / (np.abs(Rf)**2 + self.den_floor)
        GradTD = np.fft.irfft(GradF, n=2*M)
        Lw = len(w)
        w -= mu * self.BlockSize / M * GradTD[:Lw]
        self.count = 0
        return w, None
