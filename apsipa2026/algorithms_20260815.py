"""Coherence-based FD-FxNLMS extracted from the archived 2026-08-15 implementation."""

from typing import Any, Dict, Type

import numpy as np


def adv(head: int, L: int) -> int:
    head += 1
    return 0 if head == L else head


REG: Dict[str, Type] = {}


class AlgoBase:
    __aliases__: tuple[str, ...] = ()
    compute_r: bool = True

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        REG[cls.__name__.lower()] = cls
        for a in getattr(cls, "__aliases__", ()): REG[a.lower()] = cls
    def state(self) -> Dict[str, Any]: return {}


class Freq_FxNLMS_CohVSS(AlgoBase):
    __slots__ = (
        "M", "N", "BlockSize", "eps", "Hc", "x2", "x_head", "e2", "e_head", "e_pad",
        "count", "rg_alpha", "ratio", "ratio_ema", "ratio_ema_vec", "coh_pivot", "coh_gain",
        "mu_scale_min", "mu_scale_max", "mu_scale", "mu_scale_vec", "coherence", "coherence_ema",
        "coherence_vec", "Sre_ema", "Srr_ema", "See_ema", "boost_power",
        "coh_weight_min", "coh_weight_mix", "coh_weight_power", "grad_norm_mix",
        "suppress_power", "startup_blocks", "startup_boost", "update_blocks"
    )
    __aliases__ = ("freq_fxnlms_cohvss", "freqfxlms_cohvss")
    compute_r = False

    def __init__(self, params):
        M = int(params.get("M", 1024)); self.M = M; self.N = 2 * M
        self.BlockSize = int(params.get("BlockSize", M))
        self.eps = float(params.get("eps", 1e-6))

        c = params.get("c_impulse", None)
        if c is None:
            raise ValueError("Freq_FxNLMS_CohVSS requires c_impulse from anc.py")
        c = np.asarray(c, dtype=np.float64)
        self.Hc = np.fft.rfft(c, n=self.N)

        self.x2 = np.zeros(2 * self.N, dtype=np.float64); self.x_head = 0
        self.e2 = np.zeros(2 * self.M, dtype=np.float64); self.e_head = 0
        self.e_pad = np.zeros(self.N, dtype=np.float64)

        self.count = 0
        self.rg_alpha = float(params.get("rg_alpha", 0.05))
        self.ratio = 0.0
        self.ratio_ema = 0.0
        self.ratio_ema_vec = np.zeros(self.N // 2 + 1, dtype=np.float64)
        self.coherence = 0.0
        self.coherence_ema = 0.0
        self.coherence_vec = np.zeros(self.N // 2 + 1, dtype=np.float64)
        self.Sre_ema = np.zeros(self.N // 2 + 1, dtype=np.complex128)
        self.Srr_ema = np.zeros(self.N // 2 + 1, dtype=np.float64)
        self.See_ema = np.zeros(self.N // 2 + 1, dtype=np.float64)

        self.coh_pivot = float(params.get("coh_pivot", 0.50))
        self.coh_gain = float(params.get("coh_gain", 1.5))
        self.mu_scale_min = float(params.get("mu_scale_min", 0.6))
        self.mu_scale_max = float(params.get("mu_scale_max", 1.6))
        self.boost_power = float(params.get("boost_power", 0.6))
        self.suppress_power = float(params.get("suppress_power", 1.6))
        self.coh_weight_min = float(params.get("coh_weight_min", 0.55))
        self.coh_weight_mix = float(params.get("coh_weight_mix", 0.60))
        self.coh_weight_power = float(params.get("coh_weight_power", 1.6))
        self.grad_norm_mix = float(params.get("grad_norm_mix", 0.45))
        self.startup_blocks = max(1, int(params.get("startup_blocks", 48)))
        self.startup_boost = float(params.get("startup_boost", 0.7))
        self.update_blocks = 0
        self.mu_scale = 1.0
        self.mu_scale_vec = np.ones(self.N // 2 + 1, dtype=np.float64)

    def update(self, w, e, r_buf, x_buf, mu):
        M = self.M; N = self.N
        x = float(x_buf[0])

        self.x2[self.x_head] = x
        self.x2[self.x_head + N] = x
        self.x_head = adv(self.x_head, N)

        self.e2[self.e_head] = float(e)
        self.e2[self.e_head + M] = float(e)
        self.e_head = adv(self.e_head, M)

        self.count += 1
        if self.count < self.BlockSize:
            return w, None

        xv = self.x2[self.x_head : self.x_head + N]
        Xf = np.fft.rfft(xv, n=N)
        Rf = self.Hc * Xf

        ev = self.e2[self.e_head : self.e_head + M]
        ep = self.e_pad
        ep[:M] = 0.0
        ep[M:] = ev
        Ef = np.fft.rfft(ep, n=N)

        Sre = Rf * np.conj(Ef)
        Srr = np.abs(Rf) ** 2
        See = np.abs(Ef) ** 2

        # Use smoothed spectral estimates; single-block coherence would collapse to ~1.
        self.Sre_ema = (1.0 - self.rg_alpha) * self.Sre_ema + self.rg_alpha * Sre
        self.Srr_ema = (1.0 - self.rg_alpha) * self.Srr_ema + self.rg_alpha * Srr
        self.See_ema = (1.0 - self.rg_alpha) * self.See_ema + self.rg_alpha * See
        self.coherence_vec = np.clip(
            (np.abs(self.Sre_ema) ** 2) / (self.Srr_ema * self.See_ema + self.eps),
            0.0,
            1.0,
        )

        self.ratio_ema_vec = self.coherence_vec.copy()
        self.ratio = float(np.mean(self.coherence_vec))
        self.ratio_ema = float(np.mean(self.ratio_ema_vec))
        self.coherence = self.ratio
        self.coherence_ema = self.ratio_ema

        pivot = np.clip(self.coh_pivot, self.eps, 1.0 - self.eps)
        low_rel = np.clip((pivot - self.ratio_ema_vec) / pivot, 0.0, 1.0)
        high_rel = np.clip((self.ratio_ema_vec - pivot) / (1.0 - pivot), 0.0, 1.0)

        mu_low = 1.0 + (self.mu_scale_max - 1.0) * np.power(low_rel, self.boost_power)
        mu_high = 1.0 - (1.0 - self.mu_scale_min) * (1.0 - np.exp(-self.coh_gain * np.power(high_rel, self.suppress_power)))
        base_mu_scale_vec = np.where(self.ratio_ema_vec <= pivot, mu_low, mu_high)

        # Warmup is intentionally disabled: always use the base coherence mapping.
        self.mu_scale_vec = base_mu_scale_vec

        self.mu_scale_vec = np.clip(
            self.mu_scale_vec,
            self.mu_scale_min,
            self.mu_scale_max,
        )
        self.mu_scale = float(np.mean(self.mu_scale_vec))

        # A+B (refined): blend scalar/bin normalization and suppress only high-coherence excess.
        srr_scalar = float(np.mean(self.Srr_ema))
        grad_norm = (1.0 - self.grad_norm_mix) * (srr_scalar + self.eps) + self.grad_norm_mix * (self.Srr_ema + self.eps)
        coh_excess = np.clip((self.coherence_vec - pivot) / (1.0 - pivot), 0.0, 1.0)
        coh_weight = 1.0 - self.coh_weight_mix * np.power(coh_excess, self.coh_weight_power)
        coh_weight = np.clip(coh_weight, self.coh_weight_min, 1.0)
        GradF = (Rf.conj() * Ef / grad_norm) * coh_weight * self.mu_scale_vec
        g = np.fft.irfft(GradF, n=N)

        mu_ = 2.0 * mu * (self.BlockSize / M)
        Lw = min(M, len(w))
        np.subtract(w, mu_ * g[:Lw], out=w[:Lw])

        self.update_blocks += 1
        self.count = 0
        return w, None

    def state(self):
        return {
            "ratio": self.ratio,
            "ratio_ema": self.ratio_ema,
            "coherence": self.coherence,
            "coherence_ema": self.coherence_ema,
            "mu_scale": self.mu_scale,
            "mu_scale_min": float(np.min(self.mu_scale_vec)),
            "mu_scale_max": float(np.max(self.mu_scale_vec)),
        }
