"""Effective degrees of freedom of the SF direction under white error noise, with the actual secondary path and
regularisation (THEORY_v4.md, Lemma G).  The whitened reference has spectrum W = Srr / (Srr + eps * mean Srr);
the noise-induced direction has covariance ~ Toeplitz(autocorr of W), so
  dof_eff = (tr C)^2 / tr(C^2),   sd of the split coherence under noise ~ 1/sqrt(dof_eff).
Run: python3 dof.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "closedform"))
import numpy as np
from scipy import linalg
from base import hS
M, G = 512, 8192
S2 = np.abs(np.fft.rfft(hS, G)) ** 2
for eps in (1e-2, 1e-3, 1e-4, 1e-6):
    W = S2 / (S2 + eps * S2.mean())
    r = np.fft.irfft(W, G)[:M]
    C = linalg.toeplitz(r)
    dof = np.trace(C) ** 2 / np.sum(C * C)
    print(f"eps {eps:g}: mean W {W.mean():.3f}, dof_eff {dof:.1f} of {M}, predicted noise sd of coh {dof ** -.5:.3f}; "
          f"fraction of bins with W < 0.5: {np.mean(W < 0.5):.3f}")
