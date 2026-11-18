"""Theorem A check: SF mean action and fixed point in the expected-value frame model.
Frame of N samples, error window = last Me samples, M taps, causal truncation at N/2 lags (as specfac.sf_dir).
Expected circular correlation c(m) = [n1(m) psi(m) + (Me - n1(m)) psi(m - N)] / Me, n1(m) = min(Me, N - m),
psi = true cross-correlation of the error with the filtered reference (white unit input).
Direction = A w + b (normalised so that the ideal is A = I).  Run: python3 thmA.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "closedform"))
import numpy as np
from scipy import linalg
from base import hP, hS, xc, wiener, minphase, G, S2, Sw, Pw
dB = lambda x: 10 * np.log10(x + 1e-300)
Fg = minphase(S2, G)
fimp = np.fft.irfft(Fg, G)[:G // 2]                         # impulse response of F (causal)
q = np.fft.irfft(np.conj(Sw) * Pw / np.conj(Fg), G); qp = q.copy(); qp[G // 2:] = 0
winf = np.fft.irfft(-np.fft.rfft(qp) / Fg, G)

def frame(M, Me, N, eps_rel=1e-6):
    tau = np.arange(N); n1 = np.minimum(Me, N - tau).astype(float)
    big = np.arange(-N - M, N + 1); rrb = xc(hS, hS, big); pb = xc(hS, hP, np.arange(-N, N + 1))
    C = np.zeros((N, M + 1))
    for j in range(M + 1):
        if j < M:
            f = lambda la: rrb[la - j + N + M]
        else:
            f = lambda la: pb[la + N]
        C[:, j] = (n1 * f(tau) + (Me - n1) * f(tau - N)) / Me
    Sn = np.abs(np.fft.rfft(hS, N)) ** 2; Fn = minphase(Sn + eps_rel * Sn.mean(), N)
    x = np.fft.irfft(np.fft.rfft(C, axis=0) / np.conj(Fn)[:, None], N, axis=0); x[N // 2:] = 0
    g = np.fft.irfft(np.fft.rfft(x, axis=0) / Fn[:, None], N, axis=0)[:M]
    return g[:, :M], g[:, M], x[:N // 2, :M]

def report(M, Me, N):
    A, b, Y = frame(M, Me, N)
    wM, gM = wiener(M); Jo = gM @ gM
    Phi = linalg.toeplitz(xc(hS, hS, np.arange(M)))
    wfp = -np.linalg.solve(A, b)
    ex = lambda w: dB((w - wM) @ Phi @ (w - wM) / Jo)
    wi = winf[:M]
    V = -wfp                                                 # state i: w = 0
    AV = A @ V
    hE = (V @ AV) / (V @ V); hP_ = (V @ Phi @ AV) / (V @ Phi @ V)
    nsE = ((AV - hE * V) @ (AV - hE * V)) / (hE ** 2 * (V @ V))
    r = AV - hP_ * V; nsP = (r @ Phi @ r) / (hP_ ** 2 * (V @ Phi @ V))
    sv = np.linalg.svd(A - np.eye(M), compute_uv=False)[0]
    ev = np.linalg.eigvals(A)
    # y metric of the frame (Y = P_{N/2} F-part of the direction before the inverse factor) vs Phi
    yy = (Y @ V) @ (Y @ V) / (V @ Phi @ V)
    print(f"M={M} Me={Me} N={N}: |A-I|_2={sv:.3f} eig(A) re [{ev.real.min():.3f},{ev.real.max():.3f}] | state i: h_eff E {hE:.3f} Phi {hP_:.3f}, "
          f"nonscalar E {nsE:.3f} Phi {nsP:.4f} | |Y V|^2/|V|^2_Phi {yy:.3f} | excess fp {ex(wfp):.2f} dB, [w_inf]_M {ex(wi):.2f} dB, "
          f"|wfp-[w_inf]_M|^2/|[w_inf]_M-wM|^2 {dB(((wfp-wi)@(wfp-wi))/((wi-wM)@(wi-wM))):.1f} dB")

if __name__ == "__main__":
    M = 512
    # exact operator identity P_M F^-1 P_T F P_M = I_M (lower-triangular Toeplitz F), T >= M
    T = 1024; FT = linalg.toeplitz(fimp[:T], np.zeros(T))
    FM = FT[:M, :M]; I1 = np.linalg.solve(FT, np.vstack([FM, np.zeros((T - M, M))]))[:M]
    print("identity check |P_M F^-1 P_M F P_M - I| =", np.abs(I1 - np.eye(M)).max())
    Phi = linalg.toeplitz(xc(hS, hS, np.arange(M)))
    Ftall = linalg.toeplitz(fimp[:8192], np.zeros(M))[:, :M]            # F restricted to M columns, 8192 rows
    print("Phi = F^T F (all rows) rel err", np.abs(Ftall.T @ Ftall - Phi).max() / Phi[0, 0])
    Gy = FM.T @ FM
    ev = linalg.eigh(Gy, Phi, eigvals_only=True)
    print(f"y metric F_M^T F_M vs Phi_M: generalised eig min {ev.min():.4f} median {np.median(ev):.4f} max {ev.max():.4f}")
    wM, _ = wiener(M)
    for nm, a in (("w_M", wM), ("[w_inf]_M - w_M", winf[:M] - wM)):
        print(f"  |F_M a|^2/|a|^2_Phi for a = {nm}: {(a @ Gy @ a) / (a @ Phi @ a):.4f}")
    for (Me, N) in ((512, 1024), (512, 2048), (512, 4096), (512, 8192), (128, 1024), (256, 1024), (128, 4096)):
        report(M, Me, N)
    # regularisation Lam = Srr + eps * mean(Srr): the measured non-scalar mean action (protoA uses eps = 1e-2)
    wM, gM = wiener(M); Jo = gM @ gM
    for eps in (1e-6, 1e-4, 1e-3, 1e-2):
        A, b, Y = frame(M, 512, 1024, eps_rel=eps); wfp = -np.linalg.solve(A, b); V = -wfp; AV = A @ V
        hE = V @ AV / (V @ V); hPh = V @ Phi @ AV / (V @ Phi @ V)
        nsE = ((AV - hE * V) @ (AV - hE * V)) / (hE ** 2 * (V @ V)); r = AV - hPh * V
        nsP = (r @ Phi @ r) / (hPh ** 2 * (V @ Phi @ V)); ev = np.linalg.eigvals(A).real
        print(f"eps {eps:g} (M=Me=512, N=1024): h_eff E {hE:.3f} Phi {hPh:.3f}, nonscalar E {nsE:.3f} Phi {nsP:.4f}, eig(A) [{ev.min():.3f},{ev.max():.3f}], "
              f"fp excess {dB((wfp - wM) @ Phi @ (wfp - wM) / Jo):.2f} dB")
