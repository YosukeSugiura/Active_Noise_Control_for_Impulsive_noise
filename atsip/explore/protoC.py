"""protoC: prototype C, the four method changes of THEORY_vss2.md Sec. 10.6 on the SF direction (closed loop,
adaptive normaliser exactly as in vssT_lib.simulate).  A prototype, not a confirmatory study.

Changes tested
  1. y-coordinate statistics (Sec. 10.4): C, Q from y_n = KD * x_n, x_n = the SF intermediate vector after the
     causal truncation and before the inverse factor 1/F (specfac.sf_dir(..., ret_y=True)); irfft(rfft(x)/F)[:M]
     is the direction, so y = F Delta up to the truncation to M lags.  Control: the same rules with C, Q on Delta.
  2. (X_hat, N_hat) from the Sec. 9.1 / 10.3 EMA forms with the Sec. 10.2 coefficient (tau_m f -> (2 tau_m - 1) f)
     and the mean action h_eff (default 0.95 h):
        C = [1 + h mu (lbar - (2tm-1) f)] X - h mu ta N + (1 + h mu) Gbar,   Gbar = gz N + gm f X  (gz = gm = 0 default)
        Q = (1 + f) X + N - h mu [(2tm-2) f X + (ta - 1) N]
     (identical to protoA.solve91 with tm -> 2 tm - 1).  mu in the forms = mu_state, capped at mu_fmax (default:
     h mu (ta - 1) <= 0.75, i.e. the Q coefficient of N stays >= 0.25; the forms are first order in h mu).
     X_hat clipped to [0, Q/(1+f)], N_hat = max(N, 0) from the unclipped solve.
  3. step targets
     onestep   mu_1 = [X/h_eff - mu ((ta-1) N + (2tm-2) f X) / 2]_+ / Q            (Sec. 9.3 style)
     delaycap  min(mu_1, a*/h_eff),  a* = argmin_a rho(a) of the delayed recursion v+ = v - a sum_k c_k v_{-k}
     delayopt  argmin over a 40-point grid in [0, 1.5] of the predicted T-block-ahead second moment (dopt_curve)
     mu_state <- clip((1 - gamma) mu_state + gamma target, 0, 1.5), start 0.3; no update before the estimator
     has data (block L + 4, as vssT DirectEst).
  4. power correction of the applied step: none | g (1/max(g, 0.25)) | dp (Qbar_y / |y_n|^2 clipped to
     [0.1, 10]; Qbar_y = EMA_beta |y|^2 *before* block n is included, first block factor 1) | dp+g (product).
     The statistics vector is the vector that mu_state multiplies: u_n = factor_n * y_n (or * Delta_n for the
     Euclidean control).  dp always uses |y_n|^2 (also in the Euclidean control, so that only change 1 differs).

Delay-opt model (scalar, per block, constant mu over the horizon):
  mean amplitude m_t (v units), history m_{-k} = sqrt(X_hat)/h_eff for k = 0..K (v assumed unchanged over the last
  K blocks), m_{t+1} = m_t - mu h_eff sum_k c_k m_{t-k}  (Sec. 10.1, c normalised to sum 1);
  fluctuation variance p_0 = 0, p_{t+1} = rho(mu h_eff)^2 p_t + mu^2 K'_t,
  K'_t = (2tm-1) f h_eff^2 x_t + ta N_hat (+ h_eff^2 x_t with --dopt-k full), x_t = m_t^2 + p_t;
  objective J(mu) = m_T^2 + p_T (T = --dopt-T, default 8).  'excl1' (default) leaves out the '1' of
  K = [1 + (2tm-1) f] h^2 x + ta N because the mean recursion already contains the mean's own mu^2 h^2 m^2
  term exactly; 'full' takes Sec. 10.2's K literally.

protoC2 (THEORY_vss2.md Sec. 11, estimator variants for the SF-Y rules with the 1/g correction; class Est2):
  est91  the protoC estimator (L = 4, beta_d = 0.1, lbar = 13, Sec. 9.1 linear forms); bitwise = 'SF-Y <t> g <gam>'
  estL4  beta_d = 1 (ubar_{n-L} = u_{n-L}), lbar = 4, same linear forms
  estX   estL4 statistics + Sec. 11.2 (b)-(d): C = kappa rho(a)^{-L} X + (N + f X)[r(L) - a sum_{j>=1} lam^{j-1}
         (r(L-j) + r(L+j))], lam = max(1 - a, 0), r(k) = (1 - |k|/4)_+ (all nonzero terms of the sum);
         Q = kappa (1+f) X + N - a_Q [(2tm-2) f X + (ta-1) N], a_Q = min(a, h_eff mu_fmax);
         a = h_eff * mean applied mu over blocks n-L..n-1, rho from a 1e-3 grid of the delayed-recursion root;
         2x2 solve, |det| < det_rel (|cX qN| + |cN qX|) keeps the previous (X, N); X, N >= 0.
  rules 'SF-Y <est> <delayopt|onestep> g gam<.03|.1>'
  c3   closed loop as c1 (12 estimator rules + controls SF-Y delayopt g gam.03, SF-V2, SF-D g gam0.03 a0 1, SF+ramp)
  c2x  estimator diagnostic: every block X_true = H^2 |P_M F v|^2 (v relative to the SF fixed point of the active
       path) next to the driver's own X_hat and shadow est91 / estL4 / estX fed with the same u and applied mu;
       N_true / Q_frz from frozen probes at sampled blocks; C2 oracle argmin for the SF-Y drivers (--no-oracle)
  report2 -> RESULTS_protoC2.md

protoC3 (early phase from theory and adaptive smoothing; SF-Y <est91|estX> delayopt g, class SFY3 / YEstRS):
  early  internal step = mu_early (dopt model with N = 0, K 'full' = [1 + (2tm-1) f] h^2 x; 0.678 with the defaults)
         for n_e blocks after the start / each restart (n_e: trust rule, at most 64, or --n-early), then delayopt EMA
  rs     outer EMAs with beta_k = max(beta_min, 1/(k+1)); restart when |u_n|^2 > r_thr Q_{n-1} for d blocks
  rules  'SF-Y <est> delayopt g gam.03 [early|rs|early+rs]', '... gam.1 early+rs'; no suffix = the protoC2 rule
  input  'pcb' (alpha 1.45, path -> pc path at 10 s, back at 20 s; vssT_lib.path_of via data['pseg'])
  c4 (closed loop), c4x (estimator diagnostic, own X_hat, restarts), report3 -> RESULTS_protoC3.md

protoC4 (restart statistics; class SFY4 / YEstRSX; restart actions as rs; no restart within --rs-guard [64] blocks):
  rsN    z_n = sum e^2 / sum r^2 over the M-sample error window (r = data['r'] = hS * x, the filtered reference),
         zbar = EMA_{z_beta} (never reset); restart when z_n > z_thr zbar_{n-1} for z_d blocks (0.05, 4, 2)
  rsN2   rsN with sum r^2 over the 2M samples ending at the block end (also the r samples that reach the window's
         errors through the M-tap filter; rsN fires when an impulse has just left the r window)
  rsG    |y_n|^2 / g_n^2 (no floor) against its own EMA (adaptive beta, reset with Q); r_thr, d as rs
  rules  'SF-Y <est91|estX> delayopt g gam.03 early+rs<N|N2|G>'; c5 (closed loop), report4 -> RESULTS_protoC4.md

Subcommands:  check | c1 | c2 | report [--smoke] | c3 | c2x | report2 [--smoke] | c4 | c4x | report3 [--smoke]
              | c5 | report4 [--smoke]
  c1  closed loop, one job per tag (Pool <= 2), out/protoC/protoC_C1_<tag>.json/.npz (rewritten after every rule;
      a rerun resumes at the first missing rule)
  c2  tracking check: at sampled blocks, mu_state and target vs argmin of the T-step mean Phi-excess of a
      constant-internal-step continuation with the same power-correction policy (T = 4, 8, 16)
Options: --tags / --classes --seeds, --nb (smoke -> out/protoC_smoke/), --procs, --force, --rules "a;b",
  --f --tau-a --tau-m --heff --ck --gz --gm --dopt-T --dopt-k --mu-fmax --rec-db
  protoC2: --kappa --estx-a applied|state --det-rel --no-aq-cap --no-oracle
  protoC3: --early-k --n-early --ne-max --trust-q --trust-dec --beta-min --r-thr --rs-d --rs-kmin
  protoC4: --z-beta --z-thr --z-d --rs-guard
"""
import os, sys, json, time, glob, argparse, copy
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vssT_lib import *        # noqa: F401,F403
import vssT_lib as T
import vssT as VT
import protoA as PA
from multiprocessing import Pool

OUT_FULL = os.path.join(HERE, "out", "protoC")
OUT_SMOKE = os.path.join(HERE, "out", "protoC_smoke")
C_DEF = (0.238, 0.247, 0.247, 0.247, 0.010)
MU_MAX, MU_INIT, DP_LO, DP_HI = 1.5, 0.3, 0.1, 10.0
EPS_REG = 1e-2
TARGETS = ("onestep", "delaycap", "delayopt")
CORRS = ("none", "g", "dp", "dp+g")
GAMS = (0.03, 0.1)
DOPT_GRID = np.linspace(0.0, 1.5, 40)
HGRID = np.concatenate([[0.0], np.geomspace(0.01, 3.0, 40)])    # C2 continuation grid (internal step)
TS = (4, 8, 16)
BPS = FS // B                                                    # 125 blocks per second
DEF_CLASSES = {"c1": "G,BU,SM,PS,II300,a145,a120,pc", "c2": "G,a145,pc", "c3": "G,BU,a145,pc", "c2x": "G,pc"}
DEF_SEEDS = {"c1": "0,1,2", "c2": "0", "c3": "0,1,2", "c2x": "0"}


# ==============================================================================================
# Delayed recursion (Sec. 10.1)
# ==============================================================================================
def rho_delay(a, c):
    """Largest root modulus of z^{K+1} - z^K + a sum_k c_k z^{K-k} (c as given; normalise before)."""
    c = np.asarray(c, float); K = len(c) - 1
    p = np.zeros(K + 2); p[0] = 1.0; p[1] = -1.0; p[1:] += a * c
    return float(np.max(np.abs(np.roots(p))))


def a_star(c, amax=1.2, step=1e-4):
    """argmin_a rho(a) on a grid of `step`, refined once by a 100x finer grid around the minimum."""
    c = np.asarray(c, float) / np.sum(c)
    g = np.arange(0.0, amax, step); r = np.array([rho_delay(a, c) for a in g]); k = int(np.argmin(r))
    g2 = np.linspace(max(0.0, g[k] - step), g[k] + step, 201); r2 = np.array([rho_delay(a, c) for a in g2])
    k2 = int(np.argmin(r2))
    return float(g2[k2]), float(r2[k2])


class Par:
    """Constants of the protoC rules (CLI)."""

    def __init__(s, f=0.85, ta=3.8, tm=4.0, heff=0.95 * H, ck=C_DEF, gz=0.0, gm=0.0, dopt_T=8, dopt_k="excl1",
                 mu_fmax=None, L=EST["L"], bd=EST["bd"], b=EST["b"], gmin=GMIN):
        s.f, s.ta, s.tm, s.h, s.gz, s.gm, s.L, s.bd, s.b, s.gmin = f, ta, tm, heff, gz, gm, L, bd, b, gmin
        s.c = np.asarray(ck, float) / np.sum(ck); s.dopt_T, s.dopt_k = int(dopt_T), dopt_k
        s.astar, s.rho_star = a_star(s.c)
        s.mu_D = s.astar / s.h
        s.mu_fmax = 0.75 / (s.h * (ta - 1)) if mu_fmax is None else mu_fmax
        s.rho2 = np.array([rho_delay(m * s.h, s.c) for m in DOPT_GRID]) ** 2

    def desc(s):
        return dict(f=s.f, tau_a=s.ta, tau_m=s.tm, h_eff=s.h, c=s.c.tolist(), gz=s.gz, gm=s.gm, a_star=s.astar,
                    rho_star=s.rho_star, mu_D=s.mu_D, mu_fmax=s.mu_fmax, dopt_T=s.dopt_T, dopt_k=s.dopt_k,
                    L=s.L, beta_d=s.bd, beta=s.b, gmin=s.gmin)


def solve_xn(C, Q, mu, P):
    """Sec. 9.1 forms with tm -> 2 tm - 1 (Sec. 10.2) and h = h_eff, solved for (X, N).
    Returns X (clipped to [0, Q/(1+f)]), N (>= 0, from the unclipped solve), clip flag (1 if X was clipped)."""
    h, f, ta, tme = P.h, P.f, P.ta, 2 * P.tm - 1
    lbar = P.L + (1 - P.bd) / P.bd
    a1 = 1 + h * mu * (lbar - tme * f) + (1 + h * mu) * f * P.gm
    a2 = -h * mu * ta + (1 + h * mu) * P.gz
    b1 = (1 + f) - h * mu * (tme - 1) * f
    b2 = 1 - h * mu * (ta - 1)
    det = a1 * b2 - a2 * b1
    X = (C * b2 - a2 * Q) / det; Nn = (a1 * Q - b1 * C) / det
    Xc = min(max(X, 0.0), Q / (1 + P.f))
    return Xc, max(Nn, 0.0), int(Xc != X)


def mu_onestep(X, Nn, Q, mu, P):
    return max(X / P.h - mu * ((P.ta - 1) * Nn + (2 * P.tm - 2) * P.f * X) / 2.0, 0.0) / Q


def dopt_curve(X, Nn, P, grid=DOPT_GRID, rho2=None):
    """Predicted second moment x_T for each mu of the grid (model in the module docstring)."""
    h, c = P.h, P.c; K = len(c) - 1
    rho2 = P.rho2 if rho2 is None else rho2
    fK = (2 * P.tm - 1) * P.f + (1.0 if P.dopt_k == "full" else 0.0)
    m = np.full((len(grid), K + 1), np.sqrt(max(X, 0.0)) / h)      # m[:, 0] newest
    p = np.zeros(len(grid))
    for _ in range(P.dopt_T):
        x = m[:, 0] ** 2 + p
        inj = grid ** 2 * (fK * h * h * x + P.ta * Nn)
        mn = m[:, 0] - grid * h * (m @ c)
        p = rho2 * p + inj
        m = np.concatenate([mn[:, None], m[:, :-1]], axis=1)
    return m[:, 0] ** 2 + p


def mu_delayopt(X, Nn, P):
    J = dopt_curve(X, Nn, P)
    return float(DOPT_GRID[int(np.argmin(J))])


# ==============================================================================================
# Estimator and rule
# ==============================================================================================
class YEst:
    """vssT DirectEst with the statistics vector u given by the caller (same buffering, start at block L + 4):
    C <- EMA_b <u_n, ubar_{n-L}>,  Q <- EMA_b |u_n|^2,  ubar <- EMA_bd(u)."""

    def __init__(s, L=EST["L"], bd=EST["bd"], b=EST["b"]):
        s.L, s.bd, s.b = L, bd, b
        s.ubar = None; s.buf = [None] * L; s.C = s.Q = None

    def update(s, i, u):
        old = s.buf[0]
        c = 0.0 if old is None else float(u @ old); q = float(u @ u)
        if i >= s.L + 4:
            if s.C is None:
                s.C, s.Q = c, q
            else:
                s.C = (1 - s.b) * s.C + s.b * c; s.Q = (1 - s.b) * s.Q + s.b * q
        s.ubar = u.copy() if s.ubar is None else (1 - s.bd) * s.ubar + s.bd * u
        s.buf = s.buf[1:] + [s.ubar.copy()]


def sf_y(c):
    """SF direction (identical arithmetic to vssT_lib.direction('SF')), y = KD x[:N/2], F."""
    dv, x, F = SFm.sf_dir(c.R, c.E, c.Srr, c.Pbar, c.eps, EPS_REG, ret_y=True)
    return dv, KD * x[:N // 2], F


class SFY(Rule):
    """coord 'Y' (y-coordinate statistics) or 'E' (statistics on Delta); target in TARGETS; corr in CORRS.
    est (Sec. 11 variants, protoC2): None = the protoC estimator (original code path, unchanged); 'est91' = the same
    estimator through the solver interface; 'estL4' / 'estX' (see Est2)."""
    kind = "SF"

    def __init__(s, coord, target, corr, gamma, P, mu_init=MU_INIT, est=None, X2=None):
        super().__init__()
        s.coord, s.target, s.corr, s.gamma, s.P = coord, target, corr, gamma, P
        s.est = YEst(P.L, P.bd, P.b); s.mu_state = mu_init; s.Qy = None; s.last = {}
        s.tr.update({k: [] for k in ("mu_state", "target", "X", "N", "C", "Q", "ypow", "dpclip", "mufclamp",
                                     "xclip", "stclip")})
        s.e2 = None
        if est is not None:
            s.e2 = Est2(est, P, X2 or X2_DEF); s.est = s.e2.y
            s.tr.update({k: [] for k in ("a", "rho", "detkeep")})

    def mu_formula(s, i):
        return s.mu_state

    def factor(s, g, py, Qy):
        """(factor, dp clip flag) for the applied step mu_state * factor; Qy = EMA before this block."""
        fac, clp = 1.0, 0
        if s.corr in ("dp", "dp+g"):
            r = 1.0 if Qy is None else (Qy / py if py > 0 else DP_HI)
            rc = min(max(r, DP_LO), DP_HI); clp = int(rc != r); fac = rc
        if s.corr in ("g", "dp+g"):
            fac = fac / max(g, s.P.gmin)
        return fac, clp

    def ema_y(s, Qy, py):
        return py if Qy is None else (1 - s.P.b) * Qy + s.P.b * py

    def block(s, i, c):
        dv, y, _ = sf_y(c)
        py = float(y @ y)
        fac, clp = s.factor(c.g, py, s.Qy)
        Qy0 = s.Qy; s.Qy = s.ema_y(s.Qy, py)
        mus = s.mu_state; mu = mus * fac
        s.est.update(i, fac * (y if s.coord == "Y" else KD * dv))
        tgt = X = Nn = np.nan; mfc = xc = stc = 0
        if s.e2 is not None:
            return s.block2(i, mu, mus, dv, py, clp, fac, Qy0)
        if s.est.C is not None and s.est.Q > 0:
            muf = min(mus, s.P.mu_fmax); mfc = int(muf != mus)
            X, Nn, xc = solve_xn(s.est.C, s.est.Q, muf, s.P)
            if s.target == "delayopt":
                tgt = mu_delayopt(X, Nn, s.P)
            else:
                tgt = mu_onestep(X, Nn, s.est.Q, muf, s.P)
                if s.target == "delaycap":
                    tgt = min(tgt, s.P.mu_D)
            ns = (1 - s.gamma) * mus + s.gamma * tgt
            s.mu_state = min(max(ns, 0.0), MU_MAX); stc = int(s.mu_state != ns)
        s.last = dict(mu_state=mus, target=tgt, fac=fac, Qy_after=s.Qy, Qy_before=Qy0)
        t = s.tr
        for k, v in (("mu_app", mu), ("mu_state", mus), ("target", tgt), ("X", X), ("N", Nn),
                     ("C", s.est.C if s.est.C is not None else np.nan), ("Q", s.est.Q if s.est.Q is not None else np.nan),
                     ("ypow", py), ("dpclip", clp), ("mufclamp", mfc), ("xclip", xc), ("stclip", stc)):
            t[k].append(v)
        return mu, dv

    def block2(s, i, mu, mus, dv, py, clp, fac, Qy0):
        """Sec. 11 estimator variants (s.est = s.e2.y has already been updated with u_n)."""
        tgt = np.nan; stc = 0
        X, Nn, xc, mfc, keep, ok = s.e2.solve(mus)
        if ok:
            if s.target == "delayopt":
                tgt = mu_delayopt(X, Nn, s.P)
            else:
                tgt = mu_onestep(X, Nn, s.est.Q, min(mus, s.P.mu_fmax), s.P)
                if s.target == "delaycap":
                    tgt = min(tgt, s.P.mu_D)
            ns = (1 - s.gamma) * mus + s.gamma * tgt
            s.mu_state = min(max(ns, 0.0), MU_MAX); stc = int(s.mu_state != ns)
        s.e2.push_mu(mu, fac)
        s.last = dict(mu_state=mus, target=tgt, fac=fac, Qy_after=s.Qy, Qy_before=Qy0)
        t = s.tr
        for k, v in (("mu_app", mu), ("mu_state", mus), ("target", tgt), ("X", X), ("N", Nn),
                     ("C", s.est.C if s.est.C is not None else np.nan), ("Q", s.est.Q if s.est.Q is not None else np.nan),
                     ("ypow", py), ("dpclip", clp), ("mufclamp", mfc), ("xclip", xc), ("stclip", stc),
                     ("a", s.e2.a), ("rho", s.e2.rho), ("detkeep", keep)):
            t[k].append(v)
        return mu, dv


# ==============================================================================================
# Sec. 11 estimator variants (protoC2)
# ==============================================================================================
EST2 = ("est91", "estL4", "estX")
X2_DEF = dict(kappa=1.0, amode="applied", det_rel=1e-3, aq_cap=True)
_RHO_GRID = {}


def rho_table(c, amax=3.0, step=1e-3):
    """rho(a) of the delayed recursion on a grid (cached per c); np.interp between grid points, clamped at amax."""
    key = tuple(np.round(np.asarray(c, float), 12))
    if key not in _RHO_GRID:
        g = np.arange(0.0, amax + step / 2, step)
        _RHO_GRID[key] = (g, np.array([rho_delay(a, c) for a in g]))
    return _RHO_GRID[key]


def r_tri(k, width=4.0):
    return max(1.0 - abs(k) / width, 0.0)


def geo_sum(lam, L, width=4.0):
    """S = sum_{j>=1} lam^{j-1} (r(L-j) + r(L+j)), triangular r, all nonzero terms (j = 1 .. L + width - 1)."""
    S = 0.0
    for j in range(1, int(L + width)):
        S += lam ** (j - 1) * (r_tri(L - j, width) + r_tri(L + j, width))
    return S


def estx_coefs(a, aq, P, kappa, rho):
    """Sec. 11.2 (b)-(c): C = cX X + cN N, Q = qX X + qN N.
    a: h_eff * mean applied mu over the last L blocks (rho and the geometric C cross terms, lam = max(1 - a, 0));
    aq: the a of the Q cross term (capped at h_eff mu_fmax by default, as the protoC Q form)."""
    L, f = P.L, P.f
    lam = max(1.0 - a, 0.0)
    gC = r_tri(L) - a * geo_sum(lam, L)               # Gamma_L - a sum lam^{j-1}(Gamma_{L-j} + Gamma_{L+j}) per (N + fX)
    cX = kappa * rho ** (-L) + f * gC; cN = gC
    qX = kappa * (1 + f) - aq * (2 * P.tm - 2) * f; qN = 1.0 - aq * (P.ta - 1)
    return cX, cN, qX, qN


class Est2:
    """Estimator variants for the SF-Y rules (statistics u = fac * y as in SFY):
      est91  YEst(L, beta_d = 0.1) + solve_xn (lbar = 13), identical arithmetic to the protoC rule
      estL4  YEst(L, beta_d = 1): ubar_{n-L} = u_{n-L}; solve_xn with lbar = 4 (Sec. 9.1 linear forms)
      estX   YEst(L, beta_d = 1) + Sec. 11.2 (b)-(d): rho(a)^{-L} X, geometric noise cross terms with triangular
             r(k) = (1 - |k|/4)_+ for the zeta and m parts, 2x2 solve; |det| < det_rel (|cX qN| + |cN qX|) keeps
             the previous (X, N); X, N clipped at >= 0.
    a = h_eff * mean(mu over blocks n-L .. n-1), mu = applied mu (amode 'applied', the Sec. 11.2 text) or mu_state
    = applied mu / factor (amode 'state'; with the 1/g correction this is the step of the mean recursion when the
    direction scales with g)."""

    def __init__(s, kind, P, X2):
        s.kind, s.P, s.X2 = kind, P, X2
        if kind == "est91":
            s.Pe = P; s.y = YEst(P.L, P.bd, P.b)
        elif kind in ("estL4", "estX"):
            s.Pe = copy.copy(P); s.Pe.bd = 1.0; s.y = YEst(P.L, 1.0, P.b)
        else:
            raise ValueError(kind)
        s.hist = []; s.prev = None; s.a = np.nan; s.rho = np.nan
        if kind == "estX":
            s.ag, s.rg = rho_table(P.c)

    def push_mu(s, mu_app, fac):
        s.hist.append(mu_app if s.X2["amode"] == "applied" else mu_app / fac)
        if len(s.hist) > s.P.L:
            s.hist.pop(0)

    def solve(s, mus):
        """-> X, N, xclip, mufclamp, detkeep, ok (ok False: no estimate, mu_state not updated)."""
        C, Q = s.y.C, s.y.Q
        if C is None or not Q > 0:
            return np.nan, np.nan, 0, 0, 0, False
        if s.kind != "estX":
            muf = min(mus, s.P.mu_fmax)
            X, Nn, xc = solve_xn(C, Q, muf, s.Pe)
            return X, Nn, xc, int(muf != mus), 0, True
        P = s.P
        s.a = P.h * (float(np.mean(s.hist)) if s.hist else mus)
        s.rho = float(np.interp(s.a, s.ag, s.rg))
        aq = min(s.a, P.h * P.mu_fmax) if s.X2["aq_cap"] else s.a
        cX, cN, qX, qN = estx_coefs(s.a, aq, P, s.X2["kappa"], s.rho)
        det = cX * qN - cN * qX
        if abs(det) < s.X2["det_rel"] * (abs(cX * qN) + abs(cN * qX)):
            if s.prev is None:
                return np.nan, np.nan, 0, int(aq != s.a), 1, False
            return s.prev[0], s.prev[1], 0, int(aq != s.a), 1, True
        X = (C * qN - cN * Q) / det; Nn = (cX * Q - qX * C) / det
        Xc, Nc = max(X, 0.0), max(Nn, 0.0)
        s.prev = (Xc, Nc)
        return Xc, Nc, int(Xc != X or Nc != Nn), int(aq != s.a), 0, True


def gl(g):
    return "gam" + (f"{g:g}".lstrip("0"))


def rule_name(coord, target, corr, gamma):
    return f"SF-{coord} {target} {corr} {gl(gamma)}"


SFY_NAMES = [rule_name(co, t, c, g) for co in ("Y", "E") for t in TARGETS for c in CORRS for g in GAMS]
CONTROLS = ["SF-D g gam0.03 a0 1", "SF+ramp", "SF-V2", "SF mu.1", "SF mu.5"]
C1_RULES = SFY_NAMES + CONTROLS
C2_RULES = [rule_name("Y", "delayopt", "dp", 0.03), rule_name("Y", "delaycap", "dp", 0.03),
            rule_name("Y", "delayopt", "g", 0.03), rule_name("Y", "delaycap", "g", 0.03)]


def rule_name2(est, target, corr, gamma):
    return f"SF-Y {est} {target} {corr} {gl(gamma)}"


TARGETS2 = ("delayopt", "onestep")
EST2_NAMES = [rule_name2(e, t, "g", g) for e in EST2 for t in TARGETS2 for g in GAMS]
CONTROLS2 = [rule_name("Y", "delayopt", "g", 0.03), "SF-V2", "SF-D g gam0.03 a0 1", "SF+ramp"]
C3_RULES = EST2_NAMES + CONTROLS2
C2X_RULES = EST2_NAMES + ["SF+ramp"]


def make_rules(nbr, P, X2=None):
    R = {}
    for e in EST2:
        for t in TARGETS2:
            for g in GAMS:
                R[rule_name2(e, t, "g", g)] = (lambda e=e, t=t, g=g: SFY("Y", t, "g", g, P, est=e, X2=X2))
    base = VT.rules_all(nbr)
    for k in ("SF-D g gam0.03 a0 1", "SF+ramp", "SF-V2", "SF mu.1"):
        R[k] = base[k]
    R["SF mu.5"] = lambda: Fixed("SF", 0.5)
    for co in ("Y", "E"):
        for t in TARGETS:
            for cr in CORRS:
                for g in GAMS:
                    R[rule_name(co, t, cr, g)] = (lambda co=co, t=t, cr=cr, g=g: SFY(co, t, cr, g, P))
    return R


# ==============================================================================================
# Output helpers
# ==============================================================================================
def outdir(nb):
    d = OUT_FULL if nb is None else OUT_SMOKE
    os.makedirs(d, exist_ok=True); return d


def fpath(exp, tag, nb, ext="json"):
    return os.path.join(outdir(nb), f"protoC_{exp}_{tag}.{ext}")


def save(exp, tag, nb, js, arrs=None):
    p = fpath(exp, tag, nb)
    with open(p + ".tmp", "w") as f:
        json.dump(js, f, default=lambda o: float(o) if np.isscalar(o) else np.asarray(o, float).tolist())
    os.replace(p + ".tmp", p)
    if arrs:
        q = fpath(exp, tag, nb, "npz")
        np.savez_compressed(q + ".tmp.npz", **arrs); os.replace(q + ".tmp.npz", q)


def dbm(a):
    a = np.asarray(a, float)
    return float(10 * np.log10(np.mean(a))) if len(a) and np.all(np.isfinite(a)) else float("nan")


def tags_of(cmd, o):
    if o.tags:
        return o.tags.split(",")
    return [f"{c}_s{s}" for c in (o.classes or DEF_CLASSES[cmd]).split(",") for s in (o.seeds or DEF_SEEDS[cmd]).split(",")]


# ==============================================================================================
# C1: closed loop
# ==============================================================================================
PS_STEPS = (10, 15, 20, 25)
REL_DB = 3.0


def phases(nbr):
    n10 = max(1, nbr // 10)
    return {"0-1s": (0, min(nbr, BPS)), "1-5s": (min(nbr, BPS), min(nbr, 5 * BPS)),
            "5s-90%": (min(nbr, 5 * BPS), max(min(nbr, 5 * BPS), nbr - n10)), "last10": (nbr - n10, nbr)}


def summaries(sim, dd, nbr, rec_db):
    ex = sim["ex"]; cls = dd["tag"].rsplit("_s", 1)[0]
    n10 = max(1, nbr // 10)
    out = dict(ex_first1=dbm(ex[:min(nbr, BPS)]), ex_first5=dbm(ex[:min(nbr, 5 * BPS)]), ex_last10=dbm(ex[-n10:]))
    if dd["pc"]:
        out["ex_pc_15_17"] = dbm(ex[PC_BLOCK:min(nbr, PC_BLOCK + 2 * BPS)]) if nbr > PC_BLOCK else np.nan
        out["ex_pc_25_30"] = dbm(ex[25 * BPS:]) if nbr > 25 * BPS else np.nan
    if cls == "PS":
        exd = 10 * np.log10(ex)
        for k, ts in enumerate(PS_STEPS):
            b = ts * BPS; nxt = PS_STEPS[k + 1] * BPS if k + 1 < len(PS_STEPS) else nbr
            nxt = min(nxt, nbr)
            if b >= nbr or sim["div"]:
                out[f"ps_max_{ts}"] = out[f"ps_rec_{ts}"] = np.nan; continue
            seg = exd[b:min(b + 2 * BPS, nxt)]; kp = int(np.nanargmax(seg))
            out[f"ps_max_{ts}"] = float(seg[kp])
            pre = dbm(ex[max(0, b - BPS):b]) + REL_DB      # relative level: 1 s pre-step mean + 3 dB
            for key, lev in ((f"ps_rec_{ts}", rec_db), (f"ps_recrel_{ts}", pre)):
                if seg[kp] <= lev:
                    out[key] = 0.0
                else:
                    after = np.nonzero(exd[b + kp:nxt] <= lev)[0]
                    out[key] = float((kp + after[0]) / BPS) if len(after) else np.nan
    if cls == "BU" and not sim["div"]:
        p0 = 5 if nbr >= 7 * BPS else 1
        nper = nbr // BPS
        if nper > p0:
            F = ex[p0 * BPS:nper * BPS].reshape(-1, BPS).mean(0)
            out.update(bu_fold=F, bu_peak=float(10 * np.log10(F.max())), bu_burst=dbm(F[:BPS // 4]),
                       bu_base=dbm(F[BPS // 2:]))
    return out


def c1_job(tag, o, exp="C1"):
    """exp 'C1' (protoC rules) or 'C3' (protoC2: Sec. 11 estimator variants + controls); same code otherwise."""
    nb = o.nb; t0 = time.time(); P = o.P
    dd, Rall, st = T.prep(tag, nb)
    nbr = (len(dd["d"]) // B) if nb is None else nb
    RR = make_rules(nbr, P, o.X2)
    pdesc = P.desc() if exp == "C1" else par2_desc(o)
    p0 = fpath(exp, tag, nb)
    res = dict(tag=tag, nb=nbr, pc=dd["pc"], par=pdesc, rules={}, done=False)
    arrs = dict(g=np.asarray(st.g[:nbr], np.float32))
    if os.path.exists(p0) and not o.force:
        old = json.load(open(p0))
        if old.get("nb") == nbr and old.get("par") == json.loads(json.dumps(pdesc)):
            res["rules"] = old["rules"]
            q = fpath(exp, tag, nb, "npz")
            if os.path.exists(q):
                arrs.update(dict(np.load(q)))
    for name in (o.rules or (C1_RULES if exp == "C1" else C3_RULES)):
        if name in res["rules"]:
            continue
        t1 = time.time()
        sim = simulate(dd, Rall, st, RR[name](), nb=nbr)
        mt = metrics(sim, dd)
        r = dict(**mt, **summaries(sim, dd, nbr, o.rec_db), sec=time.time() - t1)
        r["amp"] = bool((not r["div"]) and np.isfinite(r["L1"][0]) and r["L1"][0] > 0)
        tr = sim["tr"]; arrs[name + "|ex"] = np.asarray(sim["ex"], np.float32)
        for k, v in tr.items():
            arrs[name + "|" + k] = np.asarray(v, np.float32)
        mua = np.asarray(tr["mu_app"], float)
        r["mu_app_ph"] = {k: (float(np.mean(mua[a:b])) if b > a and b <= len(mua) else np.nan) for k, (a, b) in phases(nbr).items()}
        if dd["pc"] and len(mua) > PC_BLOCK:
            r["mu_app_ph"]["pc15-17"] = float(np.mean(mua[PC_BLOCK:PC_BLOCK + 2 * BPS]))
        if "mu_state" in tr:
            r["zero_frac"], r["zero_run"] = zero_stats(tr["mu_state"])
            for k in ("dpclip", "mufclamp", "xclip", "stclip", "detkeep"):
                if k in tr and len(tr[k]):
                    r["frac_" + k] = float(np.mean(tr[k]))
        res["rules"][name] = r
        save(exp, tag, nb, res, arrs)
        print(f"[{exp} {tag}] {name}: L1 {np.round(r['L1'], 2)} ex1s {r['ex_first1']:.2f} last10 {r['ex_last10']:.2f} dB "
              f"div={r['div']} amp={r['amp']} ({r['sec']:.1f} s, total {time.time() - t0:.0f} s)", flush=True)
    res["done"] = True; save(exp, tag, nb, res, arrs)
    return tag


# ==============================================================================================
# C2: tracking check
# ==============================================================================================
C2_BLOCKS = [8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 2560, 3072, 3600]
C2_PC = [0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512, 1024]


def continuation_corr(data, Rall, st, w, zi, e_all, i, dv_i, fac_i, Qy, mu_int, Tn, rule):
    """vssT_lib.continuation with applied step mu_int * factor_j, factor_j from rule.factor (same policy, dp EMA
    carried on from Qy = the EMA after block i).  Returns the Phi-excess after each of the Tn updates."""
    d, xq, hS = data["d"], data["xq"], data["hS"]
    swv = sliding_window_view(xq, M)
    lo = max(0, i * B + B - M)
    eh = np.zeros((i + Tn) * B + B - lo); eh[:i * B + B - lo] = e_all[lo:i * B + B]
    w = w - 2 * mu_int * fac_i * B / M * dv_i; z = zi.copy(); out = np.full(Tn, np.inf)
    for t in range(Tn):
        j = i + t
        p = path_of(data, j); wr, _, Jr = REFS[p]
        if not np.isfinite(w).all():
            break
        out[t] = exP(w - wr, Jr)
        if t == Tn - 1 or j + 1 >= len(Rall) or not out[t] < DIV_EX:
            break
        j += 1; s0 = j * B
        Xb = swv[s0 + N - M + 1:s0 + N - M + 1 + B][:, ::-1]
        ys, z = signal.lfilter(hS, [1.0], Xb @ w, zi=z)
        eh[s0 - lo:s0 + B - lo] = d[s0:s0 + B] + ys
        E = frame_from(eh, lo, s0)
        dv, x, _ = SFm.sf_dir(Rall[j], E, st.Srr[j], st.Pbar[j], st.eps[j], EPS_REG, ret_y=True)
        y = KD * x[:N // 2]; py = float(y @ y)
        fac, _ = rule.factor(st.g[j], py, Qy); Qy = rule.ema_y(Qy, py)
        w = w - 2 * mu_int * fac * B / M * dv
    return out


class C2Hook:
    def __init__(s, data, Rall, st, blocks, Tmax):
        s.data, s.Rall, s.st, s.blocks, s.Tmax = data, Rall, st, set(blocks), Tmax; s.recs = []

    def __call__(s, i, w, mu, dv, c, rule, mf, zi=None, e_all=None):
        if i not in s.blocks or not np.isfinite(w).all():
            return
        L_ = rule.last; p = path_of(s.data, i); wr, _, Jr = REFS[p]
        exT = np.array([continuation_corr(s.data, s.Rall, s.st, w, zi, e_all, i, dv, L_["fac"], L_["Qy_after"], m,
                                          s.Tmax, rule) for m in HGRID])
        rec = dict(i=i, path=p, g=c.g, mu_state=L_["mu_state"], target=L_["target"], mu_app=mu, fac=L_["fac"],
                   X=rule.tr["X"][-1], N=rule.tr["N"][-1], Q=rule.tr["Q"][-1], ex0=exP(w - wr, Jr))
        for Tn in TS:
            if Tn <= s.Tmax:
                avg = exT[:, :Tn].mean(1)
                k = int(np.argmin(avg)); rec[f"or_T{Tn}"] = float(HGRID[k]); rec[f"or_T{Tn}_ex"] = float(avg[k])
        s.recs.append(rec)


def c2_job(tag, o):
    nb = o.nb; t0 = time.time(); P = o.P
    dd, Rall, st = T.prep(tag, nb)
    nbr = (len(dd["d"]) // B) if nb is None else nb
    Tmax = max(TS)
    blocks = [b for b in C2_BLOCKS if b < nbr - Tmax]
    if dd["pc"]:
        blocks = sorted(set(blocks + [PC_BLOCK + k for k in C2_PC if PC_BLOCK + k < nbr - Tmax]))
    RR = make_rules(nbr, P)
    res = dict(tag=tag, nb=nbr, par=P.desc(), blocks=blocks, rules={}, done=False)
    for name in (o.rules or C2_RULES):
        hk = C2Hook(dd, Rall, st, blocks, Tmax)
        sim = simulate(dd, Rall, st, RR[name](), nb=nbr, hook=hk)
        res["rules"][name] = {k: [r.get(k, np.nan) for r in hk.recs] for k in (hk.recs[0].keys() if hk.recs else [])}
        res["rules"][name]["div"] = sim["div"]
        save("C2", tag, nb, res)
        print(f"[C2 {tag}] {name}: {len(hk.recs)} records, div={sim['div']} ({time.time() - t0:.0f} s)", flush=True)
    res["done"] = True; save("C2", tag, nb, res)
    return tag


# ==============================================================================================
# C2X: estimator diagnostic (protoC2)
# ==============================================================================================
C2X_PROBE = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 2560, 3072, 3600]
PROBE_HALF = 16                     # frozen probe window: blocks [i - 16, i + 16) on the same path


def par2_desc(o):
    return dict(o.P.desc(), **{"x2_" + k: v for k, v in o.X2.items()})


def phases2(nbr, pc, fine=False):
    """Diagnostic phases (blocks): early < 64, 64-256, 256-1024, > 1024 (pc: up to 15 s), pc: after 15 s.
    fine (protoC3): pc also split into +0-63, +64-255, +256.. blocks after the change."""
    end1 = min(nbr, PC_BLOCK) if pc else nbr
    ph = {"8-63": (8, min(64, end1)), "64-255": (64, min(256, end1)), "256-1023": (256, min(1024, end1)),
          ">=1024": (1024, end1)}
    if pc:
        ph["pc >=15 s"] = (PC_BLOCK, nbr)
        if fine:
            ph.update({"pc +0-63": (PC_BLOCK, min(nbr, PC_BLOCK + 64)), "pc +64-255": (PC_BLOCK + 64, min(nbr, PC_BLOCK + 256)),
                       "pc +256..": (PC_BLOCK + 256, nbr)})
    return {k: v for k, v in ph.items() if v[1] > v[0]}


def x_true(v, F):
    """y-coordinate X of v: H^2 |P_M (F v)|^2 (F: that block's minimum-phase factor; P_M: lags 0..M-1).
    E u = kappa_y * H P_M F v with u = y / max(g, gmin); kappa_y ~ 0.96-0.98 at G_s0 (check 13)."""
    Fv = np.fft.irfft(np.fft.rfft(v, N) * F, N)[:M]
    return H * H * float(Fv @ Fv)


def probe_u(data, Rall, st, w, blocks, gmin):
    """u = y / max(g, gmin) at the given blocks with the coefficients frozen at w."""
    lo = max(0, blocks[0] * B + B - M); hi = blocks[-1] * B + B
    es = frozen_error(data, w, lo, hi); out = []
    for j in blocks:
        E = frame_from(es, lo, j * B)
        _, x, _ = SFm.sf_dir(Rall[j], E, st.Srr[j], st.Pbar[j], st.eps[j], EPS_REG, ret_y=True)
        out.append((1.0 / max(st.g[j], gmin)) * (KD * x[:N // 2]))
    return np.array(out)


class XHook:
    """Every block: X_true (x_true of v = w - (w_ref + v_fp(path))) and shadow estimators est91 / estL4 / estX fed
    with u = (1/max(g, gmin)) y and the driver's applied mu (solve mu: the driver's mu_state for SF-Y rules, else
    applied mu * max(g, gmin)).  At the probe blocks: N_true = mean |u|^2 with w frozen at w_ref + v_fp (path of
    that block) and Q_frz = mean |u|^2 with w frozen at the current w, over [i - 16, i + 16) on the same path.
    An optional inner hook (C2Hook, oracle argmin) is called as well."""

    def __init__(s, data, Rall, st, P, X2, vfp, probes, nbr, inner=None):
        s.data, s.Rall, s.st, s.P, s.X2, s.vfp, s.probes, s.nbr, s.inner = data, Rall, st, P, X2, vfp, set(probes), nbr, inner
        s.sh = {k: Est2(k, P, X2) for k in EST2}
        s.rec = {k: [] for k in ["i", "Xtrue", "ex0"] + [f"{k}_{q}" for k in EST2 for q in ("X", "N")]}
        s.prb = {k: [] for k in ("i", "Ntrue", "Qfrz")}

    def __call__(s, i, w, mu, dv, c, rule, mf, zi=None, e_all=None):
        if s.inner is not None:
            s.inner(i, w, mu, dv, c, rule, mf, zi=zi, e_all=e_all)
        if not np.isfinite(w).all():
            return
        _, y, F = sf_y(c)
        fac = 1.0 / max(c.g, s.P.gmin); u = fac * y
        mus = rule.last["mu_state"] if isinstance(rule, SFY) else mu / fac
        p = path_of(s.data, i); wr, _, Jr = REFS[p]; v = w - wr - s.vfp[p]
        s.rec["i"].append(i); s.rec["Xtrue"].append(x_true(v, F)); s.rec["ex0"].append(exP(w - wr, Jr))
        for k, e in s.sh.items():
            e.y.update(i, u)
            X, Nn, _, _, _, ok = e.solve(mus)
            e.push_mu(mu, fac)
            s.rec[f"{k}_X"].append(X if ok else np.nan); s.rec[f"{k}_N"].append(Nn if ok else np.nan)
        if i in s.probes:
            p0 = PC_BLOCK if p == 1 else 0; p1 = s.nbr if p == 1 or not s.data["pc"] else PC_BLOCK
            bl = list(range(max(8, p0, i - PROBE_HALF), min(p1, i + PROBE_HALF, len(s.Rall))))
            Uf = probe_u(s.data, s.Rall, s.st, wr + s.vfp[p], bl, s.P.gmin)
            Uw = probe_u(s.data, s.Rall, s.st, w, bl, s.P.gmin)
            s.prb["i"].append(i); s.prb["Ntrue"].append(float(np.mean(np.sum(Uf ** 2, 1))))
            s.prb["Qfrz"].append(float(np.mean(np.sum(Uw ** 2, 1))))


def xstats(rec, prb, own, nbr, pc, fine=False):
    """Per phase: median X_hat / X_true, fraction X_hat == 0, median N_hat / N_true at the probe blocks."""
    ii = np.asarray(rec["i"]); Xt = np.asarray(rec["Xtrue"], float)
    pi_ = np.asarray(prb["i"]); Nt = np.asarray(prb["Ntrue"], float)
    cols = {k: k for k in EST2}
    if own is not None:
        cols = dict(own="own", **cols)
    out = {}
    for ph, (a, b) in phases2(nbr, pc, fine).items():
        m = (ii >= a) & (ii < b); mp = (pi_ >= a) & (pi_ < b); d = {}
        for k in cols:
            X = np.asarray(own["X"] if k == "own" else rec[f"{k}_X"], float)
            Nn = np.asarray(own["N"] if k == "own" else rec[f"{k}_N"], float)
            mm = m & np.isfinite(X) & (Xt > 0)
            d[k] = dict(xr=float(np.median(X[mm] / Xt[mm])) if mm.any() else np.nan,
                        x0=float(np.mean(X[mm] == 0)) if mm.any() else np.nan)
            pos = {int(b): q for q, b in enumerate(ii)}
            nr = [Nn[pos[int(b)]] / t for b, t in zip(pi_[mp], Nt[mp])
                  if int(b) in pos and np.isfinite(Nn[pos[int(b)]]) and t > 0]
            d[k]["nr"] = float(np.median(nr)) if nr else np.nan
        d["Xtrue_med"] = float(np.median(Xt[m])) if m.any() else np.nan
        d["Ntrue_med"] = float(np.median(Nt[mp])) if mp.any() else np.nan
        out[ph] = d
    return out


def c2x_job(tag, o):
    nb = o.nb; t0 = time.time(); P = o.P
    dd, Rall, st = T.prep(tag, nb)
    nbr = (len(dd["d"]) // B) if nb is None else nb
    vfp = {0: fixed_point(("SF",), 0), 1: fixed_point(("SF",), 1)}
    probes = [b for b in C2X_PROBE if b < nbr]
    Tmax = max(TS)
    oblocks = [b for b in C2_BLOCKS if b < nbr - Tmax]
    if dd["pc"]:
        probes = sorted(set(probes + [PC_BLOCK + k for k in C2_PC if PC_BLOCK + k < nbr]))
        oblocks = sorted(set(oblocks + [PC_BLOCK + k for k in C2_PC if PC_BLOCK + k < nbr - Tmax]))
    RR = make_rules(nbr, P, o.X2)
    res = dict(tag=tag, nb=nbr, pc=dd["pc"], par=par2_desc(o), blocks=oblocks, probes=probes, rules={}, done=False)
    arrs = {}
    for name in (o.rules or C2X_RULES):
        rule = RR[name]()
        inner = C2Hook(dd, Rall, st, oblocks, Tmax) if (o.oracle and isinstance(rule, SFY)) else None
        hk = XHook(dd, Rall, st, P, o.X2, vfp, probes, nbr, inner)
        sim = simulate(dd, Rall, st, rule, nb=nbr, hook=hk)
        own = None
        if isinstance(rule, SFY):
            n_ = len(hk.rec["i"]); own = dict(X=sim["tr"]["X"][:n_], N=sim["tr"]["N"][:n_])
        x = {k: [r.get(k, np.nan) for r in inner.recs] for k in (inner.recs[0].keys() if inner and inner.recs else [])}
        x.update(div=sim["div"], probe=hk.prb, xstats=xstats(hk.rec, hk.prb, own, nbr, dd["pc"]))
        res["rules"][name] = x
        for k, v in hk.rec.items():
            arrs[name + "|" + k] = np.asarray(v, float)
        if own is not None:
            arrs[name + "|own_X"] = own["X"]; arrs[name + "|own_N"] = own["N"]
        save("C2X", tag, nb, res, arrs)
        print(f"[C2X {tag}] {name}: div={sim['div']} ({time.time() - t0:.0f} s)", flush=True)
    res["done"] = True; save("C2X", tag, nb, res, arrs)
    return tag


# ==============================================================================================
# protoC3: early phase from theory (A, 'early') and adaptive smoothing with restarts (B, 'rs')
# ==============================================================================================
X3_DEF = dict(early_k="full", n_early=0, ne_max=64, trust_q=2.0, trust_dec=0.5, beta_min=0.05, r_thr=4.0, d=2,
              rs_kmin=0)
EARLY_GRID = np.round(np.arange(0.0, MU_MAX + 1e-9, 1e-3), 6)
EST3 = ("est91", "estX")
VARS3 = ("", "early", "rs", "early+rs")
PCB_T = (10.0, 20.0)                # pcb: primary path changed at 10 s (to the pc path) and back at 20 s
RS_TRUE_WIN = BPS                   # a restart within 1 s after a true change counts as a detection


def mu_early_value(P, kmode="full"):
    """Sec. 10.1-10.2 model with N = 0 (noise-free): argmin_mu of dopt_curve(X, 0) on a 1e-3 grid in [0, 1.5].
    With N = 0 the model is homogeneous in X (m ~ sqrt X, p ~ X), so the argmin does not depend on X.
    kmode 'full': injection mu^2 [1 + (2tm-1) f] h_eff^2 x (the K of Sec. 10.2, as in the protoC3 spec);
    'excl1': the delayopt target's default (the '1' left out)."""
    Pq = copy.copy(P); Pq.dopt_k = kmode
    r2 = np.array([rho_delay(m * P.h, P.c) for m in EARLY_GRID]) ** 2
    J = dopt_curve(1.0, 0.0, Pq, grid=EARLY_GRID, rho2=r2)
    return float(EARLY_GRID[int(np.argmin(J))])


class YEstRS(YEst):
    """YEst with adaptive smoothing (rs): C, Q <- (1 - b_k) . + b_k new, b_k = max(beta_min, 1/(k+1)), k = updates
    since the last (re)start (separate counts for C and Q).  Restart: |u_n|^2 > r_thr * Q_{n-1} for d consecutive
    blocks (and at least rs_kmin Q updates since the last restart) -> C = Q = None, k = 0, ubar and the delayed buffer
    are rebuilt from the d - 1 previous trigger blocks (no statistics), then u_n is processed; C resumes once the
    lag-L partner is a post-restart vector (L - d + 1 blocks later), Q at once.  Before the first restart the
    statistics start at block L + 4 as in YEst."""

    def __init__(s, L, bd, b, X3):
        super().__init__(L, bd, b)
        s.bmin, s.rthr, s.d, s.kmin = X3["beta_min"], X3["r_thr"], int(X3["d"]), int(X3["rs_kmin"])
        s.kC = s.kQ = 0; s.cnt = 0; s.raw = []; s.restarted = False; s.nrst = 0

    def _push(s, u):
        s.ubar = u.copy() if s.ubar is None else (1 - s.bd) * s.ubar + s.bd * u
        s.buf = s.buf[1:] + [s.ubar.copy()]

    def update(s, i, u):
        q = float(u @ u); s.restarted = False
        if s.Q is not None and s.kQ >= s.kmin:
            s.cnt = s.cnt + 1 if q > s.rthr * s.Q else 0
        s.raw = (s.raw + [u.copy()])[-s.d:]
        if s.cnt >= s.d:
            s.C = s.Q = None; s.kC = s.kQ = 0; s.cnt = 0; s.ubar = None; s.buf = [None] * s.L
            for uu in s.raw[:-1]:
                s._push(uu)
            s.restarted = True; s.nrst += 1
        old = s.buf[0]
        if i >= s.L + 4 or s.nrst > 0:
            bq = max(s.bmin, 1.0 / (s.kQ + 1))
            s.Q = q if s.Q is None else (1 - bq) * s.Q + bq * q; s.kQ += 1
            if old is not None:
                c = float(u @ old); bc = max(s.bmin, 1.0 / (s.kC + 1))
                s.C = c if s.C is None else (1 - bc) * s.C + bc * c; s.kC += 1
        s._push(u)


class SFY3(SFY):
    """SF-Y <est> delayopt g <gam> with the protoC3 options (est in est91 / estX):
      early  internal step = mu_early (constant, mu_early_value) for the first n_e blocks after the start (and after
             every rs restart); n_e = X3 n_early if > 0, else the first block where X_hat is 'trusted', at most ne_max:
             estimator ok, at least 2L blocks since the (re)start, Q <= trust_q * mean(|u|^2, last L blocks) (the outer
             EMA no longer dominated by older, larger vectors) and mean(|u|^2, last L) >= trust_dec * mean(|u|^2, the L
             blocks before) (the fast decay is over).  Then the delayopt EMA takes over from mu_state = mu_early.
      rs     YEstRS (adaptive beta, restarts).  On a restart: mu_state = mu_early and the early phase restarts if early
             is on, else mu_state is kept; estX's kept (X, N) for the det test is cleared.
    The estimator (C, Q, X_hat, N_hat) runs during the early phase as well (traces), only the target is not used."""

    def __init__(s, est, gamma, P, X2, X3, early, rs, mu_e):
        super().__init__("Y", "delayopt", "g", gamma, P, mu_init=(mu_e if early else MU_INIT), est=est, X2=X2)
        s.X3, s.early, s.rs, s.mu_e = X3, early, rs, mu_e
        if rs:
            s.e2.y = YEstRS(s.e2.y.L, s.e2.y.bd, s.e2.y.b, X3); s.est = s.e2.y
        s.in_early = bool(early); s.ke = 0; s.qh = []; s.handover = []; s.restarts = []
        s.tr.update({k: [] for k in ("early", "restart")})

    def trusted(s, ok):
        X3, L = s.X3, s.P.L
        if X3["n_early"] > 0:
            return s.ke >= X3["n_early"]
        if s.ke >= X3["ne_max"]:
            return True
        if not ok or len(s.qh) < 2 * L:
            return False
        m1, m0 = float(np.mean(s.qh[-L:])), float(np.mean(s.qh[-2 * L:-L]))
        return s.est.Q <= X3["trust_q"] * m1 and m1 >= X3["trust_dec"] * m0

    def block(s, i, c):
        dv, y, _ = sf_y(c)
        py = float(y @ y)
        fac, clp = s.factor(c.g, py, s.Qy)
        Qy0 = s.Qy; s.Qy = s.ema_y(s.Qy, py)
        mus = s.mu_state; mu = mus * fac
        u = fac * y
        s.est.update(i, u)
        rst = int(bool(s.rs and s.est.restarted))
        if rst:
            s.restarts.append(i); s.e2.prev = None; s.qh = []
            if s.early:
                s.in_early = True; s.ke = 0
        s.qh = (s.qh + [float(u @ u)])[-2 * s.P.L:]
        X, Nn, xc, mfc, keep, ok = s.e2.solve(mus)
        tgt = np.nan; stc = 0; ine = int(s.in_early)
        if s.in_early:
            s.ke += 1; s.mu_state = s.mu_e; tgt = s.mu_e
            if s.trusted(ok):
                s.in_early = False; s.handover.append(i)
        elif ok:
            tgt = mu_delayopt(X, Nn, s.P)
            ns = (1 - s.gamma) * mus + s.gamma * tgt
            s.mu_state = min(max(ns, 0.0), MU_MAX); stc = int(s.mu_state != ns)
        s.e2.push_mu(mu, fac)
        s.last = dict(mu_state=mus, target=tgt, fac=fac, Qy_after=s.Qy, Qy_before=Qy0)
        t = s.tr
        for k, v in (("mu_app", mu), ("mu_state", mus), ("target", tgt), ("X", X), ("N", Nn),
                     ("C", s.est.C if s.est.C is not None else np.nan), ("Q", s.est.Q if s.est.Q is not None else np.nan),
                     ("ypow", py), ("dpclip", clp), ("mufclamp", mfc), ("xclip", xc), ("stclip", stc),
                     ("a", s.e2.a), ("rho", s.e2.rho), ("detkeep", keep), ("early", ine), ("restart", rst)):
            t[k].append(v)
        return mu, dv


def rule_name3(est, gamma, var):
    return rule_name2(est, "delayopt", "g", gamma) + (f" {var}" if var else "")


C4_NEW = ([rule_name3(e, 0.03, v) for e in EST3 for v in VARS3] + [rule_name3(e, 0.1, "early+rs") for e in EST3])
C4_RULES = C4_NEW + ["SF-V2", "SF+ramp", "SF-D g gam0.03 a0 1"]
C4X_RULES = list(C4_NEW)
DEF_CLASSES.update({"c4": "G,BU,PS,a145,a120,pc,pcb", "c4x": "G,pc"})
DEF_SEEDS.update({"c4": "0,1,2", "c4x": "0"})


def make_rules3(nbr, P, X2, X3, mu_e):
    R = make_rules(nbr, P, X2)
    for e in EST3:
        for g in GAMS:
            for v in VARS3[1:]:
                R[rule_name3(e, g, v)] = (lambda e=e, g=g, v=v: SFY3(e, g, P, X2, X3, "early" in v, "rs" in v, mu_e))
    return R


def pcb_blocks():
    return [int(t * FS) // B for t in PCB_T]


def prep3(tag, nb=None):
    """T.prep, plus class 'pcb' (data_pcb)."""
    if cls_of(tag) != "pcb":
        return T.prep(tag, nb)
    dd = data_pcb(tag)
    Rall = ML.prepass(dd)["Rall"]
    if nb is not None:
        Rall = Rall[:min(len(Rall), nb + 64)]
    return dd, Rall, Stats(dd, Rall)


def data_pcb(tag):
    """Class 'pcb': alpha 1.45 input as pc (same x for the same seed); primary path -> the pc path (0.8 x, +8 samples)
    at 10 s and back to the original at 20 s.  path_of via data['pseg']; data['pc'] = False so that the 15 s pc
    windows / score_pc are not used."""
    seed = int(tag.rsplit("_s", 1)[1])
    x = O.cms_symmetric_stable(1.45, O.N_NOISE, seed)
    dd = D.build(x)
    hP = O.pad_or_trim(np.loadtxt(os.path.join(O.ROOT, "impulse_response/primary_ir.dat"), ndmin=1), M)
    hPn = np.zeros(M); hPn[MFm.PC_DELAY:] = MFm.PC_SCALE * hP[:M - MFm.PC_DELAY]
    n1, n2 = (int(t * FS) for t in PCB_T)
    d_new = signal.lfilter(hPn, [1.0], x)
    dd["d"] = np.concatenate([dd["d"][:n1], d_new[n1:n2], dd["d"][n2:]])
    dd["w_opt"] = O.wiener_opt(dd["r"][n1:n2], dd["d"][n1:n2])
    dd["pc"] = False; dd["pseg"] = [(n1 // B, 1), (n2 // B, 0)]; dd["tag"] = tag
    return dd


def change_blocks(dd):
    if dd.get("pseg"):
        return [b for b, _ in dd["pseg"]]
    return [PC_BLOCK] if dd.get("pc") else []


def restart_stats(rst, changes, nbr):
    """Restart blocks vs true change blocks: delay (blocks) of the first restart within RS_TRUE_WIN after each change
    (NaN: missed), false restarts = all others, per 30 s of run."""
    rst = list(rst); used = set(); delays = []
    for cb in changes:
        if cb >= nbr:
            continue
        hit = [r for r in rst if cb <= r < cb + RS_TRUE_WIN]
        delays.append(float(hit[0] - cb) if hit else np.nan)
        if hit:
            used.add(hit[0])
    nf = sum(1 for r in rst if r not in used)
    return dict(restarts=[int(r) for r in rst], delays=delays, n_false=nf, false_per30=nf / (nbr / BPS / 30.0))


def pcb_summ(sim, dd, nbr):
    ex = sim["ex"]; out = {}
    b1, b2 = pcb_blocks()
    win = {"10-12": (b1, b1 + 2 * BPS), "15-20": (15 * BPS, b2), "20-22": (b2, b2 + 2 * BPS), "25-30": (25 * BPS, 30 * BPS)}
    for k, (a, b) in win.items():
        out["ex_" + k] = dbm(ex[a:b]) if nbr >= b else np.nan
    e = sim["e"]
    if (not sim["div"]) and np.isfinite(e).all() and len(e) == len(dd["d"]):
        anr = O.anr_curve(e, dd["d"]); ts = (63 + 64 * np.arange(len(anr))) / FS
        for k, (a, b) in win.items():
            m = (ts >= a / BPS) & (ts < b / BPS); out["L1_" + k] = float(anr[m].mean())
    else:
        for k in win:
            out["L1_" + k] = np.nan
    return out


def par3_desc(o):
    return dict(par2_desc(o), **{"x3_" + k: v for k, v in o.X3.items()}, x3_mu_early=o.mu_e)


def c4_job(tag, o, exp="C4"):
    """protoC3 closed loop (as c1/c3; rewritten after every rule, resumes at the first missing rule).
    exp 'C5' (protoC4): the rsN / rsG rules (make_rules4, C5_RULES, par4_desc)."""
    nb = o.nb; t0 = time.time(); P = o.P
    dd, Rall, st = prep3(tag, nb)
    nbr = (len(dd["d"]) // B) if nb is None else nb
    if exp == "C5":
        RR = make_rules4(nbr, P, o.X2, o.X3, o.X4, o.mu_e, dd); pdesc = par4_desc(o); rdef = C5_RULES
    else:
        RR = make_rules3(nbr, P, o.X2, o.X3, o.mu_e); pdesc = par3_desc(o); rdef = C4_RULES
    p0 = fpath(exp, tag, nb); chg = change_blocks(dd)
    res = dict(tag=tag, nb=nbr, pc=dd["pc"], pcb=bool(dd.get("pseg")), changes=chg, par=pdesc, rules={}, done=False)
    arrs = dict(g=np.asarray(st.g[:nbr], np.float32))
    if os.path.exists(p0) and not o.force:
        old = json.load(open(p0))
        if old.get("nb") == nbr and old.get("par") == json.loads(json.dumps(pdesc)):
            res["rules"] = old["rules"]
            q = fpath(exp, tag, nb, "npz")
            if os.path.exists(q):
                arrs.update(dict(np.load(q)))
    for name in (o.rules or rdef):
        if name in res["rules"]:
            continue
        t1 = time.time(); rule = RR[name]()
        sim = simulate(dd, Rall, st, rule, nb=nbr)
        mt = metrics(sim, dd)
        r = dict(**mt, **summaries(sim, dd, nbr, o.rec_db), sec=time.time() - t1)
        r["amp"] = bool((not r["div"]) and np.isfinite(r["L1"][0]) and r["L1"][0] > 0)
        if res["pcb"]:
            r["pcb"] = pcb_summ(sim, dd, nbr)
        tr = sim["tr"]; arrs[name + "|ex"] = np.asarray(sim["ex"], np.float32)
        for k, v in tr.items():
            arrs[name + "|" + k] = np.asarray(v, np.float32)
        mua = np.asarray(tr["mu_app"], float)
        r["mu_app_ph"] = {k: (float(np.mean(mua[a:b])) if b > a and b <= len(mua) else np.nan) for k, (a, b) in phases(nbr).items()}
        if dd["pc"] and len(mua) > PC_BLOCK:
            r["mu_app_ph"]["pc15-17"] = float(np.mean(mua[PC_BLOCK:PC_BLOCK + 2 * BPS]))
        if "mu_state" in tr:
            r["zero_frac"], r["zero_run"] = zero_stats(tr["mu_state"])
            for k in ("dpclip", "mufclamp", "xclip", "stclip", "detkeep"):
                if k in tr and len(tr[k]):
                    r["frac_" + k] = float(np.mean(tr[k]))
        if isinstance(rule, SFY3):
            r["handover"] = [int(b) for b in rule.handover]
            r["early_frac"] = float(np.mean(tr["early"])) if len(tr["early"]) else np.nan
            if rule.rs:
                r["rs"] = restart_stats(rule.restarts, chg, len(mua))
        res["rules"][name] = r
        save(exp, tag, nb, res, arrs)
        extra = ""
        if isinstance(rule, SFY3):
            extra = f" handover {r['handover'][:4]}" + (f" restarts {r['rs']['restarts'][:6]}" if rule.rs else "")
        print(f"[{exp} {tag}] {name}: L1 {np.round(r['L1'], 2)} ex1s {r['ex_first1']:.2f} last10 {r['ex_last10']:.2f} dB "
              f"div={r['div']} amp={r['amp']}{extra} ({r['sec']:.1f} s, total {time.time() - t0:.0f} s)", flush=True)
    res["done"] = True; save(exp, tag, nb, res, arrs)
    return tag


def c4x_job(tag, o):
    """Estimator diagnostic for the protoC3 rules (C2X without the oracle; XHook shadows are computed as in c2x but
    report3 shows the driver's own X_hat, N_hat).  Also restart times vs the true change."""
    nb = o.nb; t0 = time.time(); P = o.P
    dd, Rall, st = prep3(tag, nb)
    if dd.get("pseg"):
        raise ValueError("c4x supports G / pc style inputs only (pcb probes would straddle the changes)")
    nbr = (len(dd["d"]) // B) if nb is None else nb
    vfp = {0: fixed_point(("SF",), 0), 1: fixed_point(("SF",), 1)}
    probes = [b for b in C2X_PROBE if b < nbr]
    if dd["pc"]:
        probes = sorted(set(probes + [PC_BLOCK + k for k in C2_PC if PC_BLOCK + k < nbr]))
    RR = make_rules3(nbr, P, o.X2, o.X3, o.mu_e)
    chg = change_blocks(dd)
    res = dict(tag=tag, nb=nbr, pc=dd["pc"], changes=chg, par=par3_desc(o), probes=probes, rules={}, done=False)
    arrs = {}
    for name in (o.rules or C4X_RULES):
        rule = RR[name]()
        hk = XHook(dd, Rall, st, P, o.X2, vfp, probes, nbr, None)
        sim = simulate(dd, Rall, st, rule, nb=nbr, hook=hk)
        n_ = len(hk.rec["i"]); own = dict(X=sim["tr"]["X"][:n_], N=sim["tr"]["N"][:n_])
        x = dict(div=sim["div"], probe=hk.prb, xstats=xstats(hk.rec, hk.prb, own, nbr, dd["pc"], fine=True))
        if isinstance(rule, SFY3):
            x["handover"] = [int(b) for b in rule.handover]
            if rule.rs:
                x["rs"] = restart_stats(rule.restarts, chg, len(sim["tr"]["mu_app"]))
        res["rules"][name] = x
        for k, v in hk.rec.items():
            arrs[name + "|" + k] = np.asarray(v, float)
        arrs[name + "|own_X"] = own["X"]; arrs[name + "|own_N"] = own["N"]
        for k in ("early", "restart", "mu_state"):
            if k in sim["tr"]:
                arrs[name + "|" + k] = np.asarray(sim["tr"][k], float)
        save("C4X", tag, nb, res, arrs)
        print(f"[C4X {tag}] {name}: div={sim['div']} handover {x.get('handover')} "
              f"restarts {x.get('rs', {}).get('restarts')} ({time.time() - t0:.0f} s)", flush=True)
    res["done"] = True; save("C4X", tag, nb, res, arrs)
    return tag


# ==============================================================================================
# protoC4: restart statistics that do not fire on bursts / SaS impulses ('rsN', 'rsG'; class SFY4 / YEstRSX)
# ==============================================================================================
X4_DEF = dict(z_beta=0.05, z_thr=4.0, z_d=2, guard=64)
VARS4 = ("early+rsN", "early+rsN2", "early+rsG")
REF_COMMIT4 = "0073387"     # protoC.py before the protoC4 extension


def rwin_energy(dd, nbr, W=M):
    """sum r^2 over the error window of every block: samples [i B + B - W, i B + B); W = M (default, rsN) = the M
    errors in Ctx.E; W = 2M (rsN2) also covers the r samples that reach the window's errors through the M-tap filter.
    r = data['r'] = hS * x, the filtered reference (Rall = rfft(hS, N) x rfft(x frame), the same hS)."""
    r = np.asarray(dd["r"], float); cs = np.concatenate([[0.0], np.cumsum(r * r)])
    out = np.empty(nbr)
    for i in range(nbr):
        hi = min(i * B + B, len(r)); lo = max(0, i * B + B - W)
        out[i] = cs[hi] - cs[lo]
    return out


def ewin_energy(E):
    """sum e^2 of the M errors in the zero-padded N-point frame E = rfft (Parseval, N even)."""
    p = np.abs(E) ** 2
    return float((p[0] + p[-1] + 2.0 * np.sum(p[1:-1])) / N)


class YEstRSX(YEstRS):
    """YEstRS (same adaptive beta, same restart actions) with another restart statistic, set by the rule before
    update() through s.zin:
      'N' (rsN)  s.zin = z_n = sum e^2 / sum r^2 over the error window; zbar = EMA_{z_beta} z, never reset (start: z of
                 the first block); trigger z_n > z_thr * zbar_{n-1} (zbar before its update with z_n).
      'G' (rsG)  s.zin = g_n; statistic |y_n|^2 / g_n^2 = |u_n|^2 max(g_n, gmin)^2 / g_n^2 (u = y / max(g, gmin)),
                 compared with its own EMA (adaptive beta as Q, started and reset together with Q), r_thr, d as rs.
    Restart after d consecutive triggers, not within `guard` blocks of the previous restart (trigger counting held at
    0 inside the guard).  Restart actions = YEstRS (C = Q = None, k = 0, ubar / buffer rebuilt from the d - 1
    previous blocks)."""

    def __init__(s, L, bd, b, X3, X4, mode, gmin):
        super().__init__(L, bd, b, X3)
        s.mode, s.gmin, s.guard = mode, gmin, int(X4["guard"])
        if mode == "N":
            s.rthr, s.d, s.zb = X4["z_thr"], int(X4["z_d"]), X4["z_beta"]
        elif mode != "G":
            raise ValueError(mode)
        s.zin = None; s.zbar = None; s.Zg = None; s.kZ = 0; s.last_rst = None; s.stat = np.nan; s.ref = np.nan

    def update(s, i, u):
        q = float(u @ u); s.restarted = False
        if s.mode == "N":
            z = float(s.zin); ref = s.zbar
            if np.isfinite(z):
                s.zbar = z if s.zbar is None else (1 - s.zb) * s.zbar + s.zb * z
        else:
            g = float(s.zin)
            z = q * max(g, s.gmin) ** 2 / g ** 2 if g > 0 else np.nan; ref = s.Zg
        s.stat, s.ref = z, (np.nan if ref is None else ref)
        trig = ref is not None and np.isfinite(z) and z > s.rthr * ref
        if s.last_rst is not None and i - s.last_rst < s.guard:
            trig = False
        s.cnt = s.cnt + 1 if trig else 0
        s.raw = (s.raw + [u.copy()])[-s.d:]
        if s.cnt >= s.d:
            s.C = s.Q = None; s.kC = s.kQ = 0; s.cnt = 0; s.ubar = None; s.buf = [None] * s.L
            s.Zg = None; s.kZ = 0
            for uu in s.raw[:-1]:
                s._push(uu)
            s.restarted = True; s.nrst += 1; s.last_rst = i
        old = s.buf[0]
        if i >= s.L + 4 or s.nrst > 0:
            bq = max(s.bmin, 1.0 / (s.kQ + 1))
            s.Q = q if s.Q is None else (1 - bq) * s.Q + bq * q; s.kQ += 1
            if s.mode == "G" and np.isfinite(z):
                bz = max(s.bmin, 1.0 / (s.kZ + 1))
                s.Zg = z if s.Zg is None else (1 - bz) * s.Zg + bz * z; s.kZ += 1
            if old is not None:
                c = float(u @ old); bc = max(s.bmin, 1.0 / (s.kC + 1))
                s.C = c if s.C is None else (1 - bc) * s.C + bc * c; s.kC += 1
        s._push(u)


class SFY4(SFY3):
    """SFY3 with the rs restart statistic replaced (mode 'N' = rsN, 'G' = rsG; see YEstRSX).  rw = rwin_energy(data)
    (rsN only).  Restart actions as rs: estimator reset, estX's kept (X, N) cleared, mu_state = mu_early and the early
    phase restarted with early.  Extra traces: 'rsstat' (z_n or |y|^2/g^2), 'rsref' (zbar_{n-1} or its EMA)."""

    def __init__(s, est, gamma, P, X2, X3, X4, early, mode, mu_e, rw=None):
        super().__init__(est, gamma, P, X2, X3, early, True, mu_e)
        if mode == "N" and rw is None:
            raise ValueError("rsN needs rw (rwin_energy of the input)")
        s.mode, s.rw, s.X4 = mode, rw, X4
        s.e2.y = YEstRSX(s.e2.y.L, s.e2.y.bd, s.e2.y.b, X3, X4, mode, P.gmin); s.est = s.e2.y
        s.tr.update({k: [] for k in ("rsstat", "rsref")})

    def block(s, i, c):
        if s.mode == "N":
            ee, rr = ewin_energy(c.E), float(s.rw[i])
            s.est.zin = ee / rr if rr > 0 else np.nan
        else:
            s.est.zin = c.g
        out = SFY3.block(s, i, c)
        s.tr["rsstat"].append(s.est.stat); s.tr["rsref"].append(s.est.ref)
        return out


def rule_name4(est, gamma, var):
    return rule_name3(est, gamma, var)


C5_NEW = [rule_name4(e, 0.03, v) for e in EST3 for v in VARS4]
C5_CTRL = [rule_name3(e, 0.03, v) for e in EST3 for v in ("early", "early+rs")]
C5_RULES = C5_NEW + C5_CTRL + ["SF-V2", "SF+ramp"]
DEF_CLASSES.update({"c5": "G,BU,PS,a145,a120,pc,pcb,II300"})
DEF_SEEDS.update({"c5": "0,1,2"})


def make_rules4(nbr, P, X2, X3, X4, mu_e, dd):
    """make_rules3 plus 'SF-Y <est91|estX> delayopt g <gam> [early+]rs<N|G>'."""
    R = make_rules3(nbr, P, X2, X3, mu_e)
    rws = {"N": rwin_energy(dd, nbr), "N2": rwin_energy(dd, nbr, 2 * M)}
    for e in EST3:
        for g in GAMS:
            for v in ("early+rsN", "early+rsN2", "early+rsG", "rsN", "rsN2", "rsG"):
                md = v.split("rs")[-1]
                R[rule_name4(e, g, v)] = (lambda e=e, g=g, v=v, md=md: SFY4(e, g, P, X2, X3, X4, v.startswith("early"),
                                                                            md[0], mu_e, rws.get(md)))
    return R


def par4_desc(o):
    return dict(par3_desc(o), **{"x4_" + k: v for k, v in o.X4.items()})


def c5_job(tag, o):
    return c4_job(tag, o, "C5")


# ==============================================================================================
# Job dispatch
# ==============================================================================================
def c3_job(tag, o):
    return c1_job(tag, o, "C3")


JOBS = {"c1": c1_job, "c2": c2_job, "c3": c3_job, "c2x": c2x_job, "c4": c4_job, "c4x": c4x_job, "c5": c5_job}


def _dispatch(a):
    cmd, tag, o = a
    try:
        return JOBS[cmd](tag, o)
    except Exception as ex:
        import traceback; traceback.print_exc()
        return f"{tag} FAILED: {ex}"


def main_jobs(cmd, o):
    todo = []
    for t in tags_of(cmd, o):
        p = fpath(cmd.upper(), t, o.nb)
        if os.path.exists(p) and not o.force and json.load(open(p)).get("done") and not (cmd in ("c1", "c3", "c4", "c5") and o.rules):
            print("skip (done):", p); continue
        todo.append((cmd, t, o))
    print(f"{cmd.upper()}: {len(todo)} jobs, procs={o.procs}, nb={o.nb}, a*={o.P.astar:.4f} mu_D={o.P.mu_D:.3f}", flush=True)
    t0 = time.time()
    if o.procs <= 1:
        for a in todo:
            print("finished", _dispatch(a), flush=True)
    else:
        with Pool(min(o.procs, 2), maxtasksperchild=1) as pl:
            for r in pl.imap_unordered(_dispatch, todo):
                print("finished", r, flush=True)
    print(f"{cmd.upper()}: all done in {time.time() - t0:.0f} s", flush=True)


# ==============================================================================================
# Self-checks
# ==============================================================================================
def check(o):
    ok = True

    def rep(name, cond, msg=""):
        nonlocal ok
        ok &= bool(cond); print(f"[{'OK ' if cond else 'FAIL'}] {name} {msg}", flush=True)

    P = o.P; nbc = 300
    dd, Rall, st = T.prep("G_s0", nbc)
    rng = np.random.default_rng(0)
    # 1. sf_dir: ret_y does not change the direction; equals protoA.FixedSF (independent copy of the original code)
    E = np.fft.rfft(rng.standard_normal(N)); i0 = 50
    d0 = SFm.sf_dir(Rall[i0], E, st.Srr[i0], st.Pbar[i0], st.eps[i0], EPS_REG)
    d1, x, F = SFm.sf_dir(Rall[i0], E, st.Srr[i0], st.Pbar[i0], st.eps[i0], EPS_REG, ret_y=True)
    d2 = PA.FixedSF(st.Srr[i0], st.Pbar[i0], st.eps[i0], EPS_REG)(Rall[i0], E)
    rep("sf_dir(ret_y) direction unchanged (bitwise) and == protoA.FixedSF", np.array_equal(d0, d1) and np.array_equal(d0, d2))
    # 2. y -> Delta: 1/F applied to y (zero-padded) reproduces Delta
    y = KD * x[:N // 2]
    yp = np.concatenate([y, np.zeros(N - N // 2)])
    Dl = np.fft.irfft(np.fft.rfft(yp) / F, N)[:M]
    dr = np.max(np.abs(Dl - KD * d0)) / np.max(np.abs(KD * d0))
    rep("1/F y == Delta", dr < 1e-12 and np.all(x[N // 2:] == 0), f"rel diff {dr:.1e}")
    # 3. gamma = 0, no correction == fixed mu (bitwise), all targets, both coordinates
    b = simulate(dd, Rall, st, Fixed("SF", 0.1), nb=nbc)
    for co in ("Y", "E"):
        for t in TARGETS:
            a = simulate(dd, Rall, st, SFY(co, t, "none", 0.0, P, mu_init=0.1), nb=nbc)
            rep(f"SF-{co} {t} none gamma=0 == Fixed 0.1 (bitwise)", np.array_equal(a["w"], b["w"]) and np.array_equal(a["e"], b["e"]))
    # 4. a* for the given c_k
    a_, r_ = a_star(C_DEF)
    rep("a* (c_k of Sec. 10.1) = 0.161, rho 0.671", abs(a_ - 0.161) < 5e-4 and abs(r_ - 0.671) < 1e-3, f"a* {a_:.4f} rho {r_:.4f} "
        f"rho(0.5) {rho_delay(0.5, np.array(C_DEF) / sum(C_DEF)):.3f} (table 0.844); mu_D = a*/h_eff = {P.mu_D:.3f}")
    # 5. identity factor: y == Delta and the y statistics equal the Euclidean ones
    one = np.ones(NBIN); eA, eB = YEst(), YEst(); dE = DirectEst("E"); ok5 = True
    for i in range(40):
        R = np.fft.rfft(rng.standard_normal(N)); Ei = np.fft.rfft(rng.standard_normal(N)); gi = 0.3 + rng.random()
        dv, xx, FF = SFm.sf_dir(R, Ei, one, 1.0, 0.0, 0.0, ret_y=True)
        yy = KD * xx[:N // 2]; Dd = KD * dv
        ok5 &= bool(np.max(np.abs(FF - 1)) < 1e-14 and np.max(np.abs(yy - Dd)) < 1e-12 * np.max(np.abs(Dd)))
        eA.update(i, yy / max(gi, GMIN)); eB.update(i, Dd / max(gi, GMIN)); dE.update(i, dv, gi)
    rel = lambda p, q: abs(p - q) / abs(q)
    rep("identity factor: y == Delta, y-statistics == Euclidean == vssT DirectEst('E')",
        ok5 and rel(eA.C, eB.C) < 1e-10 and rel(eA.Q, eB.Q) < 1e-10 and rel(eB.C, dE.C) < 1e-12 and rel(eB.Q, dE.Q) < 1e-12,
        f"C {eA.C:.5g} / {eB.C:.5g} / {dE.C:.5g}, Q {eA.Q:.5g} / {eB.Q:.5g} / {dE.Q:.5g}")
    # 6. solve_xn == protoA.solve91 with tm -> 2 tm - 1, h -> h_eff (inside the clip range)
    C_, Q_, mu_ = 0.21, 1.0, 0.2
    Xa, Na, xc = solve_xn(C_, Q_, mu_, P)
    Xb, Nb = PA.solve91(C_, Q_, mu_, P.L, P.bd, P.f, 2 * P.tm - 1, P.ta, P.gz, P.gm, h=P.h)
    rep("solve_xn == protoA.solve91(tm -> 2tm-1)", abs(Xa - Xb) < 1e-14 and abs(Na - Nb) < 1e-14 and xc == 0,
        f"X {Xa:.4f} N {Na:.4f}")
    # forward model reproduces (C, Q)
    h, f, ta, tm = P.h, P.f, P.ta, P.tm; lbar = P.L + (1 - P.bd) / P.bd
    Cf = (1 + h * mu_ * (lbar - (2 * tm - 1) * f)) * Xa - h * mu_ * ta * Na
    Qf = (1 + f) * Xa + Na - h * mu_ * ((2 * tm - 2) * f * Xa + (ta - 1) * Na)
    rep("forward forms reproduce C, Q", abs(Cf - C_) < 1e-12 and abs(Qf - Q_) < 1e-12)
    # 7. delay-opt sanity
    t0_ = mu_delayopt(0.0, 1.0, P); t1_ = mu_delayopt(1.0, 0.0, P); t2_ = mu_delayopt(0.2, 0.3, P)
    rep("delay-opt: X=0 -> 0; N=0 -> > 0", t0_ == 0.0 and t1_ > 0, f"(X,N)=(0,1): {t0_:.3f}  (1,0): {t1_:.3f} "
        f"(a*/h_eff = {P.mu_D:.3f})  (0.2,0.3): {t2_:.3f}")
    # 8. continuation with the dp policy == simulation of the same rule with gamma = 0
    snap = {}

    def grab(i, ww, mu, dv, c, rule, mf, zi=None, e_all=None):
        if i == 50:
            snap.update(w=ww.copy(), zi=zi.copy(), e=e_all.copy(), dv=dv.copy(), last=dict(rule.last))
    rl = SFY("Y", "onestep", "dp+g", 0.0, P, mu_init=0.2)
    full = simulate(dd, Rall, st, rl, nb=70, hook=grab)
    cont = continuation_corr(dd, Rall, st, snap["w"], snap["zi"], snap["e"], 50, snap["dv"], snap["last"]["fac"],
                             snap["last"]["Qy_after"], 0.2, 12, rl)
    dc = np.max(np.abs(cont - full["ex"][50:62]) / full["ex"][50:62])
    rep("continuation_corr (dp+g) == simulation (gamma = 0)", dc < 1e-9, f"rel diff {dc:.1e}")
    ok &= check2(o, dd, Rall, st)
    ok &= check3(o, dd, Rall, st)
    ok &= check4(o, dd, Rall, st)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED", flush=True)
    return ok


REF_COMMIT = "1b0f256"      # protoC.py before the protoC2 extension (reference for the bitwise check)


def _ref_module(commit=REF_COMMIT):
    import subprocess, importlib.util, tempfile
    src = subprocess.run(["git", "show", f"{commit}:atsip/explore/protoC.py"], cwd=HERE, capture_output=True,
                         text=True, check=True).stdout
    d = tempfile.mkdtemp(prefix="protoC_ref_"); f = os.path.join(d, f"protoC_ref_{commit}.py")
    open(f, "w").write(src.replace("HERE = os.path.dirname(os.path.abspath(__file__))", f"HERE = {HERE!r}"))
    spec = importlib.util.spec_from_file_location(f"protoC_ref_{commit}", f); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check2(o, dd, Rall, st):
    """protoC2 checks (9-14)."""
    ok = True

    def rep(name, cond, msg=""):
        nonlocal ok
        ok &= bool(cond); print(f"[{'OK ' if cond else 'FAIL'}] {name} {msg}", flush=True)

    P = o.P; nbq = 200
    same = lambda a, b: (np.array_equal(a["w"], b["w"]) and np.array_equal(a["e"], b["e"], equal_nan=True)
                         and a["tr"].keys() == b["tr"].keys()
                         and all(np.array_equal(a["tr"][k], b["tr"][k], equal_nan=True) for k in a["tr"]))
    # 9. existing rules bitwise unchanged against the protoC.py of REF_COMMIT
    try:
        R0 = _ref_module()
    except Exception as ex:
        rep(f"reference module {REF_COMMIT} loads", False, str(ex)); R0 = None
    if R0 is not None:
        P0 = R0.Par(f=P.f, ta=P.ta, tm=P.tm, heff=P.h, ck=P.c, gz=P.gz, gm=P.gm, dopt_T=P.dopt_T, dopt_k=P.dopt_k,
                    mu_fmax=None if abs(P.mu_fmax - 0.75 / (P.h * (P.ta - 1))) < 1e-15 else P.mu_fmax)
        RRn, RRo = make_rules(nbq, P), R0.make_rules(nbq, P0); bad = []
        for k in R0.SFY_NAMES:
            if not same(simulate(dd, Rall, st, RRn[k](), nb=nbq), simulate(dd, Rall, st, RRo[k](), nb=nbq)):
                bad.append(k)
        rep(f"all {len(R0.SFY_NAMES)} protoC SF-Y/SF-E rules bitwise == {REF_COMMIT} (w, e, traces; {nbq} blocks)",
            not bad and R0.SFY_NAMES == SFY_NAMES, f"mismatch: {bad[:3]}" if bad else "")
    # 10. est91 through the solver interface == the protoC rule (bitwise w, e and the shared traces)
    RR = make_rules(300, P, o.X2); bad = []
    for t in TARGETS2:
        for g in GAMS:
            a = simulate(dd, Rall, st, RR[rule_name2("est91", t, "g", g)](), nb=300)
            b = simulate(dd, Rall, st, RR[rule_name("Y", t, "g", g)](), nb=300)
            if not (np.array_equal(a["w"], b["w"]) and np.array_equal(a["e"], b["e"])
                    and all(np.array_equal(a["tr"][k], b["tr"][k], equal_nan=True) for k in b["tr"])):
                bad.append((t, g))
    rep("SF-Y est91 <t> g <gam> == SF-Y <t> g <gam> (bitwise, 300 blocks)", not bad, str(bad) if bad else "")
    # 11. forms: estL4 = solve_xn with lbar = 4; estX forward forms reproduce (C, Q); limits; rho table
    e4 = Est2("estL4", P, o.X2); e4.y.C, e4.y.Q = 0.21, 1.0
    X4, N4, xc4, _, _, _ = e4.solve(0.2)
    h, f, ta, tm = P.h, P.f, P.ta, P.tm
    C4 = (1 + h * 0.2 * (4 - (2 * tm - 1) * f)) * X4 - h * 0.2 * ta * N4
    Q4 = (1 + f) * X4 + N4 - h * 0.2 * ((2 * tm - 2) * f * X4 + (ta - 1) * N4)
    rep("estL4: lbar = 4 forms reproduce (C, Q)", abs(C4 - 0.21) < 1e-12 and abs(Q4 - 1.0) < 1e-12 and xc4 == 0,
        f"X {X4:.4f} N {N4:.4f}")
    ex_ = Est2("estX", P, o.X2); ex_.y.C, ex_.y.Q = 0.3, 1.0; ex_.hist = [0.5, 0.6, 0.55, 0.5]
    Xx, Nx, xcx, _, kx, okx = ex_.solve(0.5)
    cX, cN, qX, qN = estx_coefs(ex_.a, min(ex_.a, P.h * P.mu_fmax), P, o.X2["kappa"], ex_.rho)
    rep("estX: forward forms reproduce (C, Q)", okx and kx == 0 and xcx == 0 and abs(cX * Xx + cN * Nx - 0.3) < 1e-12
        and abs(qX * Xx + qN * Nx - 1.0) < 1e-12, f"a {ex_.a:.4f} rho {ex_.rho:.4f} X {Xx:.4f} N {Nx:.4f}")
    c0 = estx_coefs(0.0, 0.0, P, 1.0, 1.0)
    ag, rg = rho_table(P.c); aa = np.random.default_rng(1).uniform(0, 2.5, 50)
    dr = max(abs(np.interp(a_, ag, rg) - rho_delay(a_, P.c)) for a_ in aa)
    rep("estX limits: a = 0 -> C = kappa X, Q = (1+f) X + N; geo_sum(lam=1) = 4; rho table vs roots",
        c0 == (1.0, 0.0, 1 + P.f, 1.0) and abs(geo_sum(1.0, 4) - 4.0) < 1e-14 and dr < 1e-4
        and abs(geo_sum(0.5, 4) - sum(0.5 ** (j - 1) * r_tri(4 - j) for j in range(1, 8))) < 1e-15,
        f"(geo_sum(0.5) = {geo_sum(0.5, 4):.4f} = 0.25 + 0.5 lam + 0.75 lam^2 + lam^3 + 0.75 lam^4 + ...; the closed form "
        f"printed in Sec. 11.2 gives {0.75 + 0.25 + 0.0625:.4f}); "
        f"max |rho_interp - rho| {dr:.1e}")
    # 12. det keep
    X2k = dict(o.X2, det_rel=10.0); ek = Est2("estX", P, X2k); ek.y.C, ek.y.Q = 0.3, 1.0; ek.hist = [0.5] * 4
    r1 = ek.solve(0.5); ek.X2 = o.X2; r2 = ek.solve(0.5); ek.X2 = X2k; ek.y.C = 0.1; r3 = ek.solve(0.5)
    rep("estX det keep: no previous -> no estimate; previous kept", (not r1[5]) and r1[4] == 1 and r2[5] and r2[4] == 0
        and r3[5] and r3[4] == 1 and r3[:2] == r2[:2])
    # 13. X_true scale: frozen w = 0, E u vs H P_M F v
    vfp = fixed_point(("SF",), 0); bl = list(range(60, 300)); U = probe_u(dd, Rall, st, np.zeros(M), bl, P.gmin)
    v0 = -wM - vfp; Fv = []
    for j in bl:
        _, _, Fj = SFm.sf_dir(Rall[j], np.zeros(NBIN, complex), st.Srr[j], st.Pbar[j], st.eps[j], EPS_REG, ret_y=True)
        Fv.append(H * np.fft.irfft(np.fft.rfft(v0, N) * Fj, N)[:M])
    um, pm = U.mean(0), np.mean(Fv, 0); kap = float(um @ um) / float(np.mean([q @ q for q in Fv]))
    cs = float(um @ pm) / np.sqrt(float(um @ um) * float(pm @ pm))
    rep("X_true scale (frozen w = 0, 240 blocks): kappa_y = |E u|^2 / X_true in [0.85, 1.1], cos > 0.99",
        0.85 < kap < 1.1 and cs > 0.99, f"kappa_y {kap:.3f} cos {cs:.4f}")
    # 14. shadow estimators in XHook == the rule's own X, N (bitwise)
    probes = [40, 120]; bad = []
    for e in EST2:
        rl = RR[rule_name2(e, "delayopt", "g", 0.1)]()
        hk = XHook(dd, Rall, st, P, o.X2, {0: vfp, 1: vfp}, probes, 300)
        sim = simulate(dd, Rall, st, rl, nb=300, hook=hk)
        if not (np.array_equal(np.asarray(hk.rec[f"{e}_X"]), sim["tr"]["X"], equal_nan=True)
                and np.array_equal(np.asarray(hk.rec[f"{e}_N"]), sim["tr"]["N"], equal_nan=True)):
            bad.append(e)
    rep("XHook shadow estimators == own X_hat, N_hat (bitwise, delayopt g gam.1, 300 blocks)", not bad,
        f"{bad}; probe N_true {np.round(hk.prb['Ntrue'], 5)} Q_frz {np.round(hk.prb['Qfrz'], 5)}")
    return ok


REF_COMMIT3 = "ac8eb5b"     # protoC.py before the protoC3 extension


def check3(o, dd, Rall, st):
    """protoC3 checks (15-20)."""
    ok = True

    def rep(name, cond, msg=""):
        nonlocal ok
        ok &= bool(cond); print(f"[{'OK ' if cond else 'FAIL'}] {name} {msg}", flush=True)

    P = o.P; nbq = 200
    same = lambda a, b: (np.array_equal(a["w"], b["w"]) and np.array_equal(a["e"], b["e"], equal_nan=True)
                         and a["tr"].keys() == b["tr"].keys()
                         and all(np.array_equal(a["tr"][k], b["tr"][k], equal_nan=True) for k in a["tr"]))
    # 15. all existing rules bitwise == ac8eb5b (protoC + protoC2 rules and the protoC2 controls)
    try:
        R0 = _ref_module(REF_COMMIT3)
    except Exception as ex:
        rep(f"reference module {REF_COMMIT3} loads", False, str(ex)); R0 = None
    if R0 is not None:
        P0 = R0.Par(f=P.f, ta=P.ta, tm=P.tm, heff=P.h, ck=P.c, gz=P.gz, gm=P.gm, dopt_T=P.dopt_T, dopt_k=P.dopt_k,
                    mu_fmax=None if abs(P.mu_fmax - 0.75 / (P.h * (P.ta - 1))) < 1e-15 else P.mu_fmax)
        names = R0.SFY_NAMES + R0.C3_RULES
        RRn, RRo = make_rules3(nbq, P, o.X2, o.X3, o.mu_e), R0.make_rules(nbq, P0, o.X2); bad = []
        for k in names:
            if not same(simulate(dd, Rall, st, RRn[k](), nb=nbq), simulate(dd, Rall, st, RRo[k](), nb=nbq)):
                bad.append(k)
        rep(f"all {len(names)} protoC / protoC2 rules bitwise == {REF_COMMIT3} (w, e, traces; {nbq} blocks)",
            not bad and R0.SFY_NAMES == SFY_NAMES and R0.C3_RULES == C3_RULES, f"mismatch: {bad[:3]}" if bad else "")
    # path_of unchanged for the existing inputs (pseg absent)
    dpc = dict(pc=True); dG = dict(pc=False)
    rep("vssT_lib.path_of unchanged without 'pseg' (pc: i >= PC_BLOCK, others 0)",
        all(path_of(dpc, i) == int(i >= PC_BLOCK) and path_of(dG, i) == 0 for i in range(0, 3750)))
    # 16. SFY3 with early = rs = False == the protoC2 rule (bitwise w, e, shared traces)
    bad = []
    for e in EST3:
        for g in GAMS:
            a = simulate(dd, Rall, st, SFY3(e, g, P, o.X2, o.X3, False, False, o.mu_e), nb=300)
            b = simulate(dd, Rall, st, SFY("Y", "delayopt", "g", g, P, est=e, X2=o.X2), nb=300)
            if not (np.array_equal(a["w"], b["w"]) and np.array_equal(a["e"], b["e"])
                    and all(np.array_equal(a["tr"][k], b["tr"][k], equal_nan=True) for k in b["tr"])):
                bad.append((e, g))
    rep("SFY3(early = rs = off) == SF-Y <est> delayopt g <gam> (bitwise, 300 blocks)", not bad, str(bad) if bad else "")
    # 17. YEstRS: running mean then EMA(beta_min); restart logic on a synthetic sequence
    rng = np.random.default_rng(3); L_, b_ = P.L, P.b
    for bd in (P.bd, 1.0):
        ys = YEstRS(L_, bd, b_, o.X3); U = rng.standard_normal((60, 16)); qs, cs = [], []
        ubar = None; buf = [None] * L_; Qr = Cr = None; kq = kc = 0
        for i, u in enumerate(U):
            ys.update(i, u)
            old = buf[0]
            if i >= L_ + 4:
                bq = max(o.X3["beta_min"], 1 / (kq + 1)); Qr = u @ u if Qr is None else (1 - bq) * Qr + bq * (u @ u); kq += 1
                bc = max(o.X3["beta_min"], 1 / (kc + 1)); Cr = u @ old if Cr is None else (1 - bc) * Cr + bc * (u @ old); kc += 1
            ubar = u.copy() if ubar is None else (1 - bd) * ubar + bd * u; buf = buf[1:] + [ubar.copy()]
            if i == L_ + 4 + 9:
                qm = float(np.mean([v @ v for v in U[L_ + 4:i + 1]]))
                okm = abs(ys.Q - qm) < 1e-12 * qm
        rep(f"YEstRS (bd = {bd:g}): Q = running mean for k < 1/beta_min, then EMA; C likewise; no restart on "
            "stationary data", okm and abs(ys.Q - Qr) < 1e-12 * abs(Qr) and abs(ys.C - Cr) < 1e-9 * abs(Qr)
            and ys.nrst == 0, f"Q {ys.Q:.5g} ref {Qr:.5g}")
    ys = YEstRS(L_, 1.0, b_, o.X3); U = rng.standard_normal((80, 16)); U[40] *= 10.0; U[60:] *= 10.0
    log = []
    for i, u in enumerate(U):
        ys.update(i, u); log.append((i, ys.restarted, ys.C is None))
    rst = [i for i, r, _ in log if r]
    d_ = int(o.X3["d"]); cnone = [i for i, _, cn in log if cn and i >= 60]
    exp_r = [60 + d_ - 1] if d_ > 1 else [40, 60]
    rep(f"YEstRS restart: single spike (block 40) ignored for d = {d_}; step at 60 -> restart at {60 + d_ - 1}; "
        f"C resumes L - d + 1 blocks later", rst == exp_r and cnone == list(range(60 + d_ - 1, 60 + L_)),
        f"restarts {rst}, C None at {cnone}")
    # 18. early value: X-invariant; SFY3 early phase with fixed n_e
    me = o.mu_e; Pq = copy.copy(P); Pq.dopt_k = o.X3["early_k"]
    r2 = np.array([rho_delay(m * P.h, P.c) for m in EARLY_GRID]) ** 2
    m7 = float(EARLY_GRID[int(np.argmin(dopt_curve(7.0, 0.0, Pq, grid=EARLY_GRID, rho2=r2)))])
    rep(f"early value (dopt with N = 0, K '{o.X3['early_k']}', T = {P.dopt_T}) X-invariant", m7 == me,
        f"mu_early = {me:.3f} (a = {me * P.h:.4f}; a*/h_eff = {P.mu_D:.3f}; 'excl1': {mu_early_value(P, 'excl1'):.3f}; "
        f"40-point delayopt grid at (1, 0): {mu_delayopt(1.0, 0.0, P):.3f})")
    X3f = dict(o.X3, n_early=20)
    rl = SFY3("estX", 0.03, P, o.X2, X3f, True, False, me); sim = simulate(dd, Rall, st, rl, nb=60)
    ms = sim["tr"]["mu_state"]; ea = sim["tr"]["early"]
    rep("SFY3 early (n_early = 20): mu_state = mu_early for blocks 0-20, early flag 0-19, handover at 19, then the "
        "delayopt EMA", np.all(ms[:21] == me) and np.all(ea[:20] == 1) and np.all(ea[20:] == 0) and rl.handover == [19]
        and abs(ms[21] - min(max(0.97 * me + 0.03 * sim["tr"]["target"][20], 0), MU_MAX)) < 1e-15)
    rl = SFY3("estX", 0.03, P, o.X2, o.X3, True, True, me); sim = simulate(dd, Rall, st, rl, nb=300)
    rep("SFY3 early+rs on G_s0 (300 blocks): handover at the trust rule (<= ne_max), no restart", len(rl.handover) == 1
        and rl.handover[0] <= o.X3["ne_max"] - 1 and not rl.restarts, f"handover {rl.handover} restarts {rl.restarts}")
    # 19. pcb data: segments equal pc_s0 / original; path_of
    dp = T.data("pc_s0"); db = data_pcb("pcb_s0"); n1, n2 = (int(t * FS) for t in PCB_T); n0 = PC_BLOCK * B
    xo = D.build(dp["x"])["d"]
    rep("pcb data: d = original before 10 s and after 20 s, = pc path in 15-20 s (== pc_s0); path_of 0/1/0",
        np.array_equal(db["x"], dp["x"]) and np.array_equal(db["d"][:n1], xo[:n1]) and np.array_equal(db["d"][n2:], xo[n2:])
        and np.array_equal(db["d"][n0:n2], dp["d"][n0:n2]) and np.array_equal(dp["d"][:n0], xo[:n0])
        and [path_of(db, i) for i in (0, n1 // B - 1, n1 // B, n2 // B - 1, n2 // B, 3749)] == [0, 0, 1, 1, 0, 0])
    return ok


def check4(o, dd, Rall, st):
    """protoC4 checks (21-24)."""
    ok = True

    def rep(name, cond, msg=""):
        nonlocal ok
        ok &= bool(cond); print(f"[{'OK ' if cond else 'FAIL'}] {name} {msg}", flush=True)

    P = o.P
    same = lambda a, b: (np.array_equal(a["w"], b["w"]) and np.array_equal(a["e"], b["e"], equal_nan=True)
                         and a["tr"].keys() == b["tr"].keys()
                         and all(np.array_equal(a["tr"][k], b["tr"][k], equal_nan=True) for k in a["tr"]))
    # 21. every protoC3 rule (incl. all SFY3 variants) and every existing rule bitwise == 0073387, on G_s0 and on
    #     a120_s0 (where the old rs restarts)
    try:
        R0 = _ref_module(REF_COMMIT4)
    except Exception as ex:
        rep(f"reference module {REF_COMMIT4} loads", False, str(ex)); R0 = None
    if R0 is not None:
        X3 = o.X3
        names_new = list(make_rules3(10, P, o.X2, X3, o.mu_e).keys())
        names_old = list(R0.make_rules3(10, P, o.X2, X3, o.mu_e).keys())
        bad = []; nbq = 200
        RRn, RRo = make_rules4(nbq, P, o.X2, X3, o.X4, o.mu_e, dd), R0.make_rules3(nbq, P, o.X2, X3, o.mu_e)
        for k in names_old:
            if not same(simulate(dd, Rall, st, RRn[k](), nb=nbq), simulate(dd, Rall, st, RRo[k](), nb=nbq)):
                bad.append(k)
        rep(f"all {len(names_old)} existing rules (protoC / protoC2 / protoC3) bitwise == {REF_COMMIT4} on G_s0 "
            f"({nbq} blocks)", not bad and names_new == names_old and R0.C4_RULES == C4_RULES,
            f"mismatch: {bad[:3]}" if bad else "")
        da, Ra, sa = prep3("a120_s0", 600); bad = []; nr = []
        RRn, RRo = make_rules4(600, P, o.X2, X3, o.X4, o.mu_e, da), R0.make_rules3(600, P, o.X2, X3, o.mu_e)
        for k in [n for n in names_old if isinstance(RRo[n](), R0.SFY3)]:
            rn, ro = RRn[k](), RRo[k]()
            if not same(simulate(da, Ra, sa, rn, nb=600), simulate(da, Ra, sa, ro, nb=600)) or rn.restarts != ro.restarts:
                bad.append(k)
            if getattr(rn, "rs", False):
                nr.append(len(rn.restarts))
        rep(f"all protoC3 SFY3 rules bitwise == {REF_COMMIT4} on a120_s0 (600 blocks, old rs restarts: {nr})",
            not bad and sum(nr) > 0, f"mismatch: {bad[:3]}" if bad else "")
    # 22. YEstRSX: 'G' with g = 1 and guard 0 == YEstRS (C, Q, restarts); 'N' logic on a synthetic z sequence
    rng = np.random.default_rng(5); L_, b_ = P.L, P.b
    U = rng.standard_normal((120, 16)); U[40] *= 10.0; U[60:] *= 10.0; U[90:] *= 10.0
    a, bq = YEstRS(L_, P.bd, b_, o.X3), YEstRSX(L_, P.bd, b_, o.X3, dict(o.X4, guard=0), "G", P.gmin); good = True
    for i, u in enumerate(U):
        a.update(i, u); bq.zin = 1.0; bq.update(i, u)
        good &= (a.C == bq.C and a.Q == bq.Q and a.restarted == bq.restarted and (a.Q is None or a.Q == bq.Zg))
    rep("YEstRSX 'G' (g = 1, guard 0) == YEstRS bitwise (C, Q, restarts; its EMA == Q)", good and a.nrst == bq.nrst == 2,
        f"restarts {a.nrst}")
    X4 = o.X4; d_ = int(X4["z_d"]); gd = int(X4["guard"]); zt = X4["z_thr"]; bz = X4["z_beta"]
    z = np.exp(0.1 * rng.standard_normal(200)); z[30] *= 50.0; z[80:] *= 50.0; z[80 + gd // 2:] *= 50.0
    z[80 + gd + 20:] *= 50.0
    ys = YEstRSX(L_, P.bd, b_, o.X3, X4, "N", P.gmin); rst = []; zb = None; zok = True
    for i in range(200):
        ys.zin = z[i]; ys.update(i, rng.standard_normal(16))
        zok &= bool(np.isnan(ys.ref)) if zb is None else (ys.ref == zb)
        zb = z[i] if zb is None else (1 - bz) * zb + bz * z[i]
        zok &= ys.zbar == zb
        if ys.restarted:
            rst.append(i)
    exp_r = [80 + d_ - 1, 80 + gd + 20 + d_ - 1]
    rep(f"YEstRSX 'N': zbar = EMA({bz:g}) used before its update; single spike ignored (d = {d_}); step at 80 -> "
        f"restart at {exp_r[0]}; step inside the {gd}-block guard ignored; step after it -> {exp_r[1]}",
        zok and rst == exp_r, f"restarts {rst}")
    # 23. window energies: Parseval of the error frame, r window
    e = rng.standard_normal(5000); s0 = 20 * B
    ok23 = abs(ewin_energy(ML.frame_E(e, s0, M)) - float(e[s0 + B - M:s0 + B] @ e[s0 + B - M:s0 + B])) < 1e-9 * M
    rw = rwin_energy(dd, 30); r = dd["r"]
    ok23 &= all(abs(rw[i] - float(r[max(0, i * B + B - M):i * B + B] @ r[max(0, i * B + B - M):i * B + B])) < 1e-9 * (1 + rw[i])
                for i in (0, 1, 3, 4, 29))
    rep("ewin_energy (Parseval of Ctx.E) == sum e^2 over the M-sample error window; rwin_energy == sum r^2 there", ok23)
    # 24. SFY4 on G_s0 (no restart) and on pc_s0 up to 150 blocks after the change: restart actions (mu_state =
    #     mu_early from the next block, early flag on, statistic > threshold x reference at the restart block);
    #     rsN2 detects the change (rsN / rsG restart counts printed only)
    RR = make_rules4(300, P, o.X2, o.X3, o.X4, o.mu_e, dd); msg = []; good = True
    for v in VARS4:
        rl = RR[rule_name4("estX", 0.03, v)](); simulate(dd, Rall, st, rl, nb=300)
        msg.append(f"G_s0 {v}: {rl.restarts}"); good &= not rl.restarts
    nbp = PC_BLOCK + 150; dp, Rp, sp = prep3("pc_s0", nbp); RR = make_rules4(nbp, P, o.X2, o.X3, o.X4, o.mu_e, dp)
    for v in VARS4:
        rl = RR[rule_name4("est91", 0.03, v)](); tr = simulate(dp, Rp, sp, rl, nb=nbp)["tr"]
        thr = o.X4["z_thr"] if "rsN" in v else o.X3["r_thr"]
        for j in rl.restarts:
            good &= (tr["mu_state"][j + 1] == o.mu_e and tr["early"][j] == 1 and tr["restart"][j] == 1
                     and tr["rsstat"][j] > thr * tr["rsref"][j])
        good &= all(b - a >= o.X4["guard"] for a, b in zip(rl.restarts, rl.restarts[1:]))
        hit = [r_ for r_ in rl.restarts if PC_BLOCK <= r_ < PC_BLOCK + 10]
        if v == "early+rsN2":
            good &= bool(hit)
        msg.append(f"pc_s0 {v}: {len(rl.restarts)} restarts, at the change: {hit}")
    rep("SFY4: no restart on G_s0 (300 blocks); pc_s0: restart actions and guard; rsN2 detects the change", good,
        "; ".join(msg))
    return ok


# ==============================================================================================
# Report
# ==============================================================================================
def load(exp, smoke):
    d = OUT_SMOKE if smoke else OUT_FULL
    J = {}
    for f in sorted(glob.glob(os.path.join(d, f"protoC_{exp}_*.json"))):
        r = json.load(open(f)); J[r["tag"]] = r
    return J


def fm(x, p=2):
    try:
        return "NaN" if x is None or not np.isfinite(x) else f"{x:.{p}f}"
    except TypeError:
        return str(x)


def cls_of(tag):
    return tag.rsplit("_s", 1)[0]


def cdesc(c):
    if c == "pcb":
        return "path change at 10 s and back at 20 s (alpha 1.45)"
    if c in CLASS_DESC:
        return CLASS_DESC[c]
    if c.startswith("II"):
        return f"isolated impulses on Gaussian (1 per ~0.5 s, {c[2:]}x std)"
    return c


def rep_c1(J):
    L = []
    byc = {}
    for t, r in J.items():
        byc.setdefault(cls_of(t), []).append(r)
    for c, rs in byc.items():
        names = list(dict.fromkeys(k for r in rs for k in r["rules"]))
        pc = rs[0].get("pc")
        L += [f"### {c}: {cdesc(c)} (tags: {', '.join(sorted(r['tag'] for r in rs))}; nb = {rs[0]['nb']})", ""]
        hdr = (["L1 15-17 s", "L1 25-30 s", "L1 whole", "L2 15-17 s", "L2 25-30 s", "ex 0-1 s", "ex 0-5 s", "ex 15-17 s", "ex 25-30 s"]
               if pc else ["L1 whole", "L1 first5", "L1 last10", "L2 whole", "L2 first5", "L2 last10", "ex 0-1 s", "ex 0-5 s", "ex last10"])
        L += ["| rule | " + " | ".join(hdr) + " | n | div | amp |", "|---" * (len(hdr) + 4) + "|"]
        rows2 = []
        for k in names:
            rows = [r["rules"][k] for r in rs if k in r["rules"]]
            okr = [x for x in rows if not x["div"]]; nd = len(rows) - len(okr); na = sum(1 for x in rows if x.get("amp"))
            mean = lambda f: float(np.mean([f(x) for x in okr])) if okr else np.nan
            g_ = lambda key: (lambda x: x.get(key, np.nan) if x.get(key) is not None else np.nan)
            if pc:
                cells = [mean(lambda x: x["pc"]["L1a"]), mean(lambda x: x["pc"]["L1b"]), mean(lambda x: x["pc"]["L1w"]),
                         mean(lambda x: x["pc"]["L2a"]), mean(lambda x: x["pc"]["L2b"]), mean(g_("ex_first1")),
                         mean(g_("ex_first5")), mean(g_("ex_pc_15_17")), mean(g_("ex_pc_25_30"))]
            else:
                cells = [mean(lambda x: x["L1"][0]), mean(lambda x: x["L1"][1]), mean(lambda x: x["L1"][2]),
                         mean(lambda x: x["L2"][0]), mean(lambda x: x["L2"][1]), mean(lambda x: x["L2"][2]),
                         mean(g_("ex_first1")), mean(g_("ex_first5")), mean(g_("ex_last10"))]
            L.append(f"| {k} | " + " | ".join(fm(x) for x in cells) + f" | {len(okr)} | {nd} | {na} |")
            ph = ["0-1s", "1-5s", "5s-90%", "last10"] + (["pc15-17"] if pc else [])
            c2 = [mean(lambda x, p=p: x["mu_app_ph"].get(p, np.nan)) for p in ph]
            c2 += [mean(g_("zero_frac")), mean(g_("zero_run")), mean(g_("frac_dpclip")), mean(g_("frac_mufclamp")),
                   mean(g_("frac_xclip")), mean(g_("frac_stclip"))]
            if c == "PS":
                c2 += ([mean(g_(f"ps_max_{s}")) for s in PS_STEPS] + [mean(g_(f"ps_rec_{s}")) for s in PS_STEPS]
                       + [mean(g_(f"ps_recrel_{s}")) for s in PS_STEPS])
            if c == "BU":
                c2 += [mean(g_("bu_peak")), mean(g_("bu_burst")), mean(g_("bu_base"))]
            rows2.append((k, c2, ph))
        L.append("")
        if rows2:
            ph = rows2[0][2]
            h2 = [f"mu_app {p}" for p in ph] + ["zero frac", "longest zero (blocks)", "dp clip", "mu_f clamp", "X clip", "state clip"]
            if c == "PS":
                h2 += ([f"max ex {s}-{s + 2} s" for s in PS_STEPS] + [f"rec {s} s (s)" for s in PS_STEPS]
                       + [f"rec-rel {s} s (s)" for s in PS_STEPS])
            if c == "BU":
                h2 += ["BU fold peak", "BU 0-0.25 s", "BU 0.5-1 s"]
            L += ["| rule | " + " | ".join(h2) + " |", "|---" * (len(h2) + 1) + "|"]
            for k, c2, _ in rows2:
                L.append(f"| {k} | " + " | ".join(fm(x, 3) for x in c2) + " |")
            L.append("")
    return L


def rep_c2(J, full=True):
    L = []
    for t, r in J.items():
        for rn, x in r["rules"].items():
            if "i" not in x or not full:
                continue
            L += [f"### {t} / {rn} (div = {x.get('div')})", "",
                  "| block | path | g | mu_state | target | mu_app | X_hat | N_hat | Q | argmin T=4 | T=8 | T=16 | ex0 dB |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for j in range(len(x["i"])):
                g_ = lambda k: x[k][j] if k in x else np.nan
                L.append("| " + " | ".join([str(x["i"][j]), str(x["path"][j]), fm(g_("g")), fm(g_("mu_state"), 3), fm(g_("target"), 3),
                                             fm(g_("mu_app"), 3), fm(g_("X"), 4), fm(g_("N"), 4), fm(g_("Q"), 4),
                                             fm(g_("or_T4"), 3), fm(g_("or_T8"), 3), fm(g_("or_T16"), 3),
                                             fm(10 * np.log10(g_("ex0")))]) + " |")
            L.append("")
        if not any("i" in x for x in r["rules"].values()):
            continue
        L += [f"#### {t}: median log2(mu_state / argmin) and log2(target / argmin) by phase (0 = exact)", "",
              "| rule | T | quantity | early (<256) | mid (256-1023) | late (>=1024, path 0) | after pc |", "|---|---|---|---|---|---|---|"]
        for rn, x in r["rules"].items():
            if "i" not in x:
                continue
            ii = np.array(x["i"]); pth = np.array(x["path"])
            ph = {"early": ii < 256, "mid": (ii >= 256) & (ii < 1024), "late": (ii >= 1024) & (pth == 0), "pc": pth == 1}
            for Tn in TS:
                orc = np.array(x.get(f"or_T{Tn}", [np.nan] * len(ii)), float)
                for q in ("mu_state", "target"):
                    v = np.array(x[q], float); cells = []
                    for k in ("early", "mid", "late", "pc"):
                        m = ph[k] & np.isfinite(v) & np.isfinite(orc) & (v > 0) & (orc > 0)
                        cells.append(fm(float(np.median(np.log2(v[m] / orc[m])))) if m.any() else "NaN")
                    L.append(f"| {rn} | {Tn} | {q} | " + " | ".join(cells) + " |")
        L.append("")
    return L


def rep_c2x(J, ests=None):
    """ests: estimator rows to show (default: own (if present) + the EST2 shadows)."""
    L = []
    for t, r in J.items():
        rules = r["rules"]
        if not rules:
            continue
        phs = list(next(iter(rules.values()))["xstats"].keys())
        L += [f"### {t} (nb = {r['nb']})", "",
              "X_hat / X_true: median over the blocks of the phase (in brackets: fraction of blocks with X_hat = 0). "
              "Driver = the rule that runs the loop; estimator = 'own' (the driver's own X_hat) or a shadow estimator "
              "fed with the driver's u = y / max(g, 0.25) and applied mu.", "",
              "| driver | estimator | " + " | ".join(phs) + " |", "|---" * (len(phs) + 2) + "|"]
        for rn, x in rules.items():
            xs = x["xstats"]
            for k in (["own"] if "own" in xs[phs[0]] else []) + list(EST2):
                if ests is not None and k not in ests:
                    continue
                L.append(f"| {rn} | {k} | " + " | ".join(f"{fm(xs[p][k]['xr'], 3)} ({fm(xs[p][k]['x0'], 2)})" for p in phs) + " |")
        L += ["", "N_hat / N_true at the probe blocks (median; N_true = mean |u|^2 with w frozen at the SF fixed point "
              "of that path over the 32 blocks around the probe):", "",
              "| driver | estimator | " + " | ".join(phs) + " |", "|---" * (len(phs) + 2) + "|"]
        for rn, x in rules.items():
            xs = x["xstats"]
            for k in (["own"] if "own" in xs[phs[0]] else []) + list(EST2):
                if ests is not None and k not in ests:
                    continue
                L.append(f"| {rn} | {k} | " + " | ".join(fm(xs[p][k]["nr"], 3) for p in phs) + " |")
        L += ["", "Scale (median over the phase, per driver): X_true and N_true.", "",
              "| driver | quantity | " + " | ".join(phs) + " |", "|---" * (len(phs) + 2) + "|"]
        for rn, x in rules.items():
            xs = x["xstats"]
            for q in ("Xtrue_med", "Ntrue_med"):
                L.append(f"| {rn} | {q[:-4]} | " + " | ".join(f"{xs[p][q]:.3g}" if np.isfinite(xs[p][q]) else "NaN" for p in phs) + " |")
        L.append("")
    return L


def report2(o):
    smoke = o.smoke
    J3, JX = load("C3", smoke), load("C2X", smoke)
    par = next(iter(J3.values()), next(iter(JX.values()), {})).get("par", par2_desc(o))
    L = ["# RESULTS protoC2: Sec. 11 estimator variants on the SF-Y rules with the 1/g correction (prototype)", "",
         f"Generated by `python protoC.py report2{' --smoke' if smoke else ''}` from out/{'protoC_smoke' if smoke else 'protoC'}/"
         "protoC_C3_*.json and protoC_C2X_*.json.  dB conventions as in RESULTS_protoC.md (more negative is better).", "",
         "Constants: " + ", ".join(f"{k} = {fm(v, 4) if isinstance(v, float) else v}" for k, v in par.items()), "",
         "Estimators (THEORY_vss2.md Sec. 9.1 / 11): est91 = protoC (L = 4, beta_d = 0.1, lbar = 13, linear forms); "
         "estL4 = beta_d = 1 (ubar_{n-L} = u_{n-L}), lbar = 4, same linear forms; estX = estL4 statistics + Sec. 11.2 "
         "(b)-(d): rho(a)^{-L} X, a = h_eff * mean applied mu over the last L blocks, geometric noise cross terms with "
         "r(k) = (1 - |k|/4)_+ (all nonzero terms), Q form of Sec. 11.2 (a of the Q form capped at h_eff mu_fmax), "
         "2x2 solve, |det| below the relative threshold keeps the previous solution, X, N >= 0.  "
         "Rule 'SF-Y delayopt g gam.03' = protoC (identical to 'SF-Y est91 delayopt g gam.03'). "
         "Second C3 table column 'X clip' = X or N clipped at 0 (estX) / X clipped (est91, estL4); "
         "'mu_f clamp' = the mu (est91, estL4) or a (estX Q form) of the forms capped.", "",
         "## C3: closed loop", ""]
    L += rep_c1(J3) if J3 else ["(no outputs)", ""]
    if J3:
        L += ["estX: fraction of blocks where the previous (X, N) was kept (|det| below threshold), mean over seeds:", "",
              "| input | " + " | ".join(n for n in EST2_NAMES if "estX" in n) + " |", "|---" * (1 + sum('estX' in n for n in EST2_NAMES)) + "|"]
        byc = {}
        for t, r in J3.items():
            byc.setdefault(cls_of(t), []).append(r)
        for c, rs in byc.items():
            cells = []
            for n in EST2_NAMES:
                if "estX" not in n:
                    continue
                v = [r["rules"][n].get("frac_detkeep", np.nan) for r in rs if n in r["rules"]]
                cells.append(fm(float(np.mean(v)), 4) if v else "NaN")
            L.append(f"| {c} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## C2X: estimator diagnostic", "",
          "X_true = H^2 |P_M F v|^2 (y coordinates, the block's minimum-phase factor F, v = w - (w_ref + v_fp) with v_fp "
          "the SF model fixed point of the active path, theory/transient/model.py via vssT_lib.fixed_point); the "
          "estimators' X is the mean part |E u|^2 = kappa_y X_true with kappa_y ~ 0.96 (check 13), so a ratio near "
          "1 is exact.  Phases in blocks (8 blocks = 64 ms); pc: '>=1024' ends at 15 s.", ""]
    L += rep_c2x(JX) if JX else ["(no outputs)", ""]
    if JX and any("i" in x for r in JX.values() for x in r["rules"].values()):
        L += ["### Tracking (as C2 of RESULTS_protoC.md, summary only)", ""] + rep_c2(JX, full=False)
    p = os.path.join(HERE, "RESULTS_protoC2.md") if not smoke else os.path.join(OUT_SMOKE, "RESULTS_protoC2_smoke.md")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", p, f"({len(L)} lines)")


def rep_c4_extra(J4, vars_=VARS3[1:]):
    """pcb windows, early handover, restarts (per input class, mean over seeds)."""
    L = []
    byc = {}
    for t, r in J4.items():
        byc.setdefault(cls_of(t), []).append(r)
    pcbs = {c: rs for c, rs in byc.items() if rs[0].get("pcb")}
    for c, rs in pcbs.items():
        names = list(dict.fromkeys(k for r in rs for k in r["rules"]))
        ks = ["10-12", "15-20", "20-22", "25-30"]
        L += [f"### {c}: change windows (path -> pc path at 10 s, back at 20 s; tags: {', '.join(sorted(r['tag'] for r in rs))})", "",
              "| rule | " + " | ".join(f"L1 {k} s" for k in ks) + " | " + " | ".join(f"ex {k} s" for k in ks) + " | n |",
              "|---" * (2 * len(ks) + 2) + "|"]
        for k in names:
            okr = [r["rules"][k] for r in rs if k in r["rules"] and not r["rules"][k]["div"] and "pcb" in r["rules"][k]]
            mean = lambda f: float(np.mean([f(x) for x in okr])) if okr else np.nan
            cells = [mean(lambda x, w=w: x["pcb"]["L1_" + w]) for w in ks] + [mean(lambda x, w=w: x["pcb"]["ex_" + w]) for w in ks]
            L.append(f"| {k} | " + " | ".join(fm(x) for x in cells) + f" | {len(okr)} |")
        L.append("")
    L += ["### Early phase and restarts (protoC3 rules; mean over seeds)", "",
          "handover = first block where the early phase ends (n_e - 1); early frac = fraction of all blocks in an early "
          "phase (incl. after restarts); restarts: false = restarts not within 1 s after a true change, per 30 s; delay = "
          "blocks from the change to the first restart within 1 s (in brackets: number of missed changes over all seeds).", "",
          "| input | rule | handover (blocks) | early frac | false restarts / 30 s | delay change 1 | delay change 2 |",
          "|---|---|---|---|---|---|---|"]
    for c, rs in byc.items():
        names = [k for k in dict.fromkeys(k for r in rs for k in r["rules"]) if k.split()[-1] in vars_]
        for k in names:
            rows = [r["rules"][k] for r in rs if k in r["rules"]]
            ho = [x["handover"][0] for x in rows if x.get("handover")]
            ef = [x["early_frac"] for x in rows if "early_frac" in x]
            rsr = [x["rs"] for x in rows if "rs" in x]
            fr = fm(float(np.mean([q["false_per30"] for q in rsr]))) if rsr else "-"
            dl = []
            for j in range(2):
                v = [q["delays"][j] for q in rsr if len(q["delays"]) > j]
                if not v:
                    dl.append("-"); continue
                fin = [x for x in v if np.isfinite(x)]
                dl.append(f"{fm(float(np.mean(fin)), 1) if fin else 'NaN'} ({len(v) - len(fin)})")
            L.append(f"| {c} | {k} | {fm(float(np.mean(ho)), 1) if ho else '-'} | {fm(float(np.mean(ef)), 3) if ef else '-'} | "
                     f"{fr} | {dl[0]} | {dl[1]} |")
    L.append("")
    return L


def report3(o):
    smoke = o.smoke
    J4, JX = load("C4", smoke), load("C4X", smoke)
    par = next(iter(J4.values()), next(iter(JX.values()), {})).get("par", par3_desc(o))
    X3 = {k[3:]: v for k, v in par.items() if k.startswith("x3_")}
    L = ["# RESULTS protoC3: early phase from theory and adaptive smoothing with restarts (SF-Y, 1/g, delayopt; prototype)", "",
         f"Generated by `python protoC.py report3{' --smoke' if smoke else ''}` from out/{'protoC_smoke' if smoke else 'protoC'}/"
         "protoC_C4_*.json and protoC_C4X_*.json.  dB conventions as in RESULTS_protoC.md (more negative is better).", "",
         "Constants: " + ", ".join(f"{k} = {fm(v, 4) if isinstance(v, float) else v}" for k, v in par.items()), "",
         f"Variants (suffix of 'SF-Y <est91|estX> delayopt g gam<.03|.1>'; no suffix = the protoC2 rule, bitwise):", "",
         f"- early: internal step = mu_early = {fm(X3.get('mu_early', np.nan), 3)} (delayopt model with N = 0, injection "
         f"mu^2 [1 + (2tau_m-1) f] h_eff^2 x for K '{X3.get('early_k')}', T = {par.get('dopt_T')}, 1e-3 grid; a constant) "
         f"for the first n_e blocks after the start and after every restart; n_e = "
         + (f"{X3.get('n_early')} (fixed)" if X3.get("n_early") else
            f"the first block where X_hat is trusted (estimator ok, >= 2L blocks, Q <= {X3.get('trust_q')} x mean |u|^2 of "
            f"the last L blocks and that mean >= {X3.get('trust_dec')} x the mean of the L blocks before), at most "
            f"{X3.get('ne_max')}") + "; then the delayopt EMA from mu_state = mu_early.",
         f"- rs: outer EMAs of C and Q with beta_k = max({X3.get('beta_min')}, 1/(k+1)), k = updates since the last "
         f"(re)start; restart when |u_n|^2 > {X3.get('r_thr')} x Q_(n-1) for {X3.get('d')} consecutive blocks: C, Q reset, "
         "ubar / delayed buffer rebuilt from the previous trigger blocks (C resumes after L - d + 1 blocks), mu_state = "
         "mu_early with early, else kept.", "",
         "## C4: closed loop", ""]
    L += rep_c1(J4) if J4 else ["(no outputs)", ""]
    if J4:
        L += rep_c4_extra(J4)
    L += ["## C4X: estimator diagnostic (driver's own X_hat, N_hat)", "",
          "As C2X in RESULTS_protoC2.md (X_true, N_true definitions there); pc also split after the change "
          "(+0-63, +64-255, +256.. blocks).", ""]
    L += rep_c2x(JX, ests=("own",)) if JX else ["(no outputs)", ""]
    if JX:
        L += ["### Restarts and handover (C4X runs)", "", "| tag | driver | handover | restarts (blocks) | true change | "
              "delay | false / 30 s |", "|---|---|---|---|---|---|---|"]
        for t, r in JX.items():
            for rn, x in r["rules"].items():
                q = x.get("rs")
                L.append(f"| {t} | {rn} | {x.get('handover', [])[:4]} | "
                         + (f"{q['restarts'][:12]}{' ...' if len(q['restarts']) > 12 else ''} | {r.get('changes')} | "
                            f"{q['delays']} | {fm(q['false_per30'])} |" if q else "- | " + f"{r.get('changes')} | - | - |"))
        L.append("")
    p = os.path.join(HERE, "RESULTS_protoC3.md") if not smoke else os.path.join(OUT_SMOKE, "RESULTS_protoC3_smoke.md")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", p, f"({len(L)} lines)")


def rep_c5_rs(J5):
    """Restart table per input class and rule (seeds pooled)."""
    L = ["### Restarts (rs / rsN / rsG rules; per input, over seeds)", "",
         "false / 30 s = restarts not within 1 s after a true change, per 30 s of run (mean over seeds; [min, max]); "
         "total = all restarts per 30 s; delay = blocks from the change to the first restart within 1 s (mean over the "
         "detected changes); missed = changes with no restart within 1 s (count over seeds / number of changes).", "",
         "| input | rule | false / 30 s | [min, max] | total / 30 s | delay change 1 | missed 1 | delay change 2 | missed 2 | n |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    byc = {}
    for t, r in J5.items():
        byc.setdefault(cls_of(t), []).append(r)
    for c, rs in byc.items():
        names = [k for k in dict.fromkeys(k for r in rs for k in r["rules"]) if k.split()[-1] in VARS3[1:] + VARS4]
        for k in names:
            q = [r["rules"][k]["rs"] for r in rs if k in r["rules"] and "rs" in r["rules"][k]]
            if not q:
                continue
            fr = [x["false_per30"] for x in q]
            tot = [len(x["restarts"]) / (r["nb"] / BPS / 30.0) for x, r in zip(q, [r for r in rs if k in r["rules"] and "rs" in r["rules"][k]])]
            cells = []
            for j in range(2):
                v = [x["delays"][j] for x in q if len(x["delays"]) > j]
                fin = [x for x in v if np.isfinite(x)]
                cells += ([fm(float(np.mean(fin)), 1) if fin else "NaN", f"{len(v) - len(fin)} / {len(v)}"] if v else ["-", "-"])
            L.append(f"| {c} | {k} | {fm(float(np.mean(fr)))} | [{fm(min(fr))}, {fm(max(fr))}] | {fm(float(np.mean(tot)))} | "
                     + " | ".join(cells) + f" | {len(q)} |")
    L.append("")
    return L


def report4(o):
    smoke = o.smoke
    J5 = load("C5", smoke)
    par = next(iter(J5.values()), {}).get("par", par4_desc(o))
    X3 = {k[3:]: v for k, v in par.items() if k.startswith("x3_")}
    X4 = {k[3:]: v for k, v in par.items() if k.startswith("x4_")}
    L = ["# RESULTS protoC4: restart statistics rsN / rsG (SF-Y, 1/g, delayopt, early; prototype)", "",
         f"Generated by `python protoC.py report4{' --smoke' if smoke else ''}` from out/{'protoC_smoke' if smoke else 'protoC'}/"
         "protoC_C5_*.json.  dB conventions as in RESULTS_protoC.md (more negative is better).", "",
         "Constants: " + ", ".join(f"{k} = {fm(v, 4) if isinstance(v, float) else v}" for k, v in par.items()), "",
         "Variants (suffix of 'SF-Y <est91|estX> delayopt g gam.03'); early and rs as in RESULTS_protoC3.md "
         f"(mu_early = {fm(X3.get('mu_early', np.nan), 3)}, n_e = "
         + (f"{X3.get('n_early')} fixed" if X3.get("n_early") else f"trust rule, at most {X3.get('ne_max')}") + "):", "",
         f"- rs (old): restart when |u_n|^2 > {X3.get('r_thr')} x Q_(n-1) for {X3.get('d')} consecutive blocks, "
         "u = y / max(g, 0.25); no guard.",
         f"- rsN: z_n = sum e^2 / sum r^2 over the M = {M}-sample error window of block n (e = the errors in the "
         "direction's error frame; r = data['r'] = hS * x, the filtered reference whose spectrum the method uses); "
         f"zbar = EMA (beta_z = {X4.get('z_beta')}) of z, never reset; restart when z_n > {X4.get('z_thr')} x zbar_(n-1) "
         f"for {X4.get('z_d')} consecutive blocks.",
         f"- rsG: |y_n|^2 / g_n^2 (no floor on g) against its own EMA (adaptive beta, started / reset with Q); "
         f"r_thr = {X3.get('r_thr')}, d = {X3.get('d')}.",
         f"- rsN and rsG: no restart within {X4.get('guard')} blocks of the previous restart; restart actions as rs "
         "(C, Q, k, ubar / delayed buffer reset, estX's kept (X, N) cleared, mu_state = mu_early and a new early phase).", "",
         "## C5: closed loop", ""]
    L += rep_c1(J5) if J5 else ["(no outputs)", ""]
    if J5:
        L += rep_c4_extra(J5, VARS3[1:] + VARS4)
        L += rep_c5_rs(J5)
    p = os.path.join(HERE, "RESULTS_protoC4.md") if not smoke else os.path.join(OUT_SMOKE, "RESULTS_protoC4_smoke.md")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", p, f"({len(L)} lines)")


def report(o):
    smoke = o.smoke
    J1, J2 = load("C1", smoke), load("C2", smoke)
    par = next(iter(J1.values()), next(iter(J2.values()), {})).get("par", o.P.desc())
    L = ["# RESULTS protoC: Sec. 10.6 method changes on the SF direction (prototype)", "",
         f"Generated by `python protoC.py report{' --smoke' if smoke else ''}` from out/{'protoC_smoke' if smoke else 'protoC'}/protoC_*.json. "
         "All dB values: more negative is better (L1 = ANR mean in dB, L2 = 10log10(sum e^2/sum d^2), "
         "ex = 10log10 of the mean Phi-excess v'Phi v/Jo over the window; div = numeric divergence, amp = L1 whole > 0 "
         "without divergence; means over non-diverged seeds).", "",
         "Constants: " + ", ".join(f"{k} = {fm(v, 4) if isinstance(v, float) else v}" for k, v in par.items()), "",
         "Rule names: SF-Y = statistics in y coordinates, SF-E = the same rule with statistics on Delta; "
         "targets onestep / delaycap (min(onestep, a*/h_eff)) / delayopt (T-block model); correction none / g / dp / dp+g; "
         "gam = gamma of the mu_state EMA.  Second table per input: mean applied mu per phase, zero dwell of mu_state, "
         "fraction of blocks with the dp ratio clipped, the formula mu capped at mu_fmax, X_hat clipped, mu_state clipped "
         "to [0, 1.5]; PS: max excess (dB) in the 2 s after each step and time (s) from the step to excess <= "
         f"{o.rec_db:g} dB after the peak (rec) and to the 1 s pre-step mean + {REL_DB:g} dB (rec-rel); NaN: not reached "
         "before the next step / end of run; BU: excess folded over the 1 s burst "
         "period (seconds 5..), peak and means over phases 0-0.25 s and 0.5-1 s (dB).", "",
         "## C1: closed loop", ""]
    L += rep_c1(J1) if J1 else ["(no outputs)", ""]
    L += ["## C2: tracking check", "",
          "argmin T: internal step (mu_state units) minimising the mean Phi-excess over the first T updates of a "
          "constant-internal-step continuation of the actual realisation from that block, with the rule's own power "
          "correction (dp EMA carried on); grid 0 and 40 log-spaced values in [0.01, 3].  target = the rule's target at "
          "that block (NaN before the estimator starts at block 8).", ""]
    L += rep_c2(J2) if J2 else ["(no outputs)", ""]
    p = os.path.join(HERE, "RESULTS_protoC.md") if not smoke else os.path.join(OUT_SMOKE, "RESULTS_protoC_smoke.md")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", p, f"({len(L)} lines)")


# ==============================================================================================
def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "report", "c1", "c2", "c3", "c2x", "report2", "c4", "c4x", "report3", "c5", "report4"])
    ap.add_argument("--classes"); ap.add_argument("--seeds"); ap.add_argument("--tags")
    ap.add_argument("--nb", type=int, default=None); ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--force", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rules", default=None, help="';'-separated rule names")
    ap.add_argument("--f", type=float, default=0.85); ap.add_argument("--tau-a", type=float, default=3.8)
    ap.add_argument("--tau-m", type=float, default=4.0)
    ap.add_argument("--heff", type=float, default=None, help="h_eff (default 0.95 h = 0.2375)")
    ap.add_argument("--ck", default=",".join(f"{c:g}" for c in C_DEF), help="frame-occupancy weights c_k (normalised)")
    ap.add_argument("--gz", type=float, default=0.0, help="Gbar = gz N + gm f X in the C form (Sec. 9.1); default 0")
    ap.add_argument("--gm", type=float, default=0.0)
    ap.add_argument("--dopt-T", type=int, default=8); ap.add_argument("--dopt-k", choices=["excl1", "full"], default="excl1")
    ap.add_argument("--mu-fmax", type=float, default=None, help="cap of mu in the Sec. 9.1 forms (default 0.75/(h_eff (tau_a-1)))")
    ap.add_argument("--rec-db", type=float, default=-20.0, help="PS recovery level (dB)")
    ap.add_argument("--kappa", type=float, default=X2_DEF["kappa"], help="estX: kappa of Sec. 11.2 (default 1)")
    ap.add_argument("--estx-a", choices=["applied", "state"], default=X2_DEF["amode"],
                    help="estX: a = h_eff * mean of applied mu (default) or of mu_state over the last L blocks")
    ap.add_argument("--det-rel", type=float, default=X2_DEF["det_rel"], help="estX: relative |det| threshold (keep previous)")
    ap.add_argument("--no-aq-cap", action="store_true", help="estX: do not cap the a of the Q cross term at h_eff mu_fmax")
    ap.add_argument("--no-oracle", action="store_true", help="c2x: skip the continuation argmin (oracle) at the sampled blocks")
    ap.add_argument("--early-k", choices=["full", "excl1"], default=X3_DEF["early_k"],
                    help="protoC3 early value: injection with the '1' of K (full, spec) or without (excl1)")
    ap.add_argument("--n-early", type=int, default=X3_DEF["n_early"], help="protoC3: fixed early length (0 = trust rule)")
    ap.add_argument("--ne-max", type=int, default=X3_DEF["ne_max"], help="protoC3: max early length (blocks)")
    ap.add_argument("--trust-q", type=float, default=X3_DEF["trust_q"])
    ap.add_argument("--trust-dec", type=float, default=X3_DEF["trust_dec"])
    ap.add_argument("--beta-min", type=float, default=X3_DEF["beta_min"])
    ap.add_argument("--r-thr", type=float, default=X3_DEF["r_thr"], help="protoC3 restart: |u|^2 > r_thr * Q")
    ap.add_argument("--rs-d", type=int, default=X3_DEF["d"], help="protoC3 restart: consecutive blocks")
    ap.add_argument("--rs-kmin", type=int, default=X3_DEF["rs_kmin"], help="protoC3 restart: min Q updates since restart")
    ap.add_argument("--z-beta", type=float, default=X4_DEF["z_beta"], help="protoC4 rsN: EMA beta of z")
    ap.add_argument("--z-thr", type=float, default=X4_DEF["z_thr"], help="protoC4 rsN: z_n > z_thr * zbar_(n-1)")
    ap.add_argument("--z-d", type=int, default=X4_DEF["z_d"], help="protoC4 rsN: consecutive blocks")
    ap.add_argument("--rs-guard", type=int, default=X4_DEF["guard"], help="protoC4 rsN / rsG: no restart within this "
                    "many blocks of the previous one")
    o = ap.parse_args(argv)
    o.rules = o.rules.split(";") if o.rules else None
    o.X2 = dict(kappa=o.kappa, amode=o.estx_a, det_rel=o.det_rel, aq_cap=not o.no_aq_cap)
    o.oracle = not o.no_oracle
    o.P = Par(f=o.f, ta=o.tau_a, tm=o.tau_m, heff=0.95 * H if o.heff is None else o.heff,
              ck=[float(v) for v in o.ck.split(",")], gz=o.gz, gm=o.gm, dopt_T=o.dopt_T, dopt_k=o.dopt_k, mu_fmax=o.mu_fmax)
    o.X3 = dict(early_k=o.early_k, n_early=o.n_early, ne_max=o.ne_max, trust_q=o.trust_q, trust_dec=o.trust_dec,
                beta_min=o.beta_min, r_thr=o.r_thr, d=o.rs_d, rs_kmin=o.rs_kmin)
    o.X4 = dict(z_beta=o.z_beta, z_thr=o.z_thr, z_d=o.z_d, guard=o.rs_guard)
    o.mu_e = mu_early_value(o.P, o.early_k)
    if o.cmd in ("c4", "c4x", "c5", "check"):
        print(f"protoC3 early step mu_early = {o.mu_e:.3f} (K '{o.early_k}', N = 0, T = {o.P.dopt_T})", flush=True)
    return o


if __name__ == "__main__":
    o = parse(sys.argv[1:])
    if o.cmd == "check":
        sys.exit(0 if check(o) else 1)
    elif o.cmd == "report":
        report(o)
    elif o.cmd == "report2":
        report2(o)
    elif o.cmd == "report3":
        report3(o)
    elif o.cmd == "report4":
        report4(o)
    else:
        main_jobs(o.cmd, o)
