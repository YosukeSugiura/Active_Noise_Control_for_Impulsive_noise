"""Core library for the vssT verification experiments (THEORY_vss2.md, sections 0-8).

Contents
  data(tag)            input builders (Gaussian, power steps, bursts, sinusoidal log-power modulation,
                       isolated impulses, alpha 1.45, alpha 1.2, path change)
  Stats                per-block shared statistics (Srr, Pbar, eps, gain g_n); depend on the reference only
  direction()          SF / W(a, om) directions, identical arithmetic to simx.run
  pi_seq()             participation ratio pi_n for several window definitions
  xhat/psi_muhat       Sec. 2 formulas
  DirectEst            delayed inner-product estimator (C, Q) on u = Delta / max(g, g_min)
  rules                Fixed, Power (mu0/g, mu0/(g rho)), Direct (SF-D / GS-D), Switch (Sec. 5), wrappers
  simulate()           block loop identical to simx.run, with Phi-excess every block and a pre-update hook
  frozen_*             frozen-coefficient probes
  metrics()            L1 / L2 ANR, divergence

Conventions: the update is w <- w - 2 mu B/M dir = w - mu Delta with Delta = KD dir, KD = 2B/M, so that
E Delta = h v with h = KD * M/N = 0.25 (THEORY Sec. 0).  Excess = v' Phi v / Jo (white unit-power model),
v relative to the M-tap Wiener solution of the path that is active in that block.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "theory", "transient"))
sys.path.insert(0, HERE)
from simx import *          # noqa: F401,F403  np, signal, linalg, D, X, ML, SFm, MFm, IC, BS, O, FS, M, N, B, wM, gM, Jo, Phi, ramp, run
import simx as SX
import re

KD = 2.0 * B / M                     # direction -> Delta scale (0.5)
H = KD * M / N                       # h = 0.25
TAU_A = 3.8
NBIN = N // 2 + 1
NB_FULL = O.N_NOISE // B             # 3750
PC_BLOCK = int(MFm.PC_T * FS) // B   # 1875: first block of the new primary path
GMIN = 0.25                          # floor of g_n in 1/g and in u = Delta/max(g, g_min)  (not fixed by THEORY)
RHO_MIN = 0.1                        # floor of rho_n (not fixed by THEORY)
DIV_EX = 1e8                         # Phi-excess above this counts as divergence
ZERO_THR = 1e-3                      # mu_state below this counts as "zero" for the dwell statistics
EST = dict(L=4, bd=0.1, b=0.05)      # SF-D estimator constants (task spec)
CONST = dict(f=0.9, tm=2.5, ta=TAU_A)   # default constants for the step rules (E4-E6)


# ----------------------------------------------------------------------------------------------
# Wiener references (original and path-change primary) and model fixed points
# ----------------------------------------------------------------------------------------------
def wiener_for(hP):
    """Same arithmetic as closedform/base.wiener, for an arbitrary primary IR (length 512)."""
    lags = np.arange(M)
    rrr = BS.xc(BS.hS, BS.hS, lags); prd = BS.xc(BS.hS, hP, lags)
    w = linalg.solve_toeplitz(rrr, -prd)
    g = np.zeros(max(len(hP), len(BS.hS) + M - 1)); g[:len(hP)] += hP; g[:len(BS.hS) + M - 1] += np.convolve(BS.hS, w)
    return w, g


def _hP_pc():
    hPn = np.zeros(M); hPn[MFm.PC_DELAY:] = MFm.PC_SCALE * BS.hP[:M - MFm.PC_DELAY]; return hPn


wM_pc, gM_pc = wiener_for(_hP_pc())
Jo_pc = float(gM_pc @ gM_pc)
REFS = {0: (wM, gM, Jo), 1: (wM_pc, gM_pc, Jo_pc)}
_FP = {}


def fixed_point(theta, path=0):
    """Mean fixed point v (relative to that path's wM) from theory/transient/model.py.
    theta: ('SF',) or ('W', a, om).  path 0: original primary, 1: path-change primary.
    model.py hard-codes gM / p at import; they are swapped temporarily and its cache is restored."""
    theta = tuple(theta) if theta[0] == "SF" else ("W", float(theta[1]), round(float(theta[2]), 2))
    key = (theta, path)
    if key in _FP:
        return _FP[key]
    import model as MD
    gm = REFS[path][1]
    old = (MD.gM, MD.p, dict(MD._cache))
    try:
        MD.gM = gm; MD.p = BS.xc(BS.hS, gm, MD.lags); MD._cache.clear()
        if theta[0] == "SF":
            v = MD.sf_v()
        else:
            A, b, _ = MD.op(theta); v = -np.linalg.solve(A, b)
    finally:
        MD.gM, MD.p = old[0], old[1]; MD._cache.clear(); MD._cache.update(old[2])
    _FP[key] = v
    return v


def path_of(data, i):
    sg = data.get("pseg")             # optional [(first block, path), ...] (protoC3 'pcb'); absent for all other inputs
    if sg is not None:
        p = 0
        for b0, q in sg:
            if i >= b0:
                p = q
        return p
    return 1 if (data.get("pc") and i >= PC_BLOCK) else 0


def exP(v, Jref):
    return float(v @ (Phi @ v)) / Jref


# ----------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------
CLASSES = ["G", "PS", "BU", "SM", "II", "a145", "a120", "pc"]
CLASS_DESC = {"G": "Gaussian", "PS": "power steps (+20 dB 10-15 s, -20 dB 20-25 s)",
              "BU": "bursts (+20 dB, 50 ms, 1/s)", "SM": "sinusoidal log-power modulation (1 s, +-10 dB)",
              "II": "isolated impulses on Gaussian (1 per ~0.5 s, 30x std)", "a145": "SaS alpha 1.45",
              "a120": "SaS alpha 1.2", "pc": "path change at 15 s (alpha 1.45)"}


def impulses(seed, amp=30.0):
    L = O.N_NOISE
    z = O.cms_symmetric_stable(2.0, L, seed)
    rng = np.random.default_rng(10_000 + seed)
    sd = float(np.std(z))
    k = np.arange(1, int(L / FS / 0.5))
    pos = np.round((0.5 * k + rng.uniform(-0.1, 0.1, len(k))) * FS).astype(int)
    pos = pos[(pos >= 0) & (pos < L)]
    z[pos] += amp * sd * rng.choice([-1.0, 1.0], len(pos))
    return z


def data(tag):
    """tag = <class>_s<seed>, class in CLASSES; 'II<amp>_s<seed>' sets the impulse amplitude (default 30)."""
    m = re.fullmatch(r"([A-Za-z]+)(\d*(?:\.\d+)?)_s(\d+)", tag)
    if not m:
        raise ValueError(tag)
    cls, par, seed = m.group(1), m.group(2), int(m.group(3))
    if cls == "a" and par in ("145", "120"):
        cls = "a" + par; par = ""
    pc = False
    if cls == "G":
        x = IC.make_input("A", seed); dd = D.build(x)
    elif cls == "PS":
        dd = D.build(IC.make_input("B", seed))
    elif cls == "BU":
        dd = D.build(IC.make_input("C", seed))
    elif cls == "SM":
        t = np.arange(O.N_NOISE) / FS
        dd = D.build(10 ** (10 * np.sin(2 * np.pi * t) / 20) * O.cms_symmetric_stable(2.0, O.N_NOISE, seed))
    elif cls == "II":
        dd = D.build(impulses(seed, float(par) if par else 30.0))
    elif cls == "a145":
        dd = O.load_source("cms", seed)
    elif cls == "a120":
        dd = D.load_data("cms12", seed)
    elif cls == "pc":
        dd = MFm.build_pc(O.cms_symmetric_stable(1.45, O.N_NOISE, seed)); pc = True
    else:
        raise ValueError(tag)
    dd["pc"] = pc; dd["tag"] = tag
    return dd


# ----------------------------------------------------------------------------------------------
# Shared statistics (reference only) and participation ratio
# ----------------------------------------------------------------------------------------------
class Stats:
    """Srr, Pbar, eps per block, exactly as X.Shared.step (E does not enter these), and g_n = mean|R|^2 / Pbar."""

    def __init__(s, data, Rall):
        sh = X.Shared(data["S2"]); nb = len(Rall); E0 = np.zeros(NBIN, complex)
        s.Srr = np.empty((nb, NBIN)); s.Pbar = np.empty(nb); s.eps = np.empty(nb)
        # Shared also computes coherence, which is unused here and undefined during leading silence.
        with np.errstate(invalid="ignore"):
            for i in range(nb):
                q = sh.step(Rall[i], E0)
                s.Srr[i], s.Pbar[i], s.eps[i] = q["Srr"], q["Pbar"], q["eps"]
        ref_power = np.mean(np.abs(Rall) ** 2, axis=1)
        s.g_zero = (ref_power == 0) & (s.Pbar == 0)
        s.g = np.divide(ref_power, s.Pbar, out=np.ones_like(ref_power), where=~s.g_zero)
        s.pi = {k: pi_seq(data, nb, k) for k in PI_DEFS}


PI_DEFS = ["x", "xM", "r"]
PI_DESC = {"x": "x over the N-sample reference frame", "xM": "x over the last M samples",
           "r": "filtered reference r over the N-sample frame"}


def pi_seq(data, nb, which):
    """pi_n = sum s^4 / (sum s^2)^2 over the window ending at the end of block n (same frame as R_n)."""
    sig = data["r"] if which == "r" else data["x"]
    W = M if which == "xM" else N
    out = np.empty(nb)
    for i in range(nb):
        e = i * B + B; seg = sig[max(0, e - W):e]
        p2 = float(seg @ seg)
        out[i] = float(np.sum(seg ** 4)) / p2 ** 2 if p2 > 0 else 0.0
    return out


class Ctx:
    __slots__ = ("i", "R", "E", "Srr", "Pbar", "eps", "g", "w", "pi")

    def __init__(s, i, R, E, st, w):
        s.i, s.R, s.E, s.w = i, R, E, w
        s.Srr, s.Pbar, s.eps, s.g = st.Srr[i], st.Pbar[i], st.eps[i], st.g[i]
        s.pi = {k: v[i] for k, v in st.pi.items()}


def direction(kind, R, E, Srr, Pbar, eps, om=1.0, a=0.5):
    """'SF' (eps_reg 1e-2), 'W' (Lam=(1-om)Srr+om Srr^a Pbar^(1-a)+eps), 'B0' = W om=0, 'geo' = W om=1 a=0.5."""
    if kind == "SF":
        return SFm.sf_dir(R, E, Srr, Pbar, eps, 1e-2)
    if kind == "B0":
        om = 0.0
    elif kind == "geo":
        om, a = 1.0, 0.5
    Lam = (1 - om) * Srr + om * Srr ** a * Pbar ** (1 - a) + eps
    return np.fft.irfft(np.conj(R) * E / Lam, N)[:M]


def gs_lam(Srr, Pbar, eps, om):
    return (1 - om) * Srr + om * np.sqrt(Srr * Pbar) + eps


# ----------------------------------------------------------------------------------------------
# Sec. 2 formulas
# ----------------------------------------------------------------------------------------------
def xhat(C, Q, mu, f, tm, ta, h=H):
    return (C + h * mu * ta * Q) / (1.0 + h * mu * (ta * (1 + f) - tm * f))


def psi_muhat(C, Q, mu, a0, f, tm, ta, h=H):
    """psi = Xhat/Q and mu_hat (Sec. 2).  Xhat is clipped to [0, Q/(1+f)] (THEORY states the range,
    not what to do when the estimate leaves it).  Returns (nan, nan) when no estimate exists yet."""
    if C is None or Q is None or not (Q > 0):
        return float("nan"), float("nan")
    Xh = min(max(xhat(C, Q, mu, f, tm, ta, h), 0.0), Q / (1 + f))
    if Xh <= 0:
        return 0.0, 0.0
    den = (a0 + tm * f) * Xh + ta * (Q - (1 + f) * Xh)
    return Xh / Q, (Xh / h) / den


CV = np.full(NBIN, 2.0); CV[0] = 1.0; CV[-1] = 1.0


def ipw(A, Bf, w):
    """Weighted spectral inner product, identical to vssI2.ipw; w=None is the Euclidean product."""
    p = (A * np.conj(Bf)).real
    if w is not None:
        p = p * w
    return float(np.sum(CV * p) / N)


class DirectEst:
    """C <- EMA_b <u_n, ubar_{n-L}>,  Q <- EMA_b |u_n|^2,  ubar <- EMA_bd(u),  u = Delta/max(g, gmin).
    ip: 'E' (Euclidean), 'Phi' (time-domain Phi), 'spec' (bin weights wc for C, wq for Q; None = unweighted).
    Same buffering as vss2.Ctrl2 / vssI (statistics start at block L+4)."""

    def __init__(s, ip="E", L=EST["L"], bd=EST["bd"], b=EST["b"], gmin=GMIN):
        s.ip, s.L, s.bd, s.b, s.gmin = ip, L, bd, b, gmin
        s.ubar = None; s.buf = [None] * L; s.C = s.Q = None

    def update(s, i, dv, g, wc=None, wq=None):
        u = KD * dv / max(g, s.gmin)
        old = s.buf[0]
        if s.ip == "spec":
            U = np.fft.rfft(u, N)
            c = 0.0 if old is None else ipw(U, old, wc); q = ipw(U, U, wq)
        elif s.ip == "Phi":
            Pu = Phi @ u
            c = 0.0 if old is None else float(old @ Pu); q = float(u @ Pu)
        else:
            c = 0.0 if old is None else float(u @ old); q = float(u @ u)
        if i >= s.L + 4:
            if s.C is None:
                s.C, s.Q = c, q
            else:
                s.C = (1 - s.b) * s.C + s.b * c; s.Q = (1 - s.b) * s.Q + s.b * q
        s.ubar = u.copy() if s.ubar is None else (1 - s.bd) * s.ubar + s.bd * u
        s.buf = s.buf[1:] + [np.fft.rfft(s.ubar, N) if s.ip == "spec" else s.ubar.copy()]

    def psi_mu(s, mu, a0=1.0, f=CONST["f"], tm=CONST["tm"], ta=CONST["ta"]):
        return psi_muhat(s.C, s.Q, mu, a0, f, tm, ta)


# ----------------------------------------------------------------------------------------------
# Step rules.  block(i, ctx) -> (mu_applied, dir);  mu_formula(): the mu that enters Sec. 2 formulas.
# ----------------------------------------------------------------------------------------------
def _arr(v, i):
    return float(v[min(i, len(v) - 1)]) if np.ndim(v) else float(v)


class Rule:
    kind = "SF"

    def __init__(s):
        s.tr = {"mu_app": []}

    def om_at(s, i):
        return 1.0

    def mu_formula(s, i):
        return float("nan")

    def traces(s):
        return {k: np.asarray(v, dtype=float) for k, v in s.tr.items()}


class Fixed(Rule):
    def __init__(s, kind, mu, om=1.0, a=0.5):
        super().__init__(); s.kind, s.mu, s.om, s.a = kind, mu, om, a

    def om_at(s, i):
        return _arr(s.om, i)

    def mu_formula(s, i):
        return _arr(s.mu, i)

    def block(s, i, c):
        mu = _arr(s.mu, i)
        dv = direction(s.kind, c.R, c.E, c.Srr, c.Pbar, c.eps, s.om_at(i), s.a)
        s.tr["mu_app"].append(mu)
        return mu, dv


class Power(Rule):
    """mu_n = mu0 / max(g, gmin) ('g') or mu0 / (max(g, gmin) rho_n) ('grho'), rho = max(1 - pi (1 - psi), rho_min);
    psi from a passive DirectEst on the applied direction with mu = mu0 (psi = 1/(1+f) before it exists)."""

    def __init__(s, kind, mu0, mode, pidef="x", gmin=GMIN, rho_min=RHO_MIN, om=1.0, ip="E", **const):
        super().__init__(); s.kind, s.mu0, s.mode, s.pidef, s.gmin, s.rho_min, s.om = kind, mu0, mode, pidef, gmin, rho_min, om
        s.c = dict(CONST, **const); s.est = DirectEst(ip, gmin=gmin); s.tr.update(psi=[], rho=[])
        s.psi = 1.0 / (1 + s.c["f"])

    def om_at(s, i):
        return _arr(s.om, i)

    def mu_formula(s, i):
        return s.mu0

    def block(s, i, c):
        rho = max(1.0 - c.pi[s.pidef] * (1.0 - s.psi), s.rho_min) if s.mode == "grho" else 1.0
        mu = s.mu0 / (max(c.g, s.gmin) * rho)
        dv = direction(s.kind, c.R, c.E, c.Srr, c.Pbar, c.eps, s.om_at(i))
        s.est.update(i, dv, c.g)
        p, _ = s.est.psi_mu(s.mu0, 1.0, s.c["f"], s.c["tm"], s.c["ta"])
        if np.isfinite(p):
            s.psi = p
        s.tr["mu_app"].append(mu); s.tr["psi"].append(s.psi); s.tr["rho"].append(rho)
        return mu, dv


class Direct(Rule):
    """Direct form (THEORY Sec. 2, Sec. 6, Sec. 8a-b):
       mu_state <- (1-gamma) mu_state + gamma clip(mu_hat(C, Q, mu_state), 0, mu_max)
       applied   = mu_state x {1 ('none'), 1/max(g,gmin) ('g'), 1/(max(g,gmin) rho) ('grho')}
    kind 'SF' or 'W' (om schedule, a=0.5).  ip 'E' / 'Phi' / weight mode 'w:none|C|Q|both' (spectral, E5)."""

    def __init__(s, kind="SF", corr="g", gamma=0.03, a0=1.0, mu_max=1.0, mu_init=0.3, om=1.0, gmin=GMIN,
                 rho_min=RHO_MIN, pidef="x", ip="E", **const):
        super().__init__()
        s.kind, s.corr, s.gamma, s.a0, s.mu_max, s.om, s.gmin, s.rho_min, s.pidef = kind, corr, gamma, a0, mu_max, om, gmin, rho_min, pidef
        s.c = dict(CONST, **const)
        s.wmode = ip[2:] if ip.startswith("w:") else None
        s.est = DirectEst("spec" if s.wmode else ip, gmin=gmin)
        s.mu_state = mu_init; s.psi = 1.0 / (1 + s.c["f"])
        s.tr.update(mu_state=[], psi=[], muhat=[])

    def om_at(s, i):
        return _arr(s.om, i)

    def mu_formula(s, i):
        return s.mu_state

    def factor(s, c):
        if s.corr == "none":
            return 1.0
        f = 1.0 / max(c.g, s.gmin)
        if s.corr == "grho":
            f /= max(1.0 - c.pi[s.pidef] * (1.0 - s.psi), s.rho_min)
        return f

    def weights(s, c, om):
        if not s.wmode or s.wmode == "none":
            return None, None
        wc = gs_lam(c.Srr, c.Pbar, c.eps, om) / c.Pbar if s.wmode in ("C", "both") else None
        wq = c.Srr / c.Pbar if s.wmode in ("Q", "both") else None
        return wc, wq

    def block(s, i, c):
        om = s.om_at(i)
        mu = s.mu_state * s.factor(c)
        dv = direction(s.kind, c.R, c.E, c.Srr, c.Pbar, c.eps, om)
        wc, wq = s.weights(c, om)
        s.est.update(i, dv, c.g, wc, wq)
        p, mh = s.est.psi_mu(s.mu_state, s.a0, s.c["f"], s.c["tm"], s.c["ta"])
        if np.isfinite(mh):
            s.psi = p
            s.mu_state = (1 - s.gamma) * s.mu_state + s.gamma * min(max(mh, 0.0), s.mu_max)
        s.tr["mu_app"].append(mu); s.tr["mu_state"].append(s.mu_state); s.tr["psi"].append(s.psi); s.tr["muhat"].append(mh)
        return mu, dv


class Track(Rule):
    """Wraps a B0 rule (Sec. 5 open loop): also computes the geo direction (not applied) and psi for both."""

    def __init__(s, base, gmin=GMIN):
        super().__init__(); s.base = base; s.kind = base.kind
        s.eB, s.eG = DirectEst("E", gmin=gmin), DirectEst("E", gmin=gmin)
        s.tr = {"psi_B0": [], "psi_geo": []}

    def om_at(s, i):
        return s.base.om_at(i)

    def mu_formula(s, i):
        return s.base.mu_formula(i)

    def block(s, i, c):
        mu_f = s.base.mu_formula(i)
        mu, dv = s.base.block(i, c)
        dg = direction("geo", c.R, c.E, c.Srr, c.Pbar, c.eps)
        s.eB.update(i, dv, c.g); s.eG.update(i, dg, c.g)
        s.tr["psi_B0"].append(s.eB.psi_mu(mu_f)[0]); s.tr["psi_geo"].append(s.eG.psi_mu(mu_f)[0])
        return mu, dv

    def traces(s):
        t = s.base.traces(); t.update({k: np.asarray(v, float) for k, v in s.tr.items()}); return t


class Switch(Direct):
    """Sec. 5 closed loop: omega_n = EMA_bom( psi_geo / (psi_geo + psi_B0) ), omega_0 = 0; applied direction
    W(a=0.5, omega_n); step = direct form on the applied direction.  psi_B0, psi_geo from their own estimators
    on the B0 and geo directions (computed every block), with mu = mu_state."""

    def __init__(s, bom=0.05, **kw):
        kw.setdefault("kind", "W"); super().__init__(**kw)
        s.bom, s.omv = bom, 0.0
        s.eB, s.eG = DirectEst("E", gmin=s.gmin), DirectEst("E", gmin=s.gmin)
        s.tr.update(om=[], psi_B0=[], psi_geo=[])

    def om_at(s, i):
        return s.omv

    def block(s, i, c):
        om = s.omv
        mu, dv = super().block(i, c)
        dB = direction("B0", c.R, c.E, c.Srr, c.Pbar, c.eps); dG = direction("geo", c.R, c.E, c.Srr, c.Pbar, c.eps)
        s.eB.update(i, dB, c.g); s.eG.update(i, dG, c.g)
        pB = s.eB.psi_mu(s.mu_state)[0]; pG = s.eG.psi_mu(s.mu_state)[0]
        if np.isfinite(pB) and np.isfinite(pG) and pB + pG > 0:
            s.omv = (1 - s.bom) * s.omv + s.bom * pG / (pG + pB)
        s.tr["om"].append(om); s.tr["psi_B0"].append(pB); s.tr["psi_geo"].append(pG)
        return mu, dv


class CtrlW(Rule):
    """vss2.Ctrl2 (SF-V2, GS-V2) driven exactly as in simx.run."""

    def __init__(s, ctrl, kind):
        super().__init__(); s.ctrl, s.kind = ctrl, kind; s.tr["om"] = []; s._om = 1.0

    def om_at(s, i):
        return s._om

    def block(s, i, c):
        mu, om = s.ctrl.step(i, dict(R=c.R, Pbar=c.Pbar)); s._om = om
        dv = direction(s.kind, c.R, c.E, c.Srr, c.Pbar, c.eps, om, 0.5)
        mu = s.ctrl.post(i, dv, mu)
        s.tr["mu_app"].append(mu); s.tr["om"].append(om)
        return mu, dv


class Wrap(Rule):
    """vssI.Rule / vssI2.Rule2 (they apply the update themselves); dir is recovered as (w - w_new)/(KD mu)."""

    def __init__(s, r, key, kind):
        super().__init__(); s.r, s.key, s.kind = r, key, kind

    def block(s, i, c):
        wn = s.r.block(i, c.R, c.E, c.Srr, c.Pbar, c.eps, c.w)
        mu = float(s.r.mus[s.key][-1])
        dv = (c.w - wn) / (KD * mu) if mu > 0 else np.zeros(M)
        s.tr["mu_app"].append(mu)
        return mu, dv


# ----------------------------------------------------------------------------------------------
# Simulation loop (identical arithmetic to simx.run) with Phi-excess every block
# ----------------------------------------------------------------------------------------------
def simulate(data, Rall, st, rule, nb=None, w0=None, hook=None, fps=None, error_frame=None):
    """fps: optional dict name -> (v_fp path 0, v_fp path 1): extra Phi-excess curves relative to w = wM + v_fp.
    error_frame: optional callable (block, w, actual_errors, actual_secondary_block) -> error spectrum.
    Returns e (length nb*B), w, ex (per block, after the update), ex_<name>, div, traces."""
    d, xq, hS = data["d"], data["xq"], data["hS"]
    nbmax = len(d) // B; nb = nbmax if nb is None else min(nb, nbmax)
    swv = sliding_window_view(xq, M)
    w = np.zeros(M) if w0 is None else w0.copy()
    zi = np.zeros(len(hS) - 1); e_all = np.zeros(nb * B)
    ex = np.full(nb, np.nan); exf = {k: np.full(nb, np.nan) for k in (fps or {})}
    div = False
    for i in range(nb):
        s0 = i * B
        Xb = swv[s0 + N - M + 1:s0 + N - M + 1 + B][:, ::-1]
        ys, zi = signal.lfilter(hS, [1.0], Xb @ w, zi=zi)
        e_all[s0:s0 + B] = d[s0:s0 + B] + ys
        E = ML.frame_E(e_all, s0, M) if error_frame is None else error_frame(i, w, e_all, ys)
        c = Ctx(i, Rall[i], E, st, w)
        mf = rule.mu_formula(i)
        mu, dv = rule.block(i, c)
        if hook is not None:
            hook(i, w, mu, dv, c, rule, mf, zi=zi, e_all=e_all)
        w = w - 2 * mu * B / M * dv
        p = path_of(data, i); wr, _, Jr = REFS[p]
        if not (np.isfinite(w).all() and np.isfinite(e_all[s0:s0 + B]).all()):
            div = True
        else:
            ex[i] = exP(w - wr, Jr)
            for k, vv in exf.items():
                vv[i] = exP(w - wr - fps[k][p], Jr)
            div = not np.isfinite(ex[i]) or ex[i] > DIV_EX
        if div:
            e_all[s0 + B:] = np.nan; ex[i:] = np.nan
            for vv in exf.values():
                vv[i:] = np.nan
            break
    out = dict(e=e_all, w=w, ex=ex, div=div, nb=nb, tr=rule.traces())
    out.update({"ex_" + k: v for k, v in exf.items()})
    return out


# ----------------------------------------------------------------------------------------------
# Frozen-coefficient probes
# ----------------------------------------------------------------------------------------------
def frozen_error(data, w, lo, hi):
    """e_t = d_t + (r * w)_t for t in [lo, hi) with w frozen for all time."""
    r = data["r"]; a = lo - M + 1
    seg = r[max(0, a):hi]
    if a < 0:
        seg = np.concatenate([np.zeros(-a), seg])
    return data["d"][lo:hi] + np.convolve(seg, w, "valid")


def frame_from(eseg, lo, s0, Lf=M):
    """ML.frame_E for a segment eseg that starts at sample lo."""
    a = max(0, s0 + B - Lf)
    seg = eseg[a - lo:s0 + B - lo]
    ev = np.zeros(Lf); ev[Lf - len(seg):] = seg
    return np.fft.rfft(np.concatenate([np.zeros(N - Lf), ev]))


def frozen_dirs(data, Rall, st, w, blocks, kind, om_fn=lambda j: 1.0):
    """Directions at the given blocks with the coefficients frozen at w."""
    blocks = list(blocks)
    lo = max(0, blocks[0] * B + B - M); hi = blocks[-1] * B + B
    es = frozen_error(data, w, lo, hi)
    out = []
    for j in blocks:
        E = frame_from(es, lo, j * B)
        out.append(direction(kind, Rall[j], E, st.Srr[j], st.Pbar[j], st.eps[j], om_fn(j)))
    return out


def continuation(data, Rall, st, w, zi, e_all, i, dv_i, mu, T, kind, om_fn):
    """Constant-mu continuation of the actual realisation: apply w - mu KD dv_i at block i, then T-1 further
    blocks with the same mu (directions recomputed from the simulated errors, as in simulate()).
    Returns the Phi-excess after each of the T updates (inf on divergence).  State is copied, not modified."""
    d, xq, hS = data["d"], data["xq"], data["hS"]
    swv = sliding_window_view(xq, M)
    lo = max(0, i * B + B - M)
    eh = np.zeros((i + T) * B + B - lo); eh[:i * B + B - lo] = e_all[lo:i * B + B]
    w = w - 2 * mu * B / M * dv_i; z = zi.copy(); out = np.full(T, np.inf)
    for t in range(T):
        j = i + t
        p = path_of(data, j); wr, _, Jr = REFS[p]
        if not np.isfinite(w).all():
            break
        out[t] = exP(w - wr, Jr)
        if t == T - 1 or j + 1 >= len(Rall) or not np.isfinite(out[t]) or out[t] > DIV_EX:
            break
        j += 1; s0 = j * B
        Xb = swv[s0 + N - M + 1:s0 + N - M + 1 + B][:, ::-1]
        ys, z = signal.lfilter(hS, [1.0], Xb @ w, zi=z)
        eh[s0 - lo:s0 + B - lo] = d[s0:s0 + B] + ys
        E = frame_from(eh, lo, s0)
        dv = direction(kind, Rall[j], E, st.Srr[j], st.Pbar[j], st.eps[j], om_fn(j))
        w = w - 2 * mu * B / M * dv
    return out


def full_error(data, w):
    """Frozen error over the whole signal, as probe.py: d + lfilter(w, r)."""
    return data["d"] + signal.lfilter(w, [1.0], data["r"])


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
def l2db(e, d):
    power = float(np.sum(d ** 2))
    error_power = float(np.sum(e ** 2))
    if power == 0:
        return np.nan if error_power == 0 else np.inf
    return float(10 * np.log10(error_power / power)) if error_power > 0 else -np.inf


def metrics(sim, data):
    """L1 = ANR mean (dB) whole / first 5 s / last 10 %; L2 = 10 log10(sum e^2 / sum d^2) on the same windows.
    amp flags positive L1 last10. For a single change, also return the complete 15–17 s and 25–30 s windows."""
    nan3 = [float("nan")] * 3
    e = sim["e"]; d = data["d"][:len(e)]
    if sim["div"] or not np.isfinite(e).all():
        out = dict(L1=nan3, L2=nan3, div=True, amp=False)
        if data.get("pc"):
            out["pc"] = dict(L1a=np.nan, L1b=np.nan, L1w=np.nan, L2a=np.nan, L2b=np.nan)
        return out
    anr = O.anr_curve(e, d); n10 = len(anr) // 10; n5 = 5 * FS // 64
    L = len(e); s5 = min(L, 5 * FS); s10 = L - L // 10
    last10 = float(anr[-n10:].mean()) if n10 else np.nan
    out = dict(L1=[float(anr.mean()), float(anr[:n5].mean()), last10],
               L2=[l2db(e, d), l2db(e[:s5], d[:s5]), l2db(e[s10:], d[s10:])], div=False, amp=bool(last10 > 0))
    if data.get("pc"):
        times = (63 + 64 * np.arange(len(anr))) / FS
        out["pc"] = dict(L1a=np.nan, L1b=np.nan, L1w=np.nan, L2a=np.nan, L2b=np.nan)
        for key, start, end in (("a", 15, 17), ("b", 25, 30)):
            if len(e) >= end * FS:
                win = slice(start * FS, end * FS)
                out["pc"]["L1" + key] = float(anr[(times >= start) & (times < end)].mean())
                out["pc"]["L2" + key] = l2db(e[win], d[win])
        if len(e) == len(data["d"]):
            out["pc"]["L1w"] = out["L1"][0]
    return out


def zero_stats(mu_state, thr=ZERO_THR):
    z = np.asarray(mu_state) < thr
    if len(z) == 0:
        return 0.0, 0
    longest = cur = 0
    for b in z:
        cur = cur + 1 if b else 0; longest = max(longest, cur)
    return float(z.mean()), int(longest)


def prep(tag, nb=None):
    dd = data(tag); Rall = ML.prepass(dd)["Rall"]
    if nb is not None:
        Rall = Rall[:min(len(Rall), nb + 64)]
    return dd, Rall, Stats(dd, Rall)
