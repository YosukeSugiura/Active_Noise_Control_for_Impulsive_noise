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

Subcommands:  check | c1 | c2 | report [--smoke]
  c1  closed loop, one job per tag (Pool <= 2), out/protoC/protoC_C1_<tag>.json/.npz (rewritten after every rule;
      a rerun resumes at the first missing rule)
  c2  tracking check: at sampled blocks, mu_state and target vs argmin of the T-step mean Phi-excess of a
      constant-internal-step continuation with the same power-correction policy (T = 4, 8, 16)
Options: --tags / --classes --seeds, --nb (smoke -> out/protoC_smoke/), --procs, --force, --rules "a;b",
  --f --tau-a --tau-m --heff --ck --gz --gm --dopt-T --dopt-k --mu-fmax --rec-db
"""
import os, sys, json, time, glob, argparse
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
DEF_CLASSES = {"c1": "G,BU,SM,PS,II300,a145,a120,pc", "c2": "G,a145,pc"}
DEF_SEEDS = {"c1": "0,1,2", "c2": "0"}


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
    """coord 'Y' (y-coordinate statistics) or 'E' (statistics on Delta); target in TARGETS; corr in CORRS."""
    kind = "SF"

    def __init__(s, coord, target, corr, gamma, P, mu_init=MU_INIT):
        super().__init__()
        s.coord, s.target, s.corr, s.gamma, s.P = coord, target, corr, gamma, P
        s.est = YEst(P.L, P.bd, P.b); s.mu_state = mu_init; s.Qy = None; s.last = {}
        s.tr.update({k: [] for k in ("mu_state", "target", "X", "N", "C", "Q", "ypow", "dpclip", "mufclamp",
                                     "xclip", "stclip")})

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


def gl(g):
    return "gam" + (f"{g:g}".lstrip("0"))


def rule_name(coord, target, corr, gamma):
    return f"SF-{coord} {target} {corr} {gl(gamma)}"


SFY_NAMES = [rule_name(co, t, c, g) for co in ("Y", "E") for t in TARGETS for c in CORRS for g in GAMS]
CONTROLS = ["SF-D g gam0.03 a0 1", "SF+ramp", "SF-V2", "SF mu.1", "SF mu.5"]
C1_RULES = SFY_NAMES + CONTROLS
C2_RULES = [rule_name("Y", "delayopt", "dp", 0.03), rule_name("Y", "delaycap", "dp", 0.03),
            rule_name("Y", "delayopt", "g", 0.03), rule_name("Y", "delaycap", "g", 0.03)]


def make_rules(nbr, P):
    R = {}
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


def c1_job(tag, o):
    nb = o.nb; t0 = time.time(); P = o.P
    dd, Rall, st = T.prep(tag, nb)
    nbr = (len(dd["d"]) // B) if nb is None else nb
    RR = make_rules(nbr, P)
    p0 = fpath("C1", tag, nb)
    res = dict(tag=tag, nb=nbr, pc=dd["pc"], par=P.desc(), rules={}, done=False)
    arrs = dict(g=np.asarray(st.g[:nbr], np.float32))
    if os.path.exists(p0) and not o.force:
        old = json.load(open(p0))
        if old.get("nb") == nbr and old.get("par") == json.loads(json.dumps(P.desc())):
            res["rules"] = old["rules"]
            q = fpath("C1", tag, nb, "npz")
            if os.path.exists(q):
                arrs.update(dict(np.load(q)))
    for name in (o.rules or C1_RULES):
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
            for k in ("dpclip", "mufclamp", "xclip", "stclip"):
                if k in tr and len(tr[k]):
                    r["frac_" + k] = float(np.mean(tr[k]))
        res["rules"][name] = r
        save("C1", tag, nb, res, arrs)
        print(f"[C1 {tag}] {name}: L1 {np.round(r['L1'], 2)} ex1s {r['ex_first1']:.2f} last10 {r['ex_last10']:.2f} dB "
              f"div={r['div']} amp={r['amp']} ({r['sec']:.1f} s, total {time.time() - t0:.0f} s)", flush=True)
    res["done"] = True; save("C1", tag, nb, res, arrs)
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
# Job dispatch
# ==============================================================================================
JOBS = {"c1": c1_job, "c2": c2_job}


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
        if os.path.exists(p) and not o.force and json.load(open(p)).get("done") and not (cmd == "c1" and o.rules):
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
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED", flush=True)
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


def rep_c2(J):
    L = []
    for t, r in J.items():
        for rn, x in r["rules"].items():
            if "i" not in x:
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
    ap.add_argument("cmd", choices=["check", "report", "c1", "c2"])
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
    o = ap.parse_args(argv)
    o.rules = o.rules.split(";") if o.rules else None
    o.P = Par(f=o.f, ta=o.tau_a, tm=o.tau_m, heff=0.95 * H if o.heff is None else o.heff,
              ck=[float(v) for v in o.ck.split(",")], gz=o.gz, gm=o.gm, dopt_T=o.dopt_T, dopt_k=o.dopt_k, mu_fmax=o.mu_fmax)
    return o


if __name__ == "__main__":
    o = parse(sys.argv[1:])
    if o.cmd == "check":
        sys.exit(0 if check(o) else 1)
    elif o.cmd == "report":
        report(o)
    else:
        main_jobs(o.cmd, o)
