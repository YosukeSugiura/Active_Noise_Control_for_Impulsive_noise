"""vssT: verification experiments for THEORY_vss2.md (Sec. 7 items 1-6).  Core code in vssT_lib.py.

Subcommands (see RUN_vssT.md for the exact commands):
  check                      self-checks (reproduction of simx.run, formulas, probes, inner products)
  e1 .. e6                   experiments; one job per input tag, Pool(<=2); each job writes
                             out/vssT_E<k>_<tag>.json (+ .npz) and rewrites it after every rule (incremental)
  report                     RESULTS_vssT.md from the saved outputs (tables only)

Common options:  --classes G,PS,...  --seeds 0,1,2  (or --tags G_s0,pc_s1)  --nb <blocks> (smoke: outputs go to
out/vssT_smoke/)  --procs 1|2  --force (recompute finished jobs)  --rules name1;name2 (E1, E4-E6 subsets)
  --pidef x|xM|r  --gmin 0.25
"""
import os, sys, json, time, glob, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vssT_lib import *        # noqa: F401,F403
import vssT_lib as T
import vss2 as V2
import vssI as V1
import vssI2 as V12
from multiprocessing import Pool

OUT_FULL = os.path.join(HERE, "out")
OUT_SMOKE = os.path.join(HERE, "out", "vssT_smoke")
A0S = (1.0, TAU_A)
FS_ = (0.9, 1.0)
TMS = (2.5, 3.8)


def outdir(nb):
    d = OUT_FULL if nb is None else OUT_SMOKE
    os.makedirs(d, exist_ok=True); return d


def fpath(exp, tag, nb, ext="json"):
    return os.path.join(outdir(nb), f"vssT_{exp}_{tag}.{ext}")


def save(exp, tag, nb, js, arrs=None):
    p = fpath(exp, tag, nb)
    with open(p + ".tmp", "w") as f:
        json.dump(js, f, default=lambda o: float(o) if np.isscalar(o) else np.asarray(o, float).tolist())
    os.replace(p + ".tmp", p)
    if arrs:
        q = fpath(exp, tag, nb, "npz")
        np.savez_compressed(q + ".tmp.npz", **arrs); os.replace(q + ".tmp.npz", q)


def theta_of(kind, om):
    if kind == "SF":
        return ("SF",)
    if kind == "B0":
        return ("W", 1.0, 1.0)
    if kind == "geo":
        return ("W", 0.5, 1.0)
    return ("W", 0.5, om)


def f32(x):
    return np.asarray(x, dtype=np.float32)


# ==============================================================================================
# Rule registries
# ==============================================================================================
def rules_all(nb, pidef="x", gmin=GMIN):
    """name -> factory.  nb only sets the length of the ramp schedules."""
    wr, mm = ramp(nb, 4, 3); sfr = 0.05 * ramp(nb, 2, 3)[1]
    R = {
        "SF mu.05": lambda: Fixed("SF", 0.05),
        "SF mu.1": lambda: Fixed("SF", 0.1),
        "SF+ramp": lambda: Fixed("SF", sfr),
        "SF-V2": lambda: CtrlW(V2.Ctrl2(**V2.VAR["SF-V2"][1]), "SF"),
        "SF-I": lambda: Wrap(V1.Rule(*V1.RULES["SF-I g.026 m.3"]), "SF", "SF"),
        "SF 1/g mu.05": lambda: Power("SF", 0.05, "g", pidef, gmin),
        "SF 1/g mu.1": lambda: Power("SF", 0.1, "g", pidef, gmin),
        "SF 1/(g rho) mu.05": lambda: Power("SF", 0.05, "grho", pidef, gmin),
        "SF 1/(g rho) mu.1": lambda: Power("SF", 0.1, "grho", pidef, gmin),
    }
    for corr in ("none", "g", "grho"):
        for gam in (0.03, 0.1):
            for a0 in A0S:
                R[f"SF-D {corr} gam{gam:g} a0 {a0:g}"] = (lambda corr=corr, gam=gam, a0=a0:
                                                         Direct("SF", corr, gam, a0, gmin=gmin, pidef=pidef))
    R["GS"] = lambda: Fixed("W", 0.05 * mm, om=wr)
    R["GS-V2 k2 .05/.2"] = lambda: CtrlW(V2.Ctrl2(**V2.VAR["GS-V2 k2 .05/.2"][1]), "W")
    R["GS-T2 fixed-om"] = lambda: Wrap(V12.Rule2("fixed_w", om_sched=wr), "GS", "W")
    R["GS-D g gam0.03 a0 1"] = lambda: Direct("W", "g", 0.03, 1.0, om=wr, gmin=gmin, pidef=pidef)
    for wm in ("none", "C", "Q", "both"):
        for a0 in A0S:
            R[f"GS-D w:{wm} a0 {a0:g}"] = (lambda wm=wm, a0=a0:
                                           Direct("W", "g", 0.03, a0, om=wr, gmin=gmin, pidef=pidef, ip="w:" + wm))
    R["B0 mu.1 +track"] = lambda: Track(Fixed("B0", 0.1), gmin)
    R["B0-D g +track"] = lambda: Track(Direct("B0", "g", 0.03, 1.0, gmin=gmin, pidef=pidef), gmin)
    R["Switch"] = lambda: Switch(bom=0.05, corr="g", gamma=0.03, a0=1.0, gmin=gmin, pidef=pidef)
    return R


E4_RULES = (["SF mu.05", "SF mu.1", "SF+ramp", "SF-V2", "SF-I", "SF 1/g mu.05", "SF 1/g mu.1",
             "SF 1/(g rho) mu.05", "SF 1/(g rho) mu.1"]
            + [f"SF-D {c} gam{g:g} a0 {a:g}" for c in ("none", "g", "grho") for g in (0.03, 0.1) for a in A0S]
            + ["GS", "GS-D g gam0.03 a0 1"])
E5_RULES = ["GS", "GS-V2 k2 .05/.2", "GS-T2 fixed-om"] + [f"GS-D w:{w} a0 {a:g}" for w in ("none", "C", "Q", "both") for a in A0S]
E6_RULES = ["B0 mu.1 +track", "B0-D g +track", "GS", "GS-V2 k2 .05/.2", "GS-D g gam0.03 a0 1", "Switch"]
E1_RULES = ["SF mu.1", "SF+ramp", "SF-D g gam0.03 a0 1", "GS"]


# ==============================================================================================
# E1: one-step oracle
# ==============================================================================================
E1_BLOCKS = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 2560, 3072, 3600]
E1_PC = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
MUGRID = np.linspace(0.0, 3.0, 601)
HGRID = np.concatenate([[0.0], np.geomspace(0.01, 3.0, 40)])


def onestep(v, Dl, Jr, mug=MUGRID):
    """True Phi-excess after one step w - mu Delta, on a grid of mu (brute force) and analytically."""
    PD = Phi @ Dl; a = float(v @ PD); b = float(Dl @ PD)
    V = v[None, :] - mug[:, None] * Dl[None, :]
    exg = np.einsum("ij,ij->i", V @ Phi, V) / Jr
    k = int(np.argmin(exg))
    return dict(ex0=exP(v, Jr), a=a / Jr, b=b / Jr, mu_or=a / b if b > 0 else np.nan, mu_grid=float(mug[k]),
                ex_min_grid=float(exg[k]), aE=float(v @ Dl), bE=float(Dl @ Dl), mu_or_E=float(v @ Dl) / float(Dl @ Dl))


class OracleHook:
    def __init__(s, data, Rall, st, blocks, K, gmin, T=0):
        s.data, s.Rall, s.st, s.blocks, s.K, s.gmin, s.T = data, Rall, st, set(blocks), K, gmin, T
        s.est = {"E": DirectEst("E", gmin=gmin), "Phi": DirectEst("Phi", gmin=gmin)}
        s.recs = []

    def __call__(s, i, w, mu, dv, c, rule, mf, zi=None, e_all=None):
        if i in s.blocks and np.isfinite(w).all():
            s.recs.append(s.record(i, w, mu, dv, c, rule, mf))
            if s.T > 0:
                s.recs[-1].update(s.horizon(i, w, dv, c, rule, zi, e_all))
        for e in s.est.values():
            e.update(i, dv, c.g)

    def horizon(s, i, w, dv, c, rule, zi, e_all):
        """T-step constant-mu continuation of the actual realisation (an addition: a one-step oracle is exactly
        quadratic with curvature |Delta|^2, i.e. a0 = 1 and no tau terms, so it cannot separate a0 / tau)."""
        exT = np.array([continuation(s.data, s.Rall, s.st, w, zi, e_all, i, dv, m, s.T, rule.kind, rule.om_at)
                        for m in HGRID])
        last, avg = exT[:, -1], exT.mean(1)
        gg = max(c.g, s.gmin)
        return dict(hT=s.T, hT_mu_last=float(HGRID[np.argmin(last)]), hT_mu_avg=float(HGRID[np.argmin(avg)]),
                    hT_mu_last_u=float(HGRID[np.argmin(last)]) * gg, hT_mu_avg_u=float(HGRID[np.argmin(avg)]) * gg,
                    hT_ex_last_min=float(last.min()), hT_ex_avg_min=float(avg.min()))

    def record(s, i, w, mu, dv, c, rule, mf):
        p = path_of(s.data, i); wr, _, Jr = REFS[p]
        om = rule.om_at(i)
        vfp = fixed_point(theta_of(rule.kind, om), p)
        Dl = KD * dv; gg = max(c.g, s.gmin)
        rec = dict(i=i, path=p, g=c.g, om=om, mu_app=mu, mu_form=mf, **{"pi_" + k: v for k, v in c.pi.items()})
        refs = {"wM": w - wr, "fp": w - wr - vfp}
        for rn, v in refs.items():
            o = onestep(v, Dl, Jr)
            rec.update({f"{rn}_{k}": x for k, x in o.items()})
            rec[f"{rn}_mu_or_u"] = o["mu_or"] * gg           # oracle in u-units (u = Delta / max(g, gmin))
        # frozen-state ensemble over K blocks starting at i
        K = min(s.K, len(s.Rall) - i)
        dirs = np.array(frozen_dirs(s.data, s.Rall, s.st, w, range(i, i + K), rule.kind, rule.om_at))
        Ds = KD * dirs; gs = np.maximum(s.st.g[i:i + K], s.gmin); Us = Ds / gs[:, None]
        DsP, UsP = Ds @ Phi, Us @ Phi
        rec["ens_K"] = K; rec["ens_g"] = float(np.mean(s.st.g[i:i + K]))
        for rn, v in refs.items():
            for nm, A, AP in (("D", Ds, DsP), ("u", Us, UsP)):
                vd = float(np.mean(AP @ v)); dd = float(np.mean(np.einsum("ij,ij->i", AP, A)))
                vdE = float(np.mean(A @ v)); ddE = float(np.mean(np.einsum("ij,ij->i", A, A)))
                rec[f"{rn}_ens{nm}_vD"] = vd / Jr; rec[f"{rn}_ens{nm}_DD"] = dd / Jr
                rec[f"{rn}_ens{nm}_mu_or"] = vd / dd if dd > 0 else np.nan
                rec[f"{rn}_ens{nm}_mu_or_E"] = vdE / ddE if ddE > 0 else np.nan
        for ipn, e in s.est.items():
            rec[f"C_{ipn}"] = e.C; rec[f"Q_{ipn}"] = e.Q
            for a0 in A0S:
                for f in FS_:
                    for tm in TMS:
                        ps, mh = psi_muhat(e.C, e.Q, mf, a0, f, tm, TAU_A)
                        key = f"{ipn}_a{a0:g}_f{f:g}_t{tm:g}"
                        rec["psi_" + key] = ps; rec["muhat_" + key] = mh
        return rec


def e1_job(tag, o):
    nb = o.nb; t0 = time.time()
    dd, Rall, st = T.prep(tag, nb)
    nbr = (len(dd["d"]) // B) if nb is None else nb
    blocks = [b for b in E1_BLOCKS if b < nbr - 1]
    if dd["pc"]:
        blocks = sorted(set(blocks + [PC_BLOCK + k for k in E1_PC if PC_BLOCK + k < nbr - 1]))
    RR = rules_all(nbr, o.pidef, o.gmin)
    res = dict(tag=tag, nb=nbr, K=o.K, blocks=blocks, rules={}, done=False)
    for name in (o.rules or E1_RULES):
        hk = OracleHook(dd, Rall, st, blocks, o.K, o.gmin, o.T)
        sim = simulate(dd, Rall, st, RR[name](), nb=nbr, hook=hk)
        recs = hk.recs
        res["rules"][name] = {k: [r.get(k, np.nan) for r in recs] for k in (recs[0].keys() if recs else [])}
        res["rules"][name]["div"] = sim["div"]
        save("E1", tag, nb, res)
        print(f"[E1 {tag}] {name}: {len(recs)} records, div={sim['div']} ({time.time() - t0:.0f} s)", flush=True)
    res["done"] = True; save("E1", tag, nb, res)
    return tag


# ==============================================================================================
# E2: correlation times (frozen-coefficient probe)
# ==============================================================================================
LMAX = 12


def lagcorr(A, Bm, Lmax=LMAX):
    """c[l + Lmax] = mean_n <A_n, B_{n+l}>, l = -Lmax..Lmax."""
    n = len(A); out = np.zeros(2 * Lmax + 1)
    for l in range(-Lmax, Lmax + 1):
        out[l + Lmax] = np.mean(np.einsum("ij,ij->i", A[:n - l], Bm[l:])) if l >= 0 else \
            np.mean(np.einsum("ij,ij->i", A[-l:], Bm[:n + l]))
    return out


def e2_job(tag, o):
    nb = o.nb; t0 = time.time()
    dd, Rall, st = T.prep(tag, nb)
    nbr = (len(dd["d"]) // B) if nb is None else nb
    late = simulate(dd, Rall, st, Fixed("SF", 0.1), nb=nbr)
    rng = np.random.default_rng(1)
    # late: deviation of a real SF (mu = 0.1) run at its last block from the SF model fixed point (v = w - w_fp)
    probes = {"init": -wM, "rand": rng.standard_normal(M) * np.linalg.norm(wM) / np.sqrt(M),
              "late": late["w"] - wM - fixed_point(("SF",), 0)}
    eo = full_error(dd, wM)
    yc = {p: signal.lfilter(v, [1.0], dd["r"]) for p, v in probes.items()}
    i0 = min(100, nbr // 4); blocks = range(i0, nbr)
    res = dict(tag=tag, nb=nbr, i0=i0, Lmax=LMAX, late_ex=exP(probes["late"], Jo), kinds={}, done=False)
    prof = {}
    for kind in o.kinds:
        Z = np.array([KD * direction(kind, Rall[i], ML.frame_E(eo, i * B, M), st.Srr[i], st.Pbar[i], st.eps[i]) for i in blocks])
        Z -= Z.mean(0)
        ZP = Z @ Phi
        cz_E, cz_P = lagcorr(Z, Z), lagcorr(Z, ZP)
        r = dict(N_E=cz_E[LMAX] / Jo, N_P=cz_P[LMAX] / Jo, tau_a_E=cz_E.sum() / cz_E[LMAX], tau_a_P=cz_P.sum() / cz_P[LMAX],
                 probes={})
        prof[f"{kind}|zeta|E"] = cz_E / cz_E[LMAX]; prof[f"{kind}|zeta|P"] = cz_P / cz_P[LMAX]
        for p, v in probes.items():
            Y = np.array([KD * direction(kind, Rall[i], ML.frame_E(yc[p], i * B, M), st.Srr[i], st.Pbar[i], st.eps[i]) for i in blocks])
            mean = Y.mean(0); Y -= mean
            vP = Phi @ v; xE = float(v @ v); xP = float(v @ vP)
            hE = float(mean @ v) / xE; hP_ = float(mean @ vP) / xP
            cy_E, cy_P = lagcorr(Y, Y), lagcorr(Y, Y @ Phi)
            cx_E, cx_P = lagcorr(Z, Y), lagcorr(ZP, Y)
            SE, SP = cy_E[LMAX], cy_P[LMAX]
            mb = mean - hE * v
            r["probes"][p] = dict(h_E=hE, h_P=hP_, f_E=SE / (H ** 2 * xE), f_P=SP / (H ** 2 * xP),
                                  f_eff_E=SE / (hE ** 2 * xE), f_eff_P=SP / (hP_ ** 2 * xP),
                                  tau_m_E=cy_E.sum() / SE, tau_m_P=cy_P.sum() / SP,
                                  cross_E=cx_E.sum() / np.sqrt(cz_E[LMAX] * SE), cross_P=cx_P.sum() / np.sqrt(cz_P[LMAX] * SP),
                                  cross0_E=cx_E[LMAX] / np.sqrt(cz_E[LMAX] * SE), cross0_P=cx_P[LMAX] / np.sqrt(cz_P[LMAX] * SP),
                                  mean_nonscalar=float(mb @ mb) / (hE ** 2 * xE))
            prof[f"{kind}|{p}|fluct|E"] = cy_E / SE; prof[f"{kind}|{p}|fluct|P"] = cy_P / SP
            prof[f"{kind}|{p}|cross|E"] = cx_E / np.sqrt(cz_E[LMAX] * SE); prof[f"{kind}|{p}|cross|P"] = cx_P / np.sqrt(cz_P[LMAX] * SP)
            del Y
        res["kinds"][kind] = r
        save("E2", tag, nb, res, prof)
        print(f"[E2 {tag}] {kind}: tau_a E/P {r['tau_a_E']:.2f}/{r['tau_a_P']:.2f} ({time.time() - t0:.0f} s)", flush=True)
        del Z, ZP
    res["done"] = True; save("E2", tag, nb, res, prof)
    return tag


# ==============================================================================================
# E3: gain / participation-ratio law (frozen probes)
# ==============================================================================================
G_EDGES = [0, 0.5, 0.8, 1.25, 2, 4, 8, 16, 64, np.inf]
PI_EDGES = [0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0001]


def bin2d(x, y, val, xe, ye):
    ix = np.digitize(x, xe) - 1; iy = np.digitize(y, ye) - 1
    mean = np.full((len(xe) - 1, len(ye) - 1), np.nan); cnt = np.zeros_like(mean, dtype=int)
    for a in range(len(xe) - 1):
        for b in range(len(ye) - 1):
            m = (ix == a) & (iy == b)
            cnt[a, b] = int(m.sum())
            if cnt[a, b]:
                mean[a, b] = float(np.mean(val[m]))
    return mean, cnt


def bin1d(x, val, xe):
    ix = np.digitize(x, xe) - 1
    return [float(np.mean(val[ix == a])) if np.any(ix == a) else np.nan for a in range(len(xe) - 1)], \
           [int(np.sum(ix == a)) for a in range(len(xe) - 1)]


def r2o(y, Xm):
    co = np.linalg.lstsq(Xm, y, rcond=None)[0]
    return float(1 - np.sum((y - Xm @ co) ** 2) / np.sum((y - y.mean()) ** 2)), [float(c) for c in co]


def e3_job(tag, o):
    nb = o.nb; t0 = time.time()
    dd, Rall, st = T.prep(tag, nb)
    nbr = (len(dd["d"]) // B) if nb is None else nb
    i0 = min(100, nbr // 4); blocks = np.arange(i0, nbr)
    g = st.g[blocks]; pis = {k: st.pi[k][blocks] for k in PI_DEFS}
    res = dict(tag=tag, nb=nbr, i0=i0, g_edges=G_EDGES, pi_edges=PI_EDGES, kinds={}, done=False,
               Eg=float(g.mean()), Eg2=float((g ** 2).mean()))
    arrs = dict(g=f32(g), **{"pi_" + k: f32(v) for k, v in pis.items()})
    for kind in o.kinds:
        vfp = fixed_point(theta_of(kind, 1.0), 0); wfp = wM + vfp
        efp = full_error(dd, wfp); ec = signal.lfilter(-wfp, [1.0], dd["r"]); v = -wfp; vP = Phi @ v
        nP = np.empty(len(blocks)); nE = np.empty_like(nP); sP = np.empty_like(nP); sE = np.empty_like(nP)
        fP = np.empty_like(nP); fE = np.empty_like(nP)
        for j, i in enumerate(blocks):
            a = (Rall[i], st.Srr[i], st.Pbar[i], st.eps[i])
            n = KD * direction(kind, a[0], ML.frame_E(efp, i * B, M), *a[1:])
            cdir = KD * direction(kind, a[0], ML.frame_E(ec, i * B, M), *a[1:])
            nPv = Phi @ n; nP[j] = float(n @ nPv) / Jo; nE[j] = float(n @ n) / Jo
            sP[j] = float(cdir @ vP) / float(v @ vP); sE[j] = float(cdir @ v) / float(v @ v)
            rP = cdir - sP[j] * v; rE = cdir - sE[j] * v
            fP[j] = float(rP @ (Phi @ rP)) / float(v @ vP); fE[j] = float(rE @ rE) / float(v @ v)
        arrs.update({f"{kind}_{k}": f32(x) for k, x in dict(noiseP=nP, noiseE=nE, sigP=sP, sigE=sE, flP=fP, flE=fE).items()})
        r = dict(noiseP_mean=float(nP.mean()), sigP_over_g=float(np.mean(sP / g)), tables={}, reg={})
        for nm, val in (("noiseP/g2", nP / g ** 2), ("sigP/g", sP / g), ("flP/g2", fP / g ** 2), ("noiseE/g2", nE / g ** 2)):
            m1, c1 = bin1d(g, val, G_EDGES)
            r["tables"][nm] = dict(by_g=m1, n_g=c1)
            for k, pv in pis.items():
                m2, c2 = bin2d(g, pv, val, G_EDGES, PI_EDGES)
                mp, cp = bin1d(pv, val, PI_EDGES)
                r["tables"][nm]["g_pi_" + k] = m2.tolist(); r["tables"][nm]["n_g_pi_" + k] = c2.tolist()
                r["tables"][nm]["by_pi_" + k] = mp; r["tables"][nm]["n_pi_" + k] = cp
        for tgt, y in (("noiseP", nP), ("flP", fP)):
            r["reg"][tgt] = {"g": r2o(y, g[:, None]), "g^2": r2o(y, (g ** 2)[:, None])}
            for k, pv in pis.items():
                r["reg"][tgt]["g^2(1-pi_" + k + ")"] = r2o(y, (g ** 2 * (1 - pv))[:, None])
        r["reg"]["sigP"] = {"g": r2o(sP, g[:, None])}
        res["kinds"][kind] = r
        save("E3", tag, nb, res, arrs)
        print(f"[E3 {tag}] {kind} ({time.time() - t0:.0f} s)", flush=True)
    res["done"] = True; save("E3", tag, nb, res, arrs)
    return tag


# ==============================================================================================
# E4 / E5 / E6: closed-loop runs
# ==============================================================================================
def exwin(ex, nbr, pc):
    n5 = min(nbr, int(5 * FS / B)); n10 = max(1, nbr // 10)
    m = lambda a: float(10 * np.log10(np.mean(a))) if len(a) and np.all(np.isfinite(a)) else float("nan")
    out = dict(ex_first5=m(ex[:n5]), ex_last10=m(ex[-n10:]), ex_half2=m(ex[nbr // 2:]))
    if pc:
        out["ex_pc_15_17"] = m(ex[PC_BLOCK:min(nbr, PC_BLOCK + int(2 * FS / B))])
        out["ex_pc_25_30"] = m(ex[int(25 * FS / B):])
    return out


def steady_win(nbr, pc):
    return np.arange(int(25 * FS / B), nbr) if pc else np.arange(nbr // 2, nbr)


def run_rules(exp, tag, o, names, fps_thetas=None):
    nb = o.nb; t0 = time.time()
    dd, Rall, st = T.prep(tag, nb)
    nbr = (len(dd["d"]) // B) if nb is None else nb
    RR = rules_all(nbr, o.pidef, o.gmin)
    fps = None
    if fps_thetas:
        fps = {k: (fixed_point(th, 0), fixed_point(th, 1)) for k, th in fps_thetas.items()}
    p0 = fpath(exp, tag, nb)
    res = dict(tag=tag, nb=nbr, pc=dd["pc"], pidef=o.pidef, gmin=o.gmin, rules={}, done=False)
    arrs = dict(g=f32(st.g[:nbr]), **{"pi_" + k: f32(v[:nbr]) for k, v in st.pi.items()})
    if os.path.exists(p0) and not o.force:            # resume a partially finished job
        old = json.load(open(p0))
        if old.get("nb") == nbr and old.get("pidef") == o.pidef and old.get("gmin") == o.gmin:
            res["rules"] = old["rules"]
            q = fpath(exp, tag, nb, "npz")
            if os.path.exists(q):
                arrs.update(dict(np.load(q)))
    for name in names:
        if name in res["rules"]:
            continue
        t1 = time.time()
        sim = simulate(dd, Rall, st, RR[name](), nb=nbr, fps=fps)
        mt = metrics(sim, dd)
        r = dict(**mt, **exwin(sim["ex"], nbr, dd["pc"]), sec=time.time() - t1)
        tr = sim["tr"]
        arrs[name + "|ex"] = f32(sim["ex"])
        for k in (fps or {}):
            arrs[name + "|ex_" + k] = f32(sim["ex_" + k])
        for k, v in tr.items():
            arrs[name + "|" + k] = f32(v)
        if "mu_state" in tr:
            r["zero_frac"], r["zero_run"] = zero_stats(tr["mu_state"])
            r["mu_state_last10"] = float(np.mean(tr["mu_state"][-max(1, nbr // 10):]))
        r["mu_app_last10"] = float(np.mean(tr["mu_app"][-max(1, nbr // 10):])) if len(tr["mu_app"]) else np.nan
        res["rules"][name] = r
        save(exp, tag, nb, res, arrs)
        print(f"[{exp} {tag}] {name}: L1 {np.round(r['L1'], 2)} ex_last10 {r['ex_last10']:.2f} dB div={r['div']} "
              f"({r['sec']:.1f} s, total {time.time() - t0:.0f} s)", flush=True)
    if exp == "E4":
        res["loss"] = loss_pred(res, arrs, st, nbr, dd["pc"])
    res["done"] = True; save(exp, tag, nb, res, arrs)
    return tag


def loss_pred(res, arrs, st, nbr, pc):
    """Sec. 3: predicted x_ss(fixed)/x_ss(1/g-type rule) = E[g^2 rho]/E[g] vs measured Phi-excess ratios."""
    win = steady_win(nbr, pc)
    if len(win) == 0:
        return {}
    g = st.g[win]; out = dict(window=[int(win[0]), int(win[-1])], Eg=float(g.mean()), pred_g2=float((g ** 2).mean() / g.mean()))
    psi = arrs.get("SF 1/(g rho) mu.1|psi")
    for k, pv in st.pi.items():
        if psi is not None:
            rho = 1 - pv[win] * (1 - np.asarray(psi, float)[win])
            out["pred_g2rho_" + k] = float((g ** 2 * rho).mean() / g.mean())
    for mu in ("05", "1"):
        fx = arrs.get(f"SF mu.{mu}|ex")
        for nm, key in (("g", f"SF 1/g mu.{mu}|ex"), ("grho", f"SF 1/(g rho) mu.{mu}|ex")):
            pr = arrs.get(key)
            if fx is not None and pr is not None:
                a, b = np.asarray(fx, float)[win].mean(), np.asarray(pr, float)[win].mean()
                out[f"meas_mu.{mu}_fixed/{nm}"] = float(a / b) if np.isfinite(a / b) else np.nan
    return out


def e4_job(tag, o):
    return run_rules("E4", tag, o, o.rules or E4_RULES)


def e5_job(tag, o):
    return run_rules("E5", tag, o, o.rules or E5_RULES)


def e6_job(tag, o):
    return run_rules("E6", tag, o, o.rules or E6_RULES, {"B0": ("W", 1.0, 1.0), "geo": ("W", 0.5, 1.0)})


JOBS = {"e1": e1_job, "e2": e2_job, "e3": e3_job, "e4": e4_job, "e5": e5_job, "e6": e6_job}
DEF_CLASSES = {"e1": "G,a145,a120,pc", "e2": "G,a145", "e3": "G,PS,BU,SM,II,a145,a120",
               "e4": "G,PS,BU,SM,II,a145,a120,pc", "e5": "G,a145,a120,pc", "e6": "G,a145,a120,pc"}
DEF_SEEDS = {"e1": "0", "e2": "0,1", "e3": "0", "e4": "0,1,2", "e5": "0,1,2", "e6": "0,1,2"}


def _dispatch(a):
    cmd, tag, o = a
    try:
        return JOBS[cmd](tag, o)
    except Exception as ex:          # keep the other jobs running; the error is printed and the job stays unfinished
        import traceback; traceback.print_exc()
        return f"{tag} FAILED: {ex}"


def main_jobs(cmd, o):
    tags = o.tags.split(",") if o.tags else [f"{c}_s{s}" for c in (o.classes or DEF_CLASSES[cmd]).split(",")
                                                for s in (o.seeds or DEF_SEEDS[cmd]).split(",")]
    exp = cmd.upper()
    todo = []
    for t in tags:
        p = fpath(exp, t, o.nb)
        if os.path.exists(p) and not o.force and json.load(open(p)).get("done"):
            print("skip (done):", p); continue
        todo.append((cmd, t, o))
    print(f"{exp}: {len(todo)} jobs, procs={o.procs}, nb={o.nb}", flush=True)
    t0 = time.time()
    if o.procs <= 1:
        for a in todo:
            print("finished", _dispatch(a), flush=True)
    else:
        with Pool(min(o.procs, 2), maxtasksperchild=1) as pl:
            for r in pl.imap_unordered(_dispatch, todo):
                print("finished", r, flush=True)
    print(f"{exp}: all done in {time.time() - t0:.0f} s", flush=True)


# ==============================================================================================
# Self-checks
# ==============================================================================================
def check(o):
    ok = True

    def rep(name, cond, msg=""):
        nonlocal ok
        ok &= bool(cond); print(f"[{'OK ' if cond else 'FAIL'}] {name} {msg}", flush=True)

    nbc = 400
    dd, Rall, st = T.prep("G_s0", nbc)
    Rfull = ML.prepass(dd)["Rall"]
    # 1. reproduction of simx.run (fixed mu, ramp GS, Ctrl2)
    wr, mm = ramp(nbc, 4, 3)
    cases = [("SF mu.1", Fixed("SF", 0.1), dict(kind="SF", mu=0.1)),
             ("GS ramp", Fixed("W", 0.05 * mm, om=wr), dict(kind="W", mu=0.05 * mm, a=0.5, om=wr)),
             ("B0 mu.1", Fixed("B0", 0.1), dict(kind="W", mu=0.1, om=0.0)),
             ("SF-V2", CtrlW(V2.Ctrl2(), "SF"), dict(kind="SF", mu=0.0, ctrl=V2.Ctrl2())),
             ("GS-V2", CtrlW(V2.Ctrl2(**V2.VAR["GS-V2 k2 .05/.2"][1]), "W"),
              dict(kind="W", mu=0.0, a=0.5, ctrl=V2.Ctrl2(**V2.VAR["GS-V2 k2 .05/.2"][1])))]
    for nm, rule, kw in cases:
        a = simulate(dd, Rall, st, rule, nb=nbc)
        b = SX.run(dd, Rfull, nb=nbc, **kw)
        de = np.max(np.abs(a["e"] - b["e"][:nbc * B])) / np.max(np.abs(b["e"][:nbc * B]))
        dw = np.max(np.abs(a["w"] - b["w"])) / np.max(np.abs(b["w"]))
        dx = np.max(np.abs(a["ex"][7::8] - b["ex"]) / b["ex"])
        rep(f"reproduce simx.run [{nm}]", max(de, dw, dx) < 1e-9, f"rel diff e {de:.1e} w {dw:.1e} ex {dx:.1e}")
    # 2. direct rule with gamma=0, no correction = fixed mu (bitwise)
    a = simulate(dd, Rall, st, Direct("SF", "none", 0.0, 1.0, mu_init=0.1), nb=nbc)
    b = simulate(dd, Rall, st, Fixed("SF", 0.1), nb=nbc)
    rep("Direct(gamma=0, none) == Fixed", np.array_equal(a["w"], b["w"]))
    a = simulate(dd, Rall, st, Switch(bom=0.0, corr="g", gamma=0.03, a0=1.0), nb=nbc)
    b = simulate(dd, Rall, st, Direct("B0", "g", 0.03, 1.0), nb=nbc)
    rep("Switch(omega frozen at 0) == Direct on B0", np.array_equal(a["w"], b["w"]))
    # 3. Sec. 2 numerical check (C/Q = -0.048 at mu = 0.1 -> psi 0.042, mu_hat 0.046)
    ps, mh = psi_muhat(-0.048, 1.0, 0.1, 1.0, 0.9, 2.5, 3.8)
    rep("Sec.2 check psi/mu_hat", abs(ps - 0.042) < 1e-3 and abs(mh - 0.046) < 1e-3, f"psi {ps:.4f} mu_hat {mh:.4f}")
    # 4. weighted inner products: w=None and w=1 reproduce the Euclidean product; same as vssI2.ipw
    rng = np.random.default_rng(0); u, v = rng.standard_normal(M), rng.standard_normal(M)
    U, Vv = np.fft.rfft(u, N), np.fft.rfft(v, N)
    e1, e2, e3 = ipw(U, Vv, None), ipw(U, Vv, np.ones(NBIN)), float(u @ v)
    rep("ipw(w=None)=ipw(w=1)=Euclid", abs(e1 - e3) < 1e-9 * abs(e3) + 1e-12 and abs(e2 - e3) < 1e-9 * abs(e3) + 1e-12
        and ipw(U, Vv, None) == V12.ipw(U, Vv, None), f"{e1:.6f} {e2:.6f} {e3:.6f}")
    estE, estS = DirectEst("E"), DirectEst("spec")
    for i in range(30):
        dv = rng.standard_normal(M); gi = 0.5 + rng.random()
        estE.update(i, dv, gi); estS.update(i, dv, gi, None, None)
    rep("DirectEst spec(unweighted) == Euclid", abs(estE.C - estS.C) < 1e-9 * abs(estE.Q) and abs(estE.Q - estS.Q) < 1e-9 * estE.Q)
    # 5. E1 one-step oracle on a noiseless synthetic quadratic: grid argmin = analytic optimum
    v = rng.standard_normal(M); Dl = 0.37 * v + 0.05 * rng.standard_normal(M)
    r = onestep(v, Dl, 1.0)
    PD = Phi @ Dl; mu_an = float(v @ PD) / float(Dl @ PD); ex_an = exP(v - mu_an * Dl, 1.0)
    step = MUGRID[1] - MUGRID[0]
    rep("E1 oracle grid == analytic", abs(r["mu_grid"] - mu_an) <= step / 2 + 1e-12 and abs(r["mu_or"] - mu_an) < 1e-12
        and r["ex_min_grid"] - ex_an < 1e-3 * ex_an + 1e-12, f"mu grid {r['mu_grid']:.4f} analytic {mu_an:.4f}")
    # 6. frozen probe == simulator with mu=0 and w0 = w
    w = 0.3 * rng.standard_normal(M) * np.linalg.norm(wM) / np.sqrt(M)
    rec = []
    simulate(dd, Rall, st, Fixed("SF", 0.0), nb=60, w0=w, hook=lambda i, ww, mu, dv, c, rule, mf, **k: rec.append(dv))
    fd = frozen_dirs(dd, Rall, st, w, range(40, 56), "SF")
    dmax = max(np.max(np.abs(fd[k] - rec[40 + k])) / np.max(np.abs(rec[40 + k])) for k in range(16))
    rep("frozen_dirs == mu=0 simulation", dmax < 1e-9, f"rel diff {dmax:.1e}")
    # 6b. constant-mu continuation from block 50 == the same fixed-mu simulation
    snap = {}

    def grab(i, ww, mu, dv, c, rule, mf, zi=None, e_all=None):
        if i == 50:
            snap.update(w=ww.copy(), zi=zi.copy(), e=e_all.copy(), dv=dv.copy())
    full = simulate(dd, Rall, st, Fixed("SF", 0.1), nb=70, hook=grab)
    cont = continuation(dd, Rall, st, snap["w"], snap["zi"], snap["e"], 50, snap["dv"], 0.1, 12, "SF", lambda j: 1.0)
    dc = np.max(np.abs(cont - full["ex"][50:62]) / full["ex"][50:62])
    rep("continuation == simulation (fixed mu)", dc < 1e-9, f"rel diff {dc:.1e}")
    # 7. E2 correlation sum on a vector AR(1) sequence: two-sided sum -> (1+a)/(1-a) = 3 for a = 0.5
    a_ = 0.5; Z = np.zeros((20000, 32)); Z[0] = rng.standard_normal(32)
    for n in range(1, len(Z)):
        Z[n] = a_ * Z[n - 1] + np.sqrt(1 - a_ ** 2) * rng.standard_normal(32)
    c = lagcorr(Z, Z); tau = c.sum() / c[LMAX]
    rep("E2 lagcorr AR(1) tau", abs(tau - 3.0) < 0.1, f"tau {tau:.3f} (expect 3.0)")
    # 8. participation ratio
    seg = np.ones(N); p_flat = np.sum(seg ** 4) / np.sum(seg ** 2) ** 2
    seg2 = np.zeros(N); seg2[5] = 3.0; p_imp = np.sum(seg2 ** 4) / np.sum(seg2 ** 2) ** 2
    pv = pi_seq(dict(x=np.ones(4 * N), r=np.ones(4 * N)), 20, "x")
    rep("pi flat = 1/N, impulse = 1", abs(p_flat - 1 / N) < 1e-15 and abs(p_imp - 1) < 1e-15 and abs(pv[-1] - 1 / N) < 1e-15)
    # 9. references and model fixed points
    w0_, g0_ = wiener_for(BS.hP)
    rep("wiener_for(hP) == simx.wM", np.max(np.abs(w0_ - wM)) < 1e-12 and abs(g0_ @ g0_ - Jo) < 1e-12 * Jo)
    import model as MD
    v1 = fixed_point(("SF",), 0); v_pc = fixed_point(("SF",), 1); v2 = MD.sf_v()
    rep("fixed_point(SF) == model.sf_v, cache restored", np.max(np.abs(v1 - v2)) < 1e-12 and np.max(np.abs(MD.sf_v() - v2)) < 1e-12,
        f"SF bias {10 * np.log10(exP(v1, Jo)):.2f} dB (path 0), {10 * np.log10(exP(v_pc, Jo_pc)):.2f} dB (path change)")
    vb = fixed_point(("W", 1.0, 1.0), 0); vg = fixed_point(("W", 0.5, 1.0), 0)
    rep("B0 / geo bias (prop3: -0.8 / -9.5 dB)", abs(10 * np.log10(exP(vb, Jo)) + 0.81) < 0.2 and abs(10 * np.log10(exP(vg, Jo)) + 9.55) < 0.2,
        f"B0 {10 * np.log10(exP(vb, Jo)):.2f} dB, geo {10 * np.log10(exP(vg, Jo)):.2f} dB")
    # 10. metrics: full-length scores equal simx.scores
    a = simulate(dd, Rfull, Stats(dd, Rfull), Fixed("SF", 0.1))
    b = SX.run(dd, Rfull, "SF", 0.1)
    m = metrics(a, dd); s_ = SX.scores(b, dd)
    rep("metrics L1 == simx.scores (full length)", np.max(np.abs(np.array(m["L1"]) - s_)) < 1e-9, f"{np.round(m['L1'], 3)} vs {np.round(s_, 3)}")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED", flush=True)
    return ok


# ==============================================================================================
# Report
# ==============================================================================================
def load(exp, smoke):
    d = OUT_SMOKE if smoke else OUT_FULL
    J = {}
    for f in sorted(glob.glob(os.path.join(d, f"vssT_{exp}_*.json"))):
        r = json.load(open(f)); J[r["tag"]] = r
    return J


def cls_of(tag):
    return tag.rsplit("_s", 1)[0]


def fm(x, p=2):
    try:
        return "NaN" if x is None or not np.isfinite(x) else f"{x:.{p}f}"
    except TypeError:
        return str(x)


def rep_runs(J, exp, title):
    L = [f"## {title}", ""]
    if not J:
        return L + ["(no outputs)", ""]
    byc = {}
    for t, r in J.items():
        byc.setdefault(cls_of(t), []).append(r)
    for c, rs in byc.items():
        names = list(dict.fromkeys(k for r in rs for k in r["rules"]))
        pc = rs[0].get("pc")
        L += [f"### {c}: {CLASS_DESC.get(c, c)} (tags: {', '.join(sorted(r['tag'] for r in rs))})", ""]
        if pc:
            L += ["| rule | L1 15-17 s | L1 25-30 s | L1 whole | L2 15-17 s | L2 25-30 s | ex 15-17 s | ex 25-30 s | zero frac | n | div |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        else:
            L += ["| rule | L1 whole | L1 first5 | L1 last10 | L2 whole | L2 first5 | L2 last10 | ex first5 | ex last10 | mu_app last10 | zero frac | longest zero | n | div |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in names:
            rows = [r["rules"][k] for r in rs if k in r["rules"]]
            ok = [x for x in rows if not x["div"]]; nd = len(rows) - len(ok)
            mean = lambda f: float(np.mean([f(x) for x in ok])) if ok else np.nan
            zf = mean(lambda x: x.get("zero_frac", np.nan)); zr = mean(lambda x: x.get("zero_run", np.nan))
            if pc:
                cells = [mean(lambda x: x["pc"]["L1a"]), mean(lambda x: x["pc"]["L1b"]), mean(lambda x: x["pc"]["L1w"]),
                         mean(lambda x: x["pc"]["L2a"]), mean(lambda x: x["pc"]["L2b"]),
                         mean(lambda x: x.get("ex_pc_15_17", np.nan)), mean(lambda x: x.get("ex_pc_25_30", np.nan)), zf]
            else:
                cells = [mean(lambda x: x["L1"][0]), mean(lambda x: x["L1"][1]), mean(lambda x: x["L1"][2]),
                         mean(lambda x: x["L2"][0]), mean(lambda x: x["L2"][1]), mean(lambda x: x["L2"][2]),
                         mean(lambda x: x["ex_first5"]), mean(lambda x: x["ex_last10"]),
                         mean(lambda x: x.get("mu_app_last10", np.nan)), zf, zr]
            L.append(f"| {k} | " + " | ".join(fm(x, 3 if i >= len(cells) - 3 and not pc else 2) for i, x in enumerate(cells))
                     + f" | {len(ok)} | {nd} |")
        L.append("")
    return L


def report(o):
    smoke = o.smoke
    L = ["# RESULTS vssT: verification of THEORY_vss2.md", "",
         f"Generated by `python vssT.py report{' --smoke' if smoke else ''}` from out/{'vssT_smoke/' if smoke else ''}vssT_*.json.",
         "All dB values: more negative is better (L1 = ANR mean in dB, L2 = 10log10(sum e^2/sum d^2), "
         "ex = 10log10 of the mean Phi-excess v'Phi v/Jo over the window).", ""]
    # ---------------- E1
    J = load("E1", smoke)
    L += ["## E1: one-step oracle (Sec. 7 item 1)", "",
          "mu_or: argmin of the true Phi-excess after one step from the current state with the block's own direction "
          "(Delta units, w <- w - mu Delta). ens(u): frozen-state ensemble over K blocks, E<v,u>_Phi / E|u|^2_Phi with "
          "u = Delta/max(g,g_min), the units of mu_hat. mu_hat columns: Euclidean C,Q, f=0.9, tau_m=2.5, tau_a=3.8, "
          "a0 = 1 or tau_a; mu in the formula = mu_form (the rule's mu state / fixed mu). ref wM: v = w - wM; "
          "ref fp: v relative to the method's model fixed point.", ""]
    for t, r in J.items():
        for rn, x in r["rules"].items():
            if "i" not in x:
                continue
            L += [f"### {t} / {rn} (K = {r['K']}, div = {x.get('div')})", "",
                  "| block | g | mu_app | mu_form | mu_or wM (u) | mu_or fp (u) | ens(u) wM | ens(u) fp | ens(D) fp | T-step avg (u) | T-step last (u) | mu_hat a0=1 | mu_hat a0=3.8 | mu_hat Phi a0=1 | psi E | psi Phi | ex0 wM dB | ex0 fp dB |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for j in range(len(x["i"])):
                g_ = lambda k: x[k][j] if k in x else np.nan
                L.append("| " + " | ".join([str(x["i"][j]), fm(g_("g")), fm(g_("mu_app"), 3), fm(g_("mu_form"), 3),
                                             fm(g_("wM_mu_or_u"), 3), fm(g_("fp_mu_or_u"), 3), fm(g_("wM_ensu_mu_or"), 3),
                                             fm(g_("fp_ensu_mu_or"), 3), fm(g_("fp_ensD_mu_or"), 3),
                                             fm(g_("hT_mu_avg_u"), 3), fm(g_("hT_mu_last_u"), 3),
                                             fm(g_("muhat_E_a1_f0.9_t2.5"), 3), fm(g_("muhat_E_a3.8_f0.9_t2.5"), 3),
                                             fm(g_("muhat_Phi_a1_f0.9_t2.5"), 3), fm(g_("psi_E_a1_f0.9_t2.5"), 3),
                                             fm(g_("psi_Phi_a1_f0.9_t2.5"), 3),
                                             fm(10 * np.log10(g_("wM_ex0"))), fm(10 * np.log10(g_("fp_ex0")))]) + " |")
            L.append("")
        # summary: log-ratio of mu_hat to the ensemble oracle (fp reference), early (< 256) / late (>= 1024) / after pc
        L += [f"#### {t}: median of log2(mu_hat / oracle) by phase (0 = exact; Euclidean C,Q, f=0.9). "
              "Oracle: frozen ensemble ens(u) (fp ref) / T-step continuation avg (u)", "",
              "| rule | variant | oracle | early (<256) | mid | late (>=1024, before pc) | after pc |", "|---|---|---|---|---|---|---|"]
        for (rn, x), on in [(it, on) for it in r["rules"].items() for on in ("fp_ensu_mu_or", "hT_mu_avg_u")]:
            if "i" not in x or on not in x:
                continue
            ii = np.array(x["i"]); orc = np.array(x[on], float)
            ph = {"early": ii < 256, "mid": (ii >= 256) & (ii < 1024), "late": (ii >= 1024) & ((ii < PC_BLOCK) | (np.array(x["path"]) == 0)),
                  "pc": np.array(x["path"]) == 1}
            for a0 in A0S:
                for tm in TMS:
                    mh = np.array(x[f"muhat_E_a{a0:g}_f0.9_t{tm:g}"], float)
                    cells = []
                    for k in ("early", "mid", "late", "pc"):
                        m = ph[k] & np.isfinite(mh) & np.isfinite(orc) & (mh > 0) & (orc > 0)
                        cells.append(fm(float(np.median(np.log2(mh[m] / orc[m]))) if m.any() else np.nan))
                    L.append(f"| {rn} | a0={a0:g} tau_m={tm:g} | {'ens' if on.startswith('fp') else 'T-step'} | " + " | ".join(cells) + " |")
        L.append("")
    # ---------------- E2
    J = load("E2", smoke)
    L += ["## E2: correlation times (Sec. 7 item 2)", "",
          "Two-sided correlation sums |lag| <= 12 of zeta (direction from the optimum residual, mean removed) and of the "
          "fluctuation H~v (direction from the coherent error r*v minus its time mean); cross = sum_l E<zeta_n, H~v_{n+l}> / "
          "sqrt(N S). f = S / (h^2 |v|^2) with h = 0.25; h_eff = <mean, v>/|v|^2. E = Euclidean, P = Phi.", "",
          "| input | kind | tau_a E | tau_a P | N_P/Jo | probe | h_eff E | h_eff P | f E | f P | tau_m E | tau_m P | cross E | cross P | cross lag0 P | nonscalar mean |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, r in J.items():
        for kind, k in r["kinds"].items():
            for p, q in k["probes"].items():
                L.append("| " + " | ".join([t, kind, fm(k["tau_a_E"]), fm(k["tau_a_P"]), fm(k["N_P"], 4), p, fm(q["h_E"], 3), fm(q["h_P"], 3),
                                             fm(q["f_E"]), fm(q["f_P"]), fm(q["tau_m_E"]), fm(q["tau_m_P"]), fm(q["cross_E"], 3),
                                             fm(q["cross_P"], 3), fm(q["cross0_P"], 3), fm(q["mean_nonscalar"], 3)]) + " |")
    L.append("")
    # ---------------- E3
    J = load("E3", smoke)
    L += ["## E3: gain / participation-ratio law (Sec. 7 item 3, Sec. 3)", "",
          "Frozen probes. noiseP: |Delta|^2_Phi/Jo with w at the model fixed point (pure noise). sigP: <v, Delta_c>_Phi/|v|^2_Phi for "
          "v = -w_fp (w = 0), Delta_c from the coherent error. Theory: noise/g^2 ~ (1 - pi), sig/g ~ h = 0.25.", "",
          "### Regression through the origin, R^2 (noiseP ~ predictor)", "",
          "| input | kind | g | g^2 | g^2(1-pi_x) | g^2(1-pi_xM) | g^2(1-pi_r) | fluct: g^2 | fluct: g^2(1-pi_x) | sig ~ g: coef |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for t, r in J.items():
        for kind, k in r["kinds"].items():
            rg = k["reg"]["noiseP"]; rf = k["reg"]["flP"]
            L.append("| " + " | ".join([t, kind, fm(rg["g"][0], 3), fm(rg["g^2"][0], 3)] +
                                        [fm(rg.get(f"g^2(1-pi_{p})", [np.nan])[0], 3) for p in PI_DEFS] +
                                        [fm(rf["g^2"][0], 3), fm(rf["g^2(1-pi_x)"][0], 3), fm(k["reg"]["sigP"]["g"][1][0], 3)]) + " |")
    L.append("")
    for t, r in J.items():
        for kind, k in r["kinds"].items():
            ge = r["g_edges"]; pe = r["pi_edges"]
            gl = [f"{ge[a]:g}-{ge[a + 1]:g}" for a in range(len(ge) - 1)]
            pl = [f"{pe[b]:g}-{min(pe[b + 1], 1):g}" for b in range(len(pe) - 1)]
            L += [f"### {t} / {kind}: by g (all pi)", "", "| g bin | n | noiseP/g^2 | flP/g^2 | sigP/g |", "|---|---|---|---|---|"]
            tb = k["tables"]
            for a in range(len(gl)):
                L.append(f"| {gl[a]} | {tb['noiseP/g2']['n_g'][a]} | {fm(tb['noiseP/g2']['by_g'][a], 4)} | {fm(tb['flP/g2']['by_g'][a], 4)} | {fm(tb['sigP/g']['by_g'][a], 3)} |")
            L.append("")
            for p in PI_DEFS:
                L += [f"#### {t} / {kind}: noiseP/g^2 by g (rows) and pi_{p} (columns); cell = mean (n)", "",
                      "| g \\ pi | " + " | ".join(pl) + " |", "|---" * (len(pl) + 1) + "|"]
                m2 = tb["noiseP/g2"]["g_pi_" + p]; c2 = tb["noiseP/g2"]["n_g_pi_" + p]
                for a in range(len(gl)):
                    L.append(f"| {gl[a]} | " + " | ".join(f"{fm(m2[a][b], 4)} ({c2[a][b]})" if c2[a][b] else "" for b in range(len(pl))) + " |")
                L.append("")
            L += [f"#### {t} / {kind}: sigP/g by g and pi_x", "", "| g \\ pi | " + " | ".join(pl) + " |", "|---" * (len(pl) + 1) + "|"]
            m2 = tb["sigP/g"]["g_pi_x"]; c2 = tb["sigP/g"]["n_g_pi_x"]
            for a in range(len(gl)):
                L.append(f"| {gl[a]} | " + " | ".join(f"{fm(m2[a][b], 3)} ({c2[a][b]})" if c2[a][b] else "" for b in range(len(pl))) + " |")
            L.append("")
    # ---------------- E4-E6
    J4 = load("E4", smoke)
    L += rep_runs(J4, "E4", "E4: step rules (Sec. 7 items 3-4, Sec. 8a-b)")
    L += ["### E4: fixed-step loss, predicted E[g^2]/E[g], E[g^2 rho]/E[g] vs measured steady Phi-excess ratio fixed / rule", "",
          "Window: second half (non-pc), 25-30 s (pc). rho = 1 - pi (1 - psi) with psi from the 'SF 1/(g rho) mu.1' run.", "",
          "| input | E[g] | pred E[g^2]/E[g] | pred g^2 rho (pi_x) | pred (pi_xM) | pred (pi_r) | meas mu.05 fixed/(1/g) | meas mu.1 fixed/(1/g) | meas mu.05 fixed/(1/(g rho)) | meas mu.1 fixed/(1/(g rho)) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for t, r in J4.items():
        lo = r.get("loss", {})
        L.append("| " + " | ".join([t, fm(lo.get("Eg")), fm(lo.get("pred_g2")), fm(lo.get("pred_g2rho_x")), fm(lo.get("pred_g2rho_xM")),
                                     fm(lo.get("pred_g2rho_r")), fm(lo.get("meas_mu.05_fixed/g")), fm(lo.get("meas_mu.1_fixed/g")),
                                     fm(lo.get("meas_mu.05_fixed/grho")), fm(lo.get("meas_mu.1_fixed/grho"))]) + " |")
    L.append("")
    L += rep_runs(load("E5", smoke), "E5", "E5: GS weight separation (Sec. 7 item 5, Sec. 4)")
    J6 = load("E6", smoke)
    L += rep_runs(J6, "E6", "E6: switching statistic (Sec. 7 item 6, Sec. 5)")
    L += ["### E6 traces (first seed of each class): psi_B0, psi_geo, distance to the B0 / geo fixed points and to wM (dB), omega", ""]
    seen = set()
    for t, r in J6.items():
        c = cls_of(t)
        if c in seen:
            continue
        seen.add(c)
        q = fpath("E6", t, None if not smoke else 1, "npz")
        if not os.path.exists(q):
            continue
        A = np.load(q); nbr = r["nb"]
        idx = [b for b in [8, 16, 32, 64, 125, 250, 500, 1000, 1500, 1874, 1880, 1900, 2000, 2500, 3000, 3700] if b < nbr]
        if not r.get("pc"):
            idx = [b for b in idx if b not in (1874, 1880, 1900)]
        L += [f"#### {t}", "", "| rule | quantity | " + " | ".join(str(b) for b in idx) + " |", "|---|---|" + "---|" * len(idx)]
        for rn in r["rules"]:
            for qn, conv in (("psi_B0", None), ("psi_geo", None), ("om", None), ("ex", "db"), ("ex_B0", "db"), ("ex_geo", "db"), ("mu_app", None)):
                key = f"{rn}|{qn}"
                if key not in A.files:
                    continue
                v = np.asarray(A[key], float)
                vals = [(10 * np.log10(v[b]) if conv else v[b]) if b < len(v) else np.nan for b in idx]
                L.append(f"| {rn} | {qn}{' dB' if conv else ''} | " + " | ".join(fm(x, 3 if not conv else 1) for x in vals) + " |")
        L.append("")
    p = os.path.join(HERE, "RESULTS_vssT.md" if not smoke else os.path.join("out", "vssT_smoke", "RESULTS_vssT_smoke.md"))
    with open(p, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", p, f"({len(L)} lines)")


# ==============================================================================================
def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "report"] + list(JOBS))
    ap.add_argument("--classes"); ap.add_argument("--seeds"); ap.add_argument("--tags")
    ap.add_argument("--nb", type=int, default=None); ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--force", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rules", default=None, help="';'-separated rule names")
    ap.add_argument("--kinds", default=None, help="E2/E3 direction kinds, e.g. SF,B0,geo")
    ap.add_argument("--K", type=int, default=32, help="E1 ensemble window (blocks, <= 64)")
    ap.add_argument("--T", type=int, default=8, help="E1 constant-mu continuation horizon (blocks; 0 = off)")
    ap.add_argument("--pidef", default="x", choices=PI_DEFS); ap.add_argument("--gmin", type=float, default=GMIN)
    o = ap.parse_args(argv)
    o.rules = o.rules.split(";") if o.rules else None
    o.kinds = o.kinds.split(",") if o.kinds else (["SF", "B0", "geo"] if o.cmd == "e2" else ["SF"])
    o.K = min(o.K, 64)
    return o


if __name__ == "__main__":
    o = parse(sys.argv[1:])
    if o.cmd == "check":
        sys.exit(0 if check(o) else 1)
    elif o.cmd == "report":
        report(o)
    else:
        main_jobs(o.cmd, o)
