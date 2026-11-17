"""Step-size rules with frequency-domain weighted inner products (DERIVATION_vss.md sections 12, 18, 21-22).

Follows vssI.py (same integral step update, same directions, same caps), but the delayed correlation c_n
and the power Q_n are measured with bin weights instead of the time-domain Phi matrix:
    <a,b>_w = (1/N) sum_k cv_k w_k Re(A_k conj(B_k)),  cv_k = 2 (interior bins), 1 (DC, Nyquist)
    c_n uses w = Lam_n / Pbar_n (Lam = GS normaliser of the block, with the same eps),
    Q_n uses w = Srr_n / Pbar_n.  Both factors use the weights of block n.  w = None gives the Euclidean product.

Rules (all: mu <- max(mu + gamma kappa* c/Q, 0), kappa* = 2NB/M^2, L=4, beta_d=0.1, beta_Q=0.05, gamma=0.026,
mu0=0.3, cap min(mu, 0.5/g_n); directions divided by max(g_n,1) before statistics):
    GS-T2            GS ramp direction, om from the GS-V2 q rule (qlo .05, qhi .2), q with the weighted c, Q
    GS-T2 fixed-om   same weights and step rule, om = ramp(nb,4,3) of the GS baseline
    GS-T2 E          GS-T2 with Euclidean weights for both c and Q (control)
    GS-V2T           Ctrl2 GS-V2 k2 .05/.2 with L=4 and chi=0.5 set explicitly (fallback)
Baselines rerun in the same run: GS, GS-V2 k2 .05/.2, SF-I g.026 m.3, SF-IE g.026 m.3.

Usage:  python vssI2.py            run all jobs (Pool(2)), one json per job in out/vssI2_<tag>.json
        python vssI2.py --timing   per-block update wall time (one process)
        python vssI2.py --report   write RESULTS_vssI2.md from the json files
"""
import os, sys, json, time, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/tmp/chk6")
import vssI as V1                       # shared directions, Rule (SF-I, SF-IE), simulate, scoring, capture
from simx import *                      # get, run, ramp, scores, ML, MFm, O, B, M, N, EXP
import vss2 as V
from multiprocessing import Pool

NBIN = N // 2 + 1
CV = np.full(NBIN, 2.0); CV[0] = 1.0
if N % 2 == 0:
    CV[-1] = 1.0
KAPPA, L_LAG, BD, BQ = V1.KAPPA, V1.L_LAG, V1.BD, V1.BQ
QLO, QHI = V1.QLO, V1.QHI
OUT = V1.OUT
IDX, IDX_PC = V1.IDX, V1.IDX_PC
GROUPS = V1.GROUPS
ORDER = ["GS", "GS-V2 k2 .05/.2", "SF-I g.026 m.3", "SF-IE g.026 m.3",
         "GS-T2", "GS-T2 fixed-om", "GS-T2 E", "GS-V2T"]


def ipw(A, Bf, w):
    """Weighted inner product of two rfft spectra (length N). w=None: Euclidean (exact Parseval)."""
    p = (A * np.conj(Bf)).real
    if w is not None:
        p = p * w
    return float(np.sum(CV * p) / N)


class Rule2:
    """mode: 'q_w' (GS-T2), 'fixed_w' (GS-T2 fixed-om), 'q_E' (GS-T2 E)."""

    def __init__(s, mode, gamma=0.026, om_sched=None):
        s.mode, s.gamma, s.om_sched = mode, gamma, om_sched
        s.mu, s.ubar, s.Q, s.qC, s.q = 0.3, None, None, None, 0.15
        s.buf = [np.zeros(NBIN, dtype=complex)] * L_LAG
        s.mus = {"GS": []}; s.oms = []

    def block(s, i, R, E, Srr, Pbar, eps, w):
        g = float(np.mean(np.abs(R) ** 2) / Pbar)
        gu = max(g, 1.0)
        if s.mode == "fixed_w":
            om = float(s.om_sched[min(i, len(s.om_sched) - 1)])
        else:
            om = 1.0 - float(np.clip((s.q - QLO) / (QHI - QLO), 0, 1))
        s.oms.append(om)
        dv = V1.dir_vec("GS", R, E, Srr, Pbar, eps, om)
        cap = 0.5 / max(g, 1e-9)
        me = min(s.mu, cap); s.mus["GS"].append(me)
        wn = w - 2 * B / M * me * dv
        u = dv / gu
        U = np.fft.rfft(u, N)
        if s.mode == "q_E":
            wc = wq = None
        else:
            Lam = (1 - om) * Srr + om * np.sqrt(Srr * Pbar) + eps
            wc, wq = Lam / Pbar, Srr / Pbar
        c = ipw(U, s.buf[0], wc)          # delayed correlation with ubar_{n-L}
        qq = ipw(U, U, wq)
        if i >= L_LAG + 4:
            s.Q = qq if s.Q is None else (1 - BQ) * s.Q + BQ * qq
            s.mu = max(s.mu + s.gamma * KAPPA * c / (s.Q + 1e-300), 0.0)
            if s.mode != "fixed_w":       # q-estimator for omega, same weights
                if s.qC is None:
                    s.qC = s.q * qq
                else:
                    s.qC = (1 - BQ) * s.qC + BQ * c
                s.q = max(s.qC, 0.0) / (s.Q + 1e-300)
        s.ubar = u.copy() if s.ubar is None else (1 - BD) * s.ubar + BD * u
        s.buf = s.buf[1:] + [np.fft.rfft(s.ubar, N)]
        return wn


def _traces(s, idx):
    if isinstance(s["mu"], dict):
        mu = {n: V1.at(v, idx) for n, v in s["mu"].items()}
    else:
        mu = {"mu": V1.at(s["mu"], idx)}
    return mu, V1.at(s["om"], idx)


def job(arg):
    tag, pc = arg
    t0 = time.time()
    data = MFm.build_pc(O.cms_symmetric_stable(1.45, O.N_NOISE, int(tag[-1]))) if pc else get(tag)
    Rall = ML.prepass(data)["Rall"]; nb = len(data["d"]) // B
    idx = IDX_PC if pc else IDX
    res = {}

    def rec(k, sim, is_sim=None):
        sc, div = V1.score(sim, data, pc)
        if div:
            sc = [np.nan] * 3
        mu, om = _traces(sim, idx)
        res[k] = dict(sc=[float(x) for x in sc], div=bool(div), mu=mu, om=om)
        print(f"[{tag}] {k}: {np.round(sc, 2)} div={div}", flush=True)

    bc = V.cfgs(nb, {k: V.VAR[k] for k in ["GS-V2 k2 .05/.2"]})
    for k in ["GS", "GS-V2 k2 .05/.2"]:
        c = bc[k]
        s = run(data, Rall, **c) if isinstance(c, dict) else run(data, Rall, kind=c[0], mu=0.0, a=0.5, ctrl=V.Ctrl2(**c[1]))
        rec(k, s, False)
    for k in ["SF-I g.026 m.3", "SF-IE g.026 m.3"]:
        dirs, gam, gq = V1.RULES[k]
        rec(k, V1.simulate(data, Rall, V1.Rule(dirs, gam, gq)), True)

    wr, _ = ramp(nb, 4, 3)
    rec("GS-T2", V1.simulate(data, Rall, Rule2("q_w")), True)
    rec("GS-T2 fixed-om", V1.simulate(data, Rall, Rule2("fixed_w", om_sched=wr)), True)
    rec("GS-T2 E", V1.simulate(data, Rall, Rule2("q_E")), True)
    ct = V.Ctrl2(kmu=2.0, qlo=0.05, qhi=0.2, L=4, chi=0.5)
    rec("GS-V2T", run(data, Rall, kind="W", mu=0.0, a=0.5, ctrl=ct), False)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"vssI2_{tag}.json"), "w") as f:
        json.dump(dict(tag=tag, pc=pc, idx=idx, res=res), f)
    print(f"[{tag}] done in {time.time() - t0:.0f} s", flush=True)
    return tag


# ---------- timing (update only, as in vssI.timing) ----------
def t_rule2(cap, mode):
    r = Rule2(mode); w = np.zeros(M)
    for i, (R, E, Srr, Pbar, eps) in enumerate(cap):
        w = r.block(i, R, E, Srr, Pbar, eps, w)


def timing():
    cap = V1.capture(500)
    fns = [("GS-V2 k2 .05/.2", lambda: V1.t_ctrl_gs(cap)), ("GS-T2", lambda: t_rule2(cap, "q_w")),
           ("SF-IE g.026 m.3", lambda: V1.t_rule(cap, "SF-IE g.026 m.3"))]
    out = {}
    for name, f in fns:
        ts = []
        for _ in range(3):
            t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
        out[name] = 1000 * min(ts) / len(cap)
        print(f"{name}: {out[name]:.3f} ms/block", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "vssI2_timing.json"), "w") as f:
        json.dump(out, f)


# ---------- report ----------
def report():
    J = {}
    for f in glob.glob(os.path.join(OUT, "vssI2_*.json")):
        if "timing" in f:
            continue
        r = json.load(open(f)); J[r["tag"]] = r
    lines = ["# RESULTS vssI2: weighted-inner-product step rules (delayless FD-FxNLMS)", "",
             "L1 = ANR mean in dB, lower (more negative) is better. Mean over non-diverged runs; "
             "n = runs used, div = diverged runs (NaN in the json). Path change columns: 15-17 s / 25-30 s / whole.", ""]
    for gname, tags in GROUPS:
        path = "path" in gname
        lines += [f"## {gname}", "",
                  "| rule | 15-17 s | 25-30 s | whole | n | div |" if path else "| rule | whole | first5 | last10 | n | div |",
                  "|---|---|---|---|---|---|"]
        for k in ORDER:
            rows = [J[t]["res"][k] for t in tags if t in J]
            ok = [r["sc"] for r in rows if not r["div"]]
            nd = sum(r["div"] for r in rows)
            cells = " | ".join(f"{x:.2f}" for x in np.mean(np.array(ok), 0)) if ok else "NaN | NaN | NaN"
            lines.append(f"| {k} | {cells} | {len(ok)} | {nd} |")
        lines.append("")
    lines += ["## Step traces (applied mu per direction, and omega for GS rules)", "",
              f"Block indices: {IDX} (non-path), {IDX_PC} (path change). Values from G0 and pc0.", ""]
    for t in ["G0", "pc0"]:
        if t not in J:
            continue
        lines += [f"### {t} (idx {J[t]['idx']})", "", "| rule | direction | mu at idx | om at idx |", "|---|---|---|---|"]
        for k in ORDER:
            r = J[t]["res"][k]
            om = "" if all(np.isnan(r["om"])) else " ".join(f"{x:.2f}" for x in r["om"])
            for n, v in r["mu"].items():
                lines.append(f"| {k} | {n} | {' '.join(f'{x:.4f}' for x in v)} | {om} |")
        lines.append("")
    tp = os.path.join(OUT, "vssI2_timing.json")
    if os.path.exists(tp):
        tm = json.load(open(tp))
        lines += ["## Per-block update time (one process, 500 blocks of G0 inputs)", "",
                  "Update only: direction, step, statistics. Weighted products use FFT spectra and bin weights "
                  "(no MxM product).", "", "| method | ms per block |", "|---|---|"]
        lines += [f"| {k} | {v:.3f} |" for k, v in tm.items()]
        lines.append("")
    with open(os.path.join(EXP, "RESULTS_vssI2.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    if "--timing" in sys.argv:
        timing()
    elif "--report" in sys.argv:
        report()
    else:
        jobs = [(t, False) for t in ["G0", "G1", "cms145_s0", "cms145_s1", "cms145_s2", "a120_s0", "a120_s1"]] + \
               [(f"pc{i}", True) for i in range(3)]
        with Pool(2) as pl:
            for tag in pl.imap_unordered(job, jobs):
                print("finished", tag, flush=True)
