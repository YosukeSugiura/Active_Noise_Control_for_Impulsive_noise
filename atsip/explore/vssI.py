"""Step-size rules for the delayless FD-FxNLMS simulator (DERIVATION_vss.md sections 15-22).

Integral rules on a set of update directions, each with its own step mu_i:
    w <- w - 2 B/M sum_i mu_i D_i,   mu_i <- max(mu_i + gamma kappa* p_i / Q_i, 0),  capped by 0.5/g_n.
Statistics use u_i = D_i / max(g_n,1); probe p_i = <u_probe_n, ubar^i_{n-L}>, Q_i = EMA <u_i,u_i>.
Baselines (SF+ramp, GS, SF-V2, GS-V2 k2 .05/.2) are the vss2 controllers, reused unchanged.

Usage:  python vssI.py            run all jobs (Pool(2)), one json per job in out/
        python vssI.py --timing   per-block update wall time (one process)
        python vssI.py --report   write RESULTS_vssI.md from the json files
"""
import os, sys, json, time, glob
sys.path.insert(0, "/tmp/chk6")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simx import *
import vss2 as V
from multiprocessing import Pool

KAPPA = 2 * N * B / M**2          # = 1.0
L_LAG, BD, BQ = 4, 0.1, 0.05
QLO, QHI = 0.05, 0.2
OUT = os.path.join(EXP, "out")
IDX = [8, 31, 125, 500, 1500, 3700]
IDX_PC = [1870, 1880, 1900, 1950, 2100, 3000]
BASE = ["SF+ramp", "GS", "SF-V2", "GS-V2 k2 .05/.2"]


def D_(name, mu0, probe="self", cip="Phi", qip="Phi"):
    return dict(name=name, mu0=mu0, probe=probe, cip=cip, qip=qip)


# name -> (directions, gamma, GS-q omega flag)
RULES = {
    "SF-I g.026 m.3": ([D_("SF", .3)], .026, False),
    "SF-I g.013 m.3": ([D_("SF", .3)], .013, False),
    "SF-I g.05 m.3": ([D_("SF", .3)], .05, False),
    "SF-I g.026 m.15": ([D_("SF", .15)], .026, False),
    "SF-I g.026 m.5": ([D_("SF", .5)], .026, False),
    "SF-IE g.026 m.3": ([D_("SF", .3, cip="E", qip="E")], .026, False),
    "SF-Y g.026 m.3": ([D_("SF", .3, probe="D0", cip="E", qip="Phi")], .026, False),
    "GS-T g.026 m.3": ([D_("GS", .3)], .026, True),
    "Mix-BG": ([D_("B0", .3, probe="D0", cip="E"), D_("geo", 0., probe="D0", cip="E")], .026, False),
    "Mix-B0": ([D_("B0", .3, probe="D0", cip="E"), D_("D0", 0., probe="D0", cip="E")], .026, False),
    "Mix-BG0": ([D_("B0", .3, probe="D0", cip="E"), D_("geo", 0., probe="D0", cip="E"),
                 D_("D0", 0., probe="D0", cip="E")], .026, False),
}


def dir_vec(name, R, E, Srr, Pbar, eps, om):
    if name == "B0":
        return np.fft.irfft(np.conj(R) * E / (Srr + eps), N)[:M]
    if name == "geo":
        return np.fft.irfft(np.conj(R) * E / (np.sqrt(Srr * Pbar) + eps), N)[:M]
    if name == "D0":
        return np.fft.irfft(np.conj(R) * E / Pbar, N)[:M]
    if name == "SF":
        return SFm.sf_dir(R, E, Srr, Pbar, eps, 1e-2)
    if name == "GS":
        Lam = (1 - om) * Srr + om * np.sqrt(Srr * Pbar) + eps
        return np.fft.irfft(np.conj(R) * E / Lam, N)[:M]
    raise ValueError(name)


def ip(a, b, kind):
    return float(a @ (Phi @ b)) if kind == "Phi" else float(a @ b)


class Rule:
    """Integral step-size rule on a set of directions. block() = everything after the frame error."""

    def __init__(s, dirs, gamma, gs_q):
        s.gamma, s.gs_q = gamma, gs_q
        s.names = [d["name"] for d in dirs]
        s.need_D0 = any(d["probe"] == "D0" for d in dirs) or ("D0" in s.names)
        s.st = {d["name"]: dict(spec=d, mu=d["mu0"], ubar=None, buf=[np.zeros(M)] * L_LAG,
                                Q=None, q=0.15, qC=None, qQ=None) for d in dirs}
        s.mus = {n: [] for n in s.names}
        s.oms = []

    def block(s, i, R, E, Srr, Pbar, eps, w):
        g = float(np.mean(np.abs(R) ** 2) / Pbar)
        gu = max(g, 1.0)
        om = 1.0 - float(np.clip((s.st["GS"]["q"] - QLO) / (QHI - QLO), 0, 1)) if s.gs_q else 1.0
        s.oms.append(om if s.gs_q else np.nan)
        dv = {n: dir_vec(n, R, E, Srr, Pbar, eps, om) for n in s.names}
        if s.need_D0 and "D0" not in dv:
            dv["D0"] = dir_vec("D0", R, E, Srr, Pbar, eps, om)
        cap = 0.5 / max(g, 1e-9)
        step = np.zeros(M)
        for n in s.names:
            me = min(s.st[n]["mu"], cap)
            s.mus[n].append(me)
            step += me * dv[n]
        wn = w - 2 * B / M * step
        u = {n: v / gu for n, v in dv.items()}
        for n in s.names:
            S = s.st[n]
            sp = S["spec"]
            pv = u["D0"] if sp["probe"] == "D0" else u[n]
            c = ip(pv, S["buf"][0], sp["cip"])
            qq = ip(u[n], u[n], sp["qip"])
            if i >= L_LAG + 4:
                S["Q"] = qq if S["Q"] is None else (1 - BQ) * S["Q"] + BQ * qq
                S["mu"] = max(S["mu"] + s.gamma * KAPPA * c / (S["Q"] + 1e-300), 0.0)
                if s.gs_q and n == "GS":      # Ctrl2 q-estimator, Phi-weighted, sets omega
                    if S["qC"] is None:
                        S["qC"], S["qQ"] = S["q"] * qq, qq
                    else:
                        S["qC"] = (1 - BQ) * S["qC"] + BQ * c
                        S["qQ"] = (1 - BQ) * S["qQ"] + BQ * qq
                    S["q"] = max(S["qC"], 0.0) / (S["qQ"] + 1e-300)
            S["ubar"] = u[n].copy() if S["ubar"] is None else (1 - BD) * S["ubar"] + BD * u[n]
            S["buf"] = S["buf"][1:] + [S["ubar"].copy()]
        return wn


def simulate(data, Rall, rule):
    d, xq, hS, S2 = data["d"], data["xq"], data["hS"], data["S2"]
    L = len(d); nb = L // B
    swv = sliding_window_view(xq, M)
    w = np.zeros(M); zi = np.zeros(len(hS) - 1); e_all = np.zeros(L)
    shared = X.Shared(S2); div = False
    for i in range(nb):
        s0 = i * B
        Xb = swv[s0 + N - M + 1:s0 + N - M + 1 + B][:, ::-1]
        ys, zi = signal.lfilter(hS, [1.0], Xb @ w, zi=zi)
        e_all[s0:s0 + B] = d[s0:s0 + B] + ys
        E = ML.frame_E(e_all, s0, M)
        s = shared.step(Rall[i], E)
        wn = rule.block(i, Rall[i], E, s["Srr"], s["Pbar"], s["eps"], w)
        if not np.isfinite(wn).all():
            e_all[s0 + B:] = np.nan; div = True; break
        w = wn
    return dict(e=e_all, w=w, div=div,
                mu={n: np.array(v) for n, v in rule.mus.items()}, om=np.array(rule.oms))


def score(sim, data, pc):
    bad = (not np.isfinite(sim["w"]).all()) or (not np.isfinite(sim["e"]).all())
    if bad:
        return [np.nan] * 3, True
    if pc:
        sc = MFm.score_pc(sim, data)
        return [sc["L1a"], sc["L1b"], sc["L1w"]], bool(sc.get("div", 0))
    v = scores(sim, data)
    return [float(x) for x in v], not np.isfinite(v).all()


def at(arr, idx):
    return [float(arr[j]) if j < len(arr) else float("nan") for j in idx]


def job(arg):
    tag, pc = arg
    t0 = time.time()
    data = MFm.build_pc(O.cms_symmetric_stable(1.45, O.N_NOISE, int(tag[-1]))) if pc else get(tag)
    Rall = ML.prepass(data)["Rall"]; nb = len(data["d"]) // B
    idx = IDX_PC if pc else IDX
    res = {}
    bc = V.cfgs(nb, {k: V.VAR[k] for k in ["SF-V2", "GS-V2 k2 .05/.2"]})
    for k in BASE:
        c = bc[k]
        s = run(data, Rall, **c) if isinstance(c, dict) else run(data, Rall, kind=c[0], mu=0.0, a=0.5, ctrl=V.Ctrl2(**c[1]))
        sc, div = score(s, data, pc)
        res[k] = dict(sc=sc, div=div, mu={"mu": at(s["mu"], idx)}, om=at(s["om"], idx))
        print(f"[{tag}] {k}: {np.round(sc, 2)} div={div}", flush=True)
    for k, (dirs, gam, gq) in RULES.items():
        sim = simulate(data, Rall, Rule(dirs, gam, gq))
        sc, div = score(sim, data, pc)
        res[k] = dict(sc=sc, div=div, mu={n: at(v, idx) for n, v in sim["mu"].items()},
                      mu_full={n: np.round(v, 6).tolist() for n, v in sim["mu"].items()},
                      om=at(sim["om"], idx))
        print(f"[{tag}] {k}: {np.round(sc, 2)} div={div}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"vssI_{tag}.json"), "w") as f:
        json.dump(dict(tag=tag, pc=pc, idx=idx, res=res), f)
    print(f"[{tag}] done in {time.time() - t0:.0f} s", flush=True)
    return tag


# ---------- timing (update only: direction + step + statistics, no filter / frame / shared stats) ----------
def capture(nbk=500):
    data = get("G0"); Rall = ML.prepass(data)["Rall"]
    d, xq, hS, S2 = data["d"], data["xq"], data["hS"], data["S2"]
    swv = sliding_window_view(xq, M)
    w = np.zeros(M); zi = np.zeros(len(hS) - 1); e_all = np.zeros(len(d))
    shared = X.Shared(S2); cap = []
    for i in range(nbk):
        s0 = i * B
        Xb = swv[s0 + N - M + 1:s0 + N - M + 1 + B][:, ::-1]
        ys, zi = signal.lfilter(hS, [1.0], Xb @ w, zi=zi)
        e_all[s0:s0 + B] = d[s0:s0 + B] + ys
        E = ML.frame_E(e_all, s0, M); s = shared.step(Rall[i], E)
        cap.append((Rall[i], E, s["Srr"], s["Pbar"], s["eps"]))
        w = w - 2 * 0.1 * B / M * SFm.sf_dir(Rall[i], E, s["Srr"], s["Pbar"], s["eps"], 1e-2)
    return cap


def t_ctrl_sf(cap):
    ctrl = V.Ctrl2(**V.VAR["SF-V2"][1]); w = np.zeros(M)
    for i, (R, E, Srr, Pbar, eps) in enumerate(cap):
        mu, _ = ctrl.step(i, dict(R=R, Pbar=Pbar))
        g = SFm.sf_dir(R, E, Srr, Pbar, eps, 1e-2)
        mu = ctrl.post(i, g, mu)
        w = w - 2 * mu * B / M * g


def t_ctrl_gs(cap):
    ctrl = V.Ctrl2(**V.VAR["GS-V2 k2 .05/.2"][1]); w = np.zeros(M)
    for i, (R, E, Srr, Pbar, eps) in enumerate(cap):
        mu, om = ctrl.step(i, dict(R=R, Pbar=Pbar))
        Lam = (1 - om) * Srr + om * Srr ** 0.5 * Pbar ** 0.5 + eps
        g = np.fft.irfft(np.conj(R) * E / Lam, N)[:M]
        mu = ctrl.post(i, g, mu)
        w = w - 2 * mu * B / M * g


def t_rule(cap, name):
    dirs, gam, gq = RULES[name]; r = Rule(dirs, gam, gq); w = np.zeros(M)
    for i, (R, E, Srr, Pbar, eps) in enumerate(cap):
        w = r.block(i, R, E, Srr, Pbar, eps, w)


def timing():
    cap = capture(500)
    fns = [("SF-V2", lambda: t_ctrl_sf(cap)), ("SF-I g.026 m.3", lambda: t_rule(cap, "SF-I g.026 m.3")),
           ("GS-V2 k2 .05/.2", lambda: t_ctrl_gs(cap)), ("GS-T g.026 m.3", lambda: t_rule(cap, "GS-T g.026 m.3")),
           ("Mix-BG", lambda: t_rule(cap, "Mix-BG")), ("Mix-BG0", lambda: t_rule(cap, "Mix-BG0"))]
    out = {}
    for name, f in fns:
        ts = []
        for _ in range(3):
            t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
        out[name] = 1000 * min(ts) / len(cap)
        print(f"{name}: {out[name]:.3f} ms/block", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "vssI_timing.json"), "w") as f:
        json.dump(out, f)


# ---------- report ----------
GROUPS = [("Gaussian (2)", ["G0", "G1"]), ("alpha 1.45 (3)", ["cms145_s0", "cms145_s1", "cms145_s2"]),
          ("alpha 1.2 (2)", ["a120_s0", "a120_s1"]), ("path change (3)", ["pc0", "pc1", "pc2"])]


def report():
    J = {}
    for f in glob.glob(os.path.join(OUT, "vssI_*.json")):
        if "timing" in f:
            continue
        r = json.load(open(f)); J[r["tag"]] = r
    order = BASE + list(RULES)
    lines = ["# RESULTS vssI: step-size rules (delayless FD-FxNLMS)", "",
             "L1 = ANR mean in dB, lower (more negative) is better. Means over non-diverged runs; "
             "n = runs used, div = diverged runs. Path change columns: 15-17 s / 25-30 s / whole.", ""]
    for gname, tags in GROUPS:
        lines += [f"## {gname}", "", "| rule | whole | first5 | last10 | n | div |" if "path" not in gname
                  else "| rule | 15-17 s | 25-30 s | whole | n | div |", "|---|---|---|---|---|---|"]
        for k in order:
            rows = [J[t]["res"][k] for t in tags if t in J]
            ok = [r["sc"] for r in rows if not r["div"]]
            nd = sum(r["div"] for r in rows)
            if ok:
                m = np.mean(np.array(ok), 0)
                cells = " | ".join(f"{x:.2f}" for x in m)
            else:
                cells = "NaN | NaN | NaN"
            lines.append(f"| {k} | {cells} | {len(ok)} | {nd} |")
        lines.append("")
    lines += ["## Step traces (applied mu per direction, and omega for GS rules)", "",
              f"Block indices: {IDX} (non-path), {IDX_PC} (path change). Values from G0 and pc0.", ""]
    for t in ["G0", "pc0"]:
        if t not in J:
            continue
        ix = J[t]["idx"]
        lines += [f"### {t} (idx {ix})", "", "| rule | direction | mu at idx | om at idx |", "|---|---|---|---|"]
        for k in order:
            r = J[t]["res"][k]
            for n, v in r["mu"].items():
                om = "" if all(np.isnan(r["om"])) else " ".join(f"{x:.2f}" for x in r["om"])
                lines.append(f"| {k} | {n} | {' '.join(f'{x:.4f}' for x in v)} | {om} |")
        lines.append("")
    if os.path.exists(os.path.join(OUT, "vssI_timing.json")):
        tm = json.load(open(os.path.join(OUT, "vssI_timing.json")))
        lines += ["## Per-block update time (one process, 500 blocks of G0 inputs)", "",
                  "Update only: direction(s), step, statistics. Phi weighting is a dense MxM product here; "
                  "a practical version would use a pre-factor vector or bin weights.", "",
                  "| method | ms per block |", "|---|---|"]
        lines += [f"| {k} | {v:.3f} |" for k, v in tm.items()]
        lines.append("")
    with open(os.path.join(EXP, "RESULTS_vssI.md"), "w") as f:
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
