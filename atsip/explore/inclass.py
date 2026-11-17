#!/usr/bin/env python3
"""Scale-varying inputs: checks of claims (i)-(iv) for the normaliser family.

Inputs (30 s, 16 kHz, same primary/secondary paths as oracle.py), seeds 0-2, x = sigma(n) * xi(n):
  A   Gaussian white (CMS alpha=2), sigma = 1
  B   Gaussian, sigma x10 (+20 dB power) on [10,15) s, x0.1 (-20 dB) on [20,25) s
  C   Gaussian, sigma x10 (+20 dB) for 50 ms at t = k s, k = 0..29
  D   i.i.d. SaS (oracle.cms_symmetric_stable), alpha in {1.85, 1.65, 1.45, 1.2}
Methods (mu from RESULTS_specfac.md; W0 has no entry there, its mu is chosen by the specfac
rule (six-source white whole mean) over the specfac grid):
  B0, W0 (a=0), W0.5, GS (a=0.5, T=4, m0=3), SF (eps 1e-2, no ramp), SF+ramp (eps 1e-2, T=2, m0=3)
Stability (iii): Gaussian, sigma step at 10 s of rho in {0,10,20,30,40} dB (held), and -40 dB.
  Methods B0, W0.5, SF; mu grid geomspace(0.01, 2, 17) (extended below the 0.05 floor, see RESULTS).
  Seeds 0, 1.

alpha_E = X.ALPHA = 0.05 (Srr EMA coefficient, oracle.py). g_i = block power of R / EMA power
(mean over bins of |R_i|^2 over mean over bins of Srr_i); g depends on the input only.
Divergence: as specfac (non-finite output or end ANR > 0 dB). Diverged runs are NaN in means.
Metrics: L1-ANR (oracle.anr_curve), L2-ANR = 10 log10(sum e^2 / sum d^2) on the same windows,
NM(t) every 64 blocks, final-weight distance, E[g^2]/E[g].

Usage:
  python3 atsip/explore/inclass.py check   reproduce four specfac rows on a145 (exact match of final w)
  python3 atsip/explore/inclass.py         full run; Pool(2); writes out/inclass_*.json, out/inclass_effT.png,
                                           and RESULTS_inclass.md (tables only; conclusion appended by hand)
  python3 atsip/explore/inclass.py post    rebuild sections 3 and 5 of RESULTS_inclass.md from the saved
                                           stability json and a per-block effective-time run
"""
import os
import sys
import json
import time
import warnings
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dissect as D     # noqa: E402
import explore as X    # noqa: E402
import mixlam as ML    # noqa: E402
import geosched as G   # noqa: E402
import specfac as SF   # noqa: E402

O = X.O
FS, M, N, B = D.FS, D.M, D.N, D.B
L_SIG = O.N_NOISE                       # 480000 samples = 30 s
NBLK = L_SIG // B                       # 3750 blocks
ALPHA_E = X.ALPHA                       # Srr EMA coefficient (oracle.py), 0.05
N5S = 5 * FS                            # first 5 s in samples
N10S = L_SIG // 10                      # last 10% in samples (= last 10% of ANR samples)
N5_ANR = X.N5                           # first 5 s in ANR samples
SAMPLE_EVERY = 64                       # NM sampled every 64 blocks
OUT = os.path.join(HERE, "out")
SEEDS = [0, 1, 2]
STAB_SEEDS = [0, 1]
STAB_MU = [float(v) for v in np.geomspace(0.01, 2.0, 17)]
INPUTS = ["A", "B", "C", "D1.85", "D1.65", "D1.45", "D1.2"]
METHODS = ["B0", "W0", "W0.5", "GS", "SF", "SF+ramp"]
MU_FIXED = {"B0": 0.1, "W0.5": 0.1, "GS": 0.05, "SF": 0.1, "SF+ramp": 0.05}   # RESULTS_specfac.md
SPEC = {"B0": ("W", 1.0), "W0": ("W", 0.0), "W0.5": ("W", 0.5),
        "GS": ("GS", 0.5, 4, 3), "SF": ("SF", 1e-2, 0, 1), "SF+ramp": ("SF", 1e-2, 2, 3)}
STAB_METHODS = {"B0": ("W", 1.0), "W0.5": ("W", 0.5), "SF": ("SF", 1e-2, 0, 1)}
STAB_SIGS = ["up0", "up10", "up20", "up30", "up40", "down40"]
GRID_STEP = float(np.geomspace(0.01, 2.0, 17)[1] / np.geomspace(0.01, 2.0, 17)[0])


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
def sigma_of(kind):
    t = np.arange(L_SIG) / FS
    s = np.ones(L_SIG)
    if kind == "B":
        s[(t >= 10) & (t < 15)] = 10.0          # +20 dB power
        s[(t >= 20) & (t < 25)] = 0.1           # -20 dB power
    elif kind == "C":
        n50 = int(round(0.05 * FS))
        for k in range(30):
            s[k * FS: k * FS + n50] = 10.0      # +20 dB power bursts of 50 ms, once per second
    elif kind.startswith("up"):
        s[t >= 10] = 10 ** (float(kind[2:]) / 20)
    elif kind == "down40":
        s[t >= 10] = 10 ** (-40 / 20)
    return s


def make_input(name, seed):
    if name == "A":
        return O.cms_symmetric_stable(2.0, L_SIG, seed)
    if name in ("B", "C"):
        return sigma_of(name) * O.cms_symmetric_stable(2.0, L_SIG, seed)
    return O.cms_symmetric_stable(float(name[1:]), L_SIG, seed)


# ----------------------------------------------------------------------------
# Gain sequence g_i = block power / EMA power (depends on the reference only)
# ----------------------------------------------------------------------------
def gain_seq(Rall):
    A = np.abs(Rall) ** 2
    Pb = A.mean(axis=1)
    Srr = A[0].copy()
    Pe = np.empty(len(A))
    Pe[0] = Srr.mean()
    for i in range(1, len(A)):
        Srr = (1 - ALPHA_E) * Srr + ALPHA_E * A[i]
        Pe[i] = Srr.mean()
    return Pb / Pe


# ----------------------------------------------------------------------------
# Tracked simulation: same loop as D.simulate / G.sim_sched / SF.sim_sf, plus NM every 64 blocks
# ----------------------------------------------------------------------------
def run_tracked(data, Rall, spec, mu, every=SAMPLE_EVERY):
    d, xq, hS, S2, w_opt = data["d"], data["xq"], data["hS"], data["S2"], data["w_opt"]
    swv = sliding_window_view(xq, M)
    w = np.zeros(M)
    zi = np.zeros(len(hS) - 1)
    e_all = np.zeros(L_SIG)
    shared = X.Shared(S2)
    wo2 = float(w_opt @ w_opt)
    nm = []
    diverged = False
    for i in range(NBLK):
        s0 = i * B
        Xb = swv[s0 + N - M + 1: s0 + N - M + 1 + B][:, ::-1]
        ys, zi = signal.lfilter(hS, [1.0], Xb @ w, zi=zi)
        e_all[s0:s0 + B] = d[s0:s0 + B] + ys
        E = ML.frame_E(e_all, s0, M)
        s = shared.step(Rall[i], E)
        Srr, Pbar, eps = s["Srr"], s["Pbar"], s["eps"]
        kind = spec[0]
        if kind == "SF":
            _, eps_reg, T, m0 = spec
            wr = 1.0 if T == 0 else min(1.0, (i * B / FS) / T)
            mu_i = mu * (1 + (m0 - 1) * (1 - wr))
            g = SF.sf_dir(Rall[i], E, Srr, Pbar, eps, eps_reg)
        elif kind == "W":
            a = spec[1]
            mu_i = mu
            Lam = Srr ** a * Pbar ** (1 - a) + eps
            g = np.fft.irfft(np.conj(Rall[i]) * E / Lam, N)[:M]
        else:
            _, a, T, m0 = spec
            wr = 1.0 if T == 0 else min(1.0, (i * B / FS) / T)
            Lam = (1 - wr) * Srr + wr * Srr ** a * Pbar ** (1 - a) + eps
            mu_i = mu * (1 + (m0 - 1) * (1 - wr))
            g = np.fft.irfft(np.conj(Rall[i]) * E / Lam, N)[:M]
        w = w - 2 * mu_i * B / M * g
        if not np.isfinite(w).all():
            e_all[s0 + B:] = np.nan
            diverged = True
            break
        if (i + 1) % every == 0:
            nm.append(10 * np.log10(float((w - w_opt) @ (w - w_opt)) / wo2 + 1e-30))
    nm += [float("nan")] * (NBLK // every - len(nm))
    return dict(e=e_all, w=w, diverged=diverged, nm=nm)


def metrics(e, w, d, w_opt):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        anr = O.anr_curve(e, d)
        whole, last10 = O.summarize(anr)
        fin_anr = bool(np.isfinite(anr).all())
        first5 = float(anr[:N5_ANR].mean()) if fin_anr else float("nan")
        e2, d2 = e ** 2, d ** 2
        L2 = dict(whole=10 * np.log10(e2.sum() / d2.sum()),
                  first5=10 * np.log10(e2[:N5S].sum() / d2[:N5S].sum()),
                  last10=10 * np.log10(e2[-N10S:].sum() / d2[-N10S:].sum()))
        wfin = bool(np.isfinite(w).all())
        fnm = O.final_msd_db(w, w_opt) if wfin else float("nan")
    div = (not wfin) or (not np.isfinite(last10)) or (not np.isfinite(whole)) or last10 > 0.0
    return dict(L1=dict(whole=float(whole), first5=first5, last10=float(last10)),
                L2={k: float(v) for k, v in L2.items()}, final_nm=float(fnm), div=bool(div))


def wrel_db(w, ref):
    return float(10 * np.log10((w - ref) @ (w - ref) / (ref @ ref) + 1e-30))


# ----------------------------------------------------------------------------
# Workers
# ----------------------------------------------------------------------------
def check_refs():
    """Final weights of run_tracked vs the reused simulators, on the a145 file source."""
    data = D.load_data(*D.KIND["a145"])
    Rall = ML.prepass(data)["Rall"]
    pairs = {
        "B0": (run_tracked(data, Rall, SPEC["B0"], 0.1)["w"], D.simulate(data, Rall, D.spec_of("B0"), 0.1, M)["w"]),
        "W0.5": (run_tracked(data, Rall, SPEC["W0.5"], 0.1)["w"], D.simulate(data, Rall, D.spec_of("W0.5"), 0.1, M)["w"]),
        "GS": (run_tracked(data, Rall, SPEC["GS"], 0.05)["w"], G.sim_sched(data, Rall, 0.5, 4, 3, 0.05)["w"]),
        "SF": (run_tracked(data, Rall, SPEC["SF"], 0.1)["w"], SF.sim_sf(data, Rall, 1e-2, 0, 1, 0.1)["w"]),
        "SF+ramp": (run_tracked(data, Rall, SPEC["SF+ramp"], 0.05)["w"],
                    SF.sim_sf(data, Rall, 1e-2, 2, 3, 0.05)["w"]),
    }
    out = {k: float(np.max(np.abs(a - b))) for k, (a, b) in pairs.items()}
    # cross-check against specfac_grid.csv (a145 rows, whole-curve ANR)
    grid = {}
    p = os.path.join(OUT, "specfac_grid.csv")
    if os.path.exists(p):
        import csv
        with open(p) as fh:
            for r in csv.DictReader(fh):
                if r["source"] == "a145":
                    grid[(r["label"], float(r["mu"]))] = float(r["whole_db"]) if r["whole_db"] not in ("", "nan") else float("nan")
    mine = {"B0": ("B0", 0.1, pairs["B0"][0]), "W0.5": ("W0.5", 0.1, pairs["W0.5"][0]),
            "GS": ("GS (a0.5 T4 m0=3)", 0.05, pairs["GS"][0]), "SF": ("SF e=0.01 T=0 m0=1", 0.1, pairs["SF"][0]),
            "SF+ramp": ("SF e=0.01 T=2 m0=3", 0.05, pairs["SF+ramp"][0])}
    csv_cmp = {}
    for name, (lab, mu, w) in mine.items():
        sim = run_tracked(data, Rall, SPEC[name], mu)
        m = metrics(sim["e"], sim["w"], data["d"], data["w_opt"])
        csv_cmp[name] = (m["L1"]["whole"], grid.get((lab, mu), float("nan")))
    return dict(w_maxabs=out, csv_whole=csv_cmp)


def unit_seed(args):
    seed, mu_w0 = args
    t0 = time.time()
    mu_used = dict(MU_FIXED, W0=mu_w0)
    res = dict(seed=seed, inputs={}, fp={}, diag={}, wfin={})
    wopt10, wfull = {}, {}
    for name in INPUTS:
        data = D.build(make_input(name, seed))
        Rall = ML.prepass(data)["Rall"]
        g = gain_seq(Rall)
        cumg = np.cumsum(g)[SAMPLE_EVERY - 1::SAMPLE_EVERY]
        rec = {}
        for mth in METHODS:
            sim = run_tracked(data, Rall, SPEC[mth], mu_used[mth])
            m = metrics(sim["e"], sim["w"], data["d"], data["w_opt"])
            m.update(mu=mu_used[mth], nm=sim["nm"])
            rec[mth] = m
            res["wfin"][(name, mth)] = sim["w"]
        wopt10[name] = data["w_opt"]
        wfull[name] = O.wiener_opt(data["r"], data["d"], T=L_SIG)
        res["inputs"][name] = dict(Eg=float(g.mean()), Eg2_over_Eg=float((g ** 2).mean() / g.mean()),
                                   cumg=cumg.tolist(), rec=rec)
        print(f"[unit seed {seed}] input {name} done ({time.time() - t0:.1f} s)", flush=True)
    if seed == 0:
        res["check"] = check_refs()
    # (ii) final-weight distance to the A run of the same seed, and (i) w_opt distance to A
    for name in INPUTS[1:]:
        for mth in METHODS:
            if res["inputs"][name]["rec"][mth]["div"] or res["inputs"]["A"]["rec"][mth]["div"]:
                res["fp"][f"{name}|{mth}"] = float("nan")
            else:
                res["fp"][f"{name}|{mth}"] = wrel_db(res["wfin"][(name, mth)], res["wfin"][("A", mth)])
    for name in ["B", "C"]:
        res["diag"][f"wopt10|{name}"] = wrel_db(wopt10[name], wopt10["A"])
        res["diag"][f"woptfull|{name}"] = wrel_db(wfull[name], wfull["A"])
    res["diag"]["wopt10|A"] = 0.0
    with open(os.path.join(OUT, f"inclass_seed{seed}.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(seed=seed, mu=mu_used, inputs=res["inputs"], fp=res["fp"], diag=res["diag"],
                       check=res.get("check")), fh, default=float)
    res["runtime"] = time.time() - t0
    print(f"[done] seed {seed} {res['runtime']:.1f} s", flush=True)
    return res


def unit_stab(args):
    sig, seed = args
    t0 = time.time()
    data = D.build(sigma_of(sig) * O.cms_symmetric_stable(2.0, L_SIG, seed))
    Rall = ML.prepass(data)["Rall"]
    rows = []
    for mth, spec in STAB_METHODS.items():
        for mu in STAB_MU:
            sim = run_tracked(data, Rall, spec, mu)
            m = metrics(sim["e"], sim["w"], data["d"], data["w_opt"])
            rows.append(dict(method=mth, mu=mu, div=m["div"], whole=m["L1"]["whole"]))
    out = dict(sig=sig, seed=seed, rows=rows)
    with open(os.path.join(OUT, f"inclass_stab_{sig}_s{seed}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, default=float)
    print(f"[stab] {sig} seed {seed} {time.time() - t0:.1f} s", flush=True)
    return out


def dispatch(task):
    if task[0] == "main":
        return ("main", unit_seed((task[1], task[2])))
    return ("stab", unit_stab(task[1]))


def w0_select(tag):
    data = D.load_data(*D.KIND[tag])
    Rall = ML.prepass(data)["Rall"]
    rows = []
    for mu in SF.MUS:
        sim = run_tracked(data, Rall, SPEC["W0"], mu)
        m = metrics(sim["e"], sim["w"], data["d"], data["w_opt"])
        rows.append(dict(mu=mu, whole=m["L1"]["whole"], div=m["div"]))
    print(f"[w0sel] {tag} done", flush=True)
    return tag, rows


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
def fmt(v, p=2):
    return "nan" if v is None or not np.isfinite(v) else f"{v:.{p}f}"


def agg(vals, divs):
    if any(divs):
        return float("nan")
    return float(np.mean(vals))


def cross(curve, thr):
    idx = np.where(np.asarray(curve) <= thr)[0]
    return int(idx[0]) if len(idx) else None


# ----------------------------------------------------------------------------
# Section 3 (stability) and section 5 (effective time): functions, so that `post` can rebuild them
# ----------------------------------------------------------------------------
def stab_lines(stab_res):
    L = []
    alpha = ALPHA_E
    rhos = [("up0", 0), ("up10", 10), ("up20", 20), ("up30", 30), ("up40", 40), ("down40", -40)]
    rows_of = {}
    for sig in STAB_SIGS:
        for mth in STAB_METHODS:
            rows_of[(sig, mth)] = [r for s in stab_res if s["sig"] == sig for r in s["rows"] if r["method"] == mth]

    def mumax(rule):
        out = {}
        for key, rows in rows_of.items():
            st = []
            for mu in STAB_MU:
                rr = [r for r in rows if r["mu"] == mu]
                if rule == "task":
                    ok = not any(r["div"] for r in rr)
                else:   # whole-curve ANR <= 0 dB on both seeds, and not diverged
                    ok = all((not r["div"]) and np.isfinite(r["whole"]) and r["whole"] <= 0.0 for r in rr)
                if ok:
                    st.append(mu)
            out[key] = max(st) if st else None
        return out

    for rule, title in [("task", "3a. Task rule: no divergence (non-finite or end ANR > 0 dB)"),
                        ("transient", "3b. Stricter rule: whole-curve ANR <= 0 dB on both seeds")]:
        mm_all = mumax(rule)
        base = {m: mm_all[("up0", m)] for m in STAB_METHODS}
        L.append(f"## {title}\n")
        if rule == "task":
            L.append(f"Grid: {', '.join(fmt(v, 3) for v in STAB_MU)} (17 values; extended below 0.05). "
                     "Max stable mu = largest grid mu passing the rule on both seeds. "
                     "Theory: g(rho) = rho/((1-alpha_E) + alpha_E rho), prediction mu_max(rho)/mu_max(0) = 1/g(rho). "
                     "Downward step: g < 1, so theory gives no cap (ratio >= 1 expected).\n")
        L.append("| step | rho [dB] | g(rho) | theory 1/g | B0 mu_max | B0 ratio | W0.5 mu_max | W0.5 ratio | SF mu_max | SF ratio |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for sig, rho in rhos:
            r_lin = 10 ** (rho / 10)
            g = r_lin / ((1 - alpha) + alpha * r_lin)
            cells = []
            for m in STAB_METHODS:
                mm = mm_all[(sig, m)]
                ratio = (mm / base[m]) if (mm is not None and base[m]) else None
                cells += [fmt(mm, 3) if mm is not None else "none", fmt(ratio, 3) if ratio is not None else "none"]
            th = "no cap" if rho < 0 else fmt(1 / g, 3)
            L.append(f"| {sig} | {rho} | {fmt(g, 3)} | {th} | " + " | ".join(cells) + " |")
        L.append("")
        for m in STAB_METHODS:
            hits = []
            for sig, rho in rhos:
                if rho <= 0 or base[m] is None:
                    continue
                mm = mm_all[(sig, m)]
                r_lin = 10 ** (rho / 10)
                th = 1 / (r_lin / ((1 - alpha) + alpha * r_lin))
                if mm is None:
                    hits.append(f"{rho} dB: none")
                    continue
                rat = mm / base[m]
                ok = abs(np.log(rat / th)) <= np.log(GRID_STEP) + 1e-9
                hits.append(f"{rho} dB: ratio {fmt(rat, 3)} vs theory {fmt(th, 3)}, {'within' if ok else 'outside'} one grid step")
            L.append(f"- {m}: " + "; ".join(hits))
        L.append("")
    return L


def effT_fine():
    """Per-block NM for W0.5 and SF on A, D1.45, D1.2 (seeds 0-2). Returns seed-mean curves."""
    raw = {}
    for mth in ["W0.5", "SF"]:
        for inp in ["A", "D1.45", "D1.2"]:
            for seed in SEEDS:
                data = D.build(make_input(inp, seed))
                Rall = ML.prepass(data)["Rall"]
                cumg = np.cumsum(gain_seq(Rall))
                sim = run_tracked(data, Rall, SPEC[mth], MU_FIXED[mth], every=1)
                raw[(mth, inp, seed)] = (np.array(sim["nm"]), cumg)
    eff = {}
    for mth in ["W0.5", "SF"]:
        for inp in ["A", "D1.45", "D1.2"]:
            nm = np.mean([raw[(mth, inp, s)][0] for s in SEEDS], axis=0)
            cg = np.mean([raw[(mth, inp, s)][1] for s in SEEDS], axis=0)
            eff[(mth, inp)] = (nm, cg)
    with open(os.path.join(OUT, "inclass_effT.json"), "w", encoding="utf-8") as fh:
        json.dump({f"{m}|{i}": dict(block=list(range(1, NBLK + 1, 32)),
                                    nm_mean=[round(float(v), 3) for v in eff[(m, i)][0][::32]],
                                    cumg_mean=[round(float(v), 2) for v in eff[(m, i)][1][::32]])
                   for (m, i) in eff}, fh)
    return eff


def effT_lines(eff):
    L = []
    L.append("## 5. Effective time (iv): NM crossings, W0.5 and SF; per-block NM, mean over seeds 0-2\n")
    L.append("NM is recorded every block for this section (the 64-block grid of sections 1-2 is too coarse for SF, "
             "which is already below -15 dB at its first sample). Block index and cumulative g (sum g_i) at the first block "
             "where the seed-mean NM reaches the threshold. Spread = max/min over A, D1.45, D1.2. "
             "The g-axis collapse holds when the cumulative-g spread is clearly smaller than the block-index spread.\n")
    L.append("| method | threshold | input | block index | cum g | reached |")
    L.append("|---|---|---|---|---|---|")
    for mth in ["W0.5", "SF"]:
        for thr in [-10.0, -15.0, -20.0, -25.0]:
            bl, gl = {}, {}
            for inp in ["A", "D1.45", "D1.2"]:
                nm, cg = eff[(mth, inp)]
                idx = np.where(nm <= thr)[0]
                if len(idx) == 0:
                    L.append(f"| {mth} | {thr:.0f} dB | {inp} | - | - | no |")
                    continue
                j = int(idx[0])
                bl[inp] = j + 1
                gl[inp] = float(cg[j])
                L.append(f"| {mth} | {thr:.0f} dB | {inp} | {bl[inp]} | {fmt(gl[inp], 1)} | yes |")
            if len(bl) == 3:
                sb = max(bl.values()) / min(bl.values())
                sg = max(gl.values()) / min(gl.values())
                L.append(f"| {mth} | {thr:.0f} dB | spread max/min | {fmt(sb, 2)} | {fmt(sg, 2)} | |")
    L.append("")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    styles = {"W0.5": "-", "SF": "--"}
    colors = {"A": "C0", "D1.45": "C1", "D1.2": "C2"}
    for (mth, inp), (nm, cg) in eff.items():
        lab = f"{mth}, {inp}"
        xb = np.arange(1, NBLK + 1)[::32]
        axs[0].plot(xb, nm[::32], styles[mth], color=colors[inp], label=lab)
        axs[1].plot(cg[::32], nm[::32], styles[mth], color=colors[inp], label=lab)
    axs[0].set_xlabel("block index")
    axs[1].set_xlabel("cumulative g = sum g_i")
    axs[0].set_ylabel("NM [dB] (seed mean)")
    for ax in axs:
        ax.grid(alpha=0.3)
    axs[1].legend(fontsize=8)
    fig.suptitle("Normalised misalignment vs block index and vs cumulative gain (inclass.py)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "inclass_effT.png"), dpi=110)
    plt.close(fig)
    return L


def post():
    """Rebuild sections 3 and 5 from saved stability json and a fresh per-block effective-time run."""
    import glob
    stab_res = []
    for f in sorted(glob.glob(os.path.join(OUT, "inclass_stab_*_s*.json"))):
        with open(f, encoding="utf-8") as fh:
            stab_res.append(json.load(fh))
    s3 = stab_lines(stab_res)
    s5 = effT_lines(effT_fine())
    path = os.path.join(HERE, "RESULTS_inclass.md")
    text = open(path, encoding="utf-8").read()
    i3, i4, i5 = text.index("## 3. "), text.index("## 4. "), text.index("## 5. ")
    new = text[:i3] + "## 3. Stability (iii): max stable mu on the grid vs 1/g(rho)\n\n" + "\n".join(s3) + "\n" \
        + text[i4:i5] + "\n".join(s5) + "\n"
    open(path, "w", encoding="utf-8").write(new)
    print("[post] RESULTS_inclass.md sections 3 and 5 rebuilt", flush=True)



def main():
    os.makedirs(OUT, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()

    # phase 1: W0 mu by the specfac rule (six white sources, whole mean, no divergence)
    with Pool(processes=2) as pool:
        sel = list(pool.imap_unordered(w0_select, D.SIX, chunksize=1))
    per_mu = {mu: [] for mu in SF.MUS}
    for tag, rows in sel:
        for r in rows:
            per_mu[r["mu"]].append((r["whole"], r["div"]))
    ok = {mu: float(np.mean([v for v, _ in vs])) for mu, vs in per_mu.items() if not any(d for _, d in vs)}
    mu_w0 = min(ok, key=ok.get)
    with open(os.path.join(OUT, "inclass_w0_select.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(mu=mu_w0, six_white_whole={str(k): v for k, v in ok.items()}), fh, indent=1)
    print(f"[w0] chosen mu {mu_w0}", flush=True)

    # phase 2: main runs (3 seeds) and stability (6 signals x 2 seeds)
    tasks = [("main", s, mu_w0) for s in SEEDS] + [("stab", (sig, s)) for sig in STAB_SIGS for s in STAB_SEEDS]

    main_res, stab_res = {}, []
    with Pool(processes=2) as pool:
        for kind, r in pool.imap_unordered(dispatch, tasks, chunksize=1):
            if kind == "main":
                main_res[r["seed"]] = r
            else:
                stab_res.append(r)
    runtime = time.time() - t0
    print(f"[total] {runtime:.1f} s", flush=True)
    with open(os.path.join(OUT, "inclass_stab_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(stab_res, fh, default=float)

    # ---------------- Table 1
    L = []
    L.append("# Scale-varying inputs: checks of (i)-(iv) (atsip/explore/inclass.py)\n")
    L.append("Numbers only. Generated by `atsip/explore/inclass.py`. E_M framing, divergence as specfac "
             "(non-finite or end ANR > 0 dB; NaN in a mean if any seed diverged). Means over seeds 0-2 of dB values. "
             f"Runtime (2 workers): {runtime:.0f} s. alpha_E = {ALPHA_E} (Srr EMA, oracle.py). "
             f"W0 mu = {mu_w0} (specfac rule on the six white sources; the other mu are from RESULTS_specfac.md).\n")
    c = main_res[0]["check"]
    L.append("Checks: final-weight max |diff| of run_tracked vs the reused simulators on a145, mu as in the table: "
             + ", ".join(f"{k} {v:.1e}" for k, v in c["w_maxabs"].items()) + ". "
             "Whole-curve L1 vs `specfac_grid.csv`: " + ", ".join(
                 f"{k} {fmt(a, 3)} vs {fmt(b, 3)}" for k, (a, b) in c["csv_whole"].items()) + ".\n")

    L.append("## 1. Main table. L1 = ANR of oracle.py, L2 = 10log10(sum e^2 / sum d^2), same windows\n")
    L.append("| input | method | mu | L1 whole | L1 first5 | L1 last10 | L2 whole | L2 first5 | L2 last10 | final NM | #div (of 3) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    table = {}
    for inp in INPUTS:
        for mth in METHODS:
            recs = [main_res[s]["inputs"][inp]["rec"][mth] for s in SEEDS]
            divs = [r["div"] for r in recs]
            row = {}
            for k1 in ["L1", "L2"]:
                for k2 in ["whole", "first5", "last10"]:
                    row[f"{k1}_{k2}"] = agg([r[k1][k2] for r in recs], divs)
            row["final_nm"] = agg([r["final_nm"] for r in recs], divs)
            row["ndiv"] = int(sum(divs))
            table[(inp, mth)] = row
            L.append(f"| {inp} | {mth} | {recs[0]['mu']} | " + " | ".join(
                fmt(row[k]) for k in ["L1_whole", "L1_first5", "L1_last10", "L2_whole", "L2_first5", "L2_last10",
                                      "final_nm"]) + f" | {row['ndiv']} |")
    L.append("")

    # rankings per input
    L.append("Method order per input (best first, nan excluded): L1 whole, then L2 whole.\n")
    for inp in INPUTS:
        for key in ["L1_whole", "L2_whole"]:
            vals = {m: table[(inp, m)][key] for m in METHODS if np.isfinite(table[(inp, m)][key])}
            order = sorted(vals, key=vals.get)
            L.append(f"- {inp} {key}: " + " < ".join(f"{m} ({fmt(vals[m])})" for m in order))
    L.append("")

    # ---------------- Table 2: fixed points
    L.append("## 2. Fixed-point invariance (ii) and causal Wiener invariance (i)\n")
    L.append("Final-weight distance 10log10(||w_B/C - w_A||^2 / ||w_A||^2) for the same seed and method, mean over seeds. "
             "Baseline: A runs of different seeds, the same formula over the three seed pairs (mean of dB values). "
             "More negative is closer.\n")
    L.append("| method | B vs A | C vs A | A vs A (seed baseline) |")
    L.append("|---|---|---|---|")
    for mth in METHODS:
        vb = [main_res[s]["fp"][f"B|{mth}"] for s in SEEDS]
        vc = [main_res[s]["fp"][f"C|{mth}"] for s in SEEDS]
        pairs = [(0, 1), (0, 2), (1, 2)]
        base = []
        for a, b in pairs:
            wa = main_res[a]["wfin"][("A", mth)]
            wb = main_res[b]["wfin"][("A", mth)]
            base.append(wrel_db(wb, wa))
        L.append(f"| {mth} | {fmt(np.mean(vb) if np.all(np.isfinite(vb)) else float('nan'))} | "
                 f"{fmt(np.mean(vc) if np.all(np.isfinite(vc)) else float('nan'))} | {fmt(np.mean(base))} |")
    L.append("")
    dgn = {k: [main_res[s]["diag"][k] for s in SEEDS] for k in main_res[0]["diag"]}
    L.append("Causal Wiener solution (i), distance 10log10(||w_opt(X) - w_opt(A)||^2/||w_opt(A)||^2), same seed, mean over seeds:\n")
    L.append(f"- 10 s solver (the existing one, first 10 s): B {fmt(np.mean(dgn['wopt10|B']))} dB, "
             f"C {fmt(np.mean(dgn['wopt10|C']))} dB. (B's steps lie after 10 s, so B's window is identical to A's.)")
    L.append(f"- full 30 s solver (diagnostic): B {fmt(np.mean(dgn['woptfull|B']))} dB, "
             f"C {fmt(np.mean(dgn['woptfull|C']))} dB.\n")

    L.append("## 3. Stability (iii): max stable mu on the grid vs 1/g(rho)\n")
    L += stab_lines(stab_res)
    # ---------------- Table 4: E[g^2]/E[g]
    L.append("## 4. Gain statistics g_i = block power / EMA power, and final NM (mean over seeds)\n")
    L.append("| input | E[g] | E[g^2]/E[g] (per seed min-max) | final NM W0.5 | final NM SF | final NM B0 |")
    L.append("|---|---|---|---|---|---|")
    for inp in INPUTS:
        eg = [main_res[s]["inputs"][inp]["Eg"] for s in SEEDS]
        r2 = [main_res[s]["inputs"][inp]["Eg2_over_Eg"] for s in SEEDS]
        L.append(f"| {inp} | {fmt(np.mean(eg), 3)} | {fmt(np.mean(r2), 3)} ({fmt(min(r2), 3)}-{fmt(max(r2), 3)}) | "
                 f"{fmt(table[(inp, 'W0.5')]['final_nm'])} | {fmt(table[(inp, 'SF')]['final_nm'])} | "
                 f"{fmt(table[(inp, 'B0')]['final_nm'])} |")
    L.append("")

    L += effT_lines(effT_fine())

    # ---------------- write
    with open(os.path.join(HERE, "RESULTS_inclass.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("[written] RESULTS_inclass.md", flush=True)


def run_check():
    print(json.dumps(check_refs(), indent=1, default=float), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "check":
        run_check()
    elif mode == "post":
        post()
    else:
        main()
