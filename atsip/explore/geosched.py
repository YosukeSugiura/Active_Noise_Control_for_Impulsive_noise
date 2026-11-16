#!/usr/bin/env python3
"""Geometric schedules of the normaliser, and an exponent fit of rho_irr(k).

Exp 1 (manual schedule). Lambda_i(k) = (1-w_i) Srr(k) + w_i Srr(k)^a Pbar^(1-a) + eps, with
  w_i = min(1, t_i / T), t_i = i B / fs, and mu_i = mu (1 + (m0-1)(1-w_i)).
  Grid a in {0.5, 0.75}, T in {1, 2, 4, 8} s, m0 in {1, 2, 3}; mu from the dissect grid, chosen per
  config by the six-source white whole mean. Baselines: B0, T3 (dissect simulate), W0.5, W0.75 (w = 1).
  Sanity rows: a=0.5 and a=1.0 with T=0 (w = 1 throughout) must reproduce W0.5 and B0 exactly.

Exp 2 (exponent fit). Per block i and bin k: rho_i(k) = |E_opt,i(k)|^2 / Srr_i(k), s_i(k) = Srr_i(k)/Pbar_i.
  Pooled over bins 1..511 and the second half of the blocks (after the Srr EMA has settled).
  OLS and Theil-Sen of log rho on log s per source, with R^2. Theory: slope = -(1-a).

Sources and helpers are those of dissect.py (ten sources: six white, two alpha=1.2, two colored).
Divergence (task rule): non-finite output, or end ANR (mean of the last 10%) > 0 dB. Diverged runs are NaN.

Usage:  python3 atsip/explore/geosched.py
Writes  atsip/explore/out/geosched_*.{csv,json} and atsip/explore/RESULTS_geosched.md.
"""
import os
import sys
import csv
import json
import time
import warnings
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal, stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dissect as D  # noqa: E402
import explore as X  # noqa: E402
import mixlam as ML  # noqa: E402

FS, M, N, B, NB = D.FS, D.M, D.N, D.B, D.NB
OUT = D.OUT
MUS = D.MUS
SIX, TWO, COL, ALLTAGS, KIND = D.SIX, D.TWO, D.COL, D.ALLTAGS, D.KIND
A_LIST, T_LIST, M0_LIST = [0.5, 0.75], [1, 2, 4, 8], [1, 2, 3]
CONFIGS = [(a, T, m0) for a in A_LIST for T in T_LIST for m0 in M0_LIST]
SANITY = [(0.5, 0, 1), (1.0, 0, 1)]          # T = 0 means w = 1 throughout
BASES = ["B0", "T3", "W0.5", "W0.75"]
TOL_DB = 1.0                                  # "close" = within 1 dB (my operationalisation)


def slab(a, T, m0):
    return f"S_a{a}_T{T}_m{m0}"


# ----------------------------------------------------------------------------
# Experiment 1: scheduled simulation (same loop as dissect.simulate)
# ----------------------------------------------------------------------------
def sim_sched(data, Rall, a, T, m0, mu):
    d, xq, hS, S2 = data["d"], data["xq"], data["hS"], data["S2"]
    L = len(d)
    nb = L // B
    swv = sliding_window_view(xq, M)
    w = np.zeros(M)
    zi = np.zeros(len(hS) - 1)
    e_all = np.zeros(L)
    shared = X.Shared(S2)
    diverged = False
    for i in range(nb):
        s0 = i * B
        Xb = swv[s0 + N - M + 1: s0 + N - M + 1 + B][:, ::-1]
        y = Xb @ w
        ys, zi = signal.lfilter(hS, [1.0], y, zi=zi)
        e_all[s0:s0+ B] = d[s0:s0 + B] + ys
        E = ML.frame_E(e_all, s0, M)
        s = shared.step(Rall[i], E)
        Srr, Pbar, eps = s["Srr"], s["Pbar"], s["eps"]
        wr = 1.0 if T == 0 else min(1.0, (i * B / FS) / T)
        Lam = (1 - wr) * Srr + wr * Srr ** a * Pbar ** (1 - a) + eps
        mu_i = mu * (1 + (m0 - 1) * (1 - wr))
        g = np.fft.irfft(np.conj(Rall[i]) * E / Lam, N)[:M]
        w = w - 2 * mu_i * B / M * g
        if not np.isfinite(w).all():
            e_all[s0 + B:] = np.nan
            diverged = True
            break
    return dict(e=e_all, w=w, diverged=diverged)


def one_run(data, Rall, cfg, mu):
    """cfg: ('S', a, T, m0) scheduled, or ('D', label) dissect simulate. Returns the metric row."""
    if cfg[0] == "S":
        _, a, T, m0 = cfg
        lab = slab(a, T, m0)
        sim = sim_sched(data, Rall, a, T, m0, mu)
    else:
        lab = cfg[1]
        sim = D.simulate(data, Rall, D.spec_of(lab), mu, M)
    d, w_opt = data["d"], data["w_opt"]
    m = X.metrics(sim["e"], sim["w"], d, w_opt)
    w_fin = bool(np.isfinite(sim["w"]).all())
    # task rule: non-finite output or end ANR > 0 dB
    div_task = (not w_fin) or (not np.isfinite(m["last10"])) or (not np.isfinite(m["whole"])) \
        or m["last10"] > 0.0
    keys = ["whole", "first5", "last10", "msd"]
    vals = {k: (float("nan") if div_task else float(m[k])) for k in keys}
    return dict(label=lab, mu=mu, whole=vals["whole"], first5=vals["first5"], last10=vals["last10"],
                msd=vals["msd"], diverged=int(div_task), xrule_div=int(m["diverged"]),
                raw_whole=float(m["whole"]) if np.isfinite(m["whole"]) else float("nan"))


# ----------------------------------------------------------------------------
# Experiment 2: per-block rho_irr and s, pooled after settling
# ----------------------------------------------------------------------------
def exp2_source(data, Rall):
    x, d, hS, w_opt = data["x"], data["d"], data["hS"], data["w_opt"]
    L = len(d)
    nb = L // B
    e_opt = d + signal.lfilter(hS, [1.0], np.convolve(x, w_opt)[:L])
    shared = X.Shared(data["S2"])
    lr, ls, la = [], [], []
    sel = slice(1, NB - 1)
    for i in range(nb):
        s0 = i * B
        E = ML.frame_E(e_opt, s0, M)
        s = shared.step(Rall[i], E)
        if i < nb // 2:
            continue
        Srr = np.maximum(s["Srr"], 1e-30)[sel]
        A = np.abs(E)[sel] ** 2
        lr.append(np.log(A / Srr))
        ls.append(np.log(Srr / s["Pbar"]))
        la.append(np.log(A))
    lr, ls, la = np.concatenate(lr), np.concatenate(ls), np.concatenate(la)
    ok = np.isfinite(lr) & np.isfinite(ls) & np.isfinite(la)
    lr, ls, la = lr[ok], ls[ok], la[ok]
    ols = stats.linregress(ls, lr)
    rng = np.random.default_rng(3003)
    idx = rng.choice(len(ls), size=min(3000, len(ls)), replace=False)   # 3000: Theil-Sen forms O(n^2) slopes
    ts = stats.theilslopes(lr[idx], ls[idx])
    diag = stats.linregress(ls, la)           # slope of log|E|^2 on log s (= OLS slope + 1)
    return dict(n=int(len(ls)), ols=float(ols.slope), r2=float(ols.rvalue ** 2),
                ts=float(ts[0]), ts_lo=float(ts[2]), ts_hi=float(ts[3]), diag=float(diag.slope))


# ----------------------------------------------------------------------------
# Worker: one source runs every Exp 1 config over the mu grid, and the Exp 2 fit
# ----------------------------------------------------------------------------
def unit(tag):
    t0 = time.time()
    kind, arg = KIND[tag]
    data = D.load_data(kind, arg)
    Rall = ML.prepass(data)["Rall"]
    rows = []
    cfgs = [("S", a, T, m0) for (a, T, m0) in CONFIGS] + [("S", a, 0, m0) for (a, _T, m0) in SANITY] \
        + [("D", lab) for lab in BASES]
    for cfg in cfgs:
        for mu in MUS:
            rows.append(dict(src=tag, **one_run(data, Rall, cfg, mu)))
        print(f"[cfg] {tag} {cfg} ", flush=True)
    e2 = exp2_source(data, Rall)
    return dict(tag=tag, rows=rows, exp2=e2, runtime=time.time() - t0)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    res = []
    with Pool(processes=2) as pool:
        for rr in pool.imap_unordered(unit, ALLTAGS, chunksize=1):
            res.append(rr)
            print(f"[done] {rr['tag']} {rr['runtime']:.1f} s ({len(res)}/{len(ALLTAGS)})", flush=True)
            with open(os.path.join(OUT, f"geosched_src_{rr['tag']}.json"), "w") as fh:
                json.dump(dict(rows=rr["rows"], exp2=rr["exp2"]), fh, default=float)
    runtime = time.time() - t0

    rows = [r for rr in res for r in rr["rows"]]
    exp2 = {rr["tag"]: rr["exp2"] for rr in res}
    cell = {(r["label"], r["mu"], r["src"]): r for r in rows}
    labels = [slab(a, T, m0) for (a, T, m0) in CONFIGS] + [slab(a, 0, m0) for (a, _T, m0) in SANITY] + BASES

    def cmean(lab, mu, srcs, key):
        return D.nmean([cell[(lab, mu, t)][key] for t in srcs])

    def ndiv(lab, mu, srcs):
        return int(sum(cell[(lab, mu, t)]["diverged"] for t in srcs))

    def best_mu(lab):
        ok = {mu: cmean(lab, mu, SIX, "whole") for mu in MUS}
        ok = {mu: v for mu, v in ok.items() if np.isfinite(v)}
        return min(ok, key=ok.get) if ok else None

    chosen = {lab: best_mu(lab) for lab in labels}

    # sanity: scheduled code with w = 1 vs dissect W(a) and B0
    def maxdiff(lab_a, lab_b):
        worst = 0.0
        for t in ALLTAGS:
            for mu in MUS:
                x_, y_ = cell[(lab_a, mu, t)]["raw_whole"], cell[(lab_b, mu, t)]["raw_whole"]
                if np.isfinite(x_) and np.isfinite(y_):
                    worst = max(worst, abs(x_ - y_))
        return worst
    chk_w05 = maxdiff(slab(0.5, 0, 1), "W0.5")
    chk_b0 = maxdiff(slab(1.0, 0, 1), "B0")

    grid_path = os.path.join(OUT, "geosched_grid.csv")
    with open(grid_path, "w", newline="") as fh:
        wr_ = csv.writer(fh)
        wr_.writerow(["label", "source", "mu", "whole_db", "first5s_db", "last10_db", "final_msd_db",
                      "diverged_task", "diverged_xrule(-5dB)"])
        for r in sorted(rows, key=lambda r: (r["label"], r["src"], r["mu"])):
            wr_.writerow([r["label"], r["src"], r["mu"], r["whole"], r["first5"], r["last10"], r["msd"],
                          r["diverged"], r["xrule_div"]])
    with open(os.path.join(OUT, "geosched_exp2.csv"), "w", newline="") as fh:
        wr_ = csv.writer(fh)
        wr_.writerow(["source", "n_points", "ols_slope", "r2", "theilsen_slope", "ts_ci_lo", "ts_ci_hi",
                      "diag_slope_logE2_on_logs"])
        for t in ALLTAGS:
            e = exp2[t]
            wr_.writerow([t, e["n"], e["ols"], e["r2"], e["ts"], e["ts_lo"], e["ts_hi"], e["diag"]])

    # ---------------- report
    F = X.fmt
    L = []
    L.append("# Geometric schedules and the exponent fit (atsip/explore/geosched.py)\n")
    L.append("Numbers only. Generated by `atsip/explore/geosched.py`.\n")
    L.append(f"Runtime (wall clock, 2 worker processes): {runtime:.1f} s. Ten sources as in dissect.py "
             "(six white alpha=1.45 sets, two alpha=1.2, two colored); not expanded to ten per condition.\n")
    L.append("Framing E_M throughout. Means over sources; nan if any run in the set diverged. "
             f"Divergence (task rule): non-finite output or end ANR > 0 dB (last 10%). "
             "The dissect rule (last10 > -5 dB) is also recorded in `out/geosched_grid.csv`.\n")
    L.append("Sanity: scheduled code with T=0 (w=1) vs dissect W0.5 and B0, max |diff| of whole-curve ANR over all "
             f"sources and mu: {chk_w05:.2e} dB and {chk_b0:.2e} dB.\n")

    L.append("## Experiment 1. Manual schedule, mu chosen per row by six-source white whole mean\n")
    L.append("Column legend: white = six alpha=1.45 sources (mean); a1.2 and colored = two-source means, whole curve.\n")
    L.append("| a | T [s] | m0 | mu | white whole | white first5 | white last10 | a1.2 whole | colored whole | "
             "#div at mu (of 10) | #div sweep (of 60) | first5 <= B0+1 | abs(last10 - W(a)) <= 1 | pass |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    b0_mu = chosen["B0"]
    b0_first5 = cmean("B0", b0_mu, SIX, "first5") if b0_mu is not None else float("nan")
    for (a, T, m0) in CONFIGS:
        lab = slab(a, T, m0)
        mu = chosen[lab]
        wa_mu = chosen[f"W{a}"]
        wa_last10 = cmean(f"W{a}", wa_mu, SIX, "last10") if wa_mu is not None else float("nan")
        if mu is None:
            L.append(f"| {a} | {T} | {m0} | nan | nan | nan | nan | nan | nan | nan | "
                     f"{sum(ndiv(lab, m, ALLTAGS) for m in MUS)} | nan | nan | no |")
            continue
        f5 = cmean(lab, mu, SIX, "first5")
        l10 = cmean(lab, mu, SIX, "last10")
        ok1 = bool(np.isfinite(f5) and np.isfinite(b0_first5) and f5 <= b0_first5 + TOL_DB)
        ok2 = bool(np.isfinite(l10) and np.isfinite(wa_last10) and abs(l10 - wa_last10) <= TOL_DB)
        L.append(f"| {a} | {T} | {m0} | {mu} | {F(cmean(lab, mu, SIX, 'whole'))} | {F(f5)} | {F(l10)} | "
                 f"{F(cmean(lab, mu, TWO, 'whole'))} | {F(cmean(lab, mu, COL, 'whole'))} | "
                 f"{ndiv(lab, mu, ALLTAGS)} | {sum(ndiv(lab, m, ALLTAGS) for m in MUS)} | "
                 f"{'yes' if ok1 else 'no'} | {'yes' if ok2 else 'no'} | {'yes' if (ok1 and ok2) else 'no'} |")
    L.append("")
    L.append("Reference points (each at its own white-selected mu):\n")
    L.append("| row | mu | white whole | white first5 | white last10 | a1.2 whole | colored whole | #div at mu (of 10) | #div sweep (of 60) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for lab in BASES:
        mu = chosen[lab]
        if mu is None:
            L.append(f"| {lab} | nan | nan | nan | nan | nan | nan | nan | nan |")
            continue
        L.append(f"| {lab} | {mu} | {F(cmean(lab, mu, SIX, 'whole'))} | {F(cmean(lab, mu, SIX, 'first5'))} | "
                 f"{F(cmean(lab, mu, SIX, 'last10'))} | {F(cmean(lab, mu, TWO, 'whole'))} | "
                 f"{F(cmean(lab, mu, COL, 'whole'))} | {ndiv(lab, mu, ALLTAGS)} | "
                 f"{sum(ndiv(lab, m, ALLTAGS) for m in MUS)} |")
    L.append("")
    L.append("Success criterion (stated before the run): first5 of the row close to B0's (within 1 dB, or better) "
             "AND last10 close to W(a)'s (within 1 dB). B0 and W(a) are taken at their own white-selected mu.\n")

    L.append("## Experiment 2. Exponent fit, log rho_irr(k) on log s(k), bins 1..511, second half of blocks\n")
    L.append("Theory: slope = -(1-a), so implied a = 1 + slope. Diagnostic column: OLS slope of log|E_opt|^2 on log s "
             "(= OLS slope + 1; equals a under the theory). Caveat: rho = |E|^2/Srr and s = Srr/Pbar share Srr, "
             "so a slope near -1 can arise from the shared term alone.\n")
    L.append("| source | group | n points | OLS slope | R^2 | Theil-Sen slope [95% CI] | implied a (OLS) | implied a (TS) | diag slope log|E|^2 on log s |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    grp = {**{t: "white" for t in SIX}, **{t: "a1.2" for t in TWO}, **{t: "colored" for t in COL}}
    for t in ALLTAGS:
        e = exp2[t]
        L.append(f"| {t} | {grp[t]} | {e['n']} | {F(e['ols'], 3)} | {F(e['r2'], 3)} | "
                 f"{F(e['ts'], 3)} [{F(e['ts_lo'], 3)}, {F(e['ts_hi'], 3)}] | {F(1 + e['ols'], 3)} | "
                 f"{F(1 + e['ts'], 3)} | {F(e['diag'], 3)} |")
    L.append("")
    for g, srcs in [("white", SIX), ("a1.2", TWO), ("colored", COL)]:
        ols_m = float(np.mean([exp2[t]["ols"] for t in srcs]))
        ts_m = float(np.mean([exp2[t]["ts"] for t in srcs]))
        diag_m = float(np.mean([exp2[t]["diag"] for t in srcs]))
        L.append(f"- {g} mean: OLS slope {F(ols_m, 3)} (implied a {F(1 + ols_m, 3)}), Theil-Sen {F(ts_m, 3)} "
                 f"(implied a {F(1 + ts_m, 3)}), diagnostic {F(diag_m, 3)}.")
    L.append("")
    L.append("## Files\n")
    L.append("- `out/geosched_grid.csv`: every run (label, source, mu), both divergence rules.")
    L.append("- `out/geosched_exp2.csv`: per-source regression numbers.")
    L.append("")
    with open(os.path.join(HERE, "RESULTS_geosched.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    summ = dict(runtime_s=runtime, chosen_mu=chosen, sanity=dict(w05=chk_w05, b0=chk_b0), exp2=exp2)
    with open(os.path.join(OUT, "geosched_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1, default=float)
    print(f"[total] {runtime:.1f} s", flush=True)


if __name__ == "__main__":
    main()
