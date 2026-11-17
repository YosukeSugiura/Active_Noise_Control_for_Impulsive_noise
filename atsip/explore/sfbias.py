#!/usr/bin/env python3
"""Why does the spectral-factor (SF) normaliser keep a causal-leak bias on a145?

Stage 1: linearised mean fixed point (fp.py conventions, source a145) for the variants in
         sfbias_lib.fp_specs(): excess of the fixed-point error power over J_o (dB, more negative is better)
         and the L1 fixed-point ANR. Reference rows: geometric a=0 and the Rall base (= fp.py ER).
Stage 2: adaptive SF loop (no ramp, mu 0.1 and 0.05) on a145, a185, a120_s0 for the current SF
         (specfac.sim_sf, eps_reg 0) and the two fixed-point variants with the lowest excess.

Usage: python3 atsip/explore/sfbias.py            (writes out/sfbias_*.json, out/sfbias_adapt.csv)
       python3 atsip/explore/sfbias.py fp         (stage 1 only)
Single process, memory well under 1.5 GB.
"""
import os
import sys
import csv
import json
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dissect as D  # noqa: E402
import explore as X  # noqa: E402
import mixlam as ML  # noqa: E402
import specfac as SF  # noqa: E402
import sfbias_lib as SB  # noqa: E402

OUT = os.path.join(HERE, "out")
ADAPT_SRC = ["a145", "a185", "a120_s0"]
ADAPT_MUS = [0.1, 0.05]


def check():
    """Operator equality with specfac.sf_dir, and base fixed point = fp.py (-17.2 dB on a145)."""
    rng = np.random.default_rng(0)
    R = np.fft.rfft(rng.standard_normal(N_ := SB.N))
    E = np.fft.rfft(rng.standard_normal(N_))
    Srr = np.abs(rng.standard_normal(SB.NB)) ** 2 + 0.1
    g1 = SF.sf_dir(R, E, Srr, Srr.mean(), 1e-6, 0.0, N_ // 2)
    F = SB.minphase_bins(Srr + 1e-6)
    q = np.fft.irfft(np.conj(R) * E / np.conj(F), N_)
    q[N_ // 2:] = 0
    g2 = np.fft.irfft(np.fft.rfft(q) / F, N_)[:SB.M]
    return dict(operator_max_diff=float(np.max(np.abs(g1 - g2))))


def stage_fp(log):
    data = D.load_data(*D.KIND["a145"])
    t0 = time.time()
    specs = SB.fp_specs()
    res = SB.fixed_point_pass(data, specs)
    log(f"[fp] a145 pass {time.time() - t0:.1f} s")
    return res


def stage_adapt(fp_res, log):
    refs = ("geometric", "base", "control", "diag")
    geo = [r for r in fp_res if r["name"].startswith("geometric")][0]["excess_db"]
    base = [r for r in fp_res if r["name"].startswith("base")][0]["excess_db"]
    # "removes most of the bias": at least half of the gap between the Rall base and the geometric a=0 reference
    thr = base - 0.5 * (base - geo)
    cand = [r for r in fp_res if not r["name"].startswith(refs) and r["excess_db"] <= thr]
    cand.sort(key=lambda r: r["excess_db"])
    names = [s["name"] for s in SB.fp_specs()]
    pick = [SB.fp_specs()[names.index(r["name"])] for r in cand[:2]]
    log(f"[adapt] threshold {thr:.2f} dB, qualifying variants: {len(cand)}")
    log("[adapt] variants: " + ", ".join(p["name"] for p in pick))
    rows = []
    for tag in ADAPT_SRC:
        kind, arg = D.KIND[tag]
        data = D.load_data(kind, arg)
        Rall = ML.prepass(data)["Rall"]
        for mu in ADAPT_MUS:
            cur = SF.sim_sf(data, Rall, 0.0, 0, 1, mu)
            m = X.metrics(cur["e"], cur["w"], data["d"], data["w_opt"])
            rows.append(dict(src=tag, mu=mu, row="current SF (eps_reg 0)", **_pick(m)))
            for spec in pick:
                sim = SB.sim_variant(data, Rall, spec, mu)
                m = X.metrics(sim["e"], sim["w"], data["d"], data["w_opt"])
                rows.append(dict(src=tag, mu=mu, row=spec["name"], **_pick(m)))
            log(f"[adapt] {tag} mu {mu} done")
    return pick, rows


def _pick(m):
    div = (not np.isfinite(m["whole"])) or m["last10"] > -5.0
    f = (lambda v: float("nan") if div else float(v))
    return dict(whole=f(m["whole"]), first5=f(m["first5"]), last10=f(m["last10"]), diverged=int(div))


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    os.makedirs(OUT, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"

    def log(msg):
        print(msg, flush=True)

    chk = check()
    log(f"[check] {chk}")
    fp_res = stage_fp(log)
    for r in fp_res:
        log(f"[fp] {r['name']:<48} excess {r['excess_db']:7.2f} dB  L1 {r['L1_db']:7.2f}  floor {r['L1_floor_db']:7.2f}")
    with open(os.path.join(OUT, "sfbias_fp_a145.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(check=chk, rows=fp_res), fh, indent=1, default=float)
    if stage == "fp":
        return
    pick, rows = stage_adapt(fp_res, log)
    with open(os.path.join(OUT, "sfbias_adapt.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "mu", "row", "whole_db", "first5_db", "last10_db", "diverged"])
        for r in rows:
            w.writerow([r["src"], r["mu"], r["row"], r["whole"], r["first5"], r["last10"], r["diverged"]])
    with open(os.path.join(OUT, "sfbias_adapt.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(picked=[p["name"] for p in pick], rows=rows), fh, indent=1, default=float)
    log("[done]")


if __name__ == "__main__":
    main()
