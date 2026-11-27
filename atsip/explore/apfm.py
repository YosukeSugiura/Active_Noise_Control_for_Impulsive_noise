"""Single-channel APFM and VSS-NFBFxLMS, Yin et al. (2023), eqs. 10–15."""
import argparse
from collections import defaultdict
import itertools
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np
from scipy.signal import lfilter

import coef as C
import final as F
import final2
from paired_ci import paired

OUT = Path("out")
BETA = .95
ETA = 1e-6
CAP = 2
SCEN = ("stat", "RS-near", "RS-far", "EN0", "ENm10")
SWITCH = SCEN[1:3]
METHODS = ("APFM", "VSS-NFBFxLMS")
COH = "Freq_FxNLMS_CohVSS"
REFS = (F.PROPOSED, "nors-cap", COH)
INITIAL = tuple(range(-8, 2))
WINDOWS = (.5, 1, 2)


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    F.ab.atomic_write(path, lambda f: json.dump(result, f, allow_nan=False))


def load(path):
    return json.loads(path.read_text())


def simulate(dd, exponent, bank=None):
    start = time.perf_counter()
    b, m, n = F.B, F.M, F.N
    x, d, r = (dd[k] for k in ("x", "d", "r"))
    (_, s), = dd["hS_seq"]
    (_, sh), = dd["hSm_seq"]
    xpad, rpad = (np.pad(v, (n, 0)) for v in (x, r))
    e = np.empty_like(d)
    zs, zh = np.zeros(len(s)-1), np.zeros(len(sh)-1)
    w = np.zeros(m)
    zero = w.copy()
    delta = w.copy()
    lam = np.ones(n//2+1)
    px = pe = 0.
    powers, choices, snapshots = [], [], {}
    fail = None
    marks = {F.CB+k:k for k in C.MAIN_KS} if dd["changes"] else {}
    for block, i in enumerate(range(0, len(d), b)):
        if block in marks:
            snapshots[marks[block]] = w.copy()
        xb = xpad[i+b:i+b+n]
        rb = rpad[i+b:i+b+n]
        y = np.convolve(xb[-(m+b-1):], w, "valid")
        actual, zs = lfilter(s, [1.], y, zi=zs)
        eb = d[i:i+b] + actual
        e[i:i+b] = eb
        if F.diverged(eb, dd["primary_power"]):
            fail = i+b-1
            break

        ef = eb
        base = w
        if bank is not None:
            predicted, zh = lfilter(sh, [1.], y, zi=zh)
            primary = eb-predicted
            candidates = (bank, w, zero)
            errors = (primary + np.convolve(rb[-(m+b-1):], bank, "valid"),
                      primary + np.convolve(rb[-(m+b-1):], w, "valid"), primary)
            pick = int(np.argmin([v @ v for v in errors]))
            choices.append(pick)
            base, ef = candidates[pick], errors[pick]
            if pick == 2:
                delta *= BETA
                lam.fill(1.)
                px = pe = 0.
                w = base + delta
                continue

        X = np.fft.rfft(xb)
        R = np.fft.rfft(rb)
        spec = np.fft.rfft(np.pad(ef, (n-b, 0)))
        lam = BETA*lam + (1-BETA)*np.abs(X)**2
        px = BETA*px + (1-BETA)*(xb[-b:] @ xb[-b:])
        pe = BETA*pe + (1-BETA)*(eb @ eb)
        powers.append(px)
        mu = min(2.**exponent / np.sqrt(px+pe+ETA), CAP)
        delta = -mu*np.fft.irfft(R.conj()*spec/lam, n=n)[:m]
        candidate = base + delta
        if not np.isfinite(candidate).all():
            fail = i+b-1
            break
        w = candidate
    if fail is not None:
        e[i:] = np.nan
    failure = None if fail is None else dict(F.error_peak(e, i), sample=fail)
    return dict(e=e, w=w, div=fail is not None, n_done=len(e) if fail is None else i,
                failure=failure, seconds=time.perf_counter()-start,
                choices=choices, px_median=float(np.median(powers)) if powers else None,
                snapshots=snapshots)


def train(case):
    k, exponent = case
    path = OUT / "apfm_bank" / f"{exponent}_{k}.json"
    if path.exists():
        return
    pair = F.load_pair(k)
    rng = np.random.default_rng(900+k)
    x = rng.standard_normal(30*F.FS)
    p,s,sh = F.paths_of("stat",pair)[0]
    sh = sh[:F.LS_FINAL]
    d = lfilter(p,[1.],x)
    dd = dict(x=x,d=d,r=lfilter(sh,[1.],x),hS_seq=[(0,s)],hSm_seq=[(0,sh)],
              changes=[],primary_power=float(np.mean(d*d)))
    sim = simulate(dd,exponent)
    save(path,dict(w=sim["w"].tolist(), div=sim["div"], px_median=sim["px_median"]))


def empirical(sim, dd):
    if not dd["changes"]:
        return {}
    t = dd["changes"][0]*F.B
    return {f"power_{k:g}": F.ratio_db(sim["e"],dd["d"],t,t+int(k*F.FS))
            for k in WINDOWS if sim["n_done"] * (F.B if "nb" in sim else 1) >= t+int(k*F.FS)}


def evaluate(case):
    cls, scen, seed, exponent, method, stage = case
    path = OUT / f"apfm_{stage}" / f"{method}_{exponent}_{cls}_{scen}_s{seed}.json"
    if path.exists():
        return
    dd, ra, st = F.build(cls,scen,seed,F.LS_FINAL,final=stage != "tune")
    row = dict(cls=cls,scen=scen,seed=seed,method=method,exponent=exponent)
    if method in REFS:
        if method == COH:
            setting = load(Path("TUNE.json"))["chosen"][COH]
            sim = F.run_baseline(COH,setting["mu"],setting["params"],dd)
        else:
            x4 = F.p.parse(["c5","--preset","final"]).X4
            sim = F.run_rule(method,dd,ra,st,x4)[1]
        save(path,dict(**row,**empirical(sim,dd)))
        return
    bank = None
    if method == METHODS[0]:
        bank = np.asarray(load(OUT/"apfm_bank"/f"{exponent}_{seed%8+1}.json")["w"])
    sim = simulate(dd,exponent,bank)
    post = None
    if scen in SWITCH and not sim["div"]:
        post = simulate(F.build(cls,"post:"+scen,seed,F.LS_FINAL,final=stage != "tune")[0],exponent,bank)
    row.update(F.measures(sim,dd,post),px_median=sim["px_median"])
    if stage != "tune":
        row.update(empirical(sim,dd))
        if scen in SWITCH:
            row["J"] = C.frozen_anr(dd,sim["snapshots"])
        if bank is not None:
            picks = np.array(sim["choices"],dtype=int)
            row["picks"] = np.bincount(picks,minlength=3).tolist()
            row["switches"] = int(np.count_nonzero(np.diff(picks)))
            row["not_current_post2"] = int(np.count_nonzero(picks[F.CB:F.CB+2*F.BPS] != 1))
    save(path,row)


def parallel(fn, cases, procs):
    with Pool(procs,initializer=final2.configure) as pool:
        for i,_ in enumerate(pool.imap_unordered(fn,cases),1):
            if i % 25 == 0:
                print(f"{fn.__name__}: {i} completed",flush=True)


def tune_rows(exponent):
    cases = itertools.product(F.PILOT_CLASSES, F.TUNE_SCEN, F.TUNE_SEEDS)
    return [load(OUT/"apfm_tune"/f"APFM_{exponent}_{c}_{s}_s{sd}.json") for c,s,sd in cases]


def choose(exponents):
    grid = {}
    for exponent in exponents:
        rows = tune_rows(exponent)
        div = sum(r["div"] for r in rows)
        l1 = F.class_mean(rows,"L1_whole")
        med,cens = F.recovery_summary([r for r in rows if r["scen"]=="RS-far"])
        grid[exponent] = dict(div=div,L1=l1,med=med,cens=cens)
    best = min(grid,key=lambda k:(grid[k]["div"], grid[k]["L1"] if grid[k]["L1"] is not None else float("inf")))
    return best,grid


def tune(procs, exponents):
    cases = itertools.product(F.PILOT_CLASSES,F.TUNE_SCEN,F.TUNE_SEEDS,exponents,(METHODS[0],),("tune",))
    parallel(evaluate,cases,procs)


def extend(procs):
    best, grid = choose(INITIAL)
    extra = {-8:(-10,-9),1:(2,3)}.get(best,())
    if extra:
        parallel(train,itertools.product(range(1,9),extra),procs)
        tune(procs,extra)
        best,grid = choose(sorted((*INITIAL,*extra)))
    save(OUT/"apfm_tune"/"chosen.json",dict(best=best,grid=grid))


def test():
    dd = F.build("G","RS-far",10,F.LS_FINAL)[0]
    sim = simulate(dd,-4)
    mixed = simulate(dd,-4,np.zeros(F.M))
    assert not sim["div"] and not mixed["div"]
    assert np.sum(np.bincount(mixed["choices"],minlength=3)) == len(dd["x"])//F.B
    assert all(w.shape == (512,) for w in sim["snapshots"].values())
    assert ETA / sim["px_median"] < .001
    print("Both updates, selector counts, coefficient snapshots and eta scale passed")


def references(folder, methods):
    result = {}
    for path in (OUT/folder).glob("*.json"):
        for r in load(path)["rows"]:
            if r["method"] in methods and r.get("variant","main") == "main":
                result[r["method"],r["scen"],r["cls"],r["seed"]] = r
    return result


def fmt(x):
    return "—" if x is None else f"{x:.3f}"


def mean(rows, key):
    values = list(F.seed_table(rows,key).values())
    return fmt(float(np.mean(values)) if values else None)


def km(rows):
    med, cens = F.recovery_summary(rows)
    return f"{int(med)*F.B/F.FS:.3f} ({cens})" if med.isdecimal() else f"{med} ({cens})"


def report():
    chosen = load(OUT/"apfm_tune"/"chosen.json")
    exponent = chosen["best"]
    saved = references("final2_run",REFS)
    fixed = references("coef2_run",REFS)
    groups = defaultdict(list)
    for m, s, c, seed in itertools.product((*METHODS,*REFS),SCEN,F.FINAL_CLASSES,F.FINAL_SEEDS):
        case = m,s,c,seed
        if m in METHODS:
            row = load(OUT/"apfm_eval"/f"{m}_{exponent}_{c}_{s}_s{seed}.json")
        else:
            row = dict(saved[case])
            if s in SWITCH:
                row.update(load(OUT/"apfm_eval"/f"{m}_{exponent}_{c}_{s}_s{seed}.json"))
                row["J"] = fixed.get(case,{}).get("J",{})
        for k in C.MAIN_KS:
            j = row.get("J",{}).get(str(k))
            if j is not None:
                row[f"J{k}"] = j
        groups[m,s].append(row)
    lines = [
        "# APFM and VSS-NFBFxLMS evaluation", "",
        "## Adaptation of the reference", "",
        "- Single reference, actuator and error microphone, with Q = 1. Each path pair uses its original primary path for 30 s of white-noise pretraining (seed 900 + pair number). The secondary plant is the full S and the model is its first 512 taps. Control starts at zero.",
        "- Parallel.m supplies the candidate ordering and minimum-energy selection. Counterfactual errors use the measured error minus the modeled response to the actual output, plus the modeled response with the candidate held fixed over the relevant history. This replaces the MATLAB simulation's direct access to the primary disturbance and candidate-specific output histories. Ties use the first minimum.",
        "- Equations (10)–(15) supply the gradient, normalization, variable step and zero-selection reset. Parallel.m leaves the gradient unimplemented and uses a different step formula and reset multiplier. Here the step uses the measured current-block error power, and the gradient uses the selected reconstructed error. The zero selection uses the decayed previous increment and resets the three averages.",
        "- The constrained overlap-save implementation uses 512 causal taps, a 128-sample hop and a 1024-point FFT. Selection scores only the newest 128 samples. The error is zero-padded on the left, and the inverse-FFT gradient is projected onto the first 512 taps. The original code uses a block length equal to filter length and a 2B FFT.",
        "- The three EMA factors are 0.95, μ_max = 2 and η = 10⁻⁶. Λ starts at one, and the two block-energy EMAs start at zero. A bank run that diverges retains its last finite deployed coefficients, with the divergence logged in the bank artifact.",
        "- References use the saved evaluation and frozen coefficients where available. The short-window empirical power ratios are computed from fresh traces under the same frozen settings. FxNLMS-family frozen coefficient values absent from the saved archive are shown as unavailable.", "",
        "## Tuning", "",
        f"Selected μ̃ = {2.**exponent:g} (2^{exponent}). Each setting covers 90 design cases. Smaller L1 means more attenuation.", "",
        "| μ̃ | Diverged | L1 whole (dB) | Far KM (s) | Censored |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k, r in sorted(chosen["grid"].items(),key=lambda x:int(x[0])):
        med = f"{int(r['med'])*F.B/F.FS:.3f}" if r["med"].isdecimal() else r["med"]
        lines.append(f"| {2.**int(k):g} | {r['div']} | {fmt(r['L1'])} | {med} | {r['cens']} |")
    bank = [load(OUT/"apfm_bank"/f"{exponent}_{k}.json") for k in range(1,9)]
    lines.extend(("",f"Selected bank training divergences: {sum(r['div'] for r in bank)} / 8.",
                  "Recovery counts use all 200 cases per condition. L1 is in dB (lower is better). The power ratio is cumulative from the switch. Frozen values are medians.", "",
                  "## Final evaluation", "",
                  "| Condition | Method | Diverged | KM s (censored) | ≤0.5 s | ≤2 s | L1 whole | Last 10 s | Event +2 s | Power +0.5 s | +1 s | +2 s | Frozen 0.13 s | Frozen 0.51 s |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"))
    for s,m in itertools.product(SCEN,(*REFS,*METHODS)):
        rows = groups[m,s]
        ok = [r for r in rows if not r["div"]]
        hits = [str(sum(r.get("rec_blocks") is not None and r["rec_blocks"]*F.B/F.FS <= t for r in ok))
                if s in SWITCH else "—" for t in (.5,2)]
        cells = [mean(ok,k) for k in ("L1_whole","L1_last10","L1_ev","power_0.5","power_1","power_2")]
        for k in C.MAIN_KS:
            vals = [r[f"J{k}"] for r in ok if f"J{k}" in r]
            cells.append(fmt(float(np.median(vals)) if vals else None))
        lines.append(f"| {s} | {m} | {len(rows)-len(ok)} | {km(rows)} | "+" | ".join(hits+cells)+" |")
    lines.extend(("","## Paired differences", "",
                  "Proposed minus APFM, seed bootstrap 95% intervals, with pair counts. Recovery uses observed pairs.", "",
                  "| Condition | L1 whole (dB) | Disturbance +2 s (dB) | Recovery (s) | Frozen 0.13 s (dB) | Frozen 0.51 s (dB) |",
                  "|---|---|---|---|---|---|"))
    for s in SCEN:
        cells = []
        for key, scale in (("L1_whole",1),("L1_ev",1),("rec_blocks",F.B/F.FS),("J16",1),("J64",1)):
            v = paired(groups[F.PROPOSED,s],groups[METHODS[0],s],lambda r:r.get(key))
            if key == "L1_ev" and s not in ("EN0","ENm10"):
                v = None
            cells.append("—" if v is None else f"{v[0]*scale:+.3f} [{v[1]*scale:+.3f}, {v[2]*scale:+.3f}] ({v[3]})")
        lines.append(f"| {s} | "+" | ".join(cells)+" |")
    lines.extend(("","## APFM selections", "",
                  "Percentages pool blocks with a recorded selection. η/P_x uses the median of per-run medians over blocks that ran the gradient update.", "",
                  "| Condition | Bank | Current | Zero | Non-current in first 2 s after switch | Selector changes (no-change) | η / median P_x |",
                  "|---|---:|---:|---:|---:|---:|---:|"))
    for s in SCEN:
        rs = groups[METHODS[0],s]
        picks = np.sum([r["picks"] for r in rs],axis=0)
        pct = [f"{100*p/picks.sum():.2f}%" for p in picks]
        count = str(sum(r["not_current_post2"] for r in rs)) if s in SWITCH else "—"
        switches = str(sum(r["switches"] for r in rs)) if s == "stat" else "—"
        px = [r["px_median"] for r in rs if r["px_median"] is not None]
        ratio = f"{ETA/np.median(px):.3g}" if px else "—"
        lines.append(f"| {s} | "+" | ".join([*pct,count,switches,ratio])+" |")
    Path("RESULTS_apfm.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("check","bank","tune","extend","eval","report"))
    parser.add_argument("--procs",type=int,default=2)
    parser.add_argument("--method",choices=(*METHODS,"references"))
    parser.add_argument("--shard",type=int,default=0)
    parser.add_argument("--shards",type=int,default=10)
    args = parser.parse_args()
    final2.configure()
    if args.command == "check":
        test()
    elif args.command == "bank":
        parallel(train,itertools.product(range(1,9),INITIAL),args.procs)
    elif args.command == "tune":
        tune(args.procs,INITIAL)
    elif args.command == "extend":
        extend(args.procs)
    elif args.command == "eval":
        exponent = load(OUT/"apfm_tune"/"chosen.json")["best"]
        methods = REFS if args.method == "references" else (args.method,)
        scenarios = SWITCH if args.method == "references" else SCEN
        cases = list(itertools.product(F.FINAL_CLASSES,scenarios,F.FINAL_SEEDS,(exponent,),methods,("eval",)))
        parallel(evaluate,cases[args.shard::args.shards],args.procs)
    else:
        report()


if __name__ == "__main__":
    main()
