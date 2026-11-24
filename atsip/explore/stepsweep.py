"""ATSIP step-size sweep; existing final2 evaluator remains unchanged."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import json
import math
import os
import time

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "stepsweep_run"
PRIOR = HERE / "out" / "final2_run"
METHODS = ("FxNLMS", "Freq_FxNLMS_CohVSS", "Th_FxNLMS")
FACTORS = (0.5, 1., 2., 4., 8., 16.)
CLASSES = ("G", "II300", "a145", "a120", "machinegun", "factory1",
           "destroyerengine", "volvo", "valve", "jackhammer")
SCENARIOS = ("RS-near", "RS-far")
SEEDS = tuple(range(100, 120))
PROPOSED = "P5-early-cap"
VERSION = 1

def cases():
    return [(c, s, i) for c in CLASSES for s in SCENARIOS for i in SEEDS]

def case_name(case):
    return "{}_{}_s{}.json".format(*case)

def read_case(path, case, old=False):
    obj = json.loads(path.read_text())
    if not old and obj.get("version") != VERSION:
        raise ValueError(f"Wrong version: {path}")
    for r in obj["rows"]:
        if (r["cls"], r["scen"], r["seed"]) != case:
            raise ValueError(f"Wrong case identity: {path}")
    return obj["rows"]

def prior_rows(case):
    out = {}
    for r in read_case(PRIOR / case_name(case), case, True):
        if r.get("variant", "main") == "main":
            m = r["method"]
            if m in out:
                raise ValueError(f"Duplicate reference: {case} {m}")
            out[m] = r
    if not set(METHODS + (PROPOSED,)) <= out.keys():
        raise ValueError(f"Missing reference: {case}")
    return out

def check_one(row, prior):
    for k in ("div", "L1_whole", "rec_blocks", "mu"):
        if (k in row) != (k in prior) or row.get(k) != prior.get(k):
            raise ValueError(f"1x mismatch {row['cls']} {row['scen']} {row['seed']} {row['method']} {k}: {row.get(k)!r} != {prior.get(k)!r}")

def configure():
    import final2
    final2.configure()

def dump(path, obj, allow_nan=False):
    import final as F
    path.parent.mkdir(parents=True, exist_ok=True)
    F.ab.atomic_write(path, lambda h: json.dump(obj, h, indent=2, allow_nan=allow_nan))

def worker(job):
    import numpy as np
    import final as F
    case, factors = job
    cls, scen, seed = case
    t0 = time.monotonic()
    tune = json.loads((HERE / "TUNE.json").read_text())
    assert tune["Ls"] == F.LS_FINAL and tune["hp_hz"] == F.HP_HZ
    old = prior_rows(case)
    path = OUT / case_name(case)
    indexed = {}
    if path.exists():
        for r in read_case(path, case):
            key = r["method"], float(r["factor"])
            if key in indexed:
                raise ValueError(f"Duplicate: {case} {key}")
            indexed[key] = r
    dd, _, _ = F.build(cls, scen, seed, F.LS_FINAL, final=True)
    post, _, _ = F.build(cls, "post:" + scen, seed, F.LS_FINAL, final=True)
    limit = max(0, len(dd["d"]) // F.B - dd["changes"][0] - F.BPS + 1)
    for factor in factors:
        for method in METHODS:
            setting = tune["chosen"][method]
            mu = setting["mu"] * factor
            sim = F.run_baseline(method, mu, setting["params"], dd)
            ps = F.run_baseline(method, mu, setting["params"], post)
            pm = dict(e=ps["e"], d=post["d"][:len(ps["e"])], div=ps["div"])
            r = dict(method=method, kind="base", variant="main", key=setting["key"],
                     mu=mu, factor=factor, cls=cls, scen=scen, seed=seed,
                     post_div=int(ps["div"]), sweep_rec_limit=limit,
                     **F.measures(sim, dd, pm))
            if not sim["div"]:
                anr = F.p.O.anr_curve(sim["e"], dd["d"][:len(sim["e"])])
                ts = (63 + 64 * np.arange(len(anr))) / F.FS
                change = dd["changes"][0] / F.BPS
                r["L1_pre5"] = float(anr[(ts >= change - 5) & (ts < change)].mean())
            if factor == 1.:
                check_one(r, old[method])
            indexed[method, factor] = r
    dump(path, dict(version=VERSION, rows=list(indexed.values()),
                    wall_seconds=time.monotonic() - t0), allow_nan=True)
    print(f"{path.stem}: {len(factors)*len(METHODS)} trials checked/completed", flush=True)
    return len(factors)*len(METHODS)

def run(procs):
    if not (HERE / "EVAL_FROZEN.json").is_file():
        raise ValueError("EVAL_FROZEN.json required")
    for case in cases():
        prior_rows(case)
    with ProcessPoolExecutor(max_workers=procs, initializer=configure) as pool:
        n = sum(pool.map(worker, [(case, (1.,)) for case in cases()]))
    print(f"Exact 1x verification passed: {n} runs", flush=True)
    with ProcessPoolExecutor(max_workers=procs, initializer=configure) as pool:
        n += sum(pool.map(worker, [(case, tuple(f for f in FACTORS if f != 1.)) for case in cases()]))
    assert n == len(cases()) * len(METHODS) * len(FACTORS)
    print(f"Completed {n} switch runs (plus post-target simulations)", flush=True)
    report()

def km(rows, include_div, block_seconds, default_limit):
    counts = Counter()
    missing = 0
    for r in rows:
        if r["div"]:
            if include_div:
                counts[(r.get("sweep_rec_limit", default_limit), 1)] += 1
        elif r.get("rec_cens") is None:
            missing += 1
        else:
            cens = int(r["rec_cens"])
            t = r["rec_limit"] if cens else r["rec_blocks"]
            if t is None or t < 0:
                raise ValueError(f"Invalid recovery: {r}")
            counts[t, cens] += 1
    risk = n = sum(counts.values())
    censored = sum(v for (t, c), v in counts.items() if c)
    survival, median = 1., None
    for t in sorted({t for t, c in counts}):
        e, c = counts[t, 0], counts[t, 1]
        survival *= (risk-e)/risk
        if survival <= .5 + 1e-12:
            median = t * block_seconds
            break
        risk -= e+c
    return dict(median=median, n=n, censored=censored, missing_target=missing)

def class_mean(rows, key):
    values = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = r.get(key)
        if not r["div"] and v is not None and math.isfinite(v):
            values[r["seed"]][r["cls"]].append(v)
    means = []
    for classes in values.values():
        cm = [sum(v)/len(v) for v in classes.values()]
        means.append(sum(cm)/len(cm))
    return sum(means)/len(means) if means else None

def summarize(rows, method, factor, scen, block_seconds, limit):
    divs = Counter(r["cls"] for r in rows if r["div"])
    return dict(method=method, factor=factor, condition=scen, n=len(rows),
                divergence_count=sum(divs.values()),
                divergence_by_noise={c: divs[c] for c in CLASSES},
                recovery_with_divergence=km(rows, True, block_seconds, limit),
                recovery_without_divergence=km(rows, False, block_seconds, limit),
                L1_whole=class_mean(rows, "L1_whole"),
                L1_pre5=class_mean(rows, "L1_pre5"))

def report():
    import final as F
    block_seconds = F.B/F.FS
    limit = max(0, (30*F.FS)//F.B - F.CB - F.BPS + 1)
    groups, seen = defaultdict(list), set()
    valid_cases = set(cases())
    for path in sorted(OUT.glob("*.json")):
        obj = json.loads(path.read_text())
        if obj.get("version") != VERSION:
            raise ValueError(f"Wrong version: {path}")
        for r in obj["rows"]:
            case = r["cls"], r["scen"], r["seed"]
            if case not in valid_cases or path.name != case_name(case):
                raise ValueError(f"Unexpected case: {path}")
            key = r["method"], float(r["factor"]), r["scen"], r["cls"], r["seed"]
            if key in seen:
                raise ValueError(f"Duplicate result: {key}")
            seen.add(key)
            groups[key[:3]].append(r)
    for case in cases():
        groups[PROPOSED, None, case[1]].append(prior_rows(case)[PROPOSED])
    summary = [summarize(rows, method, factor, scen, block_seconds, limit)
               for (method, factor, scen), rows in sorted(groups.items(),
               key=lambda kv: (kv[0][2], kv[0][0], kv[0][1] or 0))]
    def fmt(x):
        return "—" if x is None else f"{x:.3f}"
    def km_fmt(v):
        med = "—" if not v["n"] else "打切" if v["median"] is None else fmt(v["median"])
        return med + f" ({v['censored']}/{v['n']}; 目標なし {v['missing_target']})"
    lines = ["# ステップサイズ掃引", "",
             "分母は実際の試行数（完全な群は200）。L1は既存の負値が良い規約。",
             "消音量は非発散試行のseed内等重みクラス平均をseed間で平均。",
             "KM欄：中央値[秒]（打切数/解析数; 回復目標なしの除外数）。",
             "発散を含む版では発散試行を共通観測終了時点で右打切りとする。",
             "提案法はfinal2_runから引用し、5秒前区間は既存結果にないため欠測。", "",
             "| 条件 | 手法 | 倍率 | 発散数 | KM含む | KM除く | L1全体[dB] | L1変化前5秒[dB] |",
             "|---|---|---:|---:|---|---|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['condition']} | {s['method']} | {s['factor'] or '—'} | "
                     f"{s['divergence_count']}/{s['n']} | {km_fmt(s['recovery_with_divergence'])} | "
                     f"{km_fmt(s['recovery_without_divergence'])} | {fmt(s['L1_whole'])} | {fmt(s['L1_pre5'])} |")
    lines += ["", "## 騒音別発散数", "", "| 条件 | 手法 | 倍率 | " + " | ".join(CLASSES) + " |",
              "|---|---|---:|" + "---:|"*len(CLASSES)]
    for s in summary:
        lines.append(f"| {s['condition']} | {s['method']} | {s['factor'] or '—'} | " +
                     " | ".join(str(s["divergence_by_noise"][c]) for c in CLASSES) + " |")
    dump(HERE / "out" / "stepsweep_summary.json", dict(version=VERSION, groups=summary))
    F.ab.atomic_write(HERE / "RESULTS_stepsweep.md", lambda h: h.write("\n".join(lines)+"\n"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=("run", "report"))
    parser.add_argument("--procs", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.procs < 1:
        parser.error("--procs must be positive")
    if args.cmd == "run":
        run(args.procs)
    else:
        report()

if __name__ == "__main__":
    main()
