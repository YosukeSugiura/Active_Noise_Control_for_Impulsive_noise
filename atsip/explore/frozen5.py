"""Measure and report frozen coefficients for the five final2 baselines."""
import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import coef as C
import final as F

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
METHODS = ("FxNLMS", "FxlogNLMS", "MFxLMM", "FxNFLOC", "EVSS_FxeLMS")
SCEN = ("RS-near", "RS-far")
VERSION = 1
TOL = 1e-9


def cases():
    return [(c, s, seed) for seed in F.FINAL_SEEDS for c in F.FINAL_CLASSES for s in SCEN]


def path(folder, case):
    return OUT / folder / "{}_{}_s{}.json".format(*case)


def read(file):
    return json.loads(file.read_text())


def store(file, payload):
    file.parent.mkdir(parents=True, exist_ok=True)
    F.ab.atomic_write(file, lambda stream: json.dump(payload, stream, allow_nan=False))


def tag(case, method, **fields):
    cls, scen, seed = case
    return dict(method=method, cls=cls, scen=scen, seed=seed, **fields)


def check_prefix(prefix, full, stop):
    expected_div = full["div"] and full["n_done"] < stop
    span = min(stop, prefix["n_done"], full["n_done"])
    a, b = prefix["e"][:span], full["e"][:span]
    exact = bool(F.np.array_equal(a, b))
    if span:
        ap, bp = float(F.np.mean(a*a)), float(F.np.mean(b*b))
        rel = abs(ap-bp) / bp if bp else (0. if ap == 0 else None)
        max_error = float(F.np.max(F.np.abs(a-b)))
    else:
        rel, max_error = None, None
    return dict(div_equal=bool(prefix["div"]) == bool(expected_div),
                valid_span_equal=prefix["n_done"] == min(stop, full["n_done"]),
                samples=span, exact=exact, power_rel=rel, max_abs_error=max_error)


def run_job(task):
    method, case = task
    target = path(f"frozen5_parts/{method}", case)
    tune = read(HERE / "TUNE.json")
    setting = tune["chosen"][method]
    if target.exists():
        old = read(target)
        if old["version"] == VERSION and old["settings"] == setting:
            return
    archive, = (r for r in read(path("final2_run", case))["rows"]
                if r["method"] == method and r["variant"] == "main")
    if (archive["key"], archive["mu"]) != (setting["key"], setting["mu"]):
        raise ValueError(f"Frozen setting differs for {method}/{case}")
    if (tune["Ls"], tune["hp_hz"]) != (F.LS_FINAL, F.HP_HZ):
        raise ValueError("TUNE preprocessing lacks the final-run settings")
    dd, _, _ = F.build(*case, F.LS_FINAL, final=True)

    def simulate(limit=None):
        return F.run_baseline(method, setting["mu"], setting["params"], dd, limit)

    full = simulate()
    fresh = F.measures(full, dd)
    differences = {k: fresh[k]-archive[k] for k in ("L1_whole", "L1_last10", "L2_last5", "L1_ev", "L1_pre")
                   if k in archive and k in fresh and math.isfinite(fresh[k]) and math.isfinite(archive[k])}
    archive_div_equal = (fresh["div"], fresh.get("div_block")) == (archive["div"], archive.get("div_block"))
    ws, prefixes = {}, {}
    for k in C.MAIN_KS:
        stop = (C.CB+k)*F.B
        prefix = simulate(stop)
        ws[k] = None if prefix["div"] or full["div"] else prefix["w"]
        prefixes[k] = check_prefix(prefix, full, stop)
    row = tag(case, method, variant="main", kind="base", div=int(full["div"]),
              J=C.frozen_anr(dd, ws), prefix=prefixes, archive_div_equal=archive_div_equal,
              archive_div=archive["div"], archive_delta_db=differences)
    store(target, dict(version=VERSION, settings=setting, rows=[row]))
    print(method, *case, flush=True)


def eligible(check):
    return check["div_equal"] and check["valid_span_equal"] and (
        check["power_rel"] <= TOL if check["power_rel"] is not None else check["samples"] == 0)


def report():
    rows = []
    settings = read(HERE / "TUNE.json")["chosen"]
    for case in cases():
        group = []
        for method in METHODS:
            source = path(f"frozen5_parts/{method}", case)
            existing = path("frozen5_coef", case)
            if source.is_file():
                payload = read(source)
                row, = payload["rows"]
                selected = payload["settings"]
            else:
                payload = read(existing)
                row, = [r for r in payload["rows"] if r["method"] == method]
                selected = payload["settings"][method]
            if (payload["version"], selected) != (VERSION, settings[method]):
                raise ValueError(f"Stale result for {method}/{case}")
            if any(row[k] != v for k, v in tag(case, method, variant="main", kind="base").items()):
                raise ValueError(f"Case identity differs for {method}/{case}")
            group.append(row)
        rows.extend(group)
    for i, case in enumerate(cases()):
        store(path("frozen5_coef", case), dict(version=VERSION, settings={m: settings[m] for m in METHODS},
                                             rows=rows[i*len(METHODS):(i+1)*len(METHODS)]))
    lines = ["# 5手法の係数固定評価", "",
             f"{len(rows)}試行。変化後16・64ブロックの係数で変化後区間を評価。発散した全長試行を除いて中央値を集計した。", "",
             "final2_run は波形を保管していないため、同じ設定で全長を再実行し、prefix と波形・誤差パワーを比較した。保存済み記録とは発散フラグ・発散ブロックと共通の全長指標を照合した。", "",
             "| 手法 | 条件 | J(16) [dB] | J(64) [dB] | 発散 / 試行 | k=64 不一致 | k=64 波形ビット不一致 | 最大相対パワー差 | 最大標本差 | 保存発散不一致 | 保存指標 最大差 [dB] |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        for scen in SCEN:
            group = [r for r in rows if (r["method"], r["scen"]) == (method, scen)]
            median = []
            for k in C.MAIN_KS:
                sample = [r["J"][str(k)] for r in group if not r["div"] and r["J"][str(k)] is not None]
                median.append(f"{F.np.median(sample):.3f}" if sample else "—")
            checks = [r["prefix"]["64"] for r in group]
            power = max((v["power_rel"] for v in checks if v["power_rel"] is not None), default=None)
            err = max((v["max_abs_error"] for v in checks if v["max_abs_error"] is not None), default=None)
            delta = max((abs(v) for r in group for v in r["archive_delta_db"].values()), default=None)
            fmt = lambda x: "—" if x is None else f"{x:.5g}"
            lines.append(f"| {method} | {scen} | {' | '.join(median)} | {sum(r['div'] for r in group)}/{len(group)} "
                         f"| {sum(not eligible(v) for v in checks)} | {sum(not v['exact'] for v in checks)} "
                         f"| {fmt(power)} | {fmt(err)} | {sum(not r['archive_div_equal'] for r in group)} | {fmt(delta)} |")
    lines += ["", "k=64の不一致条件は発散フラグまたは有効標本数の差、あるいは誤差パワーの相対差 > 1e-9。"
              "保存指標との差と、丸めに敏感な波形の標本差は別に示した。ケース別JSONにはk=16の照合も記録した。"]
    (HERE / "RESULTS_frozen5.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--shard", type=F.parse_shard, default=(0, 1))
    parser.add_argument("--procs", type=int, default=1)
    args = parser.parse_args()
    if args.command == "report":
        report()
    else:
        if not args.method:
            parser.error("run requires --method")
        tasks = [(args.method, case) for case in F.shard_jobs(cases(), args.shard)]
        with Pool(args.procs) as pool:
            list(pool.imap_unordered(run_job, tasks))


if __name__ == "__main__":
    main()
