"""Analyze recovery and frozen coefficients after a source switch.

Run with OPENBLAS_NUM_THREADS=1 python mgun.py run --procs 2.
Regenerate the tables from saved cases with python mgun.py report.
"""
import argparse
from collections import defaultdict
from itertools import combinations
import json
import math
from multiprocessing import Pool
from pathlib import Path
import tarfile
from unittest.mock import patch

import numpy as np

import coef
import final as F
import final2
from paired_ci import paired

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "mgun_run"
EXAMPLE = OUT.parent / "mgun_example.npz"
CLASSES = ("machinegun", "a145")
SCENARIOS = ("RS-near", "RS-far")
PROPOSED, ALWAYS, NO_TRIAL, COHVSS = "P5-early-cap", "TP-always-cap", "nors-cap", "Freq_FxNLMS_CohVSS"
METHODS = (PROPOSED, ALWAYS, NO_TRIAL, COHVSS)
CASES = tuple((cls, scen, seed) for cls in CLASSES for scen in SCENARIOS for seed in F.FINAL_SEEDS)
OFFSETS = (16, 32, 64, 125, 250)
WIN = 16
STEP = F.B / F.FS
VERSION = 1


def case_name(case):
    cls, scen, seed = case
    return f"{cls}_{scen}_s{seed}.json"


def references():
    requested = {case_name(case) for case in CASES}
    path = OUT.parent / "final2_run"
    sources = {name: json.loads((path / name).read_text())
               for name in requested if (path / name).is_file()}
    if sources.keys() != requested:
        with tarfile.open(HERE / "results" / "final2_run.tar.gz", "r|gz") as archive:
            for member in archive:
                name = Path(member.name).name
                if member.isfile() and name in requested and name not in sources:
                    with archive.extractfile(member) as stream:
                        sources[name] = json.load(stream)
                    if sources.keys() == requested:
                        break
    if sources.keys() != requested:
        raise ValueError(f"Missing reference cases: {sorted(requested - sources.keys())}")
    return {name: {row["method"]: row for row in source["rows"] if row["variant"] == "main"}
            for name, source in sources.items()}


def block_power(signal):
    return np.mean(signal[:len(signal) // F.B * F.B].reshape(-1, F.B) ** 2, axis=1)


def window_power(power):
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    return (cumulative[WIN:] - cumulative[:-WIN]) / WIN


def quantiles(values):
    return np.quantile(values, (0.25, 0.5, 0.75)).tolist() if len(values) else None


def input_stats(d):
    power = block_power(d)
    center = np.median(power)
    gaps = np.diff(np.flatnonzero(power >= center * 10)) * STEP
    return dict(primary_power=power.tolist(), low_fraction=float(np.mean(power <= center / 100)),
                burst_gaps_s=gaps.tolist(), burst_quartiles_s=quantiles(gaps))


def variation(power, start, stop):
    # Use windows contained in the requested time range.
    values = power[start:stop - WIN + 1]
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    lo, hi = np.quantile(10 * np.log10(np.maximum(finite, np.finfo(float).tiny)), (0.1, 0.9))
    return float(hi - lo)


def short_time(sim, row, cb, limit_blocks):
    power = block_power(sim["e"])
    # Restart the sum at the switch, exactly as recovery does for threshold tests.
    # Fill diverged tails with NaN without losing the preceding completed windows.
    pre = window_power(power[:cb])
    post = window_power(power[cb:])
    start = None
    if "rec_target" in row:
        acceptable = np.flatnonzero(post <= row["rec_target"] * 10 ** (3 / 10))
        if len(acceptable):
            start = int(acceptable[0]) + WIN - 1
    recovered = row.get("rec_blocks")
    return dict(pre_span_db=variation(pre, cb - 5 * F.BPS, cb),
                late_span_db=variation(post, 20 * F.BPS - cb, limit_blocks - cb),
                first_blocks=start,
                delay_blocks=recovered - start if recovered is not None else None)


def simulate(method, data, post, x4, setting):
    weights = {}
    wanted = {F.CB + offset: offset for offset in OFFSETS}
    if method != COHVSS:
        def capture(block, ctx, error, step):
            if block in wanted:
                weights[wanted[block]] = ctx.w.copy()
        rule, sim = F.run_rule(method, *data, x4, hook=capture)
        post_sim = F.run_rule(method, *post, x4)[1]
        trials = F.trial_stats(rule, dict(data[0], scored_blocks=sim["n_done"]))["trials"]
    else:
        make_algo = F.make_algo

        def make_with_capture(name, params):
            algo = make_algo(name, params)
            original_update = algo.update
            block = 0

            def update(w, e, r, x, mu):
                nonlocal block
                result = original_update(w, e, r, x, mu)
                block += 1
                if block in wanted:
                    weights[wanted[block]] = result[0].copy()
                return result
            algo.update = update
            return algo

        mu, params = setting["mu"], setting["params"]
        with patch.object(F, "make_algo", make_with_capture):
            sim = F.run_baseline_block(method, mu, params, data[0])
        post_sim = F.run_baseline(method, mu, params, post[0])
        trials = []
    return sim, post_sim, weights, trials


def verify(row, expected):
    actual_l1, expected_l1 = row.get("L1_whole"), expected.get("L1_whole")
    matches_l1 = (actual_l1 is expected_l1 if actual_l1 is None or expected_l1 is None
                  else math.isclose(actual_l1, expected_l1, rel_tol=1e-9, abs_tol=1e-10))
    if not matches_l1 or any(row.get(k) != expected.get(k) for k in ("div", "rec_blocks", "rec_cens")):
        keys = ("div", "L1_whole", "rec_blocks", "rec_cens")
        raise ValueError(f"{row['cls']} {row['scen']} seed {row['seed']} {row['method']}: "
                         f"final2={tuple(expected.get(k) for k in keys)}, actual={tuple(row.get(k) for k in keys)}")


def example_trace(signal, cb):
    power = block_power(signal)
    pre, post = window_power(power[:cb]), window_power(power[cb:])
    # Windows that straddle the switch are useful in the example, but excluded from recovery.
    all_windows = window_power(power)
    all_windows[:len(pre)] = pre
    all_windows[cb:] = post
    end_blocks = np.arange(WIN - 1, len(power))
    selection = (end_blocks >= cb - 2 * F.BPS) & (end_blocks <= cb + 6 * F.BPS)
    return (end_blocks[selection] - cb) * STEP, all_windows[selection]


def run_case(task):
    case, expected, setting = task
    cls, scen, seed = case
    data = F.build(cls, scen, seed, F.LS_FINAL, final=True)
    post = F.build(cls, "post:" + scen, seed, F.LS_FINAL, final=True)
    dd = data[0]
    cb, = dd["changes"]
    x4 = F.p.parse(["c5", "--preset", "final"]).X4
    verified = []
    for method in METHODS:
        sim, ps, weights, trials = simulate(method, data, post, x4, setting)
        row = dict(cls=cls, scen=scen, seed=seed, method=method, **F.measures(sim, dd, ps))
        verify(row, expected[method])
        verified.append((row, sim, weights, trials))

    rows, example = [], {}
    for row, sim, weights, trials in verified:
        ss = short_time(sim, row, cb, len(dd["d"]) // F.B)
        near = [t for t in trials if cb <= t["block"] < cb + F.BPS]
        transfers = [t["decide"] + 1 - cb for t in trials
                     if t["status"] == "confirmed" and t["decide"] + 1 >= cb]
        row.update(**ss, J=coef.frozen_anr(dd, {k: weights.get(k) for k in OFFSETS}), trials=trials,
                   started=bool(near), discarded=any(t["status"] in ("abort", "rollback") for t in near),
                   first_transfer_blocks=min(transfers, default=None))
        rows.append(row)
        if case == ("machinegun", "RS-far", 100):
            time_s, trace = example_trace(sim["e"], cb)
            example.update(time_s=time_s, cls=cls, scen=scen, seed=seed)
            example[row["method"]] = trace
            example[row["method"] + "_threshold"] = row.get("rec_target", np.nan) * 10 ** (3 / 10)

    return case_name(case), dict(version=VERSION, input=input_stats(dd["d"]), rows=rows), example


def run(procs):
    expected = references()
    tune = json.loads((HERE / "TUNE.json").read_text())
    if (tune["Ls"], tune["hp_hz"]) != (F.LS_FINAL, F.HP_HZ):
        raise ValueError("Incompatible common tuning")
    setting = tune["chosen"][COHVSS]
    tasks = ((case, expected[case_name(case)], setting) for case in CASES)
    with Pool(procs, initializer=final2.configure) as pool:
        results = []
        for result in pool.imap_unordered(run_case, tasks):
            results.append(result)
            if len(results) % 10 == 0:
                print(f"Verified {len(results)}/{len(CASES)} cases", flush=True)
    # Publish only after the entire reference comparison passes.
    OUT.mkdir(parents=True, exist_ok=True)
    for name, result, example in results:
        F.ab.atomic_write(OUT / name, lambda stream: json.dump(result, stream, allow_nan=False))
        if example:
            np.savez_compressed(EXAMPLE, **example)
    print(f"final2_run verified: {len(results) * len(METHODS)} runs", flush=True)


def values(rows, key):
    return [row[key] for row in rows if row.get(key) is not None]


def median(data, scale=1):
    return f"{np.median(data) * scale:.3f}" if len(data) else "—"


def interval(stat):
    if stat is None:
        return "—"
    mean, lo, hi, n = stat
    return f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}] ({n})"


def add_table(lines, title, headings, rows):
    lines.extend(["", f"## {title}", "",
                  "| " + " | ".join(headings) + " |",
                  "| " + " | ".join("---" for _ in headings) + " |"])
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)


def report():
    inputs, groups = defaultdict(list), defaultdict(list)
    seen = set()
    for file in sorted(OUT.rglob("*.json")):
        result = json.loads(file.read_text())
        if result["version"] != VERSION:
            raise ValueError(f"Unsupported result version: {file}")
        first = result["rows"][0]
        case = (first["cls"], first["scen"], first["seed"])
        if case in seen or {r["method"] for r in result["rows"]} != set(METHODS):
            raise ValueError(f"Duplicate case or invalid methods: {file}")
        seen.add(case)
        inputs[case[:2]].append(result["input"])
        for row in result["rows"]:
            groups[(*case[:2], row["method"])].append(row)
    if not seen:
        raise ValueError(f"No cases in {OUT}")
    lines = ["# 機関銃騒音の回復時間", "",
             f"{len(seen)} 条件、{sum(map(len, groups.values()))} 実行。"
             "各実行の全体 L1 と回復ブロックを final2_run と照合した。時間の原点は音源切替（実行開始 15 秒）。"]
    input_table, recovery_table, coef_table, paired_table, trial_table = [], [], [], [], []
    for (cls, scen), data in sorted(inputs.items()):
        gaps = [gap for entry in data for gap in entry["burst_gaps_s"]]
        q = quantiles(gaps)
        quartiles = tuple(f"{v:.3f}" for v in q) if q else ("—", "—", "—")
        input_table.append((cls, scen, len(data), median(values(data, "low_fraction"), 100),
                            len(gaps), *quartiles))
        for method in METHODS:
            rows = groups[cls, scen, method]
            km, censored = F.recovery_summary(rows)
            km_s = f"{int(km) * STEP:.3f}" if km.isdecimal() else km
            first, delay = values(rows, "first_blocks"), values(rows, "delay_blocks")
            loss = sum(row.get("first_blocks") is not None and row.get("rec_cens") == 1 for row in rows)
            recovery_table.append((cls, scen, method, median(values(rows, "pre_span_db")),
                                   median(values(rows, "late_span_db")), median(first, STEP), len(first),
                                   km_s, censored, median(delay, STEP), len(delay), loss,
                                   median(values(rows, "L1_ev"))))
            coef_table.append((cls, scen, method, *(median([r["J"][str(k)] for r in rows
                                                           if r["J"][str(k)] is not None]) for k in OFFSETS)))
            if method in (PROPOSED, ALWAYS):
                transfers = values(rows, "first_transfer_blocks")
                trial_table.append((cls, scen, method, len(rows), sum(r["started"] for r in rows),
                                    len(transfers), sum(r["discarded"] for r in rows), median(transfers)))
        for left, right in combinations(METHODS, 2):
            a, b = groups[cls, scen, left], groups[cls, scen, right]
            paired_table.append((cls, scen, f"{left} − {right}", *(
                interval(paired(a, b, lambda row: row["J"][str(k)])) for k in OFFSETS)))
    time_headers = tuple(f"{k * STEP:.2f} 秒" for k in OFFSETS)
    add_table(lines, "一次騒音", ("騒音", "条件", "実行数", "低パワーブロック [%]", "間隔数",
                                "第1四分位 [秒]", "中央値 [秒]", "第3四分位 [秒]"), input_table)
    lines += ["", "低パワー率は、各実行の一次騒音パワー中央値を 20 dB 以上下回るブロックの割合の seed 中央値。"
              "バースト間隔は、中央値を 10 dB 以上上回る隣接検出ブロック間の時間を全 seed でプールした。"]
    add_table(lines, "短時間誤差パワーと回復",
              ("騒音", "条件", "手法", "切替前5秒の幅 [dB]", "実行20〜30秒の幅 [dB]",
               "初回到達 [秒]", "到達数", "回復 KM [秒]", "打切数", "回復−初回 [秒]", "両時刻あり",
               "到達後に回復せず", "切替後2秒 L1 [dB]"), recovery_table)
    lines += ["", "幅は 16 ブロック移動平均を dB に変換したときの第90−第10パーセンタイル。"
              "各列は seed 中央値で、回復には右打ち切りを含む Kaplan–Meier 中央値を用いた。"
              "初回到達は判定基準を初めて満たした窓の最終ブロック。回復は基準を1秒間保つ最初の連続区間の先頭窓の最終ブロック（final.py の定義）。"
              "回復−初回は両時刻が得られた同一実行内の差。"]
    add_table(lines, "固定係数の消音量 J [dB]", ("騒音", "条件", "手法", *time_headers), coef_table)
    add_table(lines, "固定係数の対応差 J [dB]", ("騒音", "条件", "左手法 − 右手法", *time_headers), paired_table)
    lines += ["", "固定係数の J は seed 中央値。対応差は完走対の平均と seed ブートストラップ95%区間（2,000回、seed 20261121）。"
              "括弧は対応対の数。J と L1 は負値ほど消音量が大きい。"]
    add_table(lines, "試行と係数移行", ("騒音", "条件", "手法", "実行数", "1秒以内に試行開始",
                                      "切替後に移行", "1秒以内に開始した試行を破棄", "初回移行中央値 [ブロック]"), trial_table)
    lines += ["", "破棄には rollback と abort を含む。係数移行は決定の次ブロックから数え、切替時点で進行中の試行からの移行も含む。"]
    add_summary(lines, inputs, groups)
    F.ab.atomic_write(HERE / "RESULTS_mgun.md", lambda stream: stream.write("\n".join(lines) + "\n"))


def add_summary(lines, inputs, groups):
    if not all((cls, scen) in inputs for cls in CLASSES for scen in SCENARIOS):
        return
    lines += ["", "## 観測結果", ""]
    for scen in SCENARIOS:
        mg, comparison = (groups[cls, scen, ALWAYS] for cls in CLASSES)
        prop, coh = (groups["machinegun", scen, method] for method in (PROPOSED, COHVSS))
        def km(rows):
            result, _ = F.recovery_summary(rows)
            return f"{int(result) * STEP:.3f}" if result.isdecimal() else result
        lines.append(f"- {scen} の常時試行では、実行20〜30秒の誤差パワーの変動幅は機関銃騒音で "
                     f"{median(values(mg, 'late_span_db'))} dB、a145 で {median(values(comparison, 'late_span_db'))} dB。"
                     f"初回到達から回復までの差の中央値はそれぞれ "
                     f"{median(values(mg, 'delay_blocks'), STEP)} 秒と {median(values(comparison, 'delay_blocks'), STEP)} 秒。")
        gain = paired(mg, prop, lambda r: r.get("L1_ev"))
        fixed = paired(mg, prop, lambda r: r["J"]["64"])
        lines.append(f"- 機関銃騒音の {scen} で、常時試行−提案法の切替後2秒 L1 差は "
                     f"{interval(gain)} dB、0.51秒の固定係数 J 差は {interval(fixed)} dB。"
                     f"回復中央値は常時試行 {km(mg)}、提案法 {km(prop)}、CohVSS-FxNLMS {km(coh)} 秒。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--procs", type=int, default=1)
    args = parser.parse_args()
    if args.procs < 1:
        parser.error("--procs must be positive")
    if args.command == "run":
        run(args.procs)
    else:
        report()
