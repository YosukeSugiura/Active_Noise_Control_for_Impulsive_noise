"""Evaluate fixed secondary-path model errors with the final2 detector.

Run with single-threaded BLAS, e.g. OPENBLAS_NUM_THREADS=1 python smodel.py run --procs 2.
The run verifies all nominal cases against the published archives before applying noise.
"""
import argparse
from collections import defaultdict
import json
import math
from multiprocessing import Pool
from pathlib import Path
import tarfile
from unittest.mock import patch

import numpy as np

import cc
import final as F
import final2
from paired_ci import paired

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "smodel_run"
SCENARIOS = ("stat", "RS-near", "RS-far")
PROPOSED, NO_TRIAL, COHVSS = "P5-early-cap", "nors-cap", "Freq_FxNLMS_CohVSS"
METHODS = (PROPOSED, NO_TRIAL, COHVSS, cc.NAME)
LEVELS = (None, -20, -10)
VERSION = 1
CASES = tuple((cls, scen, seed) for cls in F.FINAL_CLASSES
              for scen in SCENARIOS for seed in F.FINAL_SEEDS)


def level_label(level):
    return "nominal" if level is None else f"{level:g}dB"


def case_name(case):
    cls, scen, seed = case
    return f"{cls}_{scen}_s{seed}.json"


def setup():
    final2.configure()


def tuning():
    settings = {}
    for filename, method in (("TUNE.json", COHVSS), ("TUNE_cc.json", cc.NAME)):
        source = json.loads((HERE / filename).read_text())
        if (source["Ls"], source["hp_hz"]) != (F.LS_FINAL, F.HP_HZ):
            raise ValueError(f"Incompatible settings: {filename}")
        settings[method] = source["chosen"][method]
    return settings


def build(cls, scen, seed, level):
    if level is None:
        return F.build(cls, scen, seed, F.LS_FINAL, final=True)
    white = np.random.default_rng(seed + 50000).standard_normal(F.LS_FINAL)
    paths_of = F.paths_of

    def with_error(scenario, pair):
        first, second = paths_of(scenario, pair)
        primary, plant, model = first
        nominal = model[:F.LS_FINAL]
        scale = np.linalg.norm(nominal) * 10 ** (level / 20) / np.linalg.norm(white)
        perturbed = nominal + scale * white
        if second is not None and (second[1] is not plant or second[2] is not model):
            raise ValueError(f"Expected a fixed secondary path and model in {scenario}")
        return (primary, plant, perturbed), None if second is None else (*second[:2], perturbed)

    # Inject the model at the path boundary so build recomputes every dependent quantity.
    with patch.object(F, "paths_of", with_error):
        return F.build(cls, scen, seed, F.LS_FINAL, final=True)


def run_case(task):
    case, level, settings = task
    cls, scen, seed = case
    path = OUT / level_label(level) / case_name(case)
    if path.is_file():
        stored = json.loads(path.read_text())
        if stored.get("version") == VERSION and stored.get("settings") == settings:
            return path
    data = build(*case, level)
    post_data = build(cls, "post:" + scen, seed, level) if scen != "stat" else None
    x4 = F.p.parse(["c5", "--preset", "final"]).X4
    rows = []
    for method in (NO_TRIAL, PROPOSED, COHVSS, cc.NAME):
        setting = settings.get(method)
        if setting is None:
            rule, sim = F.run_rule(method, *data, x4)
            post = F.run_rule(method, *post_data, x4)[1] if post_data else None
            extras = F.trial_stats(rule, dict(data[0], scored_blocks=sim["n_done"]))
            if method == NO_TRIAL:
                reference = sim
            elif not sim["div"] and not reference["div"]:
                diff = float(np.max(np.abs(sim["y"] - reference["y"])))
                extras.update(max_output_diff=diff,
                              same_as_nors=int(diff == 0 and np.array_equal(sim["e"], reference["e"])))
        else:
            mu, params = setting["mu"], setting["params"]
            if method == cc.NAME:
                sim = cc.run_baseline(F, mu, params, data[0])
                post = cc.run_baseline(F, mu, params, post_data[0]) if post_data and not sim["div"] else None
                extras = {"lam": {}}
                if post is not None:
                    extras["post_div"] = int(post["div"])
                if post_data is not None:
                    t = data[0]["changes"][0] * F.B
                    lam = sim["lam"][:sim["n_done"]]
                    for label, start, end in (("pre2", t - 2 * F.FS, t), ("post1", t, t + F.FS)):
                        if len(lam) >= end:
                            extras["lam"][label] = float(np.median(lam[start:end]))
            else:
                sim = F.run_baseline(method, mu, params, data[0])
                post = F.run_baseline(method, mu, params, post_data[0]) if post_data else None
                extras = {"key": setting["key"], "mu": mu}
        row = dict(method=method, cls=cls, scen=scen, seed=seed, variant="main",
                   kind="base" if setting is not None else "rule",
                   **F.measures(sim, data[0], post), **extras)
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = dict(version=VERSION, error_db=level, settings=settings, rows=rows)
    F.ab.atomic_write(path, lambda f: json.dump(result, f, allow_nan=False))
    return path


def reference_rows():
    requested = {case_name(case) for case in CASES}
    references = defaultdict(dict)
    for folder, methods in (("final2_run", METHODS[:-1]), ("cc_run", (cc.NAME,))):
        local = HERE / "out" / folder

        def select(name, source):
            for row in source["rows"]:
                if row["method"] in methods and row["variant"] == "main":
                    references[name][row["method"]] = row

        missing = set()
        for name in sorted(requested):
            path = local / name
            if path.is_file():
                select(name, json.loads(path.read_text()))
            else:
                missing.add(name)
        if missing:
            with tarfile.open(HERE / "results" / f"{folder}.tar.gz", "r|gz") as tar:
                for member in tar:
                    name = Path(member.name).name
                    if member.isfile() and name in missing:
                        with tar.extractfile(member) as stream:
                            select(name, json.load(stream))
                        missing.remove(name)
                        if not missing:
                            break
    for name in requested:
        if set(references[name]) != set(METHODS):
            raise ValueError(f"Incomplete reference: {name}")
    return references


def compare(expected, actual, key):
    if isinstance(expected, dict) and isinstance(actual, dict) and expected.keys() == actual.keys():
        for field in expected:
            compare(expected[field], actual[field], f"{key}.{field}")
    elif isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, f"{key}[{index}]")
    elif isinstance(expected, float) and isinstance(actual, (float, int)):
        if not math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError(f"{key}: reference {expected!r}, actual {actual!r}")
    elif expected != actual:
        raise ValueError(f"{key}: reference {expected!r}, actual {actual!r}")


def verify(path, reference):
    actual = {row["method"]: row for row in json.loads(path.read_text())["rows"]}
    if actual.keys() != reference.keys():
        raise ValueError(f"Unexpected methods in {path}")
    for method, row in actual.items():
        expected = {k: v for k, v in reference[method].items() if k != "seconds"}
        measured = {k: v for k, v in row.items() if k != "seconds"}
        compare(expected, measured, f"{path.name}/{method}")


def run(procs):
    references = reference_rows()
    settings = tuning()
    with Pool(procs, initializer=setup) as pool:
        for level in LEVELS:
            tasks = ((case, level, settings) for case in CASES)
            for index, path in enumerate(pool.imap_unordered(run_case, tasks), 1):
                if level is None:
                    verify(path, references[path.name])
                if index == 1 or index % 25 == 0:
                    print(f"{level_label(level)} {index}/{len(CASES)} {path.name}", flush=True)
            if level is None:
                print(f"Nominal verified: {len(CASES) * len(METHODS)} method runs", flush=True)


def report():
    groups = defaultdict(list)
    seen = set()
    for path in sorted(OUT.rglob("*.json")):
        result = json.loads(path.read_text())
        if result.get("version") != VERSION:
            raise ValueError(f"Unsupported result version: {path}")
        level = result["error_db"]
        for row in result["rows"]:
            key = (level, row["scen"], row["method"])
            identity = (*key, row["cls"], row["seed"])
            if identity in seen:
                raise ValueError(f"Duplicate result: {path} {identity}")
            seen.add(identity)
            groups[key].append(row)
    if not seen:
        raise ValueError(f"No results in {OUT}")
    levels = sorted({key[0] for key in groups}, key=lambda value: (value is not None, value or 0))
    lines = ["# 二次経路モデルの誤差", "",
             f"保存済みの実行数は {len(seen):,} 件。消音量は完走例の seed 内クラス平均を seed 間で平均した L1（負値ほど消音量が大きい）。",
             "回復中央値は参照が得られた実行の右打ち切りを含む。対応差は提案法−比較手法で、消音量と観測できた回復時間の完走対を seed 単位で 2,000 回再標本化した。",
             ""]
    metrics = ("L1_whole", "L1_last10", "rec_blocks")
    seconds = F.B / F.FS
    for level in levels:
        label = "誤差なし" if level is None else f"{level:g} dB"
        lines += [f"## {label}", "",
                  "| 条件 | 手法 | 実行数 | 発散 | 回復中央値 [秒] | 全体 L1 [dB] | 最後 10 秒 L1 [dB] | 移行 | 試行なしと完全一致 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        comparisons = []
        for scen in SCENARIOS:
            proposed = groups[level, scen, PROPOSED]
            for method in METHODS:
                rows = groups[level, scen, method]
                complete = [r for r in rows if not r["div"]]
                median, _ = F.recovery_summary(rows)
                if median.isdecimal():
                    median = f"{int(median) * seconds:.3f}"
                means = []
                for metric in metrics[:2]:
                    values = list(F.seed_table(complete, metric).values())
                    means.append(f"{np.mean(values):.3f}" if values else "—")
                transitions = str(sum(r["n_confirm"] for r in rows)) if method == PROPOSED else "—"
                identical = str(sum(r.get("same_as_nors", 0) for r in rows)) if method == PROPOSED and scen == "stat" else "—"
                lines.append(f"| {scen} | {method} | {len(rows)} | {sum(r['div'] for r in rows)} | {median} | "
                             f"{means[0]} | {means[1]} | {transitions} | {identical} |")
                if method == PROPOSED:
                    continue
                cells = []
                for metric, scale in zip(metrics, (1, 1, seconds)):
                    interval = paired(proposed, rows, lambda r: r.get(metric))
                    if interval is None:
                        cells.append("—")
                    else:
                        mean, lower, upper, n = interval
                        cells.append(f"{mean*scale:+.3f} [{lower*scale:+.3f}, {upper*scale:+.3f}] ({n})")
                comparisons.append(f"| {scen} | {method} | {' | '.join(cells)} |")
        lines += ["", "| 条件 | 比較手法 | 全体 L1 差 [dB] | 最後 10 秒 L1 差 [dB] | 回復時間差 [秒] |",
                  "|---|---|---:|---:|---:|", *comparisons, ""]
    F.ab.atomic_write(HERE / "RESULTS_smodel.md", lambda stream: stream.write("\n".join(lines)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--procs", type=int, default=1)
    options = parser.parse_args()
    if options.procs < 1:
        parser.error("--procs must be positive")
    if options.command == "run":
        run(options.procs)
    else:
        report()
