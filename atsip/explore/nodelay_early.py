"""Sweep startup step sizes for SF and SF-ND using the nodelay simulator."""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from time import perf_counter

import nodelay as nd
import numpy as np

CLASSES = ("G", "BU", "PS", "II300", "a145", "a120")
SEEDS = nd.SEEDS
EARLY_STEPS = {"SF": (0.5, 0.678, 0.8, 1.0), "SF-ND": (0.7, 1.0, 1.5, 2.0)}
LENGTHS = (16, 32, 64)
LATE_STEPS = (0.03, 0.05)
BLOCKS = (8, 16, 32, 64, 125)
METRICS = ("L1_whole", "L1_first5", "L1_last10", "ex_0_1s", "ex_0_5s", "ex_last10")
OBJECTIVES = {"ex_0_1s": "ex 0–1 s", "L1_first5": "L1 first5", "L1_whole": "L1 whole"}
CONFIGS = [
    (direction, mu_e, n_e, mu_l)
    for direction, early_steps in EARLY_STEPS.items()
    for mu_e, n_e, mu_l in product(early_steps, LENGTHS, LATE_STEPS)
]
INPUTS = tuple(product(CLASSES, SEEDS))
HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "nodelay_early"


def name(config):
    direction, mu_e, n_e, mu_l = config
    return f"{direction}+g+early mue{mu_e:g} ne{n_e} mu{mu_l:g}"


def simulate(data, refs, stats, config, nb):
    direction, mu_e, n_e, mu_l = config
    schedule = np.full(nb, mu_l)
    schedule[:n_e] = mu_e
    rule = nd.GainCorrectedSF("SF", schedule)
    frame = nd.ReconstructedError(data["r"]) if direction == "SF-ND" else None
    return nd.T.simulate(data, refs, stats, rule, nb=nb, error_frame=frame)


def block_excess(ex):
    """Convert each available post-update observation to dB."""
    values = {}
    for block in BLOCKS:
        if block <= len(ex) and np.isfinite(ex[block - 1]):
            values[str(block)] = float(10 * np.log10(ex[block - 1]))
        else:
            values[str(block)] = None
    return values


def check():
    data, refs, stats = nd.T.prep("G_s10", 126)
    nb = 126
    for direction in EARLY_STEPS:
        for n_e in LENGTHS:
            sim = simulate(data, refs, stats, (direction, 0.7, n_e, 0.05), nb)
            assert not sim["div"]
            steps = np.full(nb, 0.05)
            steps[:n_e] = 0.7
            np.testing.assert_array_equal(sim["tr"]["mu_app"], steps / np.maximum(stats.g[:nb], 0.25))
            result = block_excess(sim["ex"])
            assert all(np.isfinite(value) for value in result.values())
            assert result[str(n_e)] == 10 * np.log10(sim["ex"][n_e - 1])
    old = nd.simulate(data, refs, stats, "SF-ND+g+early", 0.05, 0.7, nb)
    new = simulate(data, refs, stats, ("SF-ND", 0.7, 64, 0.05), nb)
    for key in ("e", "w", "ex"):
        np.testing.assert_array_equal(new[key], old[key])
    print("Schedule boundaries and post-update samples verified; 64-block SF-ND agrees bitwise")


def read_input(tag):
    path = OUT / f"{tag}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def run_input(pair):
    cls, seed = pair
    tag = f"{cls}_s{seed}"
    records = read_input(tag)
    pending = [c for c in CONFIGS if name(c) not in records]
    if not pending:
        return
    data, refs, stats = nd.T.prep(tag)
    nb = len(data["d"]) // nd.B
    path = OUT / f"{tag}.json"
    temporary = path.with_suffix(".tmp")
    start = perf_counter()
    for config in pending:
        rule = name(config)
        sim = simulate(data, refs, stats, config, nb)
        row = {"tag": tag, "class": cls, "seed": seed, "rule": rule, **nd.measure(sim, data)}
        row = {k: None if isinstance(v, float) and np.isnan(v) else v for k, v in row.items()}
        records[rule] = {"row": row, "blocks": block_excess(sim["ex"])}
        temporary.write_text(json.dumps(records, allow_nan=False))
        temporary.replace(path)
    print(f"{tag}: {len(records)} trials, {perf_counter() - start:.1f} s", flush=True)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for _ in pool.map(run_input, INPUTS):
            pass


def class_average(values):
    """Average finite seeds within each class, then equally weight classes."""
    by_class = np.asarray(values, dtype=float).reshape(len(CLASSES), len(SEEDS))
    means = [np.mean(finite) for row in by_class if (finite := row[np.isfinite(row)]).size]
    return float(np.mean(means)) if means else None


def db(value):
    return "—" if value is None else f"{value:.3f}"


def report():
    names = [name(config) for config in CONFIGS]
    datasets = []
    for cls, seed in INPUTS:
        tag = f"{cls}_s{seed}"
        records = read_input(tag)
        if set(records) != set(names):
            raise ValueError(f"Incomplete sweep for {tag}")
        datasets.append(records)

    lines = [
        "# SF+g と SF-ND+g の初期ステップ・長さの比較",
        "",
        "- 基準コミットは `615c86068b712ae05d7524e7d51821fec0f97261`。"
        "G、BU、PS、II300、a145、a120 の6クラス、seed 10–14 の30入力に各48規則を適用した。"
        "各入力は30秒（3,750ブロック）。seed 100–119 は最終評価用。",
        "- 初回の n_e ブロックで内部ステップを μ_e、以後を μ_L とし、全区間で 1/max(g, 0.25) を掛ける。"
        "再始動なし。SF-ND は前回と同じ誤差フレームの再構成を用いる。L1 は実誤差に基づく。",
        "- 表1では各試行の区間内で線形 Φ 超過比を平均し、dB に変換する。"
        "各指標は有効な seed の dB 平均をクラスごとに取り、クラスを等重みで平均する。"
        "数値発散した試行の指標は欠測とし、全 seed が欠測のクラスはその平均から除く。"
        "増幅した試行は指標の平均に含める。",
        "- 判定は前回と同じ。数値発散は非有限または Φ 超過比 > 10⁸。"
        "増幅は数値発散していない場合の L1 last10 > 0 dB。"
        "CSV の経路変化と再始動に関する列は空欄。",
        "- 再現時は `atsip/explore/` で OpenBLAS、OpenMP、MKL を各1スレッドに設定し、"
        "`python nodelay_early.py check`、`python nodelay_early.py run`、`python nodelay_early.py report` を順に実行する。"
        "実行は最大2ワーカー。試行ごとの中間結果は `out/nodelay_early/`。",
        "",
        "## 表1　区間ごとの評価",
        "",
        "| 規則 | L1 whole (dB) | L1 first5 (dB) | L1 last10 (dB) | ex 0–1 s (dB) | ex 0–5 s (dB) | ex last10 (dB) | 数値発散 / 30 | 増幅 / 30 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    series = {}
    stable = {direction: [] for direction in EARLY_STEPS}
    for config, rule in zip(CONFIGS, names):
        rows = [data[rule]["row"] for data in datasets]
        series[rule] = {metric: [row[metric] for row in rows] for metric in METRICS}
        means = " | ".join(db(class_average(series[rule][metric])) for metric in METRICS)
        div, amp = (sum(row[key] for row in rows) for key in ("div_num", "amp"))
        lines.append(f"| {rule} | {means} | {div} | {amp} |")
        if div == amp == 0:
            stable[config[0]].append(rule)

    lines.extend([
        "",
        "## 表2　指定ブロックの Φ 超過",
        "",
        "ブロック番号は1始まり。指定ブロックの更新後の Φ 超過比を試行ごとに dB に変換し、"
        "有効な seed の平均をクラスごとに取って等重みで平均する。"
        "指定時点までに数値発散した試行はその時点で欠測とし、後で発散する試行の有効な観測は含める。"
        "全 seed が欠測のクラスはその時点の平均から除く。括弧内は有効試行数。",
        "",
        "| 規則 | #8 (dB) | #16 (dB) | #32 (dB) | #64 (dB) | #125 (dB) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for rule in names:
        cells = []
        for block in BLOCKS:
            values = [data[rule]["blocks"][str(block)] for data in datasets]
            count = np.isfinite(np.asarray(values, dtype=float)).sum()
            cells.append(f"{db(class_average(values))} ({count})")
        lines.append(f"| {rule} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## 表3　安定条件での指標別最良設定と対応差",
        "",
        "全30試行で数値発散と増幅がともに0件の規則から、方向ごとの指定格子で各指標を最小にする設定を選んだ。"
        "対応差は同じクラス・seed の SF-ND − SF を6クラスで平均してから、5 seed の平均・最小・最大を求めた。",
        "",
        "| 指標 | SF の最良規則 | SF (dB) | SF-ND の最良規則 | SF-ND (dB) | 対応差平均 (dB) | 最小 (dB) | 最大 (dB) |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ])
    conclusions = []
    for metric, label in OBJECTIVES.items():
        means = {rule: class_average(series[rule][metric]) for rule in names}
        sf, sf_nd = [min(stable[direction], key=means.get, default=None) for direction in EARLY_STEPS]
        suffix = "— | — | —"
        if sf is None or sf_nd is None:
            conclusion = f"{label} は安定条件を満たす設定が両方向でそろわず、対応差を算出できない。"
        else:
            paired = np.subtract(series[sf_nd][metric], series[sf][metric]).reshape(len(CLASSES), len(SEEDS)).mean(axis=0)
            delta = float(paired.mean())
            suffix = f"{delta:+.3f} | {paired.min():+.3f} | {paired.max():+.3f}"
            if abs(delta) < 0.1:
                conclusion = f"{label} は差なし（対応差の平均 {delta:+.3f} dB）。"
            else:
                better = "SF-ND" if delta < 0 else "SF"
                conclusion = f"{label} は {better} が {abs(delta):.3f} dB 低い。"
        lines.append(f"| {label} | {sf or 'なし'} | {db(means.get(sf))} | "
                     f"{sf_nd or 'なし'} | {db(means.get(sf_nd))} | {suffix} |")
        conclusions.append(conclusion)
    lines.extend(["", "## 結論", "", *(f"- {line}" for line in conclusions)])

    report_path = HERE / "RESULTS_nodelay_early.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (HERE / "RESULTS_nodelay_early.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, nd.COLUMNS, lineterminator="\n")
        writer.writeheader()
        for data in datasets:
            writer.writerows(data[rule]["row"] for rule in names)
    print(f"{len(datasets) * len(names)} trials; "
          + ", ".join(f"{direction}: {len(rules)} stable rules" for direction, rules in stable.items()))
    for line in conclusions:
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run", "report"))
    command = parser.parse_args().command
    {"check": check, "run": run, "report": report}[command]()


if __name__ == "__main__":
    main()
