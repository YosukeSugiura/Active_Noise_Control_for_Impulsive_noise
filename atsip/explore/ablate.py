"""Run the final-preset SF ablations and report from the per-seed CSV."""
import argparse
import csv
import json
import math
import re
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from statistics import mean

import protoC as p

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "ablate"
CLASSES = ("G", "BU", "PS", "II300", "a145", "a120", "pc", "pcb")
SEEDS = tuple(range(10, 15))
LATE_STEPS = (0.01, 0.02, 0.03, 0.05, 0.07, 0.1)
SIMPLE = (
    ("SF fix", False, False, False),
    ("SF fix+g", False, True, False),
    ("SF early+fix+g", True, True, False),
    ("SF early+fix+g+rsN2", True, True, True),
)
A4 = "SF-Y est91 delayopt g gam.03"
A5 = A4 + " early"
A6 = p.FINAL_RULE
A7 = "SF-Y est91 delayopt none gam.03 early+rsN2"
OTHER = (A4, A5, A6, A7, "SF-V2", "SF+ramp")
COLUMNS = (
    "tag", "class", "seed", "rule", "L1_whole", "L1_first5", "L1_last10",
    "ex_0_1s", "ex_0_5s", "ex_last10", "L1_pc15_17", "L1_pcb10_12",
    "L1_pcb20_22", "div_num", "amp", "false_rs", "missed_rs",
)


def fixed_name(family, step):
    return f"{SIMPLE[family][0]} mu{step:g}"


RULES = tuple(fixed_name(family, step) for family in range(len(SIMPLE)) for step in LATE_STEPS) + OTHER


class FixedSchedule(p.Rule):
    """A fixed internal step with optional finite early phase and rsN2."""

    def __init__(self, step, early, correct, detector, rw, options):
        super().__init__()
        self.step, self.correct = step, correct
        self.detector, self.rw = detector, rw
        self.gmin = options.P.gmin
        self.early_step, self.early_length = options.mu_e, options.X3["n_early"]
        self.early_until = self.early_length if early else 0
        self.restarts = []
        self.tr.update(mu_state=[], restart_apply_block=[])

    def mu_formula(self, i):
        return self.early_step if i < self.early_until else self.step

    def block(self, i, ctx):
        internal = self.mu_formula(i)
        factor = 1.0 / max(ctx.g, self.gmin) if self.correct else 1.0
        mu = internal * factor
        direction = p.direction("SF", ctx.R, ctx.E, ctx.Srr, ctx.Pbar, ctx.eps)
        applied = bool(self.restarts) and self.restarts[-1] == i - 1
        if self.detector is not None:
            z = p.rsn_stat(ctx.E, float(self.rw[i]))
            if self.detector.update(i, z):
                self.restarts.append(i)
                self.early_until = i + 1 + self.early_length
        self.tr["mu_app"].append(mu)
        self.tr["mu_state"].append(internal)
        self.tr["restart_apply_block"].append(i if applied else -1)
        return mu, direction


def factories(data, nb, o):
    rw = p.rwin_energy(data, nb, 2 * p.M)
    rules = {}
    for family, (_, early, correct, restart) in enumerate(SIMPLE):
        for step in LATE_STEPS:
            def make(step=step, early=early, correct=correct, restart=restart):
                detector = p.RSNDetector(o.X4) if restart else None
                return FixedSchedule(step, early, correct, detector, rw, o)
            rules[fixed_name(family, step)] = make
    source = p.make_rules3(nb, o.P, o.X2, o.X3, o.mu_e)
    rules.update({name: source[name] for name in (A4, A5, "SF-V2", "SF+ramp")})
    for name, correction in ((A6, "g"), (A7, "none")):
        rules[name] = partial(p.SFY4, "est91", 0.03, o.P, o.X2, o.X3, o.X4, True, "N", o.mu_e, rw, corr=correction)
    return rules


def score(tag, name, rule, sim, data):
    cls, seed = tag.rsplit("_s", 1)
    metrics = p.metrics(sim, data)
    nb, ex = sim["nb"], sim["ex"]
    pcb = p.pcb_summ(sim, data, nb) if cls == "pcb" else {}
    row = dict(tag=tag, seed=int(seed), rule=name, L1_whole=metrics["L1"][0], L1_first5=metrics["L1"][1],
               L1_last10=metrics["L1"][2], ex_0_1s=p.dbm(ex[:p.BPS]), ex_0_5s=p.dbm(ex[:5 * p.BPS]),
               ex_last10=p.dbm(ex[-max(1, nb // 10):]), L1_pc15_17=metrics.get("pc", {}).get("L1a"),
               L1_pcb10_12=pcb.get("L1_10-12"), L1_pcb20_22=pcb.get("L1_20-22"),
               div_num=int(metrics["div"]), amp=int(metrics["amp"]))
    row["class"] = cls
    restart = None
    if name in (A6, A7) or name.startswith(SIMPLE[3][0] + " "):
        tr = sim["tr"]
        restart = p.restart_stats(rule.restarts, p.change_blocks(data), len(tr["mu_app"]), tr["restart_apply_block"])
        row.update(false_rs=restart["n_false"], missed_rs=restart["n_missed"])
    return row, restart


def atomic_write(path, writer):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer(handle)
    temp.replace(path)


def experiment_tag(job):
    tag, short, o = job
    start = time.monotonic()
    folder = OUT / "smoke" if short else OUT
    file = folder / f"{tag}.json"
    data, Rall, stats = p.prep3(tag, short)
    nb = short or len(data["d"]) // p.B
    rows, restarts = [], {}
    rules = factories(data, nb, o)
    for name in RULES:
        rule = rules[name]()
        sim = p.simulate(data, Rall, stats, rule, nb=nb)
        row, restart = score(tag, name, rule, sim, data)
        rows.append(row)
        if restart is not None:
            restarts[name] = restart
        if short and name == A6:
            original = p.make_rules4(nb, o.P, o.X2, o.X3, o.X4, o.mu_e, data)[A6]()
            check = p.simulate(data, Rall, stats, original, nb=nb)
            for key in ("e", "w", "ex"):
                p.np.testing.assert_array_equal(sim[key], check[key])
    result = dict(par=p.job_par("c5", o), nb=nb, rows=rows, restarts=restarts)
    atomic_write(file, partial(json.dump, result))
    print(f"{tag}: {len(rows)} rules, nb={nb}, divergence={sum(r['div_num'] for r in rows)}, "
          f"amplification={sum(r['amp'] for r in rows)}, {time.monotonic() - start:.1f} s", flush=True)
    return rows


def run(smoke, procs):
    short = 1250 if smoke else None
    seeds = SEEDS[:1] if smoke else SEEDS
    o = p.parse(["c5", "--preset", "final"])
    if p.mu_early_value(o.P, o.early_k) != o.mu_e or o.mu_e != 0.678 or o.X3["n_early"] != 64:
        raise ValueError("The early step or length differs from the final preset")
    folder = OUT / "smoke" if smoke else OUT
    folder.mkdir(parents=True, exist_ok=True)
    jobs = [(f"{cls}_s{seed}", short, o) for cls in CLASSES for seed in seeds]
    if procs == 1:
        batches = list(map(experiment_tag, jobs))
    else:
        with Pool(procs) as workers:
            batches = list(workers.imap(experiment_tag, jobs))
    with_path = folder / "RESULTS_ablate.csv" if smoke else HERE / "RESULTS_ablate.csv"
    def write(handle):
        output = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        output.writeheader()
        for batch in batches:
            for row in batch:
                output.writerow({key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                                 for key, value in row.items()})
    atomic_write(with_path, write)
    return with_path


SCORE_LABELS = {
    "L1_whole": "L1 whole", "L1_first5": "L1 first5", "L1_last10": "L1 last10",
    "ex_0_1s": "ex 0–1 s", "L1_pc15_17": "L1 pc 15–17 s",
    "L1_pcb10_12": "L1 pcb 10–12 s", "L1_pcb20_22": "L1 pcb 20–22 s",
}
CHANGE_FIELDS = {"pc": ("L1_pc15_17",), "pcb": ("L1_pcb10_12", "L1_pcb20_22")}
SCORE_FIELDS = tuple(SCORE_LABELS)
COUNTS = ("div_num", "amp", "false_rs", "missed_rs")
THRESHOLD_DB = 0.1


def read_scores(path, native):
    groups = defaultdict(dict)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(COLUMNS) if native else {"class", "seed", "rule", "L1_whole"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"{path.name}: missing columns {required - set(reader.fieldnames or ())}")
        for row in reader:
            cls, name, seed = row["class"], row["rule"], int(row["seed"])
            group = groups[cls, name]
            if seed in group:
                raise ValueError(f"{path.name}: duplicate row {cls}, {name}, {seed}")
            if native and row["tag"] != f"{cls}_s{seed}":
                raise ValueError(f"{path.name}: inconsistent tag {row['tag']}")
            group[seed] = {field: float(row[field]) for field in (*SCORE_FIELDS, *COUNTS)
                           if row.get(field) not in ("", None)}
    if native:
        expected = {(cls, rule) for cls in CLASSES for rule in RULES}
        if set(groups) != expected or any(set(group) != set(SEEDS) for group in groups.values()):
            raise ValueError(f"{path.name}: expected all 8 classes, 30 rules and seeds 10–14")
    return groups


def values(group, key):
    return [r[key] for r in group.values() if key in r and math.isfinite(r[key])]


def averaged(group, key):
    scores = values(group, key)
    return mean(scores) if scores else math.nan


def tuned_steps(groups):
    def objective(family, step):
        name = fixed_name(family, step)
        train = [groups[cls, name] for cls in ("G", "a145")]
        if any(len(values(group, "L1_whole")) != len(SEEDS) for group in train):
            return math.inf
        return mean(averaged(group, "L1_whole") for group in train)

    selected = []
    for family in range(len(SIMPLE)):
        step = min(LATE_STEPS, key=lambda step: (objective(family, step), step))
        criterion = objective(family, step)
        if not math.isfinite(criterion):
            raise ValueError(f"A{family}: no mu has a complete tuning score")
        selected.append((step, criterion))
    return selected


def paired(groups, cls, left, right, key):
    first, second = groups[cls, left], groups[cls, right]
    pairs = []
    for seed in SEEDS:
        a = first[seed].get(key, math.nan)
        b = second[seed].get(key, math.nan)
        if math.isfinite(a) and math.isfinite(b):
            pairs.append(a - b)
    return pairs


def db(value):
    return f"{value:.3f}" if math.isfinite(value) else "—"


def difference(value):
    if not math.isfinite(value):
        return "集計不可"
    if abs(value) < THRESHOLD_DB:
        return f"差なし（{value:+.3f} dB）"
    return f"{value:+.3f} dB"


def paired_cell(deltas):
    if not deltas:
        return "—"
    avg = mean(deltas)
    status = "（差なし）" if abs(avg) < THRESHOLD_DB else ""
    return (f"{avg:+.3f}{status} [{min(deltas):+.3f}, {max(deltas):+.3f}]"
            f" · {sum(v < 0 for v in deltas)}/{len(deltas)}")


def class_differences(groups, left, right, field):
    return {cls: mean(sample) if (sample := paired(groups, cls, left, right, field)) else math.nan for cls in CLASSES}


def class_statement(deltas):
    labels = []
    for label, subset in (
        ("改善", [cls for cls, val in deltas.items() if val <= -THRESHOLD_DB]),
        ("悪化", [cls for cls, val in deltas.items() if val >= THRESHOLD_DB]),
        ("差なし", [cls for cls, val in deltas.items() if abs(val) < THRESHOLD_DB]),
        ("集計不可", [cls for cls, val in deltas.items() if not math.isfinite(val)]),
    ):
        if subset:
            labels.append(f"{label} {', '.join(subset)}")
    return "。".join(labels)


def external_families(external, cls):
    pattern = re.compile(r"^(.+?)\s+mu([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$")
    families, standalone = defaultdict(list), []
    for input_class, name in external:
        if input_class != cls:
            continue
        match = pattern.fullmatch(name)
        if match:
            families[match[1]].append((float(match[2]), name))
        else:
            standalone.append(name)
    leaders = sorted(standalone)
    all_steps = []
    for family in sorted(families):
        candidates = sorted(families[family])
        all_steps.extend(name for _, name in candidates)

        def order(candidate):
            step, name = candidate
            score = averaged(external[cls, name], "L1_whole")
            return (score if math.isfinite(score) else math.inf, step, name)

        winner = min(candidates, key=order)
        if math.isfinite(order(winner)[0]):
            leaders.append(winner[1])
    return leaders, all_steps


def table_header(fields):
    return ["| 則 | n | " + " | ".join(SCORE_LABELS[k] for k in fields) + " | 数値発散 | 増幅 | 誤作動 | 見逃し |",
            "|---|" + "---:|" * (len(fields) + 5)]


def table_row(label, group, fields):
    n = len(values(group, "L1_whole"))
    results = [db(averaged(group, field)) for field in fields]
    counts = [str(int(sum(v))) if (v := values(group, key)) else "—" for key in COUNTS]
    return f"| {label} | {n}/{len(group)} | " + " | ".join(results + counts) + " |"


def report():
    groups = read_scores(HERE / "RESULTS_ablate.csv", native=True)
    path = HERE / "RESULTS_nodelay.csv"
    external = read_scores(path, native=False) if path.is_file() else {}
    tuned = tuned_steps(groups)
    chosen = [fixed_name(family, step) for family, (step, _) in enumerate(tuned)]
    fixed_pair = fixed_name(0, tuned[1][0])
    early_pair = fixed_name(1, tuned[2][0])
    restart_pair = fixed_name(2, tuned[3][0])

    est = class_differences(groups, A6, chosen[3], "L1_whole")
    est_early = class_differences(groups, A5, chosen[2], "L1_first5")
    corrected = class_differences(groups, A6, A7, "L1_whole")
    fixed_corrected = class_differences(groups, chosen[1], fixed_pair, "L1_whole")
    restart = [(SCORE_LABELS[key], paired(groups, cls, A6, A5, key))
               for cls, names in CHANGE_FIELDS.items() for key in names]
    restart_text = "、".join(f"{label} {difference(mean(sample) if sample else math.nan)}"
                            for label, sample in restart)
    div_a6, div_a7 = (sum(int(sum(values(groups[cls, name], "div_num"))) for cls in CLASSES)
                      for name in (A6, A7))
    pc_est = paired(groups, "pc", A6, chosen[3], "L1_pc15_17")
    pc_est_text = difference(mean(pc_est) if pc_est else math.nan)
    all_win = all(len(sample) == len(SEEDS) and max(sample) < 0 for _, sample in restart)
    early_adaptive = class_differences(groups, A5, A4, "L1_first5")
    early_fixed = class_differences(groups, chosen[2], early_pair, "L1_first5")

    lines = [
        "# ATSIP 最終則のアブレーション", "", "## 結論", "",
        f"- **推定器**　A6 − A3 の L1 whole は、{class_statement(est)}。pc 15–17 s は {pc_est_text}。"
        f"A5 − A2 の L1 first5 は、{class_statement(est_early)}。",
        f"- **1/g**　A6 − A7 の有限な対応試行の L1 whole は、{class_statement(corrected)}。"
        f"数値発散は A6 が {div_a6}/40 件、A7 が {div_a7}/40 件。同じ μ の A1 − A0 は、"
        f"{class_statement(fixed_corrected)}。",
        f"- **再始動**　A6 − A5 の経路変化直後の差は、{restart_text}。"
        + ("3 窓すべてで 5/5 seed が改善。" if all_win else ""), "",
        "## 条件と μ の選び方", "",
        "- 8 クラス、seed 10–14、各 30 秒。最終評価用 seed 100–119 は未使用。"
        "A4–A6 は `--preset final`、A7 は A6 の統計と適用ステップから 1/g を外した。再始動検出器は A3・A6・A7 で共通。",
        "- A2・A3 は再計算した 0.678 を開始時に 64 回適用し、直後に μ を切り替える。"
        "A3 は再始動を検出した次のブロックから同じ 64 回を適用する。"
        "A4 は初期値 0.3 を使う。A5–A7 は既存の更新順序に従い、"
        "初回の推定器更新が反映されるのは次ブロック。",
        "- A0–A3 のそれぞれで、G と a145 の各 5 seed の L1 whole 平均をクラス間で等重み平均し、"
        "値が最小の μ を選んだ。全 10 試行の有限値が必要で、同点は小さい μ を採用する。"
        "G・a145 の最良 μ 行には選定に使った試行が含まれる。",
        "- dB は小さいほど良い。last10 は末尾 10%。絶対差 0.1 dB 未満を「差なし」とする。"
        "数値発散と増幅は [RESULTS_fix0.md](RESULTS_fix0.md) と同じ定義。再始動を評価しない欄は —。", "",
        "| 番号 | 規則名 |",
        "|---|---|",
        *[f"| A{family} | `{name} mu{{μ_L}}` |" for family, (name, _, _, _) in enumerate(SIMPLE)],
        *[f"| A{family} | `{name}` |" for family, name in enumerate((A4, A5, A6, A7), 4)],
        "",
        "| 則 | 選択 μ | G・a145 の選定平均（dB） |",
        "|---|---:|---:|",
    ]
    for family, (step, objective) in enumerate(tuned):
        lines.append(f"| A{family} | {step:g} | {objective:.3f} |")

    lines.extend([
        "",
        "前半の効果を見ると、A5 − A4 の L1 first5 のクラス平均差は "
        f"{min(early_adaptive.values()):+.3f}～{max(early_adaptive.values()):+.3f} dB。"
        f"同一 μ={tuned[2][0]:g} の A2 − A1 では "
        f"{min(early_fixed.values()):+.3f}～{max(early_fixed.values()):+.3f} dB。", "",
    ])

    source = "手元の `RESULTS_nodelay.csv` を併記した。" if path.is_file() else "手元に `RESULTS_nodelay.csv` がないため、遅れ除去則の値は未掲載。"
    lines.extend(["## 主表", "", source, "",
                  "各セルはクラス内の有限値の平均、件数は合計。n は有限な L1 whole を持つ試行数 / 全試行数。"
                  "遅れ除去則の μ 掃引は、各クラス・各規則系列で L1 whole 平均が最良の μ を代表とする。", ""])
    for cls in CLASSES:
        fields = SCORE_FIELDS[:4] + CHANGE_FIELDS.get(cls, ())
        lines.extend([f"### {cls}", "", *table_header(fields)])
        for name in RULES:
            lines.append(table_row(name, groups[cls, name], fields))
        for family, name in enumerate(chosen):
            lines.append(table_row(f"**A{family} 最良：{name}**", groups[cls, name], fields))
        representatives, _ = external_families(external, cls)
        for name in representatives:
            label = f"{name}（遅れ除去側）" if (cls, name) in groups else name
            lines.append(table_row(label, external[cls, name], fields))
        lines.append("")

    contrasts = [
        ("A6 − A3 最良", A6, chosen[3]),
        ("A6 − A2 最良", A6, chosen[2]),
        ("A5 − A2 最良", A5, chosen[2]),
        (f"A1 − A0（同一 μ={tuned[1][0]:g}）", chosen[1], fixed_pair),
        ("A6 − A7", A6, A7),
        ("A4 − A1 最良", A4, chosen[1]),
        ("A5 − A4", A5, A4),
        (f"A2 − A1（同一 μ={tuned[2][0]:g}）", chosen[2], early_pair),
        ("A6 − A5", A6, A5),
        (f"A3 − A2（同一 μ={tuned[3][0]:g}）", chosen[3], restart_pair),
    ]
    if tuned[0][0] != tuned[1][0]:
        contrasts.insert(4, ("A1 最良 − A0 最良", chosen[1], chosen[0]))
    lines.extend(["## seed 対応差（dB）", "",
                  "セルは 左 − 右の平均 [最小, 最大] · 左が小さい seed 数 / 有効な対応数。"
                  "勝数は厳密な負の差を数え、実用上の「差なし」は平均差に適用する。", ""])
    for cls in CLASSES:
        fields = SCORE_FIELDS[:4] + CHANGE_FIELDS.get(cls, ())
        lines.extend([f"### {cls}", "", "| 比較 | " + " | ".join(SCORE_LABELS[key] for key in fields) + " |",
                      "|---|" + "---:|" * len(fields)])
        for title, left, right in contrasts:
            cells = [paired_cell(paired(groups, cls, left, right, key)) for key in fields]
            lines.append(f"| {title} | " + " | ".join(cells) + " |")
        lines.append("")

    if external:
        lines.extend(["## 付録　遅れ除去則の全 μ", ""])
        for cls in sorted({input_class for input_class, _ in external}, key=lambda c: (c not in CLASSES, c)):
            _, all_steps = external_families(external, cls)
            if all_steps:
                fields = SCORE_FIELDS[:4] + CHANGE_FIELDS.get(cls, ())
                lines.extend([f"### {cls}", "", *table_header(fields)])
                lines.extend(table_row(name, external[cls, name], fields) for name in all_steps)
                lines.append("")

    lines.extend([
        "## 再現手順", "",
        "```bash",
        "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
        "python ablate.py run --smoke --procs 2",
        "python ablate.py run --procs 2",
        "python ablate.py report",
        "```", "",
        "遅れ除去側の `RESULTS_nodelay.csv` が用意できたら、このディレクトリに置いて"
        " `python ablate.py report` を再実行する。", "",
    ])
    output = HERE / "RESULTS_ablate.md"
    atomic_write(output, lambda handle: handle.write("\n".join(lines)))
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    run_cli = subs.add_parser("run")
    run_cli.add_argument("--smoke", action="store_true")
    run_cli.add_argument("--procs", type=int, choices=(1, 2), default=2)
    subs.add_parser("report")
    args = parser.parse_args()
    if args.command == "run":
        print(run(args.smoke, args.procs))
    else:
        report()


if __name__ == "__main__":
    main()
