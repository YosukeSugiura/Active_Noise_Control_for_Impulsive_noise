"""Tune and evaluate C_FXLMS with resumable per-case JSON.

Run python cc_eval.py tune, run, then report. Set BLAS threads to one when
using multiple workers. The report reads the prior final2 and coef2 archives.
"""
import argparse
from collections import Counter
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import cc
import coef as C
import final as F
import final2
from paired_ci import paired

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
TUNE = HERE / "TUNE_cc.json"
NAME = cc.NAME
PROPOSED = F.PROPOSED
APSIPA = "Freq_FxNLMS_CohVSS"
SWITCHES = ("RS-near", "RS-far")
VERSION = 2


def configure():
    final2.configure()
    cc.install(F)


def case_path(folder, job):
    cls, scen, seed = job
    return OUT / folder / f"{cls}_{scen}_s{seed}.json"


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    F.ab.atomic_write(path, lambda f: json.dump(data, f, indent=2, allow_nan=False))


def cached(path, settings):
    if path.exists():
        result = json.loads(path.read_text())
        if result.get("version") == VERSION and result.get("settings") == settings:
            return result
    return None


def tune_job(job):
    cls, scen, seed = job
    path = case_path("cc_tune", job)
    grid = cc.tune_grid(F)
    settings = [dict(key=key, mu=mu, params=params) for key, mu, params in grid]
    if cached(path, settings) is not None:
        return
    dd, _, _ = F.build(cls, scen, seed, F.LS_FINAL)
    rows = [dict(cls=cls, scen=scen, seed=seed,
                 **F.measures(F.run_baseline(NAME, mu, params, dd), dd))
            for _, mu, params in grid]
    dump(path, dict(version=VERSION, settings=settings, rows=rows))
    print(f"Tune {path.stem}: {len(rows)} settings", flush=True)


def component(setting):
    return "正規化" if setting["params"]["norm"] else "LMS"


def tune(procs):
    jobs = [(c, s, sd) for c in F.PILOT_CLASSES for s in F.TUNE_SCEN for sd in F.TUNE_SEEDS]
    with Pool(procs, initializer=configure) as pool:
        list(pool.imap_unordered(tune_job, jobs))
    cases = [json.loads(case_path("cc_tune", job).read_text()) for job in jobs]
    grid = cases[0]["settings"]
    assert len(jobs) == 90 and len(grid) == 28
    results = []
    for i, setting in enumerate(grid):
        rows = [case["rows"][i] for case in cases]
        ok = [r for r in rows if not r["div"]]
        results.append(dict(**setting, div=sum(r["div"] for r in rows),
                            **{k: F.class_mean(ok, k) for k in ("L1_whole", "L1_last10", "L1_ev")}))
    def score(r):
        return r["div"], float("inf") if r["L1_whole"] is None else r["L1_whole"]

    best_by_component = {
        label: min((r for r in results if component(r) == label), key=score)
        for label in ("正規化", "LMS")
    }
    best = min(results, key=score)
    dump(TUNE, dict(Ls=F.LS_FINAL, hp_hz=F.HP_HZ, seeds=F.TUNE_SEEDS,
                    scen=F.TUNE_SCEN, classes=F.PILOT_CLASSES, chosen={NAME: best},
                    best_by_component=best_by_component, grid=results))
    lines = ["# S2：C_FXLMS の調整", "",
             "設計用 seed 10–14、stat・RS-far、既存 S2 と同じ 9 クラスで各設定 90 実行。",
             "発散最少の設定の中で、完走分の L1 whole の等重みクラス平均が最小の設定を選択。", "",
             "| 設定 | 成分 | 速いステップ | 遅い／速い | 発散 | L1 whole | L1 last10 | L1 直後2秒 | 選択 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in results:
        key = r["key"].replace("|", r"\|")
        lines.append(f"| {key} | {component(r)} | {r['mu']:g} | {r['params']['ratio']:g} | {r['div']} | "
                     f"{F.fmt(r['L1_whole'])} | {F.fmt(r['L1_last10'])} | {F.fmt(r['L1_ev'])} | "
                     f"{'○' if r == best else ''} |")
    lines += ["", "## 成分フィルタ別の最良設定", "",
              "| 成分 | 速いステップ | 遅いステップ | 比 | 発散 | L1 whole |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, r in best_by_component.items():
        ratio = r["params"]["ratio"]
        lines.append(f"| {label} | {r['mu']:g} | {r['mu']*ratio:g} | {ratio:g} | "
                     f"{r['div']} | {F.fmt(r['L1_whole'])} |")
    (HERE / "RESULTS_cc_tune.md").write_text("\n".join(lines) + "\n")
    print(f"Selected {best}", flush=True)


def run_jobs():
    return [(c,s,sd) for c,s,sd,_ in F.final_jobs(False)]


def run_job(job):
    cls, scen, seed = job
    path = case_path("cc_run", job)
    coef_path = case_path("cc_coef", job)
    setting = json.loads(TUNE.read_text())["chosen"][NAME]
    result = cached(path, setting)
    need_coef = scen in SWITCHES and cached(coef_path, setting) is None
    if result is not None and not need_coef:
        return

    dd, _, _ = F.build(cls, scen, seed, F.LS_FINAL, final=True)

    def simulate(data, n_tot=None):
        return F.run_baseline(NAME, setting["mu"], setting["params"], data, n_tot)

    tag = dict(method=NAME, cls=cls, scen=scen, seed=seed, variant="main", kind="base")
    if result is None:
        sim = simulate(dd)
        post = None
        if scen in SWITCHES and not sim["div"]:
            post_data, _, _ = F.build(cls, "post:" + scen, seed, F.LS_FINAL, final=True)
            post = simulate(post_data)
        row = dict(**tag, **F.measures(sim, dd, post))
        if post is not None:
            row["post_div"] = int(post["div"])
        lam = sim["lam"][:sim["n_done"]]
        if scen in SWITCHES:
            t = dd["changes"][0] * F.B
            windows = {"pre2": lam[t - 2 * F.FS:t], "post1": lam[t:t + F.FS]}
            sizes = {"pre2": 2 * F.FS, "post1": F.FS}
        elif scen == "EN0":
            spans = [(round(a * F.FS), round(b * F.FS)) for a, b in F.BURSTS]
            windows = {"bursts": np.concatenate([lam[a:b] for a, b in spans])}
            sizes = {"bursts": sum(b - a for a, b in spans)}
        else:
            windows, sizes = {}, {}
        row["lam"] = {key: float(np.median(x)) for key, x in windows.items()
                      if len(x) == sizes[key]}
        dump(path, dict(version=VERSION, settings=setting, rows=[row]))
    else:
        row, = result["rows"]

    if need_coef:
        weights = {}
        for k in C.MAIN_KS:
            end = (C.CB + k) * F.B
            # The prefix ends after the last update before evaluation block CB+k.
            # Reuse the unmodified sample loop to get its exact mixed coefficients.
            prefix = simulate(dd, end)
            weights[k] = None if prefix["div"] else prefix["w"]
        point = dict(**tag, div=row["div"], J=C.frozen_anr(dd, weights))
        dump(coef_path, dict(version=VERSION, settings=setting, rows=[point]))
    print(f"Final {path.stem}: div={row['div']}", flush=True)


def run(procs):
    if not (HERE / "EVAL_FROZEN.json").is_file():
        raise SystemExit("EVAL_FROZEN.json is required")
    assert len(run_jobs()) == 1000
    with Pool(procs, initializer=configure) as pool:
        list(pool.imap_unordered(run_job, run_jobs()))


def read_rows(folder, jobs, methods, main_only=True):
    selected = []
    for job in jobs:
        path = case_path(folder, job)
        if not path.is_file():
            raise ValueError(f"Missing case: {path}")
        rows = [r for r in json.loads(path.read_text())["rows"]
                if r["method"] in methods and (not main_only or r.get("variant", "main") == "main")]
        if sorted(r["method"] for r in rows) != sorted(methods):
            raise ValueError(f"Unexpected methods in {path}")
        if any((r["cls"], r["scen"], r["seed"]) != job for r in rows):
            raise ValueError(f"Unexpected case identity in {path}")
        selected.extend(rows)
    return selected


def subset(rows, method, scen, cls=None):
    return [r for r in rows if r["method"] == method and r["scen"] == scen
            and (cls is None or r["cls"] == cls)]


def average(rows, key):
    vals = list(F.seed_table([r for r in rows if not r["div"]], key).values())
    return float(np.mean(vals)) if vals else None


def compare(left, right, metric, scale=1.0):
    result = paired(left, right, metric)
    if result is None:
        return "—", "欠測"
    mean, lo, hi, count = result
    direction = "提案法" if hi < 0 else "C_FXLMS" if lo > 0 else "0 を含む"
    return f"{mean*scale:+.3f} [{lo*scale:+.3f}, {hi*scale:+.3f}]（{count}）", direction


def report():
    jobs = run_jobs()
    coef_jobs = [job for job in jobs if job[1] in SWITCHES]
    setting = json.loads(TUNE.read_text())["chosen"][NAME]
    for folder, batch in (("cc_run", jobs), ("cc_coef", coef_jobs)):
        if any(cached(case_path(folder, job), setting) is None for job in batch):
            raise ValueError(f"{folder}: incomplete or stale results")
    cc_rows = read_rows("cc_run", jobs, (NAME,))
    prior = read_rows("final2_run", jobs, (PROPOSED, APSIPA))
    coef_rows = read_rows("cc_coef", coef_jobs, (NAME,))
    coef_rows += read_rows("coef2_run", coef_jobs, (PROPOSED,))
    final_rows = cc_rows + prior
    compared = (NAME, PROPOSED, APSIPA)
    results = {(m, s): subset(final_rows, m, s) for m in compared for s in F.FINAL_SCEN}
    ndiv = sum(r["div"] for r in cc_rows)
    npost = sum(r.get("post_div", 0) for r in cc_rows)
    ratio = setting["params"]["ratio"]
    lines = [
        "# C_FXLMS の最終評価", "",
        f"最終評価 {len(jobs)}/1000 条件、係数固定 {len(coef_jobs)}/400 条件。seed 100–119、Ŝ {F.LS_FINAL}、高域通過 {F.HP_HZ:g} Hz。",
        f"設計用 seed で選んだ設定は `{setting['key']}`（成分フィルタは {component(setting)}）。速いステップ {setting['mu']:g}、遅いステップ {setting['mu']*ratio:g}（比 {ratio:g}）。",
        f"C_FXLMS の発散は評価本体 {ndiv} 件、回復基準用の別実行 {npost} 件。",
        "L1 は完走分の等重みクラス平均を seed 間で平均（dB）。回復中央値は Kaplan–Meier 法（主表はブロック）。",
        "対応差は提案法から各比較手法を引いた値。完走対の seed ごとの平均を初期値 20261121 で 2,000 回再標本化。",
        "回復差は両方で回復を観測した対を使用。判定は丸め前の 95% 区間に基づく。", "",
    ]
    metrics = ("L1_whole", "L1_last10", "L1_ev")
    for scen in F.FINAL_SCEN:
        lines += [f"## {scen}", "",
                  "| 手法 | 試行 | 発散 | L1 whole | L1 last10 | L1 直後2秒 | rec 中央値 | 打切 | 誤確定 | 見逃し | nors-cap と一致 | 最大出力差 | 提案 − 当手法：L1 whole [95%] | 同：直後2秒 [95%] | 同：回復 [95%] |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|"]
        prop = results[PROPOSED, scen]
        for m in compared:
            rows = results[m, scen]
            med, cens = F.recovery_summary(rows)
            counts = [sum(r[k] for r in rows if r.get(k) is not None)
                      if any(r.get(k) is not None for r in rows) else "—"
                      for k in ("false_confirm", "missed")]
            same = [r["same_as_nors"] for r in rows if r.get("same_as_nors") is not None]
            max_diff = max((r["max_output_diff"] for r in rows if "max_output_diff" in r), default=None)
            diffs = [compare(prop, rows, lambda r: r.get(k))[0] if m != PROPOSED else "—"
                     for k in ("L1_whole", "L1_ev", "rec_blocks")]
            cells = [m, len(rows), sum(r["div"] for r in rows),
                     *(F.fmt(average(rows, k)) for k in metrics), med, cens, *counts,
                     f"{sum(same)}/{len(same)}" if same else "—",
                     "—" if max_diff is None else f"{max_diff:.6g}", *diffs]
            lines.append("| " + " | ".join(map(str, cells)) + " |")
        lines.append("")

    decisions = Counter()
    wins = []
    lines += ["## 提案法 − C_FXLMS の対応差", "",
              "係数固定は変化から 16・64 ブロック後の係数を変化後の信号に適用し、先頭 4096 サンプルを除いた誤差／一次騒音の電力比（dB）。", "",
              "| 条件 | クラス | 指標 | 提案法 | C_FXLMS | 対応差 [95%]（対数） | 判定 |",
              "|---|---|---|---:|---:|---|---|"]

    def row(scen, cls, title, group, metric, center, scale=1.0):
        left = subset(group, PROPOSED, scen, cls)
        right = subset(group, NAME, scen, cls)
        difference, verdict = compare(left, right, metric, scale)
        if cls is not None:
            decisions[title, verdict] += 1
        label = cls or "全クラス"
        if verdict == NAME:
            wins.append(f"- {scen}、{label}、{title} — {difference}")
        lines.append(f"| {scen} | {label} | {title} | {center(left)} | {center(right)} | {difference} | {verdict} |")

    def recovery_seconds(rows):
        med, cens = F.recovery_summary(rows)
        value = f"{int(med)*F.B/F.FS:.3f}" if med.isdigit() else med
        return f"{value}（{cens}）"

    def median_j(rows, k):
        values = [v for r in rows if not r["div"] and (v := r["J"][str(k)]) is not None]
        return F.fmt(float(np.median(values)) if values else None, 3)

    for s in F.FINAL_SCEN:
        for cls in (None, *F.FINAL_CLASSES):
            for key in metrics:
                if key == "L1_ev" and s not in ("EN0", "ENm10"):
                    continue
                row(s, cls, key, final_rows, lambda r: r.get(key),
                    lambda rows: F.fmt(average(rows, key), 3))
            if s in SWITCHES:
                row(s, cls, "回復 (s)", final_rows, lambda r: r.get("rec_blocks"),
                    recovery_seconds, F.B/F.FS)
                for k in C.MAIN_KS:
                    row(s, cls, f"固定係数 {k*F.B/F.FS:.2f} s", coef_rows,
                        lambda r: r["J"][str(k)], lambda rows: median_j(rows, k))
    lines += ["", "## クラス別の判定件数", "",
              "| 指標 | 提案法が良い | C_FXLMS が良い | 0 を含む | 欠測 |",
              "|---|---:|---:|---:|---:|"]
    totals = Counter()
    for title in dict.fromkeys(key[0] for key in decisions):
        counts = [decisions[title, v] for v in ("提案法", NAME, "0 を含む", "欠測")]
        totals.update(dict(enumerate(counts)))
        lines.append(f"| {title} | " + " | ".join(map(str, counts)) + " |")
    lines.append("| 合計 | " + " | ".join(str(totals[i]) for i in range(4)) + " |")
    lines += ["", "## 既存手法との回復中央値の比較", "",
              "| 条件 | C_FXLMS (s) | APSIPA 2026 版 (s) | C_FXLMS が速いか |",
              "|---|---:|---:|---|"]
    for s in SWITCHES:
        a, _ = F.recovery_summary(results[NAME, s])
        b, _ = F.recovery_summary(results[APSIPA, s])
        answer = ("はい" if int(a) < int(b) else "いいえ") if a.isdigit() and b.isdigit() else "中央値に到達せず"
        lines.append(f"| {s} | {recovery_seconds(results[NAME,s])} | {recovery_seconds(results[APSIPA,s])} | {answer} |")
    lines += ["", "## 混合係数 λ", "",
              "各実行の窓内のサンプル中央値を求め、その実行間中央値を記載。EN0 は 10 秒と 20 秒からの各 0.5 秒を結合。",
              "λ が大きいほど速いフィルタの比重が大きい。窓を完了した実行を集計。", "",
              "| 条件 | 窓 | 中央値 | 実行数 |", "|---|---|---:|---:|"]
    for s, key, label in [(s, k, w) for s in SWITCHES for k,w in
                          (("pre2", "変化前 2 秒"), ("post1", "変化後 0–1 秒"))] + [("EN0", "bursts", "外乱中")]:
        values = [r["lam"][key] for r in results[NAME,s] if key in r["lam"]]
        lines.append(f"| {s} | {label} | {F.fmt(float(np.median(values)) if values else None, 4)} | {len(values)} |")
    lines += ["", "## C_FXLMS が良かった比較", "", *(wins or ["該当なし。"])]
    (HERE / "RESULTS_cc.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("tune", "run", "report"))
    parser.add_argument("--procs", type=int, default=2)
    args = parser.parse_args()
    configure()
    if args.command == "tune":
        tune(args.procs)
    elif args.command == "run":
        run(args.procs)
    else:
        report()


if __name__ == "__main__":
    main()
