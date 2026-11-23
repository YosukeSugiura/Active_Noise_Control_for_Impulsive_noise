"""Class-specific C_FXLMS tuning and supplementary evaluation."""
import argparse
from collections import Counter
import json
from multiprocessing import Pool
import tarfile

import cc_eval as E
from paired_ci import paired

F = E.F
ROOT = E.HERE
SETTINGS = ROOT / "TUNE_cc_class.json"
FOLDER = "cc_class_run"
SCEN = ("stat", "RS-far")


def read(path):
    return json.loads(path.read_text())


def archive_inputs():
    for folder, count in (("final2_run", 1000), ("cc_run", 1000)):
        if len(list((E.OUT / folder).glob("*.json"))) != count:
            with tarfile.open(ROOT / "results" / f"{folder}.tar.gz") as tar:
                tar.extractall(E.OUT, members=(m for m in tar if m.name.split("/")[0] == folder),
                               filter="data")


def tune(procs):
    previous = read(ROOT / "TUNE.json")["per_class"]
    classes = tuple(F.PILOT_CLASSES)
    assert all(set(c) == set(classes) for c in previous.values())
    common = read(E.TUNE)["chosen"][E.NAME]
    grid = [dict(key=k, mu=mu, params=p) for k, mu, p in E.cc.tune_grid(F)]
    assert len(grid) == 28 and tuple(F.TUNE_SCEN) == SCEN and tuple(F.TUNE_SEEDS) == tuple(range(10, 15))
    jobs = [(c, s, sd) for c in classes for s in SCEN for sd in F.TUNE_SEEDS]
    missing = [job for job in jobs if E.cached(E.case_path("cc_tune", job), grid) is None]
    if missing:
        with Pool(procs, initializer=E.configure) as pool:
            list(pool.imap_unordered(E.tune_job, missing))
    per_class = {}
    lines = ["# C_FXLMS の入力別調整", "",
             "stat・RS-far、seed 10–14。入力ごとに発散件数、完走分の L1 whole の順で最小化。a120 は共通設定。",
             f"既存の格子探索を {len(jobs)-len(missing)}/{len(jobs)} 条件で再利用。", "",
             "| 入力 | 成分 | 速いステップ | 比 | 発散（入力別／共通） | L1 whole（入力別） | L1 whole（共通） | 入力別 − 共通 (dB) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for c in classes:
        cases = [E.cached(E.case_path("cc_tune", j), grid) for j in jobs if j[0] == c]
        scores = []
        for i, setting in enumerate(grid):
            rows = [case["rows"][i] for case in cases]
            scores.append(dict(**setting, div=sum(r["div"] for r in rows),
                               L1_whole=F.agg([r for r in rows if not r["div"]], "L1_whole")))
        best = min(scores, key=lambda r: (r["div"], r["L1_whole"] if r["L1_whole"] is not None else float("inf")))
        base = next(r for r in scores if r["key"] == common["key"])
        delta = (best["L1_whole"] - base["L1_whole"]
                 if best["L1_whole"] is not None and base["L1_whole"] is not None else None)
        per_class[c] = dict(best, common_div=base["div"], common_L1_whole=base["L1_whole"], delta=delta)
        lines.append(f"| {c} | {E.component(best)} | {best['mu']:g} | {best['params']['ratio']:g} | "
                     f"{best['div']}/{base['div']} | {F.fmt(best['L1_whole'],3)} | "
                     f"{F.fmt(base['L1_whole'],3)} | {F.fmt(delta,3)} |")
    E.dump(SETTINGS, dict(seeds=F.TUNE_SEEDS, scen=SCEN, common=common, per_class=per_class))
    (ROOT / "RESULTS_cc_class_tune.md").write_text("\n".join(lines) + "\n")


def jobs():
    return [(c, s, sd) for c in F.FINAL_CLASSES for s in SCEN for sd in F.FINAL_SEEDS]


def evaluate(job):
    cls, scen, seed = job
    config = read(SETTINGS)
    common = config["common"]
    choice = config["per_class"].get(cls, common)
    setting = {k: choice[k] for k in ("key", "mu", "params")}
    path = E.case_path(FOLDER, job)
    if E.cached(path, setting) is not None:
        return
    if setting["key"] == common["key"]:
        saved = E.cached(E.case_path("cc_run", job), common)
        if saved is None:
            raise ValueError(f"Invalid common-setting record: {job}")
        row, = saved["rows"]
        row = dict(row, reused=True)
    else:
        def simulate(s):
            dd, _, _ = F.build(cls, s, seed, F.LS_FINAL, final=True)
            return dd, F.run_baseline(E.NAME, setting["mu"], setting["params"], dd)
        dd, sim = simulate(scen)
        post = simulate("post:" + scen)[1] if F.has_post(scen) and not sim["div"] else None
        row = dict(method=E.NAME, cls=cls, scen=scen, seed=seed, reused=False,
                   **F.measures(sim, dd, post))
        if post is not None:
            row["post_div"] = int(post["div"])
    row.update(variant="class", key=setting["key"])
    E.dump(path, dict(version=E.VERSION, settings=setting, rows=[row]))
    print(f"{path.stem}: div={row['div']}, reused={row['reused']}", flush=True)


def report():
    proposed, apsipa, local, common = E.PROPOSED, E.APSIPA, "C_FXLMS（入力別）", "C_FXLMS（共通）"
    groups = {name: [] for name in (local, common, apsipa, proposed)}
    tune_base = read(ROOT / "TUNE.json")
    for job in jobs():
        saved = read(E.case_path("final2_run", job))["rows"]
        variants = list(F.final_settings(tune_base, apsipa, job[0], job[1]))
        variant = variants[-1][0]
        for m, v in ((proposed,"main"), (apsipa,variant)):
            row, = [r for r in saved if (r["method"],r["variant"]) == (m,v)]
            groups[m].append(row)
        for name, folder in ((local,FOLDER),(common,"cc_run")):
            row, = read(E.case_path(folder, job))["rows"]
            assert (row["cls"],row["scen"],row["seed"]) == job and row["method"] == E.NAME
            groups[name].append(row)

    def rows(name, scen, cls=None):
        return [r for r in groups[name] if r["scen"] == scen and (cls is None or r["cls"] == cls)]

    def contrast(scen, comparator, key, cls=None, scale=1):
        result = paired(rows(proposed,scen,cls), rows(comparator,scen,cls), lambda r: r.get(key))
        if result is None:
            return "—", "欠測"
        mean, lo, hi, n = result
        verdict = "提案法" if hi < 0 else "C_FXLMS" if lo > 0 else "0 を含む"
        return f"{mean*scale:+.3f} [{lo*scale:+.3f}, {hi*scale:+.3f}]（{n}）", verdict

    reused = sum(r["reused"] for r in groups[local])
    post_div = sum(r.get("post_div",0) for r in groups[local])
    lines = ["# C_FXLMS の入力別設定による補助評価", "",
             f"400/400 条件。新規 {400-reused} 件、共通設定から再利用 {reused} 件。回復基準用の発散 {post_div} 件。",
             "a120 は共通設定。既存比較は保存済みの記録を使用。L1 は dB、補助表の回復中央値は Kaplan–Meier 法によるブロック数。",
             "対応差は提案法 − 比較手法。95% 区間は seed 単位で 2,000 回再標本化。回復の差は両方で回復を観測した対のみ。", "",
             "## 補助表", "",
             "| 条件 | 手法 | 発散 | L1 whole | L1 直後2秒 | rec 中央値 | 打切 | 提案 − 当手法：L1 whole [95%] |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for s in SCEN:
        for m in (local, common, apsipa, proposed):
            rs = rows(m,s)
            med, cens = F.recovery_summary(rs)
            diff = contrast(s,m,"L1_whole")[0] if m != proposed else "—"
            avg = [F.fmt(E.average(rs,k)) for k in ("L1_whole","L1_ev")]
            lines.append(f"| {s} | {m} | {sum(r['div'] for r in rs)} | {' | '.join(avg)} | {med} | {cens} | {diff} |")
    lines += ["", "## 提案法 − C_FXLMS（入力別）の対応差", "",
              "| 条件 | 入力 | 指標 | 差 [95%]（対数） | 判定 |",
              "|---|---|---|---|---|"]
    comparisons = (("stat","L1_whole","L1 whole (dB)",1), ("RS-far","L1_whole","L1 whole (dB)",1),
                   ("RS-far","L1_ev","L1 直後 2 秒 (dB)",1), ("RS-far","rec_blocks","回復 (s)",F.B/F.FS))
    counts = Counter()
    wins = []
    for s,key,label,scale in comparisons:
        for c in (None,*F.FINAL_CLASSES):
            diff, verdict = contrast(s,local,key,c,scale)
            name = c or "全体"
            lines.append(f"| {s} | {name} | {label} | {diff} | {verdict} |")
            if c:
                counts[s,key,verdict] += 1
            if verdict == "C_FXLMS":
                wins.append(f"- {s}、{name}、{label} — {diff}")

    lines += ["", "## RS-far の回復中央値（秒）", "",
              "括弧内は打ち切りの件数。", "",
              f"| 入力 | 提案法 | {local} | {common} |", "|---|---:|---:|---:|"]
    for c in (None,*F.FINAL_CLASSES):
        values = []
        for m in (proposed,local,common):
            med,cens = F.recovery_summary(rows(m,"RS-far",c))
            value = f"{int(med)*F.B/F.FS:.3f}" if med.isdigit() else med
            values.append(f"{value}（{cens}）")
        lines.append(f"| {c or '全体'} | {' | '.join(values)} |")

    lines += ["", "## 入力別の判定件数", "",
              "| 条件 | 指標 | 提案法が良い | C_FXLMS が良い | 0 を含む | 欠測 |",
              "|---|---|---:|---:|---:|---:|"]
    labels = ("提案法","C_FXLMS","0 を含む","欠測")
    for s,key,label,_ in comparisons:
        vals = [str(counts[s,key,v]) for v in labels]
        lines.append(f"| {s} | {label} | {' | '.join(vals)} |")
    total = [str(sum(n for (_,_,v),n in counts.items() if v == label)) for label in labels]
    lines += [f"| 合計 |  | {' | '.join(total)} |", "", "## C_FXLMS が良い比較", "", *(wins or ["該当なし。"])]
    (ROOT / "RESULTS_cc_class.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--procs", type=int, default=2)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    E.configure()
    archive_inputs()
    if not args.report_only:
        tune(args.procs)
        with Pool(args.procs, initializer=E.configure) as pool:
            list(pool.imap_unordered(evaluate, jobs()))
    report()


if __name__ == "__main__":
    main()
