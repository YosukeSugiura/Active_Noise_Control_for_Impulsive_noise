"""Time-domain normalized modified filtered-x LMS (Bjarnason, EUSIPCO 1992).
Sign convention e=d+S*y. The normalized variant is defined here.
Run tune, eval [--scen], class, report. All caches are method-specific.
"""
import argparse
from collections import Counter
import json
from multiprocessing import Pool
import time

import numpy as np
import cc_eval as E
import final as F
import final2
import coef as C
from fastbase import njit
from paired_ci import paired

NAME = "MFxNLMS"
BASE = (0.1, {"eps": 1e-3})
MUS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
SCEN = ("stat", "RS-near", "RS-far", "EN0", "ENm10")
CLASS_SCEN = ("stat", "RS-far")
SWITCHES = ("RS-near", "RS-far")
TUNE = E.HERE / "TUNE_mfx.json"


@njit(cache=True)
def _loop(mu, eps, d, xr, rr, sw, hrevs, lengths, hm, M, B, pref, div_gain):
    T = len(d)
    L = hrevs.shape[1]
    Lm = len(hm)
    w = np.zeros(M)
    e = np.zeros(T)
    ybuf = np.zeros(2 * L)
    mbuf = np.zeros(2 * Lm)
    head = 0
    mh = 0
    seg = 0
    for n in range(T):
        if seg + 1 < len(sw) and n == sw[seg + 1]:
            seg += 1
        i = T - 1 - n
        xv = xr[i:i + M]
        rv = rr[i:i + M]
        y = np.dot(w, xv)
        ybuf[head] = y
        ybuf[head + L] = y
        head = (head + 1) % L
        mbuf[mh] = y
        mbuf[mh + Lm] = y
        mh = (mh + 1) % Lm
        lh = lengths[seg]
        en = d[n] + np.dot(ybuf[head + L - lh:head + L], hrevs[seg, :lh])
        e[n] = en
        finite = np.isfinite(en)
        if finite:
            dhat = en - np.dot(mbuf[mh:mh + Lm], hm)
            etilde = dhat + np.dot(w, rv)
            gain = mu * etilde / (np.dot(rv, rv) + eps)
            for k in range(M):
                w[k] -= gain * rv[k]
        bad = not finite
        if not bad and (n + 1) % B == 0:
            pw = 0.0
            for k in range(n + 1 - B, n + 1):
                pw += e[k] * e[k]
            pw /= B
            if not np.isfinite(pw) or pw > div_gain * pref:
                bad = True
            else:
                for k in range(M):
                    if not np.isfinite(w[k]):
                        bad = True
                        break
        if bad:
            return e, w, True, (n // B) * B, n
    return e, w, False, T, -1


def run_baseline(F, mu, params, dd, n_tot=None):
    M, B = F.M, F.B
    assert M == 512
    d, x, r = dd["d"], dd["x"], dd["r"]
    T = len(d) if n_tot is None else min(n_tot, len(d))
    plant = dd["hS_seq"]
    lengths = np.array([len(h) for _, h in plant], dtype=np.int64)
    hrevs = np.zeros((len(plant), int(lengths.max())))
    for i, (_, h) in enumerate(plant):
        hrevs[i, :lengths[i]] = h[::-1]
    sw = np.array([b0 * B for b0, _ in plant], dtype=np.int64)
    # The five evaluation scenarios and their post references have a fixed model.
    if len(dd["hSm_seq"]) != 1:
        raise ValueError("MFxNLMS: changing secondary model is outside this experiment")
    hm = np.ascontiguousarray(dd["hSm_seq"][0][1][::-1], dtype=np.float64)
    xr = np.ascontiguousarray(np.concatenate([x[:T][::-1], np.zeros(M - 1)]), dtype=np.float64)
    rr = np.ascontiguousarray(np.concatenate([r[:T][::-1], np.zeros(M - 1)]), dtype=np.float64)
    pref = dd.get("primary_power", float(np.mean(d ** 2)))
    t0 = time.perf_counter()
    e, w, div, n_done, fail_n = _loop(float(mu), float(params.get("eps", 1e-3)),
        np.ascontiguousarray(d[:T], dtype=np.float64), xr, rr, sw, hrevs,
        lengths, hm, M, B, float(pref), float(F.DIV_GAIN))
    failure = None
    if div:
        failure = dict(F.error_peak(e, n_done), sample=int(fail_n))
        e[n_done:] = np.nan
    return dict(e=e, w=w, div=bool(div), n_done=int(n_done), failure=failure,
                seconds=time.perf_counter() - t0)


def install(F):
    F.BASE[NAME] = BASE
    if getattr(F.run_baseline, "_mfx", False):
        return
    orig = F.run_baseline
    def run(name, mu, params, dd, n_tot=None):
        return run_baseline(F, mu, params, dd, n_tot) if name == NAME else orig(name, mu, params, dd, n_tot)
    run._mfx = True
    F.run_baseline = run


def configure():
    final2.configure()
    install(F)


def read(path):
    return json.loads(path.read_text())


def grid():
    return [dict(key=f"mu{mu:g}", mu=mu, params=dict(BASE[1])) for mu in MUS]


def score(row):
    return row["div"], float("inf") if row["L1_whole"] is None else row["L1_whole"]


def tune_job(job):
    cls, scen, seed = job
    path = E.case_path("mfx_tune", job)
    settings = grid()
    if E.cached(path, settings) is not None:
        return
    dd, _, _ = F.build(cls, scen, seed, F.LS_FINAL)
    rows = [dict(cls=cls, scen=scen, seed=seed,
                 **F.measures(F.run_baseline(NAME, s["mu"], s["params"], dd), dd))
            for s in settings]
    E.dump(path, dict(version=E.VERSION, settings=settings, rows=rows))
    print(f"Tune {path.stem}: {len(rows)} settings", flush=True)


def tune(procs):
    jobs = [(c, s, sd) for c in F.PILOT_CLASSES for s in F.TUNE_SCEN for sd in F.TUNE_SEEDS]
    assert len(jobs) == 90 and tuple(F.TUNE_SEEDS) == tuple(range(10, 15))
    assert tuple(F.TUNE_SCEN) == CLASS_SCEN
    with Pool(procs, initializer=configure) as pool:
        list(pool.imap_unordered(tune_job, jobs))
    cases = {job: E.cached(E.case_path("mfx_tune", job), grid()) for job in jobs}
    def scores(batch):
        result = []
        for i, setting in enumerate(grid()):
            rows = [cases[j]["rows"][i] for j in batch]
            ok = [r for r in rows if not r["div"]]
            result.append(dict(**setting, div=sum(r["div"] for r in rows),
                               **{k: F.class_mean(ok, k) for k in ("L1_whole", "L1_last10", "L1_ev")}))
        return result
    results = scores(jobs)
    common = min(results, key=score)
    class_grid = {c: scores([j for j in jobs if j[0] == c]) for c in F.PILOT_CLASSES}
    local = {c: min(rows, key=score) for c, rows in class_grid.items()}
    E.dump(TUNE, dict(Ls=F.LS_FINAL, hp_hz=F.HP_HZ, seeds=F.TUNE_SEEDS,
                      scen=F.TUNE_SCEN, classes=F.PILOT_CLASSES, chosen={NAME: common},
                      per_class=local, grid=results, per_class_grid=class_grid))
    print("Selected common:", common, flush=True)
    print("Selected per class:", {c: s["mu"] for c, s in local.items()}, flush=True)


def jobs(scen=SCEN):
    return [(c, s, sd) for c in F.FINAL_CLASSES for s in scen for sd in F.FINAL_SEEDS]


def settings_for(job, local=False):
    cfg = read(TUNE)
    common = cfg["chosen"][NAME]
    return cfg["per_class"].get(job[0], common) if local else common


def evaluate(job, local=False):
    cls, scen, seed = job
    setting = settings_for(job, local)
    path = E.case_path("mfx_class_run" if local else "mfx_run", job)
    coef_path = E.case_path("mfx_coef", job)
    saved = E.cached(path, setting)
    need_coef = not local and scen in SWITCHES and E.cached(coef_path, setting) is None
    if saved is not None and not need_coef:
        return
    tag = dict(method=NAME, cls=cls, scen=scen, seed=seed,
               variant="class" if local else "main", kind="base")
    common_saved = None
    if local and setting["key"] == settings_for(job)["key"]:
        common_saved = E.cached(E.case_path("mfx_run", job), settings_for(job))
    if saved is None and common_saved is not None:
        row = dict(common_saved["rows"][0], **tag, reused=True, key=setting["key"])
        E.dump(path, dict(version=E.VERSION, settings=setting, rows=[row]))
        print(f"{path.stem}: reused", flush=True)
        return
    dd, _, _ = F.build(cls, scen, seed, F.LS_FINAL, final=True)
    def simulate(data, n_tot=None):
        return F.run_baseline(NAME, setting["mu"], setting["params"], data, n_tot)
    if saved is None:
        sim = simulate(dd)
        post = None
        if scen in SWITCHES and not sim["div"]:
            post_data, _, _ = F.build(cls, "post:" + scen, seed, F.LS_FINAL, final=True)
            post = simulate(post_data)
        row = dict(**tag, **F.measures(sim, dd, post))
        if post is not None:
            row["post_div"] = int(post["div"])
        if local:
            row.update(reused=False, key=setting["key"])
        E.dump(path, dict(version=E.VERSION, settings=setting, rows=[row]))
    else:
        row, = saved["rows"]
    if need_coef:
        weights = {}
        for k in C.MAIN_KS:
            prefix = simulate(dd, (C.CB + k) * F.B)
            weights[k] = None if prefix["div"] else prefix["w"]
        point = dict(**tag, div=row["div"], J=C.frozen_anr(dd, weights))
        E.dump(coef_path, dict(version=E.VERSION, settings=setting, rows=[point]))
    print(f"{path.stem}: div={row['div']}", flush=True)


def class_job(job):
    evaluate(job, local=True)


def load_mfx(folder, batch, local=False):
    result = []
    for job in batch:
        record = E.cached(E.case_path(folder, job), settings_for(job, local))
        if record is None:
            raise ValueError(f"{folder}: missing/stale {job}")
        row, = record["rows"]
        if (row["cls"], row["scen"], row["seed"]) != job or row["method"] != NAME:
            raise ValueError(f"Unexpected record: {folder} {job}")
        result.append(row)
    return result


def report():
    batch = jobs()
    class_batch = jobs(CLASS_SCEN)
    coef_batch = jobs(SWITCHES)
    assert (len(batch), len(class_batch), len(coef_batch)) == (1000, 400, 400)
    common = load_mfx("mfx_run", batch)
    local = load_mfx("mfx_class_run", class_batch, True)
    fixed = load_mfx("mfx_coef", coef_batch)
    prior = E.read_rows("final2_run", batch, (F.PROPOSED, E.APSIPA))
    fixed_prior = E.read_rows("coef2_run", coef_batch, (F.PROPOSED,))
    cfg = read(TUNE)
    tune_base = read(E.HERE / "TUNE.json")
    class_prior = []
    for job in class_batch:
        variants = list(F.final_settings(tune_base, E.APSIPA, job[0], job[1]))
        variant = variants[-1][0]
        old = read(E.case_path("final2_run", job))["rows"]
        row, = [r for r in old if (r["method"], r["variant"]) == (E.APSIPA, variant)]
        class_prior.append(row)

    def subset(rows, scen, cls=None):
        return [r for r in rows if r["scen"] == scen and (cls is None or r["cls"] == cls)]
    prop = [r for r in prior if r["method"] == F.PROPOSED]
    apsipa = [r for r in prior if r["method"] == E.APSIPA]
    metrics = ("L1_whole", "L1_last10", "L1_ev")
    def contrast(left, right, metric, scale=1):
        v = paired(left, right, metric)
        if v is None:
            return "—", "欠測"
        mean, lo, hi, count = v
        verdict = "提案法" if hi < 0 else NAME if lo > 0 else "0 を含む"
        return f"{mean*scale:+.3f} [{lo*scale:+.3f}, {hi*scale:+.3f}]（{count}）", verdict
    def recsec(rows):
        med, cens = F.recovery_summary(rows)
        val = f"{int(med)*F.B/F.FS:.3f}" if med.isdigit() else med
        return f"{val}（{cens}）"
    def mj(rows, k):
        vals = [v for r in rows if not r["div"] and (v := r["J"][str(k)]) is not None]
        return F.fmt(float(np.median(vals)) if vals else None, 3)

    chosen = cfg["chosen"][NAME]
    lines = ["# MFxNLMS 最終評価", "",
             f"設計 90 条件×7 設定、共通 {len(common)}/1000、係数固定 {len(fixed)}/400、入力別 {len(local)}/400 条件。",
             f"共通ステップ: {chosen['mu']:g}、正規化定数: 0.001。a120 は共通設定。",
             f"共通評価の発散 {sum(r['div'] for r in common)} 件、回復基準用 {sum(r.get('post_div',0) for r in common)} 件。",
             f"入力別の発散 {sum(r['div'] for r in local)} 件、回復基準用 {sum(r.get('post_div',0) for r in local)} 件。",
             "L1 は完走分のクラス等重み seed 平均 (dB)。回復中央値は Kaplan–Meier 法。対応差は提案法−比較手法。",
             "両方完走の対応差を seed 単位で 2000 回ブートストラップ (default_rng(20261121))。回復差は両方で回復観測の対。", "",
             "## 選定結果", "", "| 入力 | μ | 発散 | L1 whole |", "|---|---:|---:|---:|"]
    for name, setting in [("共通", chosen), *cfg["per_class"].items()]:
        lines.append(f"| {name} | {setting['mu']:g} | {setting['div']} | {F.fmt(setting['L1_whole'])} |")
    lines.append(f"| a120（設計外） | {chosen['mu']:g} | — | — |")

    def main_table(scen, groups, label=""):
        lines.extend(["", f"## {scen}{label}", "",
                      "| 手法 | 試行 | 発散 | L1 whole | L1 last10 | L1 直後2秒 | rec 中央値 (block) | 打切 | 誤確定 | 見逃し | nors-cap と一致 | 最大出力差 | 提案−当手法 whole [95%] | 同 直後2秒 [95%] | 同 回復 [95%] |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|"])
        left = subset(prop, scen)
        for name, group in groups:
            rows = subset(group, scen)
            med, cens = F.recovery_summary(rows)
            counts = [sum(r[k] for r in rows if r.get(k) is not None)
                      if any(r.get(k) is not None for r in rows) else "—" for k in ("false_confirm", "missed")]
            same = [r["same_as_nors"] for r in rows if r.get("same_as_nors") is not None]
            md = max((r["max_output_diff"] for r in rows if "max_output_diff" in r), default=None)
            diffs = [contrast(left, rows, lambda r: r.get(k))[0] if name != F.PROPOSED else "—"
                     for k in ("L1_whole", "L1_ev", "rec_blocks")]
            cells = [name, len(rows), sum(r["div"] for r in rows),
                     *(F.fmt(E.average(rows, k)) for k in metrics), med, cens, *counts,
                     f"{sum(same)}/{len(same)}" if same else "—", "—" if md is None else f"{md:.6g}", *diffs]
            lines.append("| " + " | ".join(map(str, cells)) + " |")

    for s in SCEN:
        main_table(s, [(NAME, common), (F.PROPOSED, prop), (E.APSIPA, apsipa)])
    for s in CLASS_SCEN:
        main_table(s,
                   [(NAME+"（入力別）", local), (NAME+"（共通）", common),
                    (F.PROPOSED, prop), (E.APSIPA+"（入力別）", class_prior)], "（入力別）")

    counts = Counter()
    wins = []
    lines.extend(["", "## 提案法 − MFxNLMS：対応差", "",
                  "固定係数は変化後16・64ブロックの係数を用い、先頭4096サンプルを除いた誤差／一次騒音電力比。",
                  "| 設定 | 条件 | 入力 | 指標 | 提案法 | MFxNLMS | 差 [95%]（対数） | 判定 |",
                  "|---|---|---|---|---:|---:|---|---|"])
    def add(mode, s, c, label, left, right, metric, center, scale=1):
        a, b = subset(left, s, c), subset(right, s, c)
        diff, verdict = contrast(a, b, metric, scale)
        lines.append(f"| {mode} | {s} | {c or '全クラス'} | {label} | {center(a)} | {center(b)} | {diff} | {verdict} |")
        if c is not None:
            counts[mode, label, verdict] += 1
        if verdict == NAME:
            wins.append(f"- {mode}、{s}、{c or '全クラス'}、{label}: {diff}")
    for mode, batch_scen, rows in (("共通", SCEN, common), ("入力別", CLASS_SCEN, local)):
        for s in batch_scen:
            for c in (None, *F.FINAL_CLASSES):
                for key in metrics:
                    if key == "L1_ev" and s == "stat":
                        continue
                    add(mode, s, c, key, prop, rows, lambda r: r.get(key),
                        lambda rr: F.fmt(E.average(rr, key), 3))
                if s in SWITCHES:
                    add(mode, s, c, "回復 (s)", prop, rows, lambda r: r.get("rec_blocks"), recsec, F.B/F.FS)
                    if mode == "共通":
                        for k in C.MAIN_KS:
                            add(mode, s, c, f"固定係数 {k*F.B/F.FS:.2f} s", fixed_prior, fixed,
                                lambda r: r["J"][str(k)], lambda rr: mj(rr, k))
    lines.extend(["", "## クラス別判定件数", "",
                  "| 設定 | 指標 | 提案法が良い | MFxNLMS が良い | 0 を含む | 欠測 |",
                  "|---|---|---:|---:|---:|---:|"])
    for mode, label in dict.fromkeys((m, k) for m, k, _ in counts):
        vals = [str(counts[mode, label, v]) for v in ("提案法", NAME, "0 を含む", "欠測")]
        lines.append(f"| {mode} | {label} | " + " | ".join(vals) + " |")
    lines.extend(["", "## 回復中央値の比較 (s)", "",
                  "| 条件 | 入力 | 提案法 | MFxNLMS 共通 | MFxNLMS 入力別 | APSIPA 共通 |",
                  "|---|---|---:|---:|---:|---:|"])
    for s in SWITCHES:
        for c in (None, *F.FINAL_CLASSES):
            vals = [recsec(subset(g,s,c)) for g in (prop,common,local,apsipa)]
            if s not in CLASS_SCEN:
                vals[2] = "—"
            lines.append(f"| {s} | {c or '全体'} | " + " | ".join(vals) + " |")
    lines.extend(["", "## MFxNLMS が良かった比較", "", *(wins or ["該当なし。"])])
    (E.HERE / "RESULTS_mfx.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("tune", "eval", "class", "report"))
    parser.add_argument("--procs", type=int, default=2)
    parser.add_argument("--scen", choices=SCEN)
    args = parser.parse_args()
    configure()
    if args.scen and args.command != "eval":
        parser.error("--scen is only for eval")
    if args.command == "tune":
        tune(args.procs)
    elif args.command == "report":
        report()
    else:
        if not (E.HERE / "EVAL_FROZEN.json").is_file():
            raise SystemExit("EVAL_FROZEN.json is required")
        batch = jobs(CLASS_SCEN if args.command == "class" else (args.scen,) if args.scen else SCEN)
        worker = class_job if args.command == "class" else evaluate
        with Pool(args.procs, initializer=configure) as pool:
            list(pool.imap_unordered(worker, batch))


if __name__ == "__main__":
    main()
