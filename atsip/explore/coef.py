"""Post-hoc analysis (decided after the final evaluation; not one of the pre-set measures): recovery measured on the
coefficients.  The coefficients at k blocks after the change are frozen and applied to the whole post-change signal;
J(k) = error power / primary power over that segment.  Unlike the short-time error, J(k) does not depend on whether
the input happens to be active around block k, so it can be read for intermittent inputs.

  python coef.py run [--dry] [--procs K] [--shard k/n]     (--dry: design seeds 10-11)
  python coef.py report [--dry]
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import final as F

np, p = F.np, F.p
B, CB = F.B, F.CB
HERE = Path(__file__).resolve().parent
KS = (0, 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1500, 1874)
RULES = ("nors-cap", "P5-early-cap", "P4-cap", "SF-fix")
BLOCK = ("Freq_FxNLMS_CohVSS", "FreqFxLMS")
SCEN = ("RS-near", "RS-far", "RM-10", "RM-40")
SKIP = 4096                      # samples after the change left out of J (tail of the old path)
VERSION = 2
MAIN_KS = (16, 64)               # evaluation points of the main table: 0.13 s and 0.51 s after the change (fixed before the run)


def out_dir(dry):
    return HERE / "out" / ("coef_dry" if dry else "coef_run")


def frozen_anr(dd, ws):
    n0 = CB * B
    xs = p.signal.lfilter(F.seq_at(dd["hS_seq"], CB), [1.0], dd["x"])
    d = dd["d"][n0 + SKIP:]; pd_ = float(d @ d)
    out = {}
    for k, w in ws.items():
        if w is None or not np.isfinite(w).all():
            out[k] = None
            continue
        e = d + p.signal.oaconvolve(xs, w)[n0 + SKIP:len(dd["d"])]
        out[k] = float(10 * np.log10((e @ e) / pd_))
    return out


def run_job(job):
    cls, scen, seed, dry = job
    f = out_dir(dry) / f"{cls}_{scen}_s{seed}.json"
    if f.exists() and json.loads(f.read_text()).get("version") == VERSION:
        return
    t0 = time.monotonic()
    tune = json.loads((HERE / "TUNE.json").read_text())["chosen"]
    X4 = p.parse(["c5", "--preset", "final"]).X4
    dd, Rall, st = F.build(cls, scen, seed, F.LS_FINAL, final=not dry)
    want = {CB + k: k for k in KS}
    rows = []

    def record(method, sim, ws, confirmed):
        curve = frozen_anr(dd, {k: ws.get(k) for k in KS})
        if dry and (cls, scen, seed) == ("G", "RS-far", 10):
            n = F.FS
            e, d = sim["e"][-n:], dd["d"][-n:]
            actual = 10 * np.log10((e @ e) / (d @ d))
            assert not sim["div"] and abs(curve[KS[-1]] - actual) <= 3.0, (method, curve[KS[-1]], actual)
            print(f"check {method}: frozen={curve[KS[-1]]:.3f}, actual last second={actual:.3f} dB", flush=True)
        return dict(method=method, div=int(sim["div"]), J=curve, confirmed=confirmed)

    for label in RULES:
        ws = {}

        def hook(i, c, eb, mu):
            if i in want:
                ws[want[i]] = c.w.copy()
        rule, sim = F.run_rule(label, dd, Rall, st, X4, hook=hook)
        rows.append(record(label, sim, ws, [t["decide"] + 1 - CB for t in getattr(rule, "trials", [])
                                            if t["status"] == "confirmed" and t["decide"] >= CB]))
    orig = F.make_algo
    for name in BLOCK:
        ws = {}; cnt = [0]

        def make(nm, par):
            algo = orig(nm, par); upd = algo.update

            def update(w, e, r, x, mu):
                out = upd(w, e, r, x, mu)
                cnt[0] += 1                     # called once per block: out[0] is used from block cnt[0] on
                if cnt[0] in want:
                    ws[want[cnt[0]]] = out[0].copy()
                return out
            algo.update = update
            return algo
        F.make_algo = make
        try:
            sim = F.run_baseline_block(name, tune[name]["mu"], tune[name]["params"], dd)
        finally:
            F.make_algo = orig
        rows.append(record(name, sim, ws, []))
    for r in rows:
        r.update(cls=cls, scen=scen, seed=int(seed))
    f.parent.mkdir(parents=True, exist_ok=True)
    F.ab.atomic_write(f, lambda h: json.dump(dict(version=VERSION, rows=rows), h))
    print(f"{f.stem}: {time.monotonic() - t0:.0f} s", flush=True)


def jobs(dry):
    return [(c, s, sd, dry) for sd in ((10, 11) if dry else F.FINAL_SEEDS) for c in F.FINAL_CLASSES for s in SCEN]


def rec(J, target):
    """First grid offset from which J stays within target + 3 dB; inf if never."""
    ok = [J[str(k)] is not None and J[str(k)] <= target + 3.0 for k in KS]
    if not ok[-1]:
        return float("inf")
    i = len(KS) - 1
    while i > 0 and ok[i - 1]:
        i -= 1
    return KS[i]


def report(dry):
    d = out_dir(dry)
    rows = [r for j in jobs(dry) if (d / f"{j[0]}_{j[1]}_s{j[2]}.json").exists()
            for r in json.loads((d / f"{j[0]}_{j[1]}_s{j[2]}.json").read_text())["rows"]]
    methods = RULES + BLOCK
    cond = {}
    for r in rows:
        cond.setdefault((r["cls"], r["scen"], r["seed"]), {})[r["method"]] = r
    for c in cond.values():                     # common target: the best final level among the methods of the condition
        ends = [m["J"][str(KS[-1])] for m in c.values() if not m["div"] and m["J"][str(KS[-1])] is not None]
        tgt = min(ends) if ends else None
        for m in c.values():
            m["rec_common"] = rec(m["J"], tgt) if not m["div"] and tgt is not None else float("inf")
            own = m["J"][str(KS[-1])]
            m["rec_own"] = rec(m["J"], own) if not m["div"] and own is not None else float("inf")
    med = lambda v: "—" if not v else ("なし" if np.isinf(np.median(v)) else f"{np.median(v) * B / F.FS:.3f}")
    L = ["# 事後分析：係数で測った回復（最終評価の後に決めた指標。事前に固定した指標ではない）", "",
         f"完了 {len(cond)}/{len(jobs(dry))} 条件。変化から k ブロック後の係数を固定し、変化後の先頭4096サンプルを除いた区間に適用したときの 誤差／一次騒音 のパワー比 J(k)（dB）。",
         "回復＝J が基準 +3 dB 以内に入り、以後の格子点でとどまる最初の k（ブロック単位の格子：" + "、".join(map(str, KS)) + "）。「なし」＝最後まで入らない。",
         "基準は 2 通り：共通（その条件で全手法の最終値の最良）、手法ごと（その手法の最終値）。発散した実行は「なし」。", ""]
    sec = lambda k: f"{k * B / F.FS:.2f} 秒"
    L += [f"## 主表：変化から {sec(MAIN_KS[0])}後と {sec(MAIN_KS[1])}後の消音量（評価点は seed 100–119 の実行前に固定。2026-11-21）", "",
          f"その時点の係数を固定して変化後の先頭4096サンプルを除いた区間に適用したときの 誤差／一次騒音（dB、小さいほど良い）。{sec(MAIN_KS[0])}＝{MAIN_KS[0]} ブロック＝{MAIN_KS[0] * B} サンプル、"
          f"{sec(MAIN_KS[1])}＝{MAIN_KS[1]} ブロック＝{MAIN_KS[1] * B} サンプル（16 kHz）。", "",
          "提案 − 比較の対応差（dB、負は提案法が良い）。seed を再標本化した 95% 区間。両方が完走した対。", "",
          "| 条件 | クラス | 変化からの時間 | 提案法 | APSIPA 2026 版 | 再始動なし | 提案 − APSIPA 2026 版 [95%] | 提案 − 再始動なし [95%] |", "|---|---|---:|---:|---:|---:|---|---|"]

    def pdiff(scen, c, k, other):
        per = {}
        for (cl, sc, sd), ms in cond.items():
            if sc == scen and (c == "全クラス" or cl == c) and other in ms and "P5-early-cap" in ms:
                x, y = ms["P5-early-cap"]["J"][str(k)], ms[other]["J"][str(k)]
                if not ms["P5-early-cap"]["div"] and not ms[other]["div"] and x is not None and y is not None:
                    per.setdefault(sd, []).append(x - y)
        if not per:
            return "—"
        v = np.array([np.mean(x) for x in per.values()]); bt = np.random.default_rng(1).choice(v, size=(2000, len(v))).mean(axis=1)
        return f"{v.mean():+.2f} [{np.quantile(bt, .025):+.2f}, {np.quantile(bt, .975):+.2f}]"
    for scen in SCEN:
        for c in ("全クラス",) + F.FINAL_CLASSES:
            for k in MAIN_KS:
                mj = lambda m: (lambda v: f"{np.median(v):.2f}" if v else "—")([r["J"][str(k)] for r in rows if r["method"] == m and r["scen"] == scen
                                                                             and (c == "全クラス" or r["cls"] == c) and not r["div"] and r["J"][str(k)] is not None])
                L.append(f"| {scen} | {c} | {sec(k)} | {mj('P5-early-cap')} | {mj('Freq_FxNLMS_CohVSS')} | {mj('nors-cap')} | "
                         f"{pdiff(scen, c, k, 'Freq_FxNLMS_CohVSS')} | {pdiff(scen, c, k, 'nors-cap')} |")
    L += ["", "以下は補助（曲線の全体と、回復までの秒数）。", ""]
    for key, name in (("rec_common", "共通の基準"), ("rec_own", "手法ごとの基準")):
        for scen in SCEN:
            L += [f"## 回復の中央値（{name}、{scen}、秒）", "", "| 手法 | 全クラス | " + " | ".join(F.FINAL_CLASSES) + " |", "|---|---:|" + "---:|" * len(F.FINAL_CLASSES)]
            for m in methods:
                g = lambda c: [r[key] for r in rows if r["method"] == m and r["scen"] == scen and (c is None or r["cls"] == c)]
                L.append(f"| {m} | {med(g(None))} | " + " | ".join(med(g(c)) for c in F.FINAL_CLASSES) + " |")
            L.append("")
    L += ["## 対応差：提案法 − 比較（共通の基準での回復、両方が回復した対。seed を再標本化した 95% 区間）", "",
          "| 条件 | クラス | 比較 | 平均差（秒） | 95% 区間 | 対数 | 提案だけ回復 | 比較だけ回復 |", "|---|---|---|---:|---|---:|---:|---:|"]
    rng = np.random.default_rng(0)
    for scen in SCEN:
        for c in ("全クラス",) + F.FINAL_CLASSES:
            for other in ("Freq_FxNLMS_CohVSS", "nors-cap"):
                per = {}; only_a = only_b = 0
                for (cl, sc, sd), ms in cond.items():
                    if sc != scen or (c != "全クラス" and cl != c) or other not in ms or "P5-early-cap" not in ms:
                        continue
                    a, b = ms["P5-early-cap"]["rec_common"], ms[other]["rec_common"]
                    if np.isfinite(a) and np.isfinite(b):
                        per.setdefault(sd, []).append((a - b) * B / F.FS)
                    only_a += np.isfinite(a) and not np.isfinite(b); only_b += np.isfinite(b) and not np.isfinite(a)
                if not per:
                    L.append(f"| {scen} | {c} | {other} | — | — | 0 | {only_a} | {only_b} |"); continue
                v = np.array([np.mean(x) for x in per.values()])
                bt = rng.choice(v, size=(2000, len(v))).mean(axis=1)
                L.append(f"| {scen} | {c} | {other} | {v.mean():+.3f} | [{np.quantile(bt, .025):+.3f}, {np.quantile(bt, .975):+.3f}] | {sum(map(len, per.values()))} | {only_a} | {only_b} |")
    L += ["", "## 消音量の時間変化（中央値、dB。列は変化からの時間）", ""]
    show = (0, 8, 16, 32, 64, 128, 256, 512, 1874)
    for scen in SCEN:
        for c in F.FINAL_CLASSES:
            L += [f"### {scen}、{c}", "", "| 手法 | " + " | ".join(f"{k * B / F.FS:.2f} 秒" for k in show) + " |", "|---|" + "---:|" * len(show)]
            for m in methods:
                cells = []
                for k in show:
                    v = [r["J"][str(k)] for r in rows if r["method"] == m and r["scen"] == scen and r["cls"] == c and not r["div"] and r["J"][str(k)] is not None]
                    cells.append(f"{np.median(v):.2f}" if v else "—")
                L.append(f"| {m} | " + " | ".join(cells) + " |")
            L.append("")
    name = "RESULTS_coef_dry.md" if dry else "RESULTS_coef.md"
    (HERE / name).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", name, len(rows), "rows")


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=("run", "report")); a.add_argument("--dry", action="store_true")
    a.add_argument("--procs", type=int, default=os.cpu_count() or 1); a.add_argument("--shard", default="0/1")
    o = a.parse_args()
    if o.cmd == "run":
        k, n = (int(v) for v in o.shard.split("/"))
        js = [j for i, j in enumerate(jobs(o.dry)) if i % n == k]
        with Pool(o.procs) as pool:
            list(pool.imap_unordered(run_job, js))
    else:
        report(o.dry)
