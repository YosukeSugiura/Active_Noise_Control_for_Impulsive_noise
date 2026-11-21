"""Detector variants inside the proposed rule (design seeds only).  The rule itself is unchanged; only the detector
object is replaced.

  base   the current detector (guard 64, baseline updated every block)
  F64    baseline frozen from the trigger block to the end of the guard (guard 64)
  F32    the same with guard 32

  python det2.py run [--procs K]     (seeds 10-29; out/det2/*.json)
  python det2.py report              (RESULTS_det3.md)
"""
import argparse, json, os, time
from multiprocessing import Pool
from pathlib import Path
import final as F
import coef as C

np, p = F.np, F.p
B, CB = F.B, F.CB
HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "det2"
SEEDS = tuple(range(10, 30))
SCEN = ("stat", "RS-near", "RS-far", "EN0", "ENm10")
VARIANTS = {"base": (False, 64), "F64": (True, 64), "F32": (True, 32)}
KS = (16, 64)
VERSION = 1


class FreezeDetector(p.RSNDetector):
    """RSNDetector whose baseline is not updated from the trigger block to the end of the guard."""

    def update(s, i, z):
        in_guard = s.last_restart is not None and i - s.last_restart < s.guard
        keep = s.zbar
        fired = super().update(i, z)
        if fired or in_guard:
            s.zbar = keep
        return fired


def run_one(variant, dd, Rall, st, X4):
    freeze, guard = VARIANTS[variant]
    rule = F.P5Cap(F.q5.config(early_ok=(4, 0.25)), dd, len(Rall), X4)
    assert isinstance(rule.det, p.RSNDetector)
    X = dict(X4, guard=guard)
    rule.det = FreezeDetector(X) if freeze else p.RSNDetector(X)
    rule.frame = F.q4.Frame4(dd, rule)
    ws = {}

    def hook(i, c, eb, mu):
        if i - CB in KS:
            ws[i - CB] = c.w.copy()
    F._SF_CACHE[0] = st
    try:
        t0 = time.perf_counter()
        sim = F.simulate(dd, Rall, st, rule, error_frame=rule.frame, hook=hook)
        sim["seconds"] = time.perf_counter() - t0
    finally:
        F._SF_CACHE[0] = None
    return rule, sim, ws


def run_job(job):
    cls, scen, seed = job
    assert seed in SEEDS
    f = OUT / f"{cls}_{scen}_s{seed}.json"
    if f.exists() and json.loads(f.read_text()).get("version") == VERSION:
        return
    t0 = time.monotonic()
    X4 = p.parse(["c5", "--preset", "final"]).X4
    dd, Rall, st = F.build(cls, scen, seed, F.LS_FINAL)
    assert len(dd["hSm_seq"]) == 1
    _, ref = F.run_rule("nors-cap", dd, Rall, st, X4)
    rows = []
    for v in VARIANTS:
        rule, sim, ws = run_one(v, dd, Rall, st, X4)
        row = dict(variant=v, cls=cls, scen=scen, seed=int(seed), **F.measures(sim, dd), **F.trial_stats(rule, dict(dd, scored_blocks=sim["n_done"])))
        row["n_trials"] = len(rule.trials); row["n_transfer"] = len(rule.restarts)
        row["trial_blocks"] = int(sum((t["decide"] if t["decide"] is not None else t["block"] + 16) - t["block"] for t in rule.trials))
        if not sim["div"] and not ref["div"]:
            row["same_as_nors"] = int(np.array_equal(sim["e"], ref["e"]) and np.array_equal(sim["y"], ref["y"]))
        if dd["changes"] and not sim["div"]:
            J = C.frozen_anr(dd, {k: ws.get(k) for k in KS}); row.update({f"J{k}": J[k] for k in KS})
        row.pop("trials", None)
        rows.append(row)
    f.parent.mkdir(parents=True, exist_ok=True)
    F.ab.atomic_write(f, lambda h: json.dump(dict(version=VERSION, rows=rows), h))
    print(f"{f.stem}: {time.monotonic() - t0:.0f} s", flush=True)


def report():
    rows = [r for f in sorted(OUT.glob("*.json")) for r in json.loads(f.read_text())["rows"]]
    sec = lambda k: f"{k * B / F.FS:.2f} 秒後"
    L = ["# 検出器の変更を提案法に組み込んだ比較（設計用 seed 10–29）", "",
         f"{len(rows) // len(VARIANTS)} 条件。base＝今の検出器、F64＝検出から保護区間の終わりまで基準値を更新しない（保護 64）、F32＝同（保護 32）。", "",
         "## 条件ごとの合計", "", "| 条件 | 検出器 | 実行 | 発散 | 見逃し | 誤った移行 | 再始動なしと一致 | 試行の回数/分 | 試行中のブロックの割合 | L1 whole | "
         + " | ".join(f"消音量 {sec(k)}" for k in KS) + " |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|" + "---:|" * len(KS)]
    mean = lambda v: "—" if not v else f"{np.mean(v):.2f}"
    for scen in SCEN:
        for v in VARIANTS:
            rs = [r for r in rows if r["scen"] == scen and r["variant"] == v]; ok = [r for r in rs if not r["div"]]
            same = [r["same_as_nors"] for r in ok if "same_as_nors" in r]
            nblk = sum(r.get("n_done", 3750) if False else 3750 for r in ok)
            L.append(f"| {scen} | {v} | {len(rs)} | {sum(r['div'] for r in rs)} | {sum(r.get('missed') or 0 for r in ok)} | {sum(r.get('false_confirm') or 0 for r in ok)} | "
                     f"{sum(same)}/{len(same)} | {sum(r['n_trials'] for r in ok) / max(nblk / F.BPS / 60, 1e-9):.1f} | {100 * sum(r['trial_blocks'] for r in ok) / max(nblk, 1):.1f}% | "
                     f"{mean([r['L1_whole'] for r in ok])} | " + " | ".join(mean([r[f'J{k}'] for r in ok if r.get(f'J{k}') is not None]) for k in KS) + " |")
    for scen in ("RS-near", "RS-far"):
        L += ["", f"## クラス別（{scen}）：見逃し／{len(SEEDS)}、消音量の中央値（dB）", "",
              "| クラス | " + " | ".join(f"{v} 見逃し | {v} {sec(KS[0])} | {v} {sec(KS[1])}" for v in VARIANTS) + " |", "|---|" + "---:|" * (3 * len(VARIANTS))]
        for c in F.FINAL_CLASSES:
            cells = []
            for v in VARIANTS:
                ok = [r for r in rows if r["scen"] == scen and r["variant"] == v and r["cls"] == c and not r["div"]]
                med = lambda k: (lambda x: f"{np.median(x):.2f}" if x else "—")([r[f"J{k}"] for r in ok if r.get(f"J{k}") is not None])
                cells += [str(sum(r.get("missed") or 0 for r in ok)), med(KS[0]), med(KS[1])]
            L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["", "## クラス別：経路が変わらない条件（stat・EN0・ENm10 の合計）", "", "| クラス | " + " | ".join(f"{v} 誤った移行 | {v} 試行/分 | {v} L1 whole" for v in VARIANTS) + " |",
          "|---|" + "---:|" * (3 * len(VARIANTS))]
    for c in F.FINAL_CLASSES:
        cells = []
        for v in VARIANTS:
            ok = [r for r in rows if r["scen"] in ("stat", "EN0", "ENm10") and r["variant"] == v and r["cls"] == c and not r["div"]]
            cells += [str(sum(r.get("false_confirm") or 0 for r in ok)), f"{sum(r['n_trials'] for r in ok) / max(len(ok) * 0.5, 1e-9):.1f}", mean([r["L1_whole"] for r in ok])]
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    (HERE / "RESULTS_det3.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote RESULTS_det3.md", len(rows))


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=("run", "report")); a.add_argument("--procs", type=int, default=os.cpu_count() or 1)
    a.add_argument("--classes", default=",".join(F.FINAL_CLASSES)); a.add_argument("--seeds", default="")
    o = a.parse_args()
    if o.cmd == "run":
        seeds = [int(s) for s in o.seeds.split(",")] if o.seeds else SEEDS
        with Pool(o.procs) as pool:
            list(pool.imap_unordered(run_job, [(c, s, sd) for c in o.classes.split(",") for s in SCEN for sd in seeds]))
    else:
        report()
