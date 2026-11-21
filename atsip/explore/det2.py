"""Detector variants inside the proposed rule (design seeds only).  The rule itself is unchanged; only the detector
object is replaced.

  base   the current detector (guard 64, baseline updated every block)
  F64    baseline frozen from the trigger block to the end of the guard (guard 64)
  F32    the same with guard 32
  Clip   F64 with upward z inputs clipped at the existing threshold in the EMA
  C10    F64 plus an OR branch on normalized coherence

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
HOLD_SEEDS = tuple(range(30, 50))
HOLD_OUT = HERE / "out" / "det2_hold"
SCEN = ("stat", "RS-near", "RS-far", "EN0", "ENm10")
VARIANTS = {"base": (False, 64), "F64": (True, 64), "F32": (True, 32), "Clip": (True, 64), "C10": (True, 64)}
KS = (16, 64)
VERSION = 2


class FreezeDetector(p.RSNDetector):
    """RSNDetector whose baseline is not updated from the trigger block to the end of the guard."""

    def update(s, i, z):
        in_guard = s.last_restart is not None and i - s.last_restart < s.guard
        keep = s.zbar
        fired = super().update(i, z)
        if fired or in_guard:
            s.zbar = keep
        return fired


class ClippedDetector(FreezeDetector):
    """Reuse the trigger threshold as the upper limit on the EMA input."""

    def update(self, i, z):
        ref = self.zbar
        fired = super().update(i, z)
        armed = self.last_restart is None or i - self.last_restart >= self.guard
        if not fired and armed and ref is not None and np.isfinite(z):
            self.zbar = (1 - self.beta) * ref + self.beta * min(z, self.threshold * ref)
        return fired


class CoherenceDetector(FreezeDetector):
    """Shared frozen guard, with an independent previous EMA for coherence."""

    weight = 0.1
    delta = 0.1

    def __init__(self, config):
        super().__init__(config)
        self.cross = np.zeros(F.N // 2 + 1, complex)
        self.rr = np.zeros(F.N // 2 + 1)
        self.ee = np.zeros(F.N // 2 + 1)
        self.coherence = np.nan
        self.c_base = None
        self.c_count = 0

    def observe(self, R, E):
        pr, pe = float(np.vdot(R,R).real), float(np.vdot(E,E).real)
        self.coherence = np.nan
        if not (np.isfinite(pr) and np.isfinite(pe) and pr > 0 and pe > 0):
            return
        R, E = R / np.sqrt(pr), E / np.sqrt(pe)
        weight = self.weight
        self.cross = (1-weight)*self.cross + weight*np.conj(R)*E
        self.rr = (1-weight)*self.rr + weight*np.abs(R)**2
        self.ee = (1-weight)*self.ee + weight*np.abs(E)**2
        self.coherence = float(np.sum(np.abs(self.cross)**2)/(np.sum(self.rr*self.ee)+1e-30))

    def update(self, i, z):
        armed = self.last_restart is None or i - self.last_restart >= self.guard
        z_base = self.zbar
        fired_z = super().update(i,z)
        c = self.coherence
        crossed = armed and self.c_base is not None and np.isfinite(c) and c - self.c_base > self.delta
        self.c_count = self.c_count + 1 if crossed else 0
        if fired_z or self.c_count >= self.required:
            self.zbar = z_base
            self.count = self.c_count = 0
            self.last_restart = i
            return True
        if armed and np.isfinite(c):
            self.c_base = c if self.c_base is None else (1-self.beta)*self.c_base + self.beta*c
        return False


class CoherenceRule(F.P5Cap):
    def block(self, i, c):
        self.det.observe(c.R,c.E)
        return super().block(i,c)


def run_one(variant, dd, Rall, st, X4):
    freeze, guard = VARIANTS[variant]
    rule_type = CoherenceRule if variant == "C10" else F.P5Cap
    rule = rule_type(F.q5.config(early_ok=(4, 0.25)), dd, len(Rall), X4)
    assert isinstance(rule.det, p.RSNDetector)
    X = dict(X4, guard=guard)
    detector = {"Clip": ClippedDetector, "C10": CoherenceDetector}.get(variant)
    rule.det = (detector or (FreezeDetector if freeze else p.RSNDetector))(X)
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
    assert seed in SEEDS or seed in HOLD_SEEDS
    out = HOLD_OUT if seed in HOLD_SEEDS else OUT
    f = out / f"{cls}_{scen}_s{seed}.json"
    cached = json.loads(f.read_text()) if f.exists() else {}
    rows = cached["rows"] if cached.get("version") == VERSION else []
    done = {r["variant"] for r in rows}
    if done == set(VARIANTS):
        return
    t0 = time.monotonic()
    X4 = p.parse(["c5", "--preset", "final"]).X4
    dd, Rall, st = F.build(cls, scen, seed, F.LS_FINAL)
    assert len(dd["hSm_seq"]) == 1
    _, ref = F.run_rule("nors-cap", dd, Rall, st, X4)
    for v in VARIANTS:
        if v in done:
            continue
        rule, sim, ws = run_one(v, dd, Rall, st, X4)
        row = dict(variant=v, cls=cls, scen=scen, seed=int(seed), **F.measures(sim, dd), **F.trial_stats(rule, dict(dd, scored_blocks=sim["n_done"])))
        row["n_trials"] = len(rule.trials); row["n_transfer"] = len(rule.restarts)
        row["n_done"] = int(sim["n_done"])
        row["trial_blocks"] = sum(max(0, (t["decide"] if t["decide"] is not None else sim["n_done"] - 1)
                                      - t["block"]) for t in rule.trials)
        if not sim["div"] and not ref["div"]:
            row["same_as_nors"] = int(np.array_equal(sim["e"], ref["e"]) and np.array_equal(sim["y"], ref["y"]))
        if dd["changes"] and not sim["div"]:
            J = C.frozen_anr(dd, {k: ws.get(k) for k in KS}); row.update({f"J{k}": J[k] for k in KS})
        rows.append(row)
    f.parent.mkdir(parents=True, exist_ok=True)
    F.ab.atomic_write(f, lambda h: json.dump(dict(version=VERSION, rows=rows), h))
    print(f"{f.stem}: {time.monotonic() - t0:.0f} s", flush=True)


def report(seeds=SEEDS):
    out, name = (HOLD_OUT, "RESULTS_det3_hold.md") if seeds == HOLD_SEEDS else (OUT, "RESULTS_det3.md")
    rows = []
    for c in F.FINAL_CLASSES:
        for s in SCEN:
            for sd in seeds:
                case = json.loads((out/f"{c}_{s}_s{sd}.json").read_text())
                assert case["version"] == VERSION and {r["variant"] for r in case["rows"]} == set(VARIANTS)
                rows.extend(case["rows"])
    sec = lambda k: f"{k * B / F.FS:.2f} 秒後"
    L = [f"# 検出器の変更を提案法に組み込んだ比較（seed {min(seeds)}–{max(seeds)}）", "",
         f"{len(rows) // len(VARIANTS)} 条件。base＝今の検出器、F64＝検出から保護区間の終わりまで基準値を更新しない（保護 64）、F32＝同（保護 32）。Clip は F64 の EMA 入力を既存しきい値で上側クリップ。C10 は F64 と c0.1 の差分判定の OR。", "",
         "## 条件ごとの合計", "", "| 条件 | 検出器 | 実行 | 発散 | 見逃し | 誤った移行 | 再始動なしと一致 | 試行の回数/分 | 試行中のブロックの割合 | L1 whole | "
         + " | ".join(f"消音量 {sec(k)}" for k in KS) + " |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|" + "---:|" * len(KS)]
    mean = lambda v: "—" if not v else f"{np.mean(v):.2f}"
    for scen in SCEN:
        for v in VARIANTS:
            rs = [r for r in rows if r["scen"] == scen and r["variant"] == v]; ok = [r for r in rs if not r["div"]]
            same = [r["same_as_nors"] for r in ok if "same_as_nors" in r]
            nblk = sum(r["n_done"] for r in ok)
            L.append(f"| {scen} | {v} | {len(rs)} | {sum(r['div'] for r in rs)} | {sum(r.get('missed') or 0 for r in ok)} | {sum(r.get('false_confirm') or 0 for r in ok)} | "
                     f"{sum(same)}/{len(same)} | {sum(r['n_trials'] for r in ok) / max(nblk / F.BPS / 60, 1e-9):.1f} | {100 * sum(r['trial_blocks'] for r in ok) / max(nblk, 1):.1f}% | "
                     f"{mean([r['L1_whole'] for r in ok])} | " + " | ".join(mean([r[f'J{k}'] for r in ok if r.get(f'J{k}') is not None]) for k in KS) + " |")
    for scen in ("RS-near", "RS-far"):
        L += ["", f"## クラス別（{scen}）：見逃し／{len(seeds)}、消音量の中央値（dB）", "",
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
    (HERE / name).write_text("\n".join(L).rstrip() + "\n", encoding="utf-8")
    print("wrote", name, len(rows))


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=("run", "report")); a.add_argument("--procs", type=int, default=os.cpu_count() or 1)
    a.add_argument("--classes", default=",".join(F.FINAL_CLASSES)); a.add_argument("--seeds", default="")
    o = a.parse_args()
    if o.seeds in ("", "10-29"):
        seeds = SEEDS
    elif o.seeds == "30-49":
        seeds = HOLD_SEEDS
    else:
        a.error("--seeds must be 10-29 or 30-49")
    if o.cmd == "run":
        with Pool(o.procs) as pool:
            list(pool.imap_unordered(run_job, [(c, s, sd) for c in o.classes.split(",") for s in SCEN for sd in seeds]))
    else:
        report(seeds)
