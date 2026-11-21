"""Detector study on design seeds (10-49): log candidate statistics from runs without restart, then evaluate
detector rules offline on the logs.

Why offline is valid: the proposed rule's foreground output equals the no-restart rule until a trial is confirmed
(proposition H), so every statistic computed from the foreground (error, reference, error-frame spectrum) is the same
in both up to the first confirmed transfer.  The offline evaluation therefore reproduces when a detector would first
fire; after a real transfer the logs no longer describe the proposed rule, which is why only the first trigger
around the change and triggers without a change are scored.

Statistics per block n (all from quantities the rule already has):
  z      current statistic: error energy over the last M samples / filtered-reference energy over the last 2M.
  c{L}   coherence between the filtered reference R_n and the error frame E_n (spectra the rule computes each
         block).  Each block's spectra are scaled to unit energy first (so loud impulses do not dominate the
         averages), then S_re, S_rr, S_ee are averaged recursively with weight L, and
         c = sum_k |S_re|^2 / sum_k S_rr S_ee   (in [0, 1]).
  c{L}u  the same without the per-block scaling.
  q{L}   coherent part of the error relative to the reference: c{L} * z (error power that the reference explains).

Detector rules (all with the current guard of 64 blocks and a baseline EMA with weight 0.05 updated after use):
  ratio:  s_n > k * base_{n-1} for d consecutive blocks
  diff:   s_n - base_{n-1} > k for d consecutive blocks           (for coherence, which lies in [0, 1])

Commands:
  python det.py log [--procs K]        (runs; out/det_log/*.npz)
  python det.py eval                   (RESULTS_det.md)
"""
import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import final as F

np, p = F.np, F.p
B, M, N, CB, BPS = F.B, F.M, F.N, F.CB, F.BPS
HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "det_log"
SEEDS = tuple(range(10, 50))     # 40 seeds: every (recording segment, path pair) combination once (seed % 5, seed % 8)
CLASSES = F.FINAL_CLASSES
SCEN = ("stat", "RS-near", "RS-far", "EN0", "ENm10")
LAMS = (0.1, 0.25, 0.5)
WARM = 2 * BPS            # triggers in the first 2 s are start-up, not counted as false
WIN = 64                  # a trigger within 64 blocks after the change counts as detection (as RS_TRUE_WIN)


def run_log(job):
    cls, scen, seed = job
    assert seed in SEEDS
    f = OUT / f"{cls}_{scen}_s{seed}.npz"
    if f.exists():
        return
    t0 = time.monotonic()
    X4 = p.parse(["c5", "--preset", "final"]).X4
    dd, Rall, st = F.build(cls, scen, seed, F.LS_FINAL)
    nb = len(Rall)
    acc = {L: [np.zeros(N // 2 + 1, complex), np.zeros(N // 2 + 1), np.zeros(N // 2 + 1)] for L in LAMS}
    accu = [np.zeros(N // 2 + 1, complex), np.zeros(N // 2 + 1), np.zeros(N // 2 + 1)]
    out = {f"c{L}": np.full(nb, np.nan) for L in LAMS}
    out["c0.25u"] = np.full(nb, np.nan)
    eps = 1e-30

    def hook(i, c, eb, mu):
        R, E = c.R, c.E
        pr, pe = float(np.vdot(R, R).real), float(np.vdot(E, E).real)
        if pr <= 0 or pe <= 0:
            return
        Rn, En = R / np.sqrt(pr), E / np.sqrt(pe)
        for L in LAMS:
            a = acc[L]
            a[0] = (1 - L) * a[0] + L * np.conj(Rn) * En
            a[1] = (1 - L) * a[1] + L * np.abs(Rn) ** 2
            a[2] = (1 - L) * a[2] + L * np.abs(En) ** 2
            out[f"c{L}"][i] = float(np.sum(np.abs(a[0]) ** 2) / (np.sum(a[1] * a[2]) + eps))
        L = 0.25
        accu[0] = (1 - L) * accu[0] + L * np.conj(R) * E
        accu[1] = (1 - L) * accu[1] + L * np.abs(R) ** 2
        accu[2] = (1 - L) * accu[2] + L * np.abs(E) ** 2
        out["c0.25u"][i] = float(np.sum(np.abs(accu[0]) ** 2) / (np.sum(accu[1] * accu[2]) + eps))

    _, sim = F.run_rule("nors-cap", dd, Rall, st, X4, hook=hook)
    e, r = sim["e"], np.asarray(dd["r"], float)
    ce = np.concatenate([[0.0], np.cumsum(np.nan_to_num(e) ** 2)]); cr = np.concatenate([[0.0], np.cumsum(r ** 2)])
    hi = (np.arange(nb) + 1) * B
    ee = ce[hi] - ce[np.maximum(0, hi - M)]; rr = cr[hi] - cr[np.maximum(0, hi - 2 * M)]
    z = np.where(rr > 0, ee / np.where(rr > 0, rr, 1), np.nan)
    out["z"] = z
    for L in LAMS:
        out[f"q{L}"] = out[f"c{L}"] * z
    ref = np.where(rr > 0, rr, np.nan)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.stem + ".tmp.npz")
    np.savez(tmp, div=sim["div"], changes=np.array(dd["changes"], int), events=np.array(dd["events"], int), rr=ref, **out)
    tmp.replace(f)
    print(f"{f.stem}: {time.monotonic() - t0:.0f} s", flush=True)


def triggers(s, mode, k, d, beta=0.05, guard=64):
    """Block indices where a detector of the given form fires (same structure as protoC.RSNDetector)."""
    base = None; cnt = 0; last = None; out = []
    for i, v in enumerate(s):
        ref = base
        if np.isfinite(v):
            base = v if ref is None else (1 - beta) * ref + beta * v
        armed = last is None or i - last >= guard
        hit = armed and ref is not None and np.isfinite(v) and (v > k * ref if mode == "ratio" else v - ref > k)
        cnt = cnt + 1 if hit else 0
        if cnt >= d:
            cnt = 0; last = i; out.append(i)
    return out


def triggers_or(parts, beta=0.05, guard=64):
    """Combined detector: fires when any part (stat series, mode, k, d) reaches its count; one shared guard."""
    st = [dict(base=None, cnt=0) for _ in parts]; last = None; out = []
    for i in range(len(parts[0][0])):
        armed = last is None or i - last >= guard
        fire = False
        for s_, (s, mode, k, d) in zip(st, parts):
            v = s[i]; ref = s_["base"]
            if np.isfinite(v):
                s_["base"] = v if ref is None else (1 - beta) * ref + beta * v
            hit = armed and ref is not None and np.isfinite(v) and (v > k * ref if mode == "ratio" else v - ref > k)
            s_["cnt"] = s_["cnt"] + 1 if hit else 0
            fire = fire or s_["cnt"] >= d
        if fire:
            last = i; out.append(i)
            for s_ in st:
                s_["cnt"] = 0
    return out


RULES = ([("z", "ratio", k, d) for k in (2, 3, 4, 6) for d in (1, 2, 3)]
         + [(f"q{L}", "ratio", k, d) for L in LAMS for k in (2, 3, 4, 6) for d in (1, 2, 3)]
         + [(f"c{L}", "diff", k, d) for L in LAMS for k in (0.05, 0.1, 0.2, 0.3) for d in (1, 2, 3)]
         + [("c0.25u", "diff", k, d) for k in (0.05, 0.1, 0.2, 0.3) for d in (1, 2, 3)]
         + [(("z", "ratio", 4, 2), (f"c{L}", "diff", k, d)) for L in LAMS for k in (0.1, 0.2, 0.3) for d in (2, 3)])


def evaluate():
    logs = {}
    for f in sorted(p_ for p_ in OUT.glob("*.npz") if ".tmp" not in p_.name):
        cls, scen, s = f.stem.rsplit("_", 2)
        logs[(cls, scen, int(s[1:]))] = dict(np.load(f))
    L = ["# 検出器の比較（設計用 seed 10–49、再始動なしの実行のログから、検出の規則をオフラインで評価）", "",
         f"ログ {len(logs)} 件。見逃し＝騒音源の切り替えの後 {WIN} ブロック以内に検出なし。"
         f"誤検出＝変化のない区間（stat 全体、外乱の条件全体、切り替えの前）での検出。開始 2 秒は数えない。"
         "誤検出は 1 分あたりの回数。遅れ＝変化から最初の検出までのブロック数の中央値。", "",
         "現在の検出器は z・ratio・4・2。", ""]
    rows = []
    for rule in RULES:
        combo = isinstance(rule[0], tuple)
        name = " または ".join(f"{s}・{m}・{k:g}・{d}" for s, m, k, d in (rule if combo else (rule,)))
        miss = {}; delay = []; fa = {}; dur = {}
        for (cls, scen, seed), lg in logs.items():
            if bool(lg["div"]):
                continue
            tr = (triggers_or([(lg[s], m, k, d) for s, m, k, d in rule]) if combo
                  else triggers(lg[rule[0]], rule[1], rule[2], rule[3]))
            stat = rule[0][0] if combo else rule[0]
            ch = [int(x) for x in lg["changes"]]
            nb = len(lg[stat])
            if ch:
                hit = [t for t in tr if ch[0] <= t < ch[0] + WIN]
                miss.setdefault(cls, []).append(0 if hit else 1)
                if hit:
                    delay.append(hit[0] - ch[0])
                pre = [t for t in tr if WARM <= t < ch[0]]
                fa[cls] = fa.get(cls, 0) + len(pre); dur[cls] = dur.get(cls, 0) + (ch[0] - WARM)
            else:
                post = [t for t in tr if t >= WARM]
                fa[cls] = fa.get(cls, 0) + len(post); dur[cls] = dur.get(cls, 0) + (nb - WARM)
        mr = {c: float(np.mean(v)) for c, v in miss.items()}
        far = {c: fa[c] / (dur[c] / BPS / 60) for c in fa}
        rows.append(dict(rule=name, miss=float(np.mean(list(mr.values()))), worst_miss=max(mr.values()),
                         fa=float(np.mean(list(far.values()))), worst_fa=max(far.values()),
                         delay=float(np.median(delay)) if delay else None, mr=mr, far=far))
    rows.sort(key=lambda r: (r["miss"], r["fa"]))
    L += ["## 全規則（見逃しの少ない順）", "", "| 規則 | 見逃し（平均） | 見逃し（最悪クラス） | 誤検出/分（平均） | 誤検出/分（最悪） | 遅れ |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['rule']} | {r['miss']:.2f} | {r['worst_miss']:.2f} | {r['fa']:.2f} | {r['worst_fa']:.2f} | {F.fmt(r['delay'], 0)} |")
    cur = next(r for r in rows if r["rule"] == "z・ratio・4・2")
    pareto = [r for r in rows if not any(o["miss"] <= r["miss"] and o["fa"] <= r["fa"] and (o["miss"] < r["miss"] or o["fa"] < r["fa"]) for o in rows)]
    L += ["", "## 見逃しと誤検出の両方で他に負けない規則（パレート集合）と、現在の検出器", "",
          "| 規則 | " + " | ".join(f"見逃し {c}" for c in CLASSES) + " | " + " | ".join(f"誤検出 {c}" for c in CLASSES) + " | 遅れ |",
          "|---|" + "---:|" * (2 * len(CLASSES) + 1)]
    for r in [cur] + [r for r in pareto if r is not cur]:
        L.append(f"| {r['rule']} | " + " | ".join(F.fmt(r['mr'].get(c), 2) for c in CLASSES) + " | "
                 + " | ".join(F.fmt(r['far'].get(c), 2) for c in CLASSES) + f" | {F.fmt(r['delay'], 0)} |")
    (HERE / "RESULTS_det.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote RESULTS_det.md", len(rows), "rules")


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=("log", "eval")); a.add_argument("--procs", type=int, default=os.cpu_count() or 1)
    a.add_argument("--classes", default=",".join(CLASSES))
    o = a.parse_args()
    if o.cmd == "log":
        jobs = [(c, s, sd) for c in o.classes.split(",") for s in SCEN for sd in SEEDS]
        with Pool(o.procs) as pool:
            list(pool.imap_unordered(run_log, jobs))
    else:
        evaluate()
