"""P5: the trial of P4 run in the background (two-path style), and experiment E12.

P4 (frozen, prop4.config(K=16, win="B", abort=4.0)) applies the trial coefficients to the loudspeaker at once and
rolls back if the trial did not help.  The two-path echo cancellers (Ochiai et al. 1977 and later transfer logics)
keep the old (foreground) filter in the signal path and adapt a background filter, which is transferred only when
it is better.  In ANC a background filter can be adapted without being applied, because the error it would have
produced can be rebuilt from d-hat = e - S-hat * y.  P5 is that variant with the same detector, the same early step
and the same test as P4, so that the two differ only in which coefficients drive the loudspeaker during the trial.

P5.  When the detector fires at block n0:
  1. background copy w_b = w (foreground) at n0; the foreground goes on with the late step and the ordinary frame;
  2. blocks n0+1 .. : w_b is adapted with the early step on error frames rebuilt with w_b;
  3. from n0+2: rho = E(w_b) / E(w_f) on the newest B samples, both rebuilt (w_f = current foreground);
       rho > abort or non-finite -> drop w_b;
       at n0+K: rho <= theta  -> transfer w_f <- w_b, the foreground continues the early phase (rebuilt frames)
                                 for the remaining (32 - K) blocks; otherwise drop w_b;
       option early_ok = (k_min, theta_e): transfer as soon as k >= k_min and rho <= theta_e.
Experiment E12 compares P4 and P5 on the 13 design classes and under secondary-path model errors, with the recovery
time and the peak excess after each event as additional measures.

Commands (resumable):
  python prop5.py check
  python prop5.py run [--procs K] [--smoke]
  python prop5.py report [--smoke]
"""
import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import ablate as ab
import prop2 as q
import prop3 as q3
import prop4 as q4
import protoC as p

np, T = p.np, p.T
B, M, N, FS, BPS = p.B, p.M, p.N, p.FS, p.BPS
HERE = Path(__file__).resolve().parent
SEEDS = q.SEEDS
CLASSES = q3.E8_CLASSES
SHAT_CLASSES = ("pc", "EVpc", "EN0", "SC", "a145")
SHATS = ("g0.7", "g1.4", "n30")
ND0 = q3.ND0
COLUMNS = q.COLUMNS + ("n_trial", "n_rollback", "shat", "ex_peak_ev", "rec_blocks", "ex_pre_ev", "same_as_nors")


def config(K=16, theta=0.5, abort=4.0, early_ok=None):
    c = q.config(ND0["early"], ND0["mu_e"], ND0["end"])
    c.update(K=int(K), theta=float(theta), abort=float(abort), early_ok=early_ok)
    return c


def name_of(c):
    s = q.name_of(c).replace("P2 ", "P5 ", 1) + f" bg{c['K']}/{c['theta']:g} ab{c['abort']:g}"
    if c["early_ok"] is not None:
        s += f" ok{c['early_ok'][0]}/{c['early_ok'][1]:g}"
    return s


class P5(q.P2):
    def __init__(s, c, data, nb, X4):
        super().__init__(c, data, nb, X4)
        s.bg = None; s.trials = []
        s.tr["transfer"] = []

    def _rho(s, tb, wf):
        tf = s.frame.trial_frame(wf)
        a, b = float(tb[-B:] @ tb[-B:]), float(tf[-B:] @ tf[-B:])
        return a / b if b > 0 else float("inf")

    def block(s, i, c):
        early = s.in_early(i)
        if not early:
            s._close(s.until, "fixed")
        internal = s.c["mu_e"] if early else s.late
        gain = max(c.g, s.gmin)
        mu = internal / gain
        dv = None
        transfer = False
        busy = s.bg is not None
        if busy:
            k = i - s.bg["block"]; wb = s.bg["w"]; trial = s.trials[-1]
            tb = s.frame.trial_frame(wb)
            decided = None
            if k >= 2:
                rho = s._rho(tb, c.w)
                eo = s.c["early_ok"]
                if not np.isfinite(rho) or rho > s.c["abort"]:
                    decided = "abort"
                elif eo is not None and k >= eo[0] and rho <= eo[1]:
                    decided = "confirmed"
                elif k >= s.c["K"]:
                    decided = "confirmed" if rho <= s.c["theta"] else "rollback"
            if decided is None:                        # adapt the background filter on its own rebuilt frame
                frame = np.zeros(N); frame[-len(tb):] = tb
                db_ = p.SFm.sf_dir(c.R, np.fft.rfft(frame), c.Srr, c.Pbar, c.eps, s.c["eps"])
                mu_bg = s.c["mu_e"] / gain
                s.bg["w"] = wb - 2 * mu_bg * B / M * db_
            else:
                trial.update(decide=int(i), rho=float(rho), ok=decided == "confirmed", status=decided)
                if decided == "confirmed":
                    transfer = True
                    c.w_override = wb
                    mu = 0.0
                    s.restarts.append(trial["block"])
                    left = max(0, s.length - k)
                    s._close(i + 1, "restart")
                    s.until = i + 1 + left; s.phases.append([i + 1, None, None])
                s.bg = None
        if not transfer:
            dv = p.SFm.sf_dir(c.R, c.E, c.Srr, c.Pbar, c.eps, s.c["eps"])
        if s.det is not None:
            rw = float(s.rw[i])
            z = s.frame.energy / rw if rw > 0 else np.nan
            if s.det.update(i, z) and not busy:
                s.bg = dict(block=i, w=c.w.copy())
                s.trials.append(dict(block=int(i), decide=None, rho=None, ok=None, status="pending"))
        s.tr["mu_app"].append(mu); s.tr["mu_state"].append(internal)
        s.tr["restart_apply_block"].append(i if transfer else -1)
        s.tr["early"].append(float(early)); s.tr["transfer"].append(float(transfer))
        return mu, dv


def run_p5(data, Rall, st, c, nb, X4, hook=None):
    rule = P5(c, data, nb, X4)
    rule.frame = q4.Frame4(data, rule)
    sim = p.simulate(data, Rall, st, rule, nb=nb, error_frame=rule.frame, hook=hook)
    return rule, sim


# ----------------------------------------------------------------------------------------------
# Experiment E12
# ----------------------------------------------------------------------------------------------
def out_dir(smoke):
    return HERE / "out" / ("dev5_smoke" if smoke else "dev5")


def specs():
    """(label, kind, cfg)."""
    return [("nors", "P4", q4.config(restart=False)),
            ("ver-", "P4", q4.config()),
            ("P4", "P4", q4.config(K=16, win="B", abort=4.0)),
            ("P5", "P5", config()),
            ("P5-early", "P5", config(early_ok=(4, 0.25)))]


def event_measures(ex, cls, nb):
    """Peak Phi-excess (dB) in the 2 s after each event, the level before it (mean of the 0.5 s before, dB), and the
    number of blocks until the excess stays within 3 dB above that level for the rest of the 2 s window."""
    ev = [t for t in q.EVENTS.get(cls, ()) if nb >= (t + 2) * BPS]
    if not ev:
        return {}
    pk, pre, rec = [], [], []
    for t in ev:
        b0 = t * BPS; seg = 10 * np.log10(np.maximum(ex[b0:b0 + 2 * BPS], 1e-300))
        lvl = 10 * np.log10(max(float(np.mean(ex[b0 - BPS // 2:b0])), 1e-300))
        pk.append(float(seg.max())); pre.append(lvl)
        above = np.nonzero(seg > lvl + 3.0)[0]
        rec.append(0 if len(above) == 0 else int(above[-1]) + 1)
    return dict(ex_peak_ev=float(np.mean(pk)), ex_pre_ev=float(np.mean(pre)), rec_blocks=float(np.mean(rec)))


def run_job(job):
    tag, shat, smoke = job
    f = out_dir(smoke) / "E12" / (tag + ("" if shat is None else "_" + shat) + ".json")
    sp = specs(); names = [lab if shat is None else f"{lab} | Shat {shat}" for lab, _, _ in sp]
    if f.exists():
        old = json.loads(f.read_text())
        if old.get("version") == 1 and [r["rule"] for r in old["rows"]] == names:
            return old["rows"]
    t0 = time.monotonic()
    short = 18 * BPS if smoke else None
    data, Rall, st = q.prep(tag, short, shat)
    nb = short or len(data["d"]) // B
    o = p.parse(["c5", "--preset", "final"])
    changes = p.change_blocks(data)
    cls = tag.rsplit("_s", 1)[0]
    rows = []
    baseline = None
    for (lab, kind, c), nm in zip(sp, names):
        rule, sim = (q4.run_p4 if kind == "P4" else run_p5)(data, Rall, st, c, nb, o.X4)
        row = q.score(tag, nm, rule, sim, data, False); row["exp"] = "E12"; row["shat"] = shat or "S"
        if lab == "nors":
            baseline = sim["e"], sim["ex"]
            rs = p.restart_stats([], changes, len(sim["tr"]["mu_app"]))
            row.update(false_rs=0, missed_rs=rs["n_missed"], n_restart=0)
        row["same_as_nors"] = int(all(np.array_equal(sim[key], ref, equal_nan=True)
                                      for key, ref in zip(("e", "ex"), baseline)))
        row.update(n_trial=len(rule.trials), n_rollback=sum(t["status"] in ("rollback", "abort") for t in rule.trials))
        if not row["div_num"]:
            row.update(event_measures(sim["ex"], cls, nb))
        row["trials"] = [dict(block=t["block"], decide=t.get("decide"), rho=t.get("rho"), status=t["status"],
                              true=any(cb <= t["block"] < cb + p.RS_TRUE_WIN for cb in changes)) for t in rule.trials]
        rows.append(row)
    f.parent.mkdir(parents=True, exist_ok=True)
    ab.atomic_write(f, lambda h: json.dump(dict(version=1, nb=nb, rows=rows), h))
    print(f"E12 {f.stem}: {len(rows)} rules, div={sum(r['div_num'] for r in rows)}, {time.monotonic() - t0:.0f} s", flush=True)
    return rows


def run(procs, smoke):
    seeds = SEEDS[:1] if smoke else SEEDS
    jobs = [(f"{c}_s{s}", None, smoke) for c in CLASSES for s in seeds]
    jobs += [(f"{c}_s{s}", sh, smoke) for sh in SHATS for c in SHAT_CLASSES for s in seeds]
    if procs == 1:
        for j in jobs:
            run_job(j)
    else:
        with Pool(procs) as pool:
            list(pool.imap(run_job, jobs))


def report(smoke):
    d = out_dir(smoke) / "E12"
    rows = [r for f in sorted(d.glob("*.json")) for r in json.loads(f.read_text())["rows"]]
    dest = out_dir(smoke) if smoke else HERE
    dest.mkdir(parents=True, exist_ok=True)

    def write_csv(h):
        w = csv.DictWriter(h, fieldnames=COLUMNS, lineterminator="\n", extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None or (isinstance(v, float) and not math.isfinite(v)) else v)
                        for k, v in r.items()})
    ab.atomic_write(dest / "RESULTS_dev5.csv", write_csv)

    def write_trials(h):
        w = csv.writer(h, lineterminator="\n")
        w.writerow(["class", "seed", "shat", "rule", "block", "decide_block", "blocks_to_decision", "true_change", "rho", "status"])
        for r in rows:
            for t in r.get("trials") or []:
                dec = t.get("decide")
                w.writerow([r["class"], r["seed"], r["shat"], r["rule"], t["block"], "" if dec is None else dec,
                            "" if dec is None else dec - t["block"], int(t["true"]),
                            "" if t.get("rho") is None else t["rho"], t["status"]])
    ab.atomic_write(dest / "RESULTS_dev5_trials.csv", write_trials)

    fields = ("L1_whole", "L1_last10", "ex_last10", "L1_ev", "ex_ev", "ex_peak_ev", "rec_blocks", "L1_pc15_17",
              "L1_pcb10_12", "L1_pcb20_22", "n_trial", "n_rollback")
    L = ["# dev5：すぐ鳴らす試行（P4）と裏で行う試行（P5）の比較（E12）", "",
         "表は自動生成。値は dB（小さいほど良い）。クラス内で seed 平均、クラス間で等重み平均。件数は合計。",
         "nors：再始動なし。ver-：検出したら必ず再始動。P4：凍結した手法（試行をすぐ鳴らし、だめなら戻す）。"
         "P5：同じ試行を裏で行い、確定してから係数を移す。P5-early：検出の 4 ブロック後以降、ρ ≤ 0.25 になった時点で移す。",
         "ex_peak_ev：イベント直後 2 秒の Φ 超過の最大値。rec_blocks：Φ 超過がイベント前の水準 +3 dB 以内に戻るまでのブロック数"
         "（以降も窓内で水準を超えない最初のブロック。2 秒 = 250 で打ち切り）。SC の Φ 超過は白色入力の Wiener 解が基準の参考値。", ""]
    groups = (("経路変化あり（pc、pcb、EVpc）", ("pc", "pcb", "EVpc")), ("誤差側の外乱（ENm20、ENm10、EN0）", ("ENm20", "ENm10", "EN0")),
              ("入力の色の変化（SC）", ("SC",)), ("変化なし（G、BU、PS、II300、a145、a120）", q.BASE6))
    base = [r for r in rows if r["shat"] == "S"]
    for title, cl in groups:
        rc = [r for r in base if r["class"] in cl]
        if rc:
            L += [f"## Ŝ = S：{title}", "", q.table(rc, fields), ""]
    if base:
        L += ["## Ŝ = S：クラスごとの L1_ev", "", q.per_class(base, "L1_ev"), "",
              "## Ŝ = S：クラスごとの ex_peak_ev", "", q.per_class(base, "ex_peak_ev"), "",
              "## Ŝ = S：クラスごとの rec_blocks", "", q.per_class(base, "rec_blocks"), "",
              "## Ŝ = S：P4 − P5 の対応差（左 − 右。seed ごとにクラス平均）", "",
              "| クラス | L1_ev | ex_peak_ev | rec_blocks | L1 whole |", "|---|---|---|---|---|"]
        for cls in CLASSES:
            rc = [r for r in base if r["class"] == cls]
            if any(r.get("L1_ev") is not None for r in rc):
                L.append(f"| {cls} | " + " | ".join(q.paired(rc, "P4", "P5", f) for f in ("L1_ev", "ex_peak_ev", "rec_blocks", "L1_whole")) + " |")
        L.append("")
    for sh in SHATS:
        rs = [r for r in rows if r["shat"] == sh]
        if rs:
            L += [f"## Ŝ ≠ S（{sh}）", "", q.table(rs, fields), "",
                  f"### クラスごとの L1_ev（{sh}）", "", q.per_class(rs, "L1_ev"), "",
                  f"### クラスごとの L1 whole（{sh}）", "", q.per_class(rs, "L1_whole"), ""]
    if rows:
        L += ["## 試行の判定（件数）", "", "| Ŝ | 規則 | クラス | 種別 | 確定 | 取り消し（判定時） | 取り消し（早期） | 未判定 | 判定までの平均ブロック数 |",
              "|---|---|---|---|---:|---:|---:|---:|---:|"]
        g = defaultdict(list)
        for r in rows:
            lab = r["rule"].split(" | ")[0]
            if lab in ("P4", "P5", "P5-early"):
                for t in r.get("trials") or []:
                    g[(r["shat"], lab, r["class"], "経路変化" if t["true"] else "それ以外")].append(t)
        for (sh, lab, cls, kind), ts in g.items():
            n = lambda st: sum(t["status"] == st for t in ts)
            dl = [t["decide"] - t["block"] for t in ts if t.get("decide") is not None]
            L.append(f"| {sh} | {lab} | {cls} | {kind} | {n('confirmed')} | {n('rollback')} | {n('abort')} | {n('pending')} | "
                     f"{(np.mean(dl) if dl else float('nan')):.1f} |")
        L.append("")
    L += ["## 再現手順", "", "```sh", "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
          "python prop5.py check", "python prop5.py run --procs <コア数>", "python prop5.py report", "```", ""]
    (dest / "RESULTS_dev5.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", dest / "RESULTS_dev5.md", len(rows), "rows")


def check():
    o = p.parse(["c5", "--preset", "final"])
    # 1. No detection -> P5 equals P4 (cold start and late phase are the same code path).
    nb = 400
    data, Rall, st = q.prep("G_s10", nb)
    r4, s4 = q4.run_p4(data, Rall, st, q4.config(K=16, win="B", abort=4.0), nb, o.X4)
    r5, s5 = run_p5(data, Rall, st, config(), nb, o.X4)
    np.testing.assert_allclose(s5["ex"], s4["ex"], rtol=1e-9)
    assert not r5.trials
    print("check 1 ok: without a detection P5 equals P4")
    # 2. Path change: the background filter is transferred at n0 + K; until then the foreground is untouched,
    #    so the excess stays at the level without restart, and afterwards it drops.
    nb = 2100
    data, Rall, st = q.prep("pc_s10", nb)
    rn, sn = q4.run_p4(data, Rall, st, q4.config(restart=False), nb, o.X4)
    r5, s5 = run_p5(data, Rall, st, config(), nb, o.X4)
    t = r5.trials[0]
    assert t["status"] == "confirmed" and t["decide"] == t["block"] + 16 and r5.restarts == [t["block"]], r5.trials
    np.testing.assert_allclose(s5["ex"][:t["decide"]], sn["ex"][:t["decide"]], rtol=1e-9)
    assert s5["ex"][t["decide"]] < 0.5 * sn["ex"][t["decide"]]
    r4, s4 = q4.run_p4(data, Rall, st, q4.config(K=16, win="B", abort=4.0), nb, o.X4)
    print(f"check 2 ok: transfer at +16, rho {t['rho']:.4f}; excess after the transfer {q.db(s5['ex'][t['decide']]):.2f} dB "
          f"(no restart {q.db(sn['ex'][t['decide']]):.2f} dB, P4 {q.db(s4['ex'][t['decide']]):.2f} dB)")
    r5e, s5e = run_p5(data, Rall, st, config(early_ok=(4, 0.25)), nb, o.X4)
    te = r5e.trials[0]
    assert te["status"] == "confirmed" and 4 <= te["decide"] - te["block"] <= 16, r5e.trials
    print(f"check 2b ok: early transfer at +{te['decide'] - te['block']}, rho {te['rho']:.4f}")
    # 3. Burst at the error microphone: the background filter is dropped and the foreground never changes.
    nb = 1400
    data, Rall, st = q.prep("EN0_s10", nb)
    rn, sn = q4.run_p4(data, Rall, st, q4.config(restart=False), nb, o.X4)
    r5, s5 = run_p5(data, Rall, st, config(), nb, o.X4)
    assert r5.trials and not r5.restarts and all(t["status"] in ("rollback", "abort") for t in r5.trials), r5.trials
    np.testing.assert_allclose(s5["ex"], sn["ex"], rtol=1e-9)
    np.testing.assert_allclose(s5["e"], sn["e"], rtol=1e-9, atol=1e-12)
    print("check 3 ok: burst -> background dropped, output identical to no restart", r5.trials)
    # 4. Event measures on a hand-made excess curve.
    ex = np.full(30 * BPS, 0.01); ex[15 * BPS:15 * BPS + 40] = 10.0
    m = event_measures(ex, "pc", 30 * BPS)
    assert abs(m["ex_peak_ev"] - 10.0) < 1e-9 and m["rec_blocks"] == 40 and abs(m["ex_pre_ev"] + 20.0) < 1e-9, m
    print("check 4 ok: event measures", m)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("cmd", choices=("check", "run", "report"))
    a.add_argument("--procs", type=int, default=os.cpu_count() or 1)
    a.add_argument("--smoke", action="store_true")
    o = a.parse_args()
    if o.cmd == "check":
        check()
    elif o.cmd == "run":
        run(o.procs, o.smoke)
    else:
        report(o.smoke)


if __name__ == "__main__":
    main()
