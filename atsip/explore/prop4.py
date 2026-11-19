"""P4: restart as a trial that is verified on identical data and rolled back if it did not help.  Experiment E10.

dev2 (RESULTS_dev2.md) showed that the split-coherence gate cannot separate path changes from other detections at
the detection block: with impulsive inputs the two B-sample halves are dominated by a few samples and the coherence
of a true change falls to about 0 (range -0.11 .. 0.59, against -0.44 .. 0.12 for the other detections).

P4 (early=ND only).  When the rsN2 detector fires at block n0:
  1. keep a copy w0 of the coefficients and start the early phase at n0 + 1 (large step, rebuilt error frames),
     exactly as an ungated restart, so a true change loses no time;
  2. at block n0 + K rebuild the last M error samples twice, with the current coefficients w and with w0, from the
     same reference and the same disturbance estimate d-hat = e - S-hat * y:
         rho = E(w) / E(w0)          (energies of the two rebuilt frames);
  3. rho <= theta: the restart is confirmed and the early phase continues;
     rho >  theta: roll back, i.e. restore w0 and return to the late step at block n0 + K + 1.
Both energies contain the same disturbance and the same input samples, so the input scale (impulses) and any
disturbance that is uncorrelated with the reference cancel in the comparison: after a path change the adaptation
lowers E(w) by many dB within a few blocks; without a mismatch a large step can only raise it.
Every trial is logged with its detection block, decision block, all three ratios and outcome.

Commands (resumable):
  python prop4.py check
  python prop4.py run [--procs K] [--smoke]
  python prop4.py report [--smoke]
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
import protoC as p

np, T = p.np, p.T
B, M, N, FS, BPS = p.B, p.M, p.N, p.FS, p.BPS
HERE = Path(__file__).resolve().parent
SEEDS = q.SEEDS
CLASSES = q3.E8_CLASSES
ND0 = q3.ND0
COLUMNS = q.COLUMNS + ("n_trial", "n_rollback")


WINDOWS = {"M": M, "2B": 2 * B, "B": B}       # samples used for rho; "B" = only the samples no update has used yet


EXP = "E11"                                   # experiment run by `run` / `report` (set from --exp)
DEV = {"E10": "dev3", "E11": "dev4"}


def config(K=None, theta=0.5, restart=True, win="M", abort=None):
    """abort: while a trial is pending (2 .. K-1 blocks after the detection) roll back at once if rho > abort."""
    assert K is None or (isinstance(K, int) and 0 < K < 64), K      # shorter than the detector's guard interval
    assert win in WINDOWS, win
    c = q.config(ND0["early"], ND0["mu_e"], ND0["end"], restart=restart)
    c.update(K=K, theta=float(theta), win=win, abort=None if abort is None else float(abort))
    return c


def name_of(c):
    s = q.name_of(c).replace("P2 ", "P4 ", 1)
    if c["restart"]:
        s += " ver-" if c["K"] is None else f" ver{c['K']}/{c['theta']:g}" + ("" if c["win"] == "M" else f" w{c['win']}") + ("" if c.get("abort") is None else f" ab{c['abort']:g}")
    return s


class Frame4(q.Frame):
    """q.Frame (rebuild in the early phase) that can also rebuild the last M errors for any coefficient vector."""

    def __call__(s, i, w, e_all, ys):
        s.end, s.e_all = (i + 1) * B, e_all
        return super().__call__(i, w, e_all, ys)

    def trial_frame(s, w):
        """Last M error samples rebuilt with the coefficients w (time domain)."""
        end = s.end; start = max(0, end - M); n = end - start
        frozen = np.fft.irfft(np.fft.rfft(s.ref[end:end + 2 * M - 1], N) * np.fft.rfft(w, N), N)
        return (s.e_all[start:end] - s.sec[-n:]) + frozen[2 * M - 1 - n:2 * M - 1]


class P4(q.P2):
    def __init__(s, c, data, nb, X4):
        super().__init__(c, data, nb, X4)
        s.pending = None
        s.trials = []
        s.tr["rollback"] = []

    def block(s, i, c):
        early = s.in_early(i)
        if not early:
            s._close(s.until, "fixed")
        internal = s.c["mu_e"] if early else s.late
        applied = s.tr["restart_apply_block"]
        applied.append(-1)
        rolled = False
        final = s.pending is not None and i == s.pending["block"] + s.c["K"]
        watch = (s.pending is not None and s.c.get("abort") is not None
                 and s.pending["block"] + 2 <= i < s.pending["block"] + s.c["K"])
        if final or watch:
            w0, trial = s.pending["w0"], s.trials[-1]
            t_new, t_old = s.frame.trial_frame(c.w), s.frame.trial_frame(w0)
            rhos = {}
            for k, L in WINDOWS.items():                # all windows are logged; the configured one decides
                a, b = float(t_new[-L:] @ t_new[-L:]), float(t_old[-L:] @ t_old[-L:])
                rhos[k] = a / b if b > 0 else float("inf")
            rho = rhos[s.c["win"]]
            if final:
                ok = bool(np.isfinite(rho) and rho <= s.c["theta"])
                status = "confirmed" if ok else "rollback"
            else:                                       # early abort: the trial is clearly making things worse
                ok = None if (np.isfinite(rho) and rho <= s.c["abort"]) else False
                status = "abort"
            if ok is None:
                pass                                    # keep the trial pending
            else:
                trial.update(decide=i, rho=rho, ok=ok, status=status,
                             rho_M=rhos["M"], rho_2B=rhos["2B"], rho_B=rhos["B"])
            if ok is None:
                pass
            elif ok:
                s.restarts.append(trial["block"])
                start = trial["block"] + 1
                applied[start] = start
            else:
                rolled = True
                s.until = i + 1
                s._close(i + 1, "rollback")
            if ok is not None:
                s.pending = None
        if rolled:
            mu, dv = 1.0, (c.w - w0) / T.KD     # simulator applies w - KD * mu * dv
        else:
            mu = internal / max(c.g, s.gmin)
            dv = p.SFm.sf_dir(c.R, c.E, c.Srr, c.Pbar, c.eps, s.c["eps"])
        if s.restarts and s.restarts[-1] == i - 1:
            applied[i] = i
        if s.det is not None:
            rw = float(s.rw[i])
            z = s.frame.energy / rw if rw > 0 else np.nan
            if s.det.update(i, z) and not rolled:
                s._close(i + 1, "restart")
                s.until = i + 1 + s.length; s.phases.append([i + 1, None, None])
                trial = dict(block=i, decide=None, rho=None, ok=None, status="pending",
                             rho_M=None, rho_2B=None, rho_B=None)
                if s.c["K"] is None:
                    s.restarts.append(i)
                    trial.update(decide=i, ok=True, status="immediate")
                else:
                    s.pending = dict(block=i, w0=c.w.copy())
                s.trials.append(trial)
        s.tr["mu_app"].append(mu); s.tr["mu_state"].append(internal)
        s.tr["early"].append(float(early)); s.tr["rollback"].append(float(rolled))
        return mu, dv


def run_p4(data, Rall, st, c, nb, X4, hook=None):
    rule = P4(c, data, nb, X4)
    rule.frame = Frame4(data, rule)
    sim = p.simulate(data, Rall, st, rule, nb=nb, error_frame=rule.frame, hook=hook)
    rule.pending = None     # An unfinished trial remains pending in the log.
    return rule, sim


# ----------------------------------------------------------------------------------------------
# Experiment E10
# ----------------------------------------------------------------------------------------------
def out_dir(smoke):
    return HERE / "out" / (DEV[EXP] + ("_smoke" if smoke else ""))


def specs():
    if EXP == "E11":                                    # final check before freezing: decide at K with window B, abort early
        out = [config(restart=False), config(), config(K=16, win="B"), config(K=16, win="B", abort=4.0),
               config(K=16, win="B", abort=2.0), config(K=12, win="B", abort=4.0), config(K=16, win="M", abort=4.0)]
        return out
    out = [config(restart=False), config()]
    out += [config(K=k) for k in (4, 8, 16)]
    out += [config(K=8, win=w) for w in ("2B", "B")] + [config(K=4, win="B"), config(K=16, win="B")]
    return out


def run_job(job):
    tag, smoke = job
    f = out_dir(smoke) / EXP / (tag + ".json")
    cfgs = specs(); names = [name_of(c) for c in cfgs]
    if f.exists():
        old = json.loads(f.read_text())
        if old.get("version") == 2 and [r["rule"] for r in old["rows"]] == names:
            return old["rows"]
    t0 = time.monotonic()
    short = 1250 if smoke else None
    data, Rall, st = q.prep(tag, short)
    nb = short or len(data["d"]) // B
    o = p.parse(["c5", "--preset", "final"])
    changes = p.change_blocks(data)
    rows = []
    for c, nm in zip(cfgs, names):
        rule, sim = run_p4(data, Rall, st, c, nb, o.X4)
        row = q.score(tag, nm, rule, sim, data, False); row["exp"] = EXP
        row.update(n_trial=len(rule.trials), n_rollback=sum(t["status"] in ("rollback", "abort") for t in rule.trials))
        row["trials"] = [dict(t, true=any(cb <= t["block"] < cb + p.RS_TRUE_WIN for cb in changes)) for t in rule.trials]
        rows.append(row)
    f.parent.mkdir(parents=True, exist_ok=True)
    ab.atomic_write(f, lambda h: json.dump(dict(version=2, nb=nb, rows=rows), h))
    print(f"{EXP} {tag}: {len(rows)} rules, div={sum(r['div_num'] for r in rows)}, {time.monotonic() - t0:.0f} s", flush=True)
    return rows


def run(procs, smoke):
    jobs = [(f"{c}_s{s}", smoke) for c in CLASSES for s in (SEEDS[:1] if smoke else SEEDS)]
    if procs == 1:
        for j in jobs:
            run_job(j)
    else:
        with Pool(procs) as pool:
            list(pool.imap(run_job, jobs))


def rho_db(rho):
    return 10 * math.log10(rho) if rho is not None and 0 < rho < math.inf else None


def report(smoke):
    d = out_dir(smoke) / EXP
    rows = [r for f in (sorted(d.glob("*.json")) if d.exists() else []) for r in json.loads(f.read_text())["rows"]]
    dest = out_dir(smoke) if smoke else HERE
    dest.mkdir(parents=True, exist_ok=True)

    def write_csv(h):
        w = csv.DictWriter(h, fieldnames=COLUMNS, lineterminator="\n", extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None or (isinstance(v, float) and not math.isfinite(v)) else v)
                        for k, v in r.items()})
    ab.atomic_write(dest / f"RESULTS_{DEV[EXP]}.csv", write_csv)

    def write_trials(h):
        w = csv.writer(h, lineterminator="\n")
        w.writerow(["class", "seed", "rule", "block", "time_s", "decide_block", "true_change", "rho", "rho_db", "confirmed", "status",
                    "rho_M", "rho_2B", "rho_B"])
        for r in rows:
            for t in r.get("trials") or []:
                confirmed = None if t["ok"] is None else int(t["ok"])
                w.writerow([r["class"], r["seed"], r["rule"], t["block"], t["block"] / BPS, t["decide"], int(t["true"]),
                            t["rho"], rho_db(t["rho"]), confirmed, t["status"], t["rho_M"], t["rho_2B"], t["rho_B"]])
    ab.atomic_write(dest / f"RESULTS_{DEV[EXP]}_trials.csv", write_trials)

    L = [f"# {DEV[EXP]}：再始動の試行と検証（{EXP}）", "",
         "ab4 は、試行中（検出の 2 ブロック後から判定の前まで）に ρ > 4 になったらその場で取り消す設定。",
         "表は自動生成。値は dB（小さいほど良い）。クラス内で seed 平均、クラス間で等重み平均。件数は合計。",
         "規則名：ver- は検証なし（検出したら必ず再始動）、ver8/0.5 は検出の 8 ブロック後に ρ = E(w)/E(w0) ≤ 0.5 を確かめ、"
         "wB / w2B は ρ を最新の B / 2B サンプルだけで測る（無印は直近 M サンプル）。"
         "満たさなければ係数を w0 に戻す。nors は再始動なし。false_rs / missed_rs は確定した再始動だけを数える。", ""]
    if rows:
        L += ["## 試行ごとの ρ（dB）", "", "| 規則 | クラス | 種別 | 件数 | 最小 | 中央 | 最大 | 確定 |", "|---|---|---|---:|---:|---:|---:|---:|"]
        g = defaultdict(list)
        for r in rows:
            for t in r.get("trials") or []:
                if t["rho"] is None:
                    continue
                g[(r["rule"], r["class"], "経路変化" if t["true"] else "それ以外")].append(t)
        for (rule, cls, kind), ts in g.items():
            v = [v for t in ts if (v := rho_db(t["rho"])) is not None]
            if v:
                L.append(f"| {q.md_cell(rule)} | {cls} | {kind} | {len(ts)} | {min(v):.2f} | {np.median(v):.2f} | {max(v):.2f} | "
                         f"{sum(t['ok'] for t in ts)} |")
        L += ["", "## 窓ごとの ρ（dB。K ごとに、全試行について 3 つの窓の値を記録したもの）", "",
              "| K | 窓 | 経路変化：件数 | 最小 | 中央 | 最大 | それ以外：件数 | 最小 | 中央 | 最大 | 間隔（それ以外の最小 − 経路変化の最大） |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for K in (4, 8, 16):
            nm = name_of(config(K=K))
            ts = [t for r in rows if r["rule"] == nm for t in r.get("trials") or []]
            for k in WINDOWS:
                tv = [v for t in ts if t["true"] and (v := rho_db(t["rho_" + k])) is not None]
                fv = [v for t in ts if not t["true"] and (v := rho_db(t["rho_" + k])) is not None]
                if tv and fv:
                    L.append(f"| {K} | {k} | {len(tv)} | {min(tv):.2f} | {np.median(tv):.2f} | {max(tv):.2f} | {len(fv)} | "
                             f"{min(fv):.2f} | {np.median(fv):.2f} | {max(fv):.2f} | {min(fv) - max(tv):.2f} |")
        L.append("")
        groups = (("経路変化あり（pc、pcb、EVpc）", ("pc", "pcb", "EVpc")), ("誤差側の外乱（ENm20、ENm10、EN0）", ("ENm20", "ENm10", "EN0")),
                  ("入力の色の変化（SC）", ("SC",)), ("変化なし（G、BU、PS、II300、a145、a120）", q.BASE6))
        for title, cl in groups:
            rc = [r for r in rows if r["class"] in cl]
            if rc:
                L += [f"## {title}", "",
                      q.table(rc, ("L1_whole", "L1_first5", "L1_last10", "ex_0_1s", "ex_last10", "L1_ev", "ex_ev", "L1_pc15_17",
                                   "L1_pcb10_12", "L1_pcb20_22", "n_trial", "n_rollback")), ""]
        L += ["## クラスごとの L1 whole", "", q.per_class(rows, "L1_whole"), "",
              "## クラスごとの L1_ev（イベント直後 2 秒）", "", q.per_class(rows, "L1_ev"), "",
              "## クラスごとの ex_ev（イベント直後 2 秒の Φ 超過）", "", q.per_class(rows, "ex_ev"), ""]
    L += ["## 再現手順", "", "```sh", "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
          "python prop4.py check", "python prop4.py run --procs <コア数>", "python prop4.py report", "```", ""]
    (dest / f"RESULTS_{DEV[EXP]}.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", dest / f"RESULTS_{DEV[EXP]}.md", len(rows), "rows")


def check():
    o = p.parse(["c5", "--preset", "final"])
    # 1. Without verification P4 equals the ungated P3 (early=ND).
    nb = 2100
    data, Rall, st = q.prep("pc_s10", nb)
    r3, s3 = q3.run_p3(data, Rall, st, q3.config(ND0), nb, o.X4)
    r4, s4 = run_p4(data, Rall, st, config(), nb, o.X4)
    assert r4.restarts == r3.restarts and len(r4.restarts) == 1
    np.testing.assert_allclose(s4["ex"], s3["ex"], rtol=1e-9)
    np.testing.assert_array_equal(s4["tr"]["restart_apply_block"], s3["tr"]["restart_apply_block"])
    assert [(t["status"], t["rho"]) for t in r4.trials] == [("immediate", None)]
    print("check 1 ok: P4 without verification equals P3 without a gate; restart at", r4.restarts)
    # 2. A path change is confirmed, with rho far below the threshold, and the result is unchanged by the test.
    r4v, s4v = run_p4(data, Rall, st, config(K=8), nb, o.X4)
    assert r4v.restarts == r4.restarts and r4v.trials[0]["ok"], r4v.trials
    np.testing.assert_allclose(s4v["ex"], s4["ex"], rtol=1e-9)
    np.testing.assert_array_equal(s4v["tr"]["restart_apply_block"], s4["tr"]["restart_apply_block"])
    print("check 2 ok: path change confirmed", r4v.trials)
    unfinished, _ = run_p4(data, Rall, st, config(K=8), r4.restarts[0] + 2, o.X4)
    assert not unfinished.restarts and len(unfinished.trials) == 1
    assert unfinished.trials[0]["status"] == "pending" and unfinished.trials[0]["rho"] is None
    # 3. A burst at the error microphone is rolled back, and the coefficients return to those before the trial.
    nb = 1400
    data, Rall, st = q.prep("EN0_s10", nb)
    captured = {}

    def inspect(i, w, mu, dv, c, rule, mf, **_):
        t = rule.trials[-1] if rule.trials else None
        if t is None:
            return
        if i == t["block"]:
            captured["w0"] = w.copy()
        if i == t["decide"]:
            frame = rule.frame
            rebuilt = np.fft.irfft(c.E, N)[-M:]
            np.testing.assert_allclose(frame.trial_frame(w), rebuilt, atol=1e-12)
            end = (i + 1) * B
            start = end - B
            disturbance = frame.e_all[start:end] - frame.sec[-B:]
            ref = data["r"][start - M + 1:end]
            current = disturbance + np.convolve(ref, w, "valid")
            saved = disturbance + np.convolve(ref, captured["w0"], "valid")
            np.testing.assert_allclose(t["rho_B"], (current @ current) / (saved @ saved), rtol=1e-12)
        if t["decide"] is not None and i == t["decide"] + 1:
            captured["restored"] = w.copy()

    rule, sim = run_p4(data, Rall, st, config(K=8), nb, o.X4, hook=inspect)
    assert rule.trials and not rule.restarts and not any(t["ok"] for t in rule.trials), rule.trials
    t = rule.trials[0]
    before, after = sim["ex"][t["block"] - 1], sim["ex"][t["decide"]]
    np.testing.assert_allclose(after, before, rtol=1e-6)
    assert sim["tr"]["mu_state"][t["decide"] + 1] == q.mu_late(0.03)
    print(f"check 3 ok: burst rolled back {rule.trials}; excess before {q.db(before):.2f} dB, "
          f"during the trial {q.db(sim['ex'][t['decide'] - 1]):.2f} dB, after the roll-back {q.db(after):.2f} dB")
    np.testing.assert_allclose(captured["restored"], captured["w0"], rtol=0, atol=1e-12)
    print("check 4 ok: rebuilt frame, newest B samples, and restored simulator coefficients")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("cmd", choices=("check", "run", "report"))
    a.add_argument("--procs", type=int, default=os.cpu_count() or 1)
    a.add_argument("--smoke", action="store_true")
    a.add_argument("--exp", default="E11", choices=tuple(DEV))
    o = a.parse_args()
    global EXP
    EXP = o.exp
    if o.cmd == "check":
        check()
    elif o.cmd == "run":
        run(o.procs, o.smoke)
    else:
        report(o.smoke)


if __name__ == "__main__":
    main()
