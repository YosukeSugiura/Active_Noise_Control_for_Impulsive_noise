"""P2 with a gated restart (P3), and the dev2 experiments E8 (gate) and E9 (secondary-path model errors with baselines).

dev1 (RESULTS_dev1.md) showed that the rsN2 detector also fires on disturbances at the error microphone that are
uncorrelated with the reference (30/30 bursts) and on a change of the input colour (5/5, divergence with early=ND).
The detector statistic (error energy / reference energy) cannot tell these from a change of the primary path.

Gate.  When the detector fires at block n, the restart is carried out only if both tests pass:
  coherence test:  the last 2B actual errors are split into two disjoint halves of B samples; each half gives an SF
      direction d_a, d_b (same reference frame, same factor).  coh = <d_a, d_b> / (|d_a| |d_b|).
      A mismatch of the coefficients gives a common mean direction (expected coh about 1 / (1 + M/B) = 0.2 when the
      mismatch dominates); an uncorrelated disturbance gives coh about 0 +- 1/sqrt(M).  Pass: coh >= kappa.
  colour test:  J_n = log(mean_k rho_k) - mean_k(log rho_k), rho_k = |R_n(k)|^2 / Srr_n(k)  (>= 0; about 0.577 for a
      Gaussian frame whose spectrum matches the running estimate Srr, larger when the shape differs).
      Pass: max of J over the last 4 blocks <= jthr.
A rejected detection changes nothing except that the detector's guard interval starts (as after a restart).
Every detection is logged (block, coh, J, accepted) whatever the gate settings.

Commands (resumable):
  python prop3.py check
  python prop3.py run --exp all [--procs K] [--smoke]
  python prop3.py report [--smoke]
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
import protoC as p

np, T = p.np, p.T
B, M, N, FS, BPS = p.B, p.M, p.N, p.FS, p.BPS
HERE = Path(__file__).resolve().parent
SEEDS = q.SEEDS
E8_CLASSES = q.BASE8 + ("ENm20", "ENm10", "EN0", "EVpc", "SC")
E9_CLASSES = ("G", "a145", "pc")
E9_SHATS = (None, "g0.7", "g1.4", "n30", "n20", "f0.25", "d1")
E9_BASE = ("SF fix mu0.1", "SF-V2", q.FINAL)
ND0 = dict(early="ND", mu_e=1.0, end=("fixed", 32))        # dev1: mu_e 1.0 keeps a 1.9x margin; 32 = 64 for ND
SF0 = dict(early="SF", mu_e=0.5, end=("fixed", 64))
COLUMNS = q.COLUMNS + ("n_cand", "n_reject", "J_p50", "J_p99", "J_max")


def config(base, kappa=None, jthr=None, restart=True):
    c = q.config(base["early"], base["mu_e"], base["end"], restart=restart)
    c.update(kappa=kappa, jthr=jthr)
    return c


def name_of(c):
    s = q.name_of(c).replace("P2 ", "P3 ", 1)
    if c["restart"]:
        s += " gate" + ("-" if c["kappa"] is None else f"{c['kappa']:g}")
        s += "" if c["jthr"] is None else f" J{c['jthr']:g}"
    return s


class Frame3(q.Frame):
    """q.Frame that also keeps the last 2B actual errors, and rebuilds the frame only for early=ND."""

    def __init__(s, data, rule, rebuild):
        super().__init__(data, rule)
        s.rebuild, s.last = rebuild, np.zeros(2 * B)

    def __call__(s, i, w, e_all, ys):
        end = (i + 1) * B
        seg = e_all[max(0, end - 2 * B):end]
        s.last[:] = 0.0; s.last[2 * B - len(seg):] = seg
        if s.rebuild:
            return super().__call__(i, w, e_all, ys)
        return p.ML.frame_E(e_all, i * B, M)


def split_coherence(c, last, eps):
    fa = np.zeros(N); fb = np.zeros(N)
    fa[N - 2 * B:N - B] = last[:B]; fb[N - B:] = last[B:]
    da = p.SFm.sf_dir(c.R, np.fft.rfft(fa), c.Srr, c.Pbar, c.eps, eps)
    db_ = p.SFm.sf_dir(c.R, np.fft.rfft(fb), c.Srr, c.Pbar, c.eps, eps)
    den = math.sqrt(float(da @ da) * float(db_ @ db_))
    return float(da @ db_) / den if den > 0 else 0.0


def colour_stat(c):
    rho = np.abs(c.R) ** 2 / np.maximum(c.Srr, 1e-300)
    mean = rho.mean()
    if mean == 0:
        return 0.0  # No reference energy in this frame.
    rho[rho == 0] = np.finfo(rho.dtype).tiny
    return float(np.log(mean) - np.log(rho).mean())


class P3(q.P2):
    def __init__(s, c, data, nb, X4):
        super().__init__(c, data, nb, X4)
        s.J = []; s.cands = []                          # cands: dict(block, coh, J, ok)

    def block(s, i, c):
        early = s.in_early(i)
        if not early:
            s._close(s.until, "fixed")
        internal = s.c["mu_e"] if early else s.late
        gain = max(c.g, s.gmin)
        # Match P2's operation order in both early modes.
        mu = internal / gain if s.frame.rebuild else internal * (1.0 / gain)
        dv = p.SFm.sf_dir(c.R, c.E, c.Srr, c.Pbar, c.eps, s.c["eps"])
        s.J.append(colour_stat(c))
        applied = bool(s.restarts) and s.restarts[-1] == i - 1
        if s.det is not None:
            rw = float(s.rw[i])
            if s.frame.rebuild:
                z = s.frame.energy / rw if rw > 0 else np.nan
            else:
                z = p.rsn_stat(c.E, rw)
            if s.det.update(i, z):
                coh = split_coherence(c, s.frame.last, s.c["eps"])
                J4 = max(s.J[-4:])
                ok = (s.c["kappa"] is None or coh >= s.c["kappa"]) and (s.c["jthr"] is None or J4 <= s.c["jthr"])
                s.cands.append(dict(block=int(i), coh=coh, J=J4, ok=bool(ok)))
                if ok:
                    s.restarts.append(i); s._close(i + 1, "restart")
                    s.until = i + 1 + s.length; s.phases.append([i + 1, None, None])
        s.tr["mu_app"].append(mu); s.tr["mu_state"].append(internal)
        s.tr["restart_apply_block"].append(i if applied else -1); s.tr["early"].append(float(early))
        return mu, dv


def run_p3(data, Rall, st, c, nb, X4):
    rule = P3(c, data, nb, X4)
    rule.frame = Frame3(data, rule, c["early"] == "ND")
    sim = p.simulate(data, Rall, st, rule, nb=nb, error_frame=rule.frame)
    return rule, sim


# ----------------------------------------------------------------------------------------------
# Secondary-path model errors: those of prop2, plus n<L> at any level and a fractional delay f<samples>
# ----------------------------------------------------------------------------------------------
def prep(tag, nb=None, shat=None):
    if shat is None or shat[0] != "f":
        return q.prep(tag, nb, shat)
    delay = float(shat[1:])
    if delay == 0.0:
        return q.prep(tag, nb, "g1")
    cls = tag.rsplit("_s", 1)[0]
    dd = p.data_pcb(tag) if cls == "pcb" else T.data(tag)
    hS = dd["hS"]; L = 4 * len(hS)
    k = np.arange(L // 2 + 1)
    hm = np.fft.irfft(np.fft.rfft(hS, L) * np.exp(-2j * np.pi * k * delay / L), L)[:len(hS) + 16]
    out = dict(dd)
    out["hS_model"] = hm; out["Sf"] = np.fft.rfft(hm, N); out["r"] = p.signal.lfilter(hm, [1.0], dd["x"])
    s2 = np.abs(out["Sf"]) ** 2; out["S2"] = s2 / s2.mean()
    Rall = p.ML.prepass(out)["Rall"]
    if nb is not None:
        Rall = Rall[:min(len(Rall), nb + 64)]
    return out, Rall, p.Stats(out, Rall)


# ----------------------------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------------------------
def out_dir(smoke):
    return HERE / "out" / ("dev2_smoke" if smoke else "dev2")


def plan(exp):
    """[(classes, shat, [spec])], spec = ('P3', cfg) or ('base', name)."""
    if exp == "E8":
        specs = [("P3", config(ND0, restart=False)), ("P3", config(ND0))]
        specs += [("P3", config(ND0, kappa=k)) for k in (0.05, 0.1, 0.15)]
        specs += [("P3", config(ND0, kappa=0.1, jthr=j)) for j in (1.0, 1.5, 2.0)]
        specs += [("P3", config(SF0)), ("P3", config(SF0, kappa=0.1)), ("P3", config(SF0, kappa=0.1, jthr=1.5))]
        return [(E8_CLASSES, None, specs)]
    if exp == "E9":
        specs = [("P3", config(ND0, kappa=0.1)), ("P3", config(SF0, kappa=0.1))] + [("base", b) for b in E9_BASE]
        return [(E9_CLASSES, s, specs) for s in E9_SHATS]
    raise ValueError(exp)


def spec_name(spec, shat):
    nm = name_of(spec[1]) if spec[0] == "P3" else spec[1]
    return nm if shat is None else f"{nm} | Shat {shat}"


def run_job(job):
    exp, tag, shat, specs, smoke = job
    f = out_dir(smoke) / exp / (tag + ("" if shat is None else "_" + shat) + ".json")
    names = [spec_name(s, shat) for s in specs]
    if f.exists():
        old = json.loads(f.read_text())
        if [r["rule"] for r in old["rows"]] == names:
            return old["rows"]
    t0 = time.monotonic()
    short = 1250 if smoke else None
    data, Rall, st = prep(tag, short, shat)
    nb = short or len(data["d"]) // B
    o = p.parse(["c5", "--preset", "final"])
    base = ab.factories(data, nb, o) if any(s[0] == "base" for s in specs) else {}
    changes = p.change_blocks(data)
    rows = []
    for spec, nm in zip(specs, names):
        if spec[0] == "P3":
            rule, sim = run_p3(data, Rall, st, spec[1], nb, o.X4)
        else:
            rule = base[spec[1]](); sim = p.simulate(data, Rall, st, rule, nb=nb)
        row = q.score(tag, nm, rule, sim, data, False); row["exp"] = exp
        if spec[0] == "P3":
            J = np.asarray(rule.J)
            row.update(n_cand=len(rule.cands), n_reject=sum(not c["ok"] for c in rule.cands))
            if len(J):
                row.update(J_p50=float(np.median(J)), J_p99=float(np.percentile(J, 99)), J_max=float(J.max()))
            row["cands"] = [dict(c, true=any(cb <= c["block"] < cb + p.RS_TRUE_WIN for cb in changes))
                            for c in rule.cands]
        rows.append(row)
    f.parent.mkdir(parents=True, exist_ok=True)
    ab.atomic_write(f, lambda h: json.dump(dict(nb=nb, rows=rows), h))
    print(f"{exp} {f.stem}: {len(rows)} rules, div={sum(r['div_num'] for r in rows)}, {time.monotonic() - t0:.0f} s",
          flush=True)
    return rows


def run_exp(exp, procs, smoke):
    seeds = SEEDS[:1] if smoke else SEEDS
    jobs = [(exp, f"{c}_s{s}", shat, specs, smoke) for classes, shat, specs in plan(exp) for c in classes for s in seeds]
    if procs == 1:
        for j in jobs:
            run_job(j)
    else:
        with Pool(procs) as pool:
            list(pool.imap(run_job, jobs))


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------
def load_rows(smoke):
    rows = []
    for exp in ("E8", "E9"):
        d = out_dir(smoke) / exp
        for f in sorted(d.glob("*.json")) if d.exists() else []:
            rows += json.loads(f.read_text())["rows"]
    return rows


def report(smoke):
    rows = load_rows(smoke)
    dest = out_dir(smoke) if smoke else HERE
    dest.mkdir(parents=True, exist_ok=True)

    def write_csv(h):
        w = csv.DictWriter(h, fieldnames=COLUMNS, lineterminator="\n", extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None or (isinstance(v, float) and not math.isfinite(v)) else v)
                        for k, v in r.items()})
    ab.atomic_write(dest / "RESULTS_dev2.csv", write_csv)

    def write_cands(h):
        w = csv.writer(h, lineterminator="\n")
        w.writerow(("exp", "class", "seed", "rule", "block", "time_s", "true_change", "coh", "J", "accepted"))
        for r in rows:
            for c in r.get("cands") or []:
                w.writerow((r["exp"], r["class"], r["seed"], r["rule"], c["block"], f"{c['block'] / BPS:.3f}",
                            int(c["true"]), c["coh"], c["J"], int(c["ok"])))
    ab.atomic_write(dest / "RESULTS_dev2_cands.csv", write_cands)

    E = lambda e: [r for r in rows if r["exp"] == e]
    L = ["# dev2：再始動のゲート（E8）と二次経路の推定誤差（E9）", "",
         "表は自動生成。L1・Φ 超過は dB（小さいほど良い）。クラス内で seed 平均、クラス間で等重み平均。件数は合計。",
         "規則名：gate- はゲートなし、gate0.1 は一致度 κ = 0.1、J1.5 は色の判定のしきい値。nors は再始動なし。",
         f"基準設定：ND = {ND0}、SF = {SF0}、δ = 0.03。", ""]
    r8 = E("E8")
    if r8:
        ref = name_of(config(ND0))                       # no gate: every detection is logged and accepted
        L += ["## E8 検出ごとの一致度 coh と色の統計 J（ND・ゲートなしの記録）", "",
              "検出候補の種別は経路変化から1秒以内かどうかで判定する。誤作動・見逃しの件数は実際の再始動を各変化と1対1で対応させて数える。", "",
              "| クラス | 種別 | 件数 | coh 最小 | coh 中央 | coh 最大 | J 最小 | J 中央 | J 最大 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        g = defaultdict(list)
        for r in r8:
            if r["rule"] == ref:
                for c in r.get("cands") or []:
                    g[(r["class"], "経路変化" if c["true"] else "それ以外")].append(c)
        for (cls, kind), cs in g.items():
            co = [c["coh"] for c in cs]; J = [c["J"] for c in cs]
            L.append(f"| {cls} | {kind} | {len(cs)} | {min(co):.3f} | {np.median(co):.3f} | {max(co):.3f} | "
                     f"{min(J):.2f} | {np.median(J):.2f} | {max(J):.2f} |")
        L += ["", "### 全ブロックの J（規則ごとではなく入力ごと。ND・ゲートなしの行）", "",
              "| クラス | J 中央値 | J 99% | J 最大 |", "|---|---:|---:|---:|"]
        for cls in E8_CLASSES:
            rc = [r for r in r8 if r["rule"] == ref and r["class"] == cls]
            if rc:
                L.append(f"| {cls} | {q.fmt(q.cmean(rc, 'J_p50'))} | {q.fmt(q.cmean(rc, 'J_p99'))} | {q.fmt(q.cmean(rc, 'J_max'))} |")
        L.append("")
        groups = (("経路変化あり（pc、pcb、EVpc）", ("pc", "pcb", "EVpc")), ("誤差側の外乱（ENm20、ENm10、EN0）", ("ENm20", "ENm10", "EN0")),
                  ("入力の色の変化（SC）", ("SC",)), ("変化なし（G、BU、PS、II300、a145、a120）", q.BASE6))
        for title, cl in groups:
            rc = [r for r in r8 if r["class"] in cl]
            if rc:
                L += [f"## E8 {title}", "",
                      q.table(rc, ("L1_whole", "L1_first5", "L1_last10", "ex_0_1s", "ex_last10", "L1_ev", "ex_ev", "L1_pc15_17",
                                   "L1_pcb10_12", "L1_pcb20_22"),
                              ("n_cand", "n_reject", "div_num", "amp", "false_rs", "missed_rs")), ""]
        L += ["### クラスごとの L1 whole", "", q.per_class(r8, "L1_whole"), "",
              "### クラスごとの ex_ev（イベント直後 2 秒の Φ 超過）", "", q.per_class(r8, "ex_ev"), ""]
    r9 = E("E9")
    if r9:
        L += ["## E9 二次経路の推定誤差（G、a145、pc）", "",
              "g0.7 / g1.4：利得誤差。n30 / n20：白色タップ誤差の総エネルギー比 −30 / −20 dB。f0.25：0.25 サンプルの遅れ。d1：1 サンプルの遅れ。",
              "基準の規則（SF fix mu0.1、SF-V2、最終の則）も同じ Ŝ で動かす。", "",
              q.table(r9, ("L1_whole", "L1_first5", "L1_last10", "ex_0_1s", "ex_last10", "L1_pc15_17")), "",
              "### Ŝ ≠ S − Ŝ = S（同じ規則）", "", "| 規則 | L1 whole | L1 first5 | ex 0–1 s | L1 pc 15–17 |", "|---|---|---|---|---|"]
        for b in [n for n in q.by_rule(r9) if "Shat" not in n]:
            for s in E9_SHATS[1:]:
                a = f"{b} | Shat {s}"
                L.append(f"| {q.md_cell(a)} | " + " | ".join(q.paired(r9, a, b, f) for f in ("L1_whole", "L1_first5", "ex_0_1s", "L1_pc15_17")) + " |")
        L.append("")
    L += ["## 再現手順", "", "```sh", "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
          "python prop3.py check", "python prop3.py run --exp all --procs <コア数>", "python prop3.py report", "```", ""]
    (dest / "RESULTS_dev2.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", dest / "RESULTS_dev2.md", len(rows), "rows")


# ----------------------------------------------------------------------------------------------
# Check
# ----------------------------------------------------------------------------------------------
def check():
    o = p.parse(["c5", "--preset", "final"])
    # 1. Without a gate, P3 reproduces P2 (same restarts, same Phi-excess) for both early modes.
    nb = 2100
    data, Rall, st = prep("pc_s10", nb)
    for base in (ND0, SF0):
        r2, s2 = q.run_p2(data, Rall, st, q.config(base["early"], base["mu_e"], base["end"]), nb, o.X4)
        r3, s3 = run_p3(data, Rall, st, config(base), nb, o.X4)
        assert r3.restarts == r2.restarts and len(r3.restarts) == 1, (r3.restarts, r2.restarts)
        for key in ("e", "w", "ex"):
            np.testing.assert_array_equal(s3[key], s2[key])
        for key in ("mu_app", "restart_apply_block"):
            np.testing.assert_array_equal(s3["tr"][key], s2["tr"][key])
        assert r3.frame.n_rebuilt == (r2.frame.n_rebuilt if r2.frame is not None else 0)
        print(f"check 1 ok ({base['early']}): P3 without a gate equals P2; detection {r3.cands}")
    # 2. Split coherence: about 1/(1 + M/B) = 0.2 for a pure mismatch, about 0 for white noise at the error microphone.
    data, Rall, st = prep("G_s10", 300)
    i = 200; c = T.Ctx(i, Rall[i], None, st, None)
    end = (i + 1) * B
    e_mis = T.frozen_error(data, np.zeros(M), end - 2 * B, end)       # w = 0: the error is d, a full mismatch
    rng = np.random.default_rng(0)
    co_m = split_coherence(c, e_mis, 1e-2)
    co_n = [split_coherence(c, rng.standard_normal(2 * B), 1e-2) for _ in range(200)]
    assert co_m > 0.1 and abs(np.mean(co_n)) < 0.02 and np.std(co_n) < 0.1, (co_m, np.mean(co_n), np.std(co_n))
    J = [colour_stat(T.Ctx(k, Rall[k], None, st, None)) for k in range(100, 300)]
    print(f"check 2 ok: coh mismatch {co_m:.3f}, noise mean {np.mean(co_n):+.3f} sd {np.std(co_n):.3f}; "
          f"J on G: median {np.median(J):.2f}, max {max(J):.2f}")
    # 3. The gate rejects a burst at the error microphone and keeps a path change.
    nb = 1400
    data, Rall, st = prep("EN0_s10", nb)
    rule, sim = run_p3(data, Rall, st, config(ND0, kappa=0.1), nb, o.X4)
    assert rule.cands and not rule.restarts, (rule.cands, rule.restarts)
    print("check 3a ok: burst rejected", rule.cands)
    nb = 2100
    data, Rall, st = prep("pc_s10", nb)
    rule, sim = run_p3(data, Rall, st, config(ND0, kappa=0.1), nb, o.X4)
    assert len(rule.restarts) == 1, (rule.cands, rule.restarts)
    print("check 3b ok: path change kept", rule.cands)
    # 4. Fractional-delay model: f0 equals the true path.
    d0, R0, s0 = prep("G_s10", 64, "f0")
    np.testing.assert_array_equal(d0["hS_model"], d0["hS"])
    d_ref, R_ref, s_ref = prep("G_s10", 64)
    np.testing.assert_array_equal(d0["r"], d_ref["r"])
    np.testing.assert_array_equal(R0, R_ref)
    np.testing.assert_array_equal(s0.Srr, s_ref.Srr)
    print("check 4 ok")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("cmd", choices=("check", "run", "report"))
    a.add_argument("--exp", default="all")
    a.add_argument("--procs", type=int, default=os.cpu_count() or 1)
    a.add_argument("--smoke", action="store_true")
    o = a.parse_args()
    if o.cmd == "check":
        check()
    elif o.cmd == "run":
        for exp in (("E8", "E9") if o.exp == "all" else o.exp.split(",")):
            run_exp(exp, o.procs, o.smoke)
    else:
        report(o.smoke)


if __name__ == "__main__":
    main()
