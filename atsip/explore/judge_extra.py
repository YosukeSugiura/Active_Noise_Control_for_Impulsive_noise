"""Add proposition-4 kappa audit and actual post-change interval ANR (M7).

The original judge records, frozen settings and controller are read-only.
Run gate, run --level=... --scen ... --group ... and report.
"""
import argparse
from collections import defaultdict, deque
import json
import math
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np
import judge as J
import final as F
import cc
from paired_ci import paired

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "judge_extra_run"
INPUT = HERE / ".judge_extra_inputs"
METHODS = (J.METHOD, "Freq_FxNLMS_CohVSS", cc.NAME, "MFxNLMS")
HORIZONS = (("0.5", .5), ("1", 1.), ("2", 2.))
VERSION = 1
MARKER = "\n<!-- judge-extra-v1 -->\n"
BOUND_ATOL, BOUND_RTOL = 1e-12, 1e-10


def load(path):
    return json.loads(path.read_text())


def path_for(job):
    return OUT / J.path_for(job).relative_to(J.OUT)


def old_case(job):
    old = load(J.path_for(job))
    if old["job"] != list(job) or old["verified"] is not True:
        raise ValueError(f"Unverified original case: {job}")
    return old


def needs_m7(job):
    return job[0] == "nominal" and job[2] in ("RS-near", "RS-far")


def equal(a, b, label):
    if a != b:
        raise ValueError(f"{label}: {a!r} != {b!r}")


def finite(x):
    return x is not None and math.isfinite(float(x))


def norm(a):
    v = float(np.linalg.norm(a))
    return v if math.isfinite(v) else None


class ExtraObserver(J.Observer):
    """Keep the original rho and post-block checks; add only measurements."""
    def __init__(self, dd):
        super().__init__(dd)
        self.s = np.asarray(dd["hS_seq"][0][1])
        hm = np.asarray(dd["hSm_seq"][0][1])
        self.hm = np.pad(hm, (0, len(self.s) - len(hm)))
        self.diff = self.s - self.hm
        self.r_diff = np.pad(F.p.signal.lfilter(self.diff, [1.0], dd["x"]), (F.M - 1, 0))
        self.xpad = np.pad(dd["x"], (F.M - 1, 0))
        self.history = deque(maxlen=math.ceil((len(self.s)-1)/F.B)+2)
        self.extra = []
        self.direct_checks = 0

    def future(self, wf, wb, end, horizon, skip):
        # Already available in the immutable judge_run files.
        return None

    def capture(self, rule, tb, wf):
        rho = super().capture(rule, tb, wf)
        _, row, wf, wb = self.pending
        start, end = row["window_start"], row["window_end"]
        i = row["block"]
        delta = wb - wf
        num_signal = self.frozen(self.r_diff, delta, start, end)
        den_signal = self.frozen(self.r_model, delta, start, end)
        # Check FIR commutation using explicit candidate output history
        # on the first decision in each replayed case.
        if self.direct_checks == 0:
            lo = start - (len(self.s)-1)
            u = self.frozen(self.xpad, delta, max(0, lo), end)
            if lo < 0:
                u = np.pad(u, (-lo, 0))
            for h, target in ((self.diff, num_signal), (self.hm, den_signal)):
                direct = np.convolve(u, h, "valid")
                if not np.allclose(direct, target, rtol=1e-6, atol=1e-7, equal_nan=True):
                    raise ValueError(f"FIR commutation mismatch at block {i}")
            self.direct_checks += 1
        numerator, denominator = norm(num_signal), norm(den_signal)
        kappa = None
        if numerator is not None and denominator is not None and denominator > 0:
            value = numerator / denominator
            if math.isfinite(value):
                kappa = value
        # Current pre-update foreground is used for every sample of this
        # block. The previous pre-update coefficients come from the hook.
        hist_start = max(0, start - (len(self.s)-1))
        def changed(lo):
            first = lo // F.B
            expected = list(range(first, i))
            past = [(b, w) for b, w in self.history if first <= b < i]
            equal([b for b, _ in past], expected, "foreground history")
            return any(not np.array_equal(w, wf, equal_nan=True) for _, w in past)
        window_changed = changed(start)
        history_changed = changed(hist_start)
        threshold = None
        if row["decision"] == "transfer":
            early = rule.c["early_ok"]
            if early is not None and row["age"] >= early[0] and rho <= early[1]:
                threshold = float(early[1])
            else:
                if not (row["age"] >= rule.c["K"] and rho <= rule.c["theta"]):
                    raise ValueError("Unexpected transfer threshold")
                threshold = float(rule.c["theta"])
        self.extra.append(dict(block=i, trial_block=row["trial_block"],
                               kappa=kappa, numerator_norm=numerator, denominator_norm=denominator,
                               window_changed=window_changed, history_changed=history_changed,
                               threshold=threshold, history_start=hist_start))
        return rho

    def hook(self, i, c, eb, mu):
        super().hook(i, c, eb, mu)
        self.history.append((i, c.w.copy()))


def bound(kappa, rec, true):
    rec, true = float(rec), float(true)
    if kappa is None or not all(math.isfinite(x) and x >= 0 for x in (rec, true)):
        return None, None, None
    lhs = abs(math.sqrt(true) - math.sqrt(rec))
    rhs = kappa * (1.0 + math.sqrt(rec))
    if not math.isfinite(rhs):
        return lhs, None, None
    return lhs, rhs, bool(lhs <= rhs + BOUND_ATOL + BOUND_RTOL * max(lhs, rhs))


def verify_decisions(obs, original):
    equal(len(obs.records), len(original), "decision count")
    equal(len(obs.extra), len(original), "extra count")
    for extra, got, old in zip(obs.extra, obs.records, original):
        for key in ("block", "trial_block", "age", "window_start", "window_end",
                    "decision", "status", "decision_true", "status_true",
                    "decision_true_v", "status_true_v"):
            equal(got[key], old[key], key)
        for key in ("rho_rec", "rho_rec_td", "rho_true", "rho_true_v"):
            J.numeric_equal(got[key], old[key], key)
        for key, value in old["powers"].items():
            J.numeric_equal(got["powers"][key], value, key)
        # Use the original stored ratios for the final bound/scatter, after
        # verifying the replay. Do not silently replace earlier measurements.
        extra.update(rho_rec=old["rho_rec"], rho_true_v=old["rho_true_v"],
                     rho_true=old["rho_true"], decision=old["decision"])
        lhs, rhs, ok = bound(extra["kappa"], old["rho_rec"], old["rho_true_v"])
        lhs0, rhs0, ok0 = bound(extra["kappa"], old["rho_rec"], old["rho_true"])
        t = extra["threshold"]
        if t is not None and not float(old["rho_rec"]) <= t:
            raise ValueError("Original transfer inconsistent with threshold")
        extra.update(lhs=lhs, rhs=rhs, bound_ok=ok, lhs_no_v=lhs0,
                     rhs_no_v=rhs0, bound_ok_no_v=ok0,
                     bad_transfer=(t is not None and float(old["rho_rec"]) <= t
                                   and float(old["rho_true_v"]) >= 1))


def verify_measures(sim, dd, ref, label):
    got = F.measures(sim, dd)
    equal(got["div"], ref["div"], label + "/div")
    if got["div"]:
        equal(got["div_block"], ref["div_block"], label + "/div_block")
        equal(got["failure"]["sample"], ref["failure"]["sample"], label + "/failure.sample")
        return 0.
    maximum = 0.
    for k in ("L1_whole", "L1_last10", "L1_ev", "L1_pre", "L2_last5"):
        equal(k in got, k in ref, label + "/" + k + " presence")
        if k in ref:
            maximum = max(maximum, J.numeric_equal(got[k], ref[k], label + "/" + k))
    return maximum


def original_reference(job, method):
    _, cls, scen, seed = job
    name = f"{cls}_{scen}_s{seed}.json"
    if method in METHODS[:2]:
        path = HERE / "out" / "final2_run" / name
    elif method == cc.NAME:
        path = HERE / "out" / "cc_run" / name
    elif method == "MFxNLMS":
        path = INPUT / "out" / "mfx_run" / name
    else:
        raise ValueError(method)
    source = load(path)
    row, = [r for r in source["rows"] if r["method"] == method and r.get("variant", "main") == "main"]
    equal((row["cls"], row["scen"], row["seed"]), (cls, scen, seed), "reference identity")
    return row


def settings(method):
    if method == METHODS[1]:
        path = HERE / "TUNE.json"
    elif method == cc.NAME:
        path = HERE / "TUNE_cc.json"
    elif method == "MFxNLMS":
        path = INPUT / "TUNE_mfx.json"
    else:
        raise ValueError(method)
    data = load(path)
    equal((data["Ls"], data["hp_hz"]), (F.LS_FINAL, F.HP_HZ), "frozen setting")
    return data["chosen"][method]


def simulate_baseline(method, dd):
    setting = settings(method)
    mu, params = setting["mu"], setting["params"]
    if method == METHODS[1]:
        return F.run_baseline(method, mu, params, dd)
    if method == cc.NAME:
        return cc.run_baseline(F, mu, params, dd)
    if method == "MFxNLMS":
        # Use only the pinned comparator implementation; no patch/install.
        sys.path.insert(0, str(INPUT))
        try:
            import mfx
            return mfx.run_baseline(F, mu, params, dd)
        finally:
            sys.path.pop(0)
    raise ValueError(method)


def m7_row(job, method, sim, dd, max_diff):
    _, cls, scen, seed = job
    equal(dd["changes"], [F.CB], "change block")
    start = F.CB * F.B
    # Block method reports n_done in blocks; sample methods in samples.
    done = sim["n_done"] * F.B if "nb" in sim else sim["n_done"]
    row = dict(method=method, cls=cls, scen=scen, seed=seed, div=int(sim["div"]),
               n_done_samples=int(done), max_ref_diff=max_diff)
    for name, seconds in HORIZONS:
        end = start + int(round(seconds * F.FS))
        value = None
        if done >= end and len(sim["e"]) >= end and len(dd["d"]) >= end:
            e, d = sim["e"][start:end], dd["d"][start:end]
            a, b = float(e @ e), float(d @ d)
            if math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0:
                value = float(10 * math.log10(a / b))
        row["M7_" + name] = value
    return row


def setup():
    J.setup()


def compute(job):
    old = old_case(job)
    do_m7 = needs_m7(job)
    result = dict(version=VERSION, job=list(job), verified=True, decisions=[],
                  m7=[], replayed=False, direct_checks=0, max_ref_diff=0.)
    if not old["decisions"] and not do_m7:
        return result
    level, cls, scen, seed = job
    data = J.build(cls, scen, seed, level)
    dd = data[0]
    x4 = F.p.parse(["c5", "--preset", "final"]).X4
    observer = ExtraObserver(dd)
    rule, sim = observer.run(data, x4)
    verify_decisions(observer, old["decisions"])
    stats = F.trial_stats(rule, dict(dd, scored_blocks=sim["n_done"]))
    for k in ("n_trial", "n_confirm", "false_confirm", "missed", "detection_blocks", "decide_blocks"):
        equal(stats[k], old["measures"][k], k)
    maximum = verify_measures(sim, dd, old["measures"], f"{job}/{J.METHOD}")
    result.update(replayed=True, decisions=observer.extra,
                  direct_checks=observer.direct_checks, max_ref_diff=maximum)
    if do_m7:
        ref = original_reference(job, J.METHOD)
        maximum = max(maximum, verify_measures(sim, dd, ref, f"{job}/final2/{J.METHOD}"))
        result["m7"].append(m7_row(job, J.METHOD, sim, dd, maximum))
        del sim, rule, observer
        for method in METHODS[1:]:
            ref = original_reference(job, method)
            sim = simulate_baseline(method, dd)
            delta = verify_measures(sim, dd, ref, f"{job}/{method}")
            result["m7"].append(m7_row(job, method, sim, dd, delta))
            maximum = max(maximum, delta)
            del sim
        result["max_ref_diff"] = maximum
    return result


def checked_result(job, result, old):
    equal(result.get("version"), VERSION, "version")
    equal(result.get("job"), list(job), "job")
    equal(result.get("verified"), True, "verified")
    equal(len(result["decisions"]), len(old["decisions"]), "decision count")
    equal(result["replayed"], bool(old["decisions"] or needs_m7(job)), "replay coverage")
    for a, b in zip(result["decisions"], old["decisions"]):
        equal((a["trial_block"], a["block"], a["decision"]),
              (b["trial_block"], b["block"], b["decision"]), "decision identity")
        for key in ("rho_rec", "rho_true", "rho_true_v"):
            equal(a[key], b[key], "stored rho")
    equal([r["method"] for r in result["m7"]], list(METHODS) if needs_m7(job) else [], "M7 coverage")
    return result


def run_case(job):
    path = path_for(job)
    old = old_case(job)
    if path.is_file():
        checked_result(job, load(path), old)
        return path, True
    result = checked_result(job, compute(job), old)
    path.parent.mkdir(parents=True, exist_ok=True)
    F.ab.atomic_write(path, lambda stream: json.dump(result, stream, allow_nan=False))
    return path, False


def gate():
    setup()
    for job in (("nominal", "G", "RS-far", 100),
                ("nominal", "a145", "EN0", 100),
                ("-30dB", "jackhammer", "RS-far", 114),
                ("-20dB", "jackhammer", "stat", 101)):
        result = compute(job)
        print(f"Gate PASS {job}: {len(result['decisions'])} decisions; "
              f"{len(result['m7'])} M7 runs; max reference difference "
              f"{result['max_ref_diff']:.6g}", flush=True)


def run(level, scen, group, procs):
    batch = J.cases(level, scen, group)
    equal(len(batch), 100, "shard size")
    with Pool(procs, initializer=setup) as pool:
        for i, (path, reused) in enumerate(pool.imap_unordered(run_case, batch), 1):
            print(f"{i}/{len(batch)} {path.name}: {'cached' if reused else 'verified'}", flush=True)


def fmt(x):
    return "—" if x is None or not math.isfinite(float(x)) else f"{float(x):.6g}"


def count_rate(n, total):
    return f"{n}/{total} ({100*n/total:.3f}%)" if total else "0/0 (—)"


def distribution(rows):
    vals = [r["kappa"] for r in rows if r["kappa"] is not None]
    q = [fmt(x) for x in np.quantile(vals, [.05, .5, .95])] if vals else ["—"] * 3
    return [str(len(rows)), str(len(vals)), str(len(rows)-len(vals)), *q,
            count_rate(sum(v < .33 for v in vals), len(vals)),
            count_rate(sum(v < .17 for v in vals), len(vals))]


def report():
    batch = [job for level in J.LEVELS for job in J.cases(level)]
    equal(len(batch), 2000, "total cases")
    equal(set(OUT.rglob("*.json")), {path_for(job) for job in batch}, "complete case set")
    groups = defaultdict(list)
    m7 = []
    lookup = {}
    replayed = direct_checks = 0
    max_diff = 0.
    for job in batch:
        old = old_case(job)
        result = checked_result(job, load(path_for(job)), old)
        level, cls, scen, seed = job
        replayed += result["replayed"]
        direct_checks += result["direct_checks"]
        max_diff = max(max_diff, result["max_ref_diff"])
        m7.extend(result["m7"])
        for r in result["decisions"]:
            key = (level, cls, scen, seed, r["trial_block"], r["block"])
            if key in lookup:
                raise ValueError(f"Duplicate decision {key}")
            lookup[key] = r
            groups[level, scen].append((job, r))
    equal(len(m7), 1600, "M7 method-case count")
    equal(len({(r["method"], r["cls"], r["scen"], r["seed"]) for r in m7}), 1600, "M7 uniqueness")
    keys = [(level, scen) for level in J.LEVELS for scen in
            (J.SCENARIOS if level == "nominal" else J.SCENARIOS[:3])]
    lines = ["## 追加分析：命題4の κ と実際の変化後消音量（M7）", "",
             f"全 {len(batch)} ケース、判定 {len(lookup)} 回、再実行 {replayed} ケース、"
             f"判定なし再利用 {len(batch)-replayed} ケース。FIR直接畳み込み照合 {direct_checks} ケース。"
             f" M7 {len(m7)} 手法ケース。既存指標との最大絶対差 {max_diff:.6g}。",
             "κの分子・分母は同じ判定窓のノルム。比は外乱込みの真の残差を主とし、"
             "外乱なしの不等式もNPZ・ケースJSONに保存。分母ゼロ・非有限値は評価不能。"
             " κの割合は有限κを分母とする。", "",
             "### κの分布（全判定と移行判定）", "",
             "| モデル | 条件 | 対象 | 総数 | 有効 | 不明 | 5% | 中央値 | 95% | κ < 0.33 | κ < 0.17 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for level, scen in keys:
        rows = [r for _, r in groups[level, scen]]
        for label, rs in (("全判定", rows), ("移行", [r for r in rows if r["decision"] == "transfer"])):
            lines.append(f"| {level} | {scen} | {label} | {' | '.join(distribution(rs))} |")
    lines += ["", "### 命題4の不等式の成立状況", "",
              "LHS <= RHS + 1e-12 + 1e-10 max(LHS,RHS)。窓内係数変化と、"
              "真の二次経路2048係数の履歴も含めた係数変化を分ける。"
              " 判定窓は1ブロックで、その中では係数が一定だが過去の係数は一般に異なる。", "",
              "| モデル | 条件 | 判定数 | 評価不能 | 成立 | 不成立 | 不成立かつ窓内変化 | 不成立かつ履歴変化 | 不成立かつ履歴一定 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for level, scen in keys:
        rows = [r for _, r in groups[level, scen]]
        bad = [r for r in rows if r["bound_ok"] is False]
        hc = sum(r["history_changed"] for r in bad)
        cells = [len(rows), sum(r["bound_ok"] is None for r in rows),
                 sum(r["bound_ok"] is True for r in rows), len(bad),
                 sum(r["window_changed"] for r in bad), hc, len(bad)-hc]
        lines.append(f"| {level} | {scen} | {' | '.join(map(str,cells))} |")
    lines += ["", "### 閾値を満たす移行で真の比が1以上の事例", "",
              "早期閾値0.25（年齢4以上）を優先し、それ以外の最終移行は0.5。"
              " 件数は実際に移行した判定を母集団とする。", "",
              "| モデル | 条件 | 悪い移行/全移行 | κの5% | κ中央値 | κの95% |",
              "|---|---|---:|---:|---:|---:|"]
    for level, scen in keys:
        rows = [r for _, r in groups[level, scen]]
        bad = [r for r in rows if r["bad_transfer"]]
        vals = [r["kappa"] for r in bad if r["kappa"] is not None]
        q = [fmt(v) for v in np.quantile(vals,[.05,.5,.95])] if vals else ["—"]*3
        lines.append(f"| {level} | {scen} | {count_rate(len(bad),sum(r['decision']=='transfer' for r in rows))} | {' | '.join(q)} |")
    lines += ["", "| モデル | 条件 | 騒音 | seed | 判定block | ρ_i | ρ_rec | ρ_true_v | κ | 不等式成立 | 窓内変化 | 履歴変化 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|"]
    for level, scen in keys:
        for job, r in groups[level, scen]:
            if r["bad_transfer"]:
                lines.append(f"| {level} | {scen} | {job[1]} | {job[3]} | {r['block']} | {fmt(r['threshold'])} | "
                             f"{fmt(r['rho_rec'])} | {fmt(r['rho_true_v'])} | {fmt(r['kappa'])} | "
                             f"{r['bound_ok']} | {r['window_changed']} | {r['history_changed']} |")
    lines += ["", "### M7：変化直後からの実際の区間全体の消音量", "",
              "10 log10(sum e² / sum d²)。先頭サンプル除外なし、適応中の実出力。"
              " 区間が完走し有限かつ正の電力を持つ場合のみ有効（後で発散した場合も区間が有効なら含む）。"
              " 平均はseed内クラス平均をseed間で平均。対応差は提案法−比較法、"
              "対応seedを2000回再標本化 (default_rng(20261121))、負値が提案法優位。"
              " 全区間の発散数と窓の有効数を区別する。", "",
              "| 条件 | 騒音 | 窓 | 手法 | 有効/全数 | 全実行発散 | 平均 M7 dB | 対応差 [95%区間] dB | 対数 |",
              "|---|---|---|---|---:|---:|---:|---|---:|"]
    for scen in ("RS-near","RS-far"):
        for cls in (None,*F.FINAL_CLASSES):
            sub = [r for r in m7 if r["scen"]==scen and (cls is None or r["cls"]==cls)]
            for h, _ in HORIZONS:
                key = "M7_" + h
                left = [r for r in sub if r["method"] == J.METHOD]
                for method in METHODS:
                    rows = [r for r in sub if r["method"] == method]
                    # paired filters div; here eligibility is window completion.
                    eligible = [dict(r, div=0) for r in rows if r[key] is not None]
                    prop = [dict(r, div=0) for r in left if r[key] is not None]
                    means = F.seed_table(eligible, key)
                    avg = float(np.mean(list(means.values()))) if means else None
                    interval = paired(prop, eligible, lambda r: r[key]) if method != J.METHOD else None
                    diff = (f"{interval[0]:+.4f} [{interval[1]:+.4f}, {interval[2]:+.4f}]"
                            if interval is not None else "—")
                    n = interval[3] if interval is not None else "—"
                    lines.append(f"| {scen} | {cls or '全騒音'} | {h}s | {method} | {len(eligible)}/{len(rows)} | "
                                 f"{sum(r['div'] for r in rows)} | {fmt(avg)} | {diff} | {n} |")

    dest = HERE / "out" / "judge_scatter.npz"
    with np.load(dest, allow_pickle=False) as src:
        arrays = {k: src[k] for k in src.files}
    n = len(arrays["rho_rec"])
    equal(n, len(lookup), "scatter count")
    order = []
    seen = set()
    for i in range(n):
        key = (str(arrays["model_error"][i]), str(arrays["noise"][i]),
               str(arrays["condition"][i]), int(arrays["seed"][i]),
               int(arrays["trial_block"][i]), int(arrays["block"][i]))
        if key in seen or key not in lookup:
            raise ValueError(f"Scatter key mismatch: {key}")
        seen.add(key)
        row = lookup[key]
        for k in ("rho_rec", "rho_true", "rho_true_v"):
            J.numeric_equal(arrays[k][i], row[k], "scatter/" + k)
        equal(str(arrays["decision"][i]), row["decision"], "scatter decision")
        order.append(row)
    equal(len(seen), len(lookup), "scatter bijection")
    for key in ("kappa","numerator_norm","denominator_norm","lhs","rhs","lhs_no_v","rhs_no_v","threshold"):
        arrays[key] = np.asarray([np.nan if r[key] is None else r[key] for r in order], dtype=np.float64)
    for key in ("bound_ok", "bound_ok_no_v"):
        arrays[key] = np.asarray([-1 if r[key] is None else int(r[key]) for r in order], dtype=np.int8)
    for key in ("window_changed", "history_changed", "bad_transfer"):
        arrays[key] = np.asarray([r[key] for r in order], dtype=np.bool_)
    arrays["history_start"] = np.asarray([r["history_start"] for r in order], dtype=np.int64)
    tmp = dest.with_name(".judge_extra_scatter.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    with np.load(tmp, allow_pickle=False) as check:
        equal(set(check.files), set(arrays), "output arrays")
        for k, a in arrays.items():
            b = check[k]
            equal(a.dtype, b.dtype, k + " dtype")
            if not np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc" else not np.array_equal(a, b):
                raise ValueError("NPZ roundtrip " + k)
    lines += ["", f"散布図：{n} 判定キーの全単射を確認。既存配列を保持し、κ・分子分母・不等式・係数変化等を追加。"
              " bound_ok: 1=成立、0=不成立、-1=評価不能。", ""]
    report_path = HERE / "RESULTS_judge.md"
    original = report_path.read_text().split(MARKER, 1)[0]
    tmp.replace(dest)
    F.ab.atomic_write(report_path, lambda stream: stream.write(original + MARKER + "\n".join(lines) + "\n"))
    print(f"Report complete: {len(batch)} verified cases, {n} decisions, {len(m7)} M7 method-cases", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("gate","run","report"))
    parser.add_argument("--level", choices=J.LEVELS)
    parser.add_argument("--scen", choices=J.SCENARIOS)
    parser.add_argument("--group", type=int, choices=(0,1))
    parser.add_argument("--procs", type=int, default=4)
    args = parser.parse_args()
    if args.command == "report":
        report()
    elif args.command == "gate":
        gate()
    else:
        if args.level is None or args.scen is None or args.group is None or args.procs < 1:
            parser.error("run requires level, scen, group and positive procs")
        run(args.level,args.scen,args.group,args.procs)


if __name__ == "__main__":
    main()
