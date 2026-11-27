"""Observational audit of every P5-early-cap rho evaluation against the true plant.

Reference implementation: final.py, prop5.P5._rho/block, prop4.Frame4,
and smodel.py at 31d072c9adcf23e95889d5cf36a51c91aebf4c60.
No changes to the evaluated controller or existing results.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from multiprocessing import Pool
from pathlib import Path
from unittest.mock import patch

import numpy as np
import final as F
import final2
import coef as C

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "judge_run"
LEVELS = ("nominal", "-30dB", "-20dB")
SCENARIOS = ("stat", "RS-near", "RS-far", "EN0")
METHOD = "P5-early-cap"
NO_TRIAL = "nors-cap"
LABELS = ("discard", "transfer", "continue")
VERSION = 1
ATOL, RTOL = 1e-7, 1e-6


def cases(level, scen=None, group=None):
    scenarios = SCENARIOS if level == "nominal" else SCENARIOS[:3]
    if scen is not None:
        if scen not in scenarios:
            raise ValueError((level, scen))
        scenarios = (scen,)
    classes = tuple(F.FINAL_CLASSES)
    assert len(classes) == 10 and tuple(F.FINAL_SEEDS) == tuple(range(100, 120))
    if group is not None:
        classes = classes[group * 5:(group + 1) * 5]
    return [(level, c, s, sd) for s in scenarios for c in classes for sd in F.FINAL_SEEDS]


def path_for(job):
    level, cls, scen, seed = job
    group = tuple(F.FINAL_CLASSES).index(cls) // 5
    return OUT / level / scen / f"g{group}" / f"{cls}_{scen}_s{seed}.json"


def reference(job):
    level, cls, scen, seed = job
    base = HERE / "out"
    base = base / "final2_run" if level == "nominal" else base / "smodel_run" / level
    path = base / f"{cls}_{scen}_s{seed}.json"
    records = json.loads(path.read_text())["rows"]
    out = {}
    for method in (METHOD, NO_TRIAL):
        row, = [r for r in records if r["method"] == method and r["variant"] == "main"]
        if (row["cls"], row["scen"], row["seed"]) != (cls, scen, seed):
            raise ValueError(f"Wrong reference identity: {path}")
        out[method] = row
    return out


def build(cls, scen, seed, level):
    # Exact path-boundary perturbation and random seed of smodel.build.
    if level == "nominal":
        return F.build(cls, scen, seed, F.LS_FINAL, final=True)
    db = float(level[:-2])
    white = np.random.default_rng(seed + 50000).standard_normal(F.LS_FINAL)
    paths_of = F.paths_of

    def with_error(scenario, pair):
        first, second = paths_of(scenario, pair)
        primary, plant, model = first
        nominal = model[:F.LS_FINAL]
        scale = np.linalg.norm(nominal) * 10 ** (db / 20) / np.linalg.norm(white)
        perturbed = nominal + scale * white
        if second is not None and (second[1] is not plant or second[2] is not model):
            raise ValueError(f"Expected a fixed secondary path and model in {scenario}")
        return (primary, plant, perturbed), None if second is None else (*second[:2], perturbed)

    with patch.object(F, "paths_of", with_error):
        return F.build(cls, scen, seed, F.LS_FINAL, final=True)


def setup():
    final2.configure()


def numeric_equal(actual, expected, label):
    a, b = float(actual), float(expected)
    if not math.isfinite(a) or not math.isfinite(b):
        if a == b or (math.isnan(a) and math.isnan(b)):
            return 0.0
    elif abs(a - b) <= ATOL + RTOL * abs(b):
        return abs(a - b)
    raise ValueError(f"{label}: expected {expected!r}, actual {actual!r}")


def encode(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def ratio(a, b):
    # Exact zero-denominator behavior of P5._rho.
    a, b = float(a @ a), float(b @ b)
    return a / b if b > 0 else float("inf")


def decision(rho, k, cfg):
    if k < 2:
        raise ValueError(f"No rho evaluation at age {k}")
    if not math.isfinite(rho) or rho > cfg["abort"]:
        return "discard", "abort"
    eo = cfg["early_ok"]
    if eo is not None and k >= eo[0] and rho <= eo[1]:
        return "transfer", "confirmed"
    if k >= cfg["K"]:
        return ("transfer", "confirmed") if rho <= cfg["theta"] else ("discard", "rollback")
    return "continue", "pending"


class Observer:
    """Capture the pre-update candidates inside the original _rho, then audit
    the actual decision in final.simulate's post-block/pre-update hook."""

    def __init__(self, dd):
        self.dd = dd
        self.records = []
        self.pending = None
        self.original = F.q5.P5._rho
        assert len(dd["hS_seq"]) == len(dd["hSm_seq"]) == 1
        s = dd["hS_seq"][0][1]
        assert len(s) == 2048, len(s)
        assert len(dd["hSm_seq"][0][1]) == F.LS_FINAL == 512
        assert F.B == 128 and F.M == 512 and F.FS == 16000
        self.r_true = np.pad(F.p.signal.lfilter(s, [1.0], dd["x"]), (F.M - 1, 0))
        self.r_model = np.pad(dd["r"], (F.M - 1, 0))
        self.d_with = dd["d"]
        self.d_clean = dd["d"] - dd.get("v", 0.0)
        self.max_rho_diff = 0.0
        self.max_td_diff = 0.0

    def frozen(self, r, w, start, end):
        return np.convolve(r[start:end + F.M - 1], w, "valid")

    def true_residual(self, w, start, end, with_v=False):
        d = self.d_with if with_v else self.d_clean
        return d[start:end] + self.frozen(self.r_true, w, start, end)

    def capture(self, rule, tb, wf):
        if self.pending is not None:
            raise ValueError("Missing post-block hook")
        # The original function alone determines the algorithm's return value.
        original_rho = self.original(rule, tb, wf)
        wb = rule.bg["w"].copy()
        wf = wf.copy()
        end = rule.frame.end
        i = end // F.B - 1
        start = max(0, end - F.B)
        k = i - rule.bg["block"]
        tf = rule.frame.trial_frame(wf)
        bg_rec, fg_rec = tb[-F.B:], tf[-F.B:]
        rho_rec = ratio(bg_rec, fg_rec)
        delta = numeric_equal(rho_rec, original_rho, "rho reconstruction vs original")
        self.max_rho_diff = max(self.max_rho_diff, delta)
        # Independent direct time-domain reconstruction on the same B samples.
        offset = rule.frame.e_all[start:end] - rule.frame.sec[-(end - start):]
        bg_td = offset + self.frozen(self.r_model, wb, start, end)
        fg_td = offset + self.frozen(self.r_model, wf, start, end)
        for label, a, b in (("background", bg_td, bg_rec), ("foreground", fg_td, fg_rec)):
            if not np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True):
                raise ValueError(f"{label} direct reconstruction mismatch at {i}")
            if np.isfinite(a).all() and np.isfinite(b).all():
                self.max_td_diff = max(self.max_td_diff, float(np.max(np.abs(a - b))))
        rho_td = ratio(bg_td, fg_td)
        numeric_equal(rho_td, original_rho, "time-domain rho vs original")

        bg0, fg0 = (self.true_residual(w, start, end) for w in (wb, wf))
        bgv, fgv = (self.true_residual(w, start, end, True) for w in (wb, wf))
        true0, truev = ratio(bg0, fg0), ratio(bgv, fgv)
        rec_action, status = decision(original_rho, k, rule.c)
        action0, status0 = decision(true0, k, rule.c)
        actionv, statusv = decision(truev, k, rule.c)
        row = dict(block=i, trial_block=int(rule.bg["block"]), age=int(k),
                   window_start=start, window_end=end, rho_rec=encode(original_rho),
                   rho_rec_td=encode(rho_td), rho_true=encode(true0),
                   rho_true_v=encode(truev), decision=rec_action, status=status,
                   decision_true=action0, status_true=status0,
                   decision_true_v=actionv, status_true_v=statusv,
                   powers={key: encode(v @ v) for key, v in (
                       ("rec_fg", fg_rec), ("rec_bg", bg_rec), ("true_fg", fg0),
                       ("true_bg", bg0), ("true_v_fg", fgv), ("true_v_bg", bgv))})
        self.pending = (rule, row, wf, wb)
        return original_rho

    def future(self, wf, wb, end, horizon, skip):
        stop = end + int(round(horizon * F.FS))
        start = end + skip
        if stop > len(self.d_clean) or start >= stop:
            return None
        d = self.d_clean[start:stop]
        fg = self.true_residual(wf, start, stop)
        bg = self.true_residual(wb, start, stop)
        den, ef, eb = float(d @ d), float(fg @ fg), float(bg @ bg)
        if min(den, ef, eb) <= 0 or not all(math.isfinite(x) for x in (den, ef, eb)):
            return None
        jf, jb = 10 * math.log10(ef / den), 10 * math.log10(eb / den)
        return dict(start=start, end=stop, J_fg=jf, J_bg=jb, improvement=jf - jb)

    def hook(self, i, c, eb, mu):
        if self.pending is None:
            if c.w_override is not None:
                raise ValueError(f"Unobserved transfer at {i}")
            return
        rule, row, wf, wb = self.pending
        self.pending = None
        if row["block"] != i or not np.array_equal(c.w, wf, equal_nan=True):
            raise ValueError(f"Foreground time mismatch at {i}")
        t = rule.trials[-1]
        if t["block"] != row["trial_block"]:
            raise ValueError(f"Trial identity mismatch at {i}")
        if row["decision"] == "continue":
            if t["status"] != "pending" or t["decide"] is not None or rule.bg is None or c.w_override is not None:
                raise ValueError(f"Unexpected actual continuation at {i}")
        else:
            if t["status"] != row["status"] or t["decide"] != i or rule.bg is not None:
                raise ValueError(f"Actual terminal decision mismatch at {i}: {t}")
            numeric_equal(t["rho"], row["rho_rec"], "logged terminal rho")
        if row["decision"] == "transfer":
            if c.w_override is None or not np.array_equal(c.w_override, wb, equal_nan=True):
                raise ValueError(f"Transferred coefficients mismatch at {i}")
            end = row["window_end"]
            row["phase"] = ("no_change" if not self.dd["changes"] else
                            "post_change" if end >= self.dd["changes"][0] * F.B else "pre_change")
            # Both definitions are retained: full future window and coef.py's
            # 4096-sample washout inside that fixed horizon, not after it.
            row["future"] = {key: self.future(wf, wb, end, h, C.SKIP)
                             for key, h in (("0.5", .5), ("2", 2.))}
            row["future_full"] = {key: self.future(wf, wb, end, h, 0)
                                  for key, h in (("0.5", .5), ("2", 2.))}
        elif c.w_override is not None:
            raise ValueError(f"Unexpected coefficient override at {i}")
        self.records.append(row)

    def run(self, data, x4):
        # Patch only this bound call site in this process, preserving the
        # original return value and controller state. Never patch post run.
        def observed(rule, tb, wf):
            return self.capture(rule, tb, wf)
        with patch.object(F.q5.P5, "_rho", observed):
            rule, sim = F.run_rule(METHOD, *data, x4, hook=self.hook)
        if self.pending is not None:
            raise ValueError("Unprocessed rho evaluation")
        observed_terminal = [(r["trial_block"], r["block"], r["status"]) for r in self.records
                             if r["decision"] != "continue"]
        logged = [(t["block"], t["decide"], t["status"]) for t in rule.trials if t["decide"] is not None]
        if observed_terminal != logged:
            raise ValueError("Terminal trial coverage mismatch")
        # All intermediate evaluations must be present too, including a
        # pending trial cut off by the end or a detected divergence.
        observed_keys = [(r["trial_block"], r["block"]) for r in self.records]
        last_processed = sim["n_done"] if sim["div"] else sim["n_done"] - 1
        expected_keys = [(t["block"], i) for t in rule.trials
                         for i in range(t["block"] + 2,
                                        (t["decide"] if t["decide"] is not None else last_processed) + 1)]
        if observed_keys != expected_keys:
            raise ValueError("All rho evaluations were not captured")
        return rule, sim


def verify_metrics(actual, expected, label):
    max_diff = 0.0
    for key in ("div", "div_block", "rec_blocks", "rec_cens", "rec_limit",
                "n_trial", "n_confirm", "false_confirm", "missed",
                "detection_blocks", "decide_blocks"):
        if (key in actual) != (key in expected) or actual.get(key) != expected.get(key):
            raise ValueError(f"{label}/{key}: expected {expected.get(key)!r}, actual {actual.get(key)!r}")
    for key in ("L1_whole", "L1_last10", "L1_ev", "L1_pre", "L2_last5", "rec_target"):
        if (key in actual) != (key in expected):
            raise ValueError(f"{label}/{key}: missing field")
        if key in expected:
            max_diff = max(max_diff, numeric_equal(actual[key], expected[key], f"{label}/{key}"))
    if actual["div"]:
        if actual["failure"]["sample"] != expected["failure"]["sample"]:
            raise ValueError(f"{label}: divergence sample differs")
    at, et = actual["trials"], expected["trials"]
    if len(at) != len(et):
        raise ValueError(f"{label}: trial count differs")
    for a, e in zip(at, et):
        for key in ("block", "decide", "status", "false"):
            if a[key] != e[key]:
                raise ValueError(f"{label}: trial {key} differs")
        if a["rho"] is None or e["rho"] is None:
            if a["rho"] != e["rho"]:
                raise ValueError(f"{label}: trial rho presence differs")
        else:
            max_diff = max(max_diff, numeric_equal(a["rho"], e["rho"], f"{label}/trial rho"))
    return max_diff


def audit(job, full_gate=False):
    level, cls, scen, seed = job
    refs = reference(job)
    data = build(cls, scen, seed, level)
    x4 = F.p.parse(["c5", "--preset", "final"]).X4
    observer = Observer(data[0])
    rule, sim = observer.run(data, x4)
    post = None
    if F.has_post(scen) and not sim["div"]:
        post_data = build(cls, "post:" + scen, seed, level)
        post = F.run_rule(METHOD, *post_data, x4)[1]
    actual = dict(**F.measures(sim, data[0], post),
                  **F.trial_stats(rule, dict(data[0], scored_blocks=sim["n_done"])))
    max_ref_diff = verify_metrics(actual, refs[METHOD], str(job))
    if full_gate:
        other_data = build(cls, scen, seed, level)
        plain_rule, plain = F.run_rule(METHOD, *other_data, x4)
        for key in ("e", "y", "w"):
            if not np.array_equal(sim[key], plain[key], equal_nan=True):
                raise ValueError(f"{job}: hook changed {key}")
        for key in ("div", "n_done", "nb"):
            if sim[key] != plain[key]:
                raise ValueError(f"{job}: hook changed {key}")
        plain_stats = F.trial_stats(plain_rule, dict(other_data[0], scored_blocks=plain["n_done"]))
        for key, val in plain_stats.items():
            if actual[key] != val:
                raise ValueError(f"{job}: hook changed trial stats {key}")
    return dict(version=VERSION, job=list(job), verified=True,
                max_ref_diff=max_ref_diff, max_rho_diff=observer.max_rho_diff,
                max_td_diff=observer.max_td_diff, measures=actual,
                ref_no_trial_div=int(refs[NO_TRIAL]["div"]), decisions=observer.records)


def run_case(job):
    path = path_for(job)
    if path.is_file():
        saved = json.loads(path.read_text())
        if saved.get("version") == VERSION and saved.get("job") == list(job) and saved.get("verified") is True:
            return path, True
        raise ValueError(f"Invalid checkpoint {path}")
    result = audit(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    F.ab.atomic_write(path, lambda stream: json.dump(result, stream, allow_nan=False))
    return path, False


def gate():
    setup()
    jobs = [("nominal", "G", "RS-far", 100),
            ("nominal", "a145", "EN0", 100),
            ("nominal", "G", "stat", 100),
            ("-30dB", "jackhammer", "RS-far", 114),
            ("-20dB", "jackhammer", "stat", 101),
            ("-20dB", "G", "RS-near", 100)]
    for job in jobs:
        r = audit(job, full_gate=True)
        print(f"Gate PASS {job}: e/y/w bitwise equal; decisions={len(r['decisions'])}, "
              f"div={r['measures']['div']}, max reference diff={r['max_ref_diff']:.3g}", flush=True)
    print(f"Gate verified {len(jobs)}/{len(jobs)} cases", flush=True)


def run(level, scen, group, procs):
    batch = cases(level, scen, group)
    assert len(batch) == 100
    with Pool(procs, initializer=setup) as pool:
        for i, (path, reused) in enumerate(pool.imap_unordered(run_case, batch), 1):
            print(f"{i}/{len(batch)} {path.name}: {'cached' if reused else 'verified'}", flush=True)


def fmt(x):
    return "—" if x is None or not math.isfinite(float(x)) else f"{x:.6g}"


def quant(values, ps):
    if not values:
        return ["—"] * len(ps)
    return [fmt(v) for v in np.quantile(values, ps)]


def correlation(a, b):
    return None if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0 else float(np.corrcoef(a, b)[0, 1])


def report():
    batch = [job for level in LEVELS for job in cases(level)]
    assert len(batch) == 2000
    runs = []
    for job in batch:
        path = path_for(job)
        if not path.is_file():
            raise ValueError(f"Missing result {path}")
        r = json.loads(path.read_text())
        if r["version"] != VERSION or r["job"] != list(job) or r["verified"] is not True:
            raise ValueError(f"Invalid result {path}")
        runs.append(r)
    found = set(OUT.rglob("*.json"))
    if found != {path_for(job) for job in batch}:
        raise ValueError("Unexpected result files")
    groups = defaultdict(list)
    scatter = defaultdict(list)
    transfers = defaultdict(list)
    for r in runs:
        level, cls, scen, seed = r["job"]
        for row in r["decisions"]:
            groups[level, scen].append(row)
            for key in ("rho_rec", "rho_true", "rho_true_v"):
                scatter[key].append(float(row[key]))
            for key in ("decision", "decision_true", "decision_true_v", "status", "block", "trial_block", "age"):
                scatter[key].append(row[key])
            scatter["condition"].append(scen)
            scatter["model_error"].append(level)
            scatter["model_error_db"].append(np.nan if level == "nominal" else float(level[:-2]))
            scatter["noise"].append(cls)
            scatter["seed"].append(seed)
            if row["decision"] == "transfer":
                transfers[level, scen, row["phase"]].append(row)
    keys = [(level, scen) for level in LEVELS for scen in
            (SCENARIOS if level == "nominal" else SCENARIOS[:3])]
    total = sum(map(len, groups.values()))
    lines = ["# 移行判定：再構成誤差と真の残差", "",
             f"保存済み {len(runs)}/2000 実行。全実行で保存済み指標との一致確認に合格。判定 {total} 回。",
             f"指標の最大絶対差 {max(r['max_ref_diff'] for r in runs):.6g}、"
             f"rho の再計算最大絶対差 {max(r['max_rho_diff'] for r in runs):.6g}、"
             f"時間領域再構成の最大サンプル絶対差 {max(r['max_td_diff'] for r in runs):.6g}。",
             "一致条件：離散量は完全一致、連続量は絶対差 ≤ 1e-7 + 1e-6 |参照値|。"
             " nominal は512係数のモデルに追加誤差なし（真の経路は2048係数）。",
             "判定は同じ直近128サンプルを使用。真の残差は候補係数をその区間以前の全参照履歴にも固定した反実仮想。"
             " true_v は誤差マイク外乱を含む（EN0以外は同一）。各判定は実際の試行年齢における局所比較であり、"
             " oracle による判定を実行し直した軌跡ではない。", "",
             "## 判定一致表", "",
             "行が実際の判定、列が真の残差による判定。順番は破棄・移行・継続。"
             " 括弧内は不一致率。中間判定を含む。", "",
             "| モデル | 条件 | v | 判定数 | 破棄→破棄/移行/継続 | 移行→破棄/移行/継続 | 継続→破棄/移行/継続 | 不一致 |",
             "|---|---|---|---:|---|---|---|---:|"]
    summary = {}
    for level, scen in keys:
        rows = groups[level, scen]
        for suffix, label in (("", "なし"), ("_v", "あり")):
            mat = Counter((r["decision"], r["decision_true" + suffix]) for r in rows)
            n = len(rows)
            bad = n - sum(mat[k, k] for k in LABELS)
            if suffix == "":
                summary[level, scen] = (n, bad)
            cells = ["/".join(str(mat[a, b]) for b in LABELS) for a in LABELS]
            frac = f"{100*bad/n:.3f}%" if n else "—"
            lines.append(f"| {level} | {scen} | {label} | {n} | {' | '.join(cells)} | {bad} ({frac}) |")
    lines += ["", "## rho の関係", "",
              "log は自然対数。有限正値の対を log 比に使用し、相関は有限値の対の Pearson 相関。除外数を明示。",
              "| モデル | 条件 | v | 有効log対/全数 | 5% | 中央値 | 95% | 有限相関対 | Pearson r |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for level, scen in keys:
        rows = groups[level, scen]
        for suffix, label in (("", "なし"), ("_v", "あり")):
            a = np.array([float(r["rho_rec"]) for r in rows])
            b = np.array([float(r["rho_true" + suffix]) for r in rows])
            finite = np.isfinite(a) & np.isfinite(b)
            positive = finite & (a > 0) & (b > 0)
            logs = (np.log(a[positive]) - np.log(b[positive])).tolist()
            cells = quant(logs, [.05, .5, .95])
            corr = correlation(a[finite], b[finite])
            lines.append(f"| {level} | {scen} | {label} | {len(logs)}/{len(rows)} | {' | '.join(cells)} | {int(finite.sum())} | {fmt(corr)} |")
    lines += ["", "## 固定係数による移行の改善", "",
              "J = 10 log10(真の残差電力 / 一次騒音電力)。改善 = J_fg − J_bg（正が改善）。"
              " 移行直後からの固定0.5秒・2秒の窓のうち先頭4096サンプルを除く主定義と、"
              " 先頭を除かない補足定義の両方を掲載。外乱vは除外。"
              " 終端までに窓が揃わない場合は欠測。移行時点での分類：経路変化なし、変化前、変化後。"
              " 経路変化なしの移行を検出上の誤移行とするが、負の改善とは区別する。", "",
              "| モデル | 条件 | 移行時期 | 窓 | 定義 | 有効/移行 | Q1 | 中央値 | Q3 | 悪化数 (率) |",
              "|---|---|---|---|---|---:|---:|---:|---:|---:|"]
    total_valid = total_worse = 0
    for level, scen in keys:
        phases = ("no_change",) if scen in ("stat", "EN0") else ("pre_change", "post_change")
        for phase in phases:
            rows = transfers[level, scen, phase]
            for horizon in ("0.5", "2"):
                for field, label in (("future", "4096除外"), ("future_full", "全窓")):
                    vals = [r[field][horizon]["improvement"] for r in rows if r[field][horizon] is not None]
                    bad = sum(v < 0 for v in vals)
                    if horizon == "2" and field == "future":
                        total_valid += len(vals)
                        total_worse += bad
                    frac = f"{100*bad/len(vals):.3f}%" if vals else "—"
                    lines.append(f"| {level} | {scen} | {phase} | {horizon}s | {label} | {len(vals)}/{len(rows)} | "
                                 f"{' | '.join(quant(vals,[.25,.5,.75]))} | {bad} ({frac}) |")
    lines += ["", "## 提案法のみ発散した実行（nors-cap 比）", "",
              "発散検出サンプルより厳密に前に終了した最後の移行。未来窓は入力・真の経路による反実仮想であり発散後も計算可能。",
              "| モデル | 騒音 | 条件 | seed | 発散時刻 s | 最後の移行 s | rho_rec | rho_true | rho_true_v | 改善0.5s dB | 改善2s dB |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    isolated = Counter()
    for r in runs:
        level, cls, scen, seed = r["job"]
        if level == "nominal" or not r["measures"]["div"] or r["ref_no_trial_div"]:
            continue
        isolated[level] += 1
        fail = r["measures"]["failure"]["sample"]
        candidates = [a for a in r["decisions"] if a["decision"] == "transfer" and a["window_end"] - 1 < fail]
        if candidates:
            a = candidates[-1]
            vals = [fmt(a["window_end"]/F.FS), str(a["rho_rec"]), str(a["rho_true"]), str(a["rho_true_v"])]
            vals += [fmt(a["future"][h]["improvement"]) if a["future"][h] is not None else "—" for h in ("0.5", "2")]
        else:
            vals = ["—"] * 6
        lines.append(f"| {level} | {cls} | {scen} | {seed} | {fail/F.FS:.6f} | {' | '.join(vals)} |")
    if any(isolated[level] != 3 for level in LEVELS[1:]):
        raise ValueError(f"Mismatch with smodel divergence report: {isolated}")

    lines += ["", "## 数値から言えること（5行以内）", ""]
    for level in LEVELS:
        n = sum(summary[k][0] for k in keys if k[0] == level)
        bad = sum(summary[k][1] for k in keys if k[0] == level)
        lines.append(f"- {level}：外乱なしの真の判定との不一致 {bad}/{n} ({100*bad/n:.3f}%)。" if n
                     else f"- {level}：判定なし。")
    frac = f"{100*total_worse/total_valid:.3f}%" if total_valid else "—"
    lines += [f"- 全水準の2秒固定係数評価（4096サンプル除外）では悪化 {total_worse}/{total_valid} ({frac})。",
              f"- 提案法のみの発散は -30 dB で {isolated['-30dB']} 件、-20 dB で {isolated['-20dB']} 件。"]
    # Non-object arrays keep the scatter portable without allow_pickle.
    arrays = {k: np.asarray(v) for k, v in scatter.items()}
    assert all(len(v) == total for v in arrays.values())
    dest = HERE / "out" / "judge_scatter.npz"
    tmp = dest.with_name(".judge_scatter.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(dest)
    F.ab.atomic_write(HERE / "RESULTS_judge.md", lambda stream: stream.write("\n".join(lines) + "\n"))
    print(f"Report complete: {len(runs)} verified cases, {total} decisions", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("gate", "run", "report"))
    parser.add_argument("--level", choices=LEVELS)
    parser.add_argument("--scen", choices=SCENARIOS)
    parser.add_argument("--group", type=int, choices=(0, 1))
    parser.add_argument("--procs", type=int, default=4)
    args = parser.parse_args()
    setup()
    if args.command == "gate":
        gate()
    elif args.command == "report":
        report()
    else:
        if args.level is None or args.scen is None or args.group is None or args.procs < 1:
            parser.error("run requires --level --scen --group and positive --procs")
        run(args.level, args.scen, args.group, args.procs)


if __name__ == "__main__":
    main()
