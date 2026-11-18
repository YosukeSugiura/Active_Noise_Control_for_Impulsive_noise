"""Paired SF/SF-ND sweep. Run: check, smoke, run, report (in this order)."""

import argparse
import csv
import json
import platform
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from time import perf_counter

import vssT_lib as T
import numpy as np
import scipy

from protoC import C_DEF, Par, dbm, mu_early_value, prep3, rho_delay
from scipy.optimize import brentq

B, M, N = T.B, T.M, T.N
CLASSES = ("G", "BU", "PS", "II300", "a145", "a120", "pc", "pcb")
SEEDS = (10, 11, 12, 13, 14)
MUS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
RULES = ("SF", "SF-ND", "SF+g", "SF-ND+g", "SF-ND+g+early")
EARLY_BLOCKS = 64
NO_DELAY = (1, 0, 0, 0, 0)
COLUMNS = (
    "tag", "class", "seed", "rule", "L1_whole", "L1_first5", "L1_last10",
    "ex_0_1s", "ex_0_5s", "ex_last10", "L1_pc15_17", "L1_pcb10_12",
    "L1_pcb20_22", "div_num", "amp", "false_rs", "missed_rs",
)
SCORES = ("L1_whole", "L1_first5", "ex_0_1s")
HERE = Path(__file__).resolve().parent


class ReconstructedError:
    """Causal modified filtered-x frame with exact secondary-path model."""

    def __init__(self, filtered_reference):
        self.reference = np.pad(filtered_reference, (2 * M - 1, 0))
        self.secondary = np.zeros(M)
        self.frame = np.zeros(N)

    def __call__(self, block, w, actual_errors, secondary_block):
        end = (block + 1) * B
        start = max(0, end - M)
        self.secondary[:-B] = self.secondary[B:]
        self.secondary[-B:] = secondary_block
        disturbance = actual_errors[start:end] - self.secondary[-(end - start):]

        # r = S*x. Applying w to r gives the frozen S*(w*x) by FIR
        # commutativity. Keep the M unaliased samples of the circular result.
        r_window = self.reference[end:end + 2 * M - 1]
        frozen = np.fft.irfft(np.fft.rfft(r_window, N) * np.fft.rfft(w, N), N)
        samples = end - start
        self.frame[-samples:] = disturbance + frozen[2 * M - 1 - samples:2 * M - 1]
        return np.fft.rfft(self.frame)


class GainCorrectedSF(T.Fixed):
    def block(self, i, c):
        mu = self.mu_formula(i) / max(c.g, T.GMIN)
        dv = T.direction(self.kind, c.R, c.E, c.Srr, c.Pbar, c.eps)
        self.tr["mu_app"].append(mu)
        return mu, dv


def simulate(dd, references, stats, kind, mu, mu_e, nb):
    schedule = mu
    if kind.endswith("+early"):
        schedule = np.full(nb, mu)
        schedule[:EARLY_BLOCKS] = mu_e
    rule = GainCorrectedSF("SF", schedule) if "+g" in kind else T.Fixed("SF", schedule)
    frame = ReconstructedError(dd["r"]) if kind.startswith("SF-ND") else None
    return T.simulate(dd, references, stats, rule, nb=nb, error_frame=frame)


def measure(sim, data):
    # score_pc treats positive last10 ANR as divergence. Reuse the common
    # metrics and score the path windows using the same ANR timestamps.
    metric = T.metrics(sim, dict(data, pc=False))
    div = metric["div"]
    whole, first5, last10 = metric["L1"]
    ex = sim["ex"]
    bps = T.FS // B
    windows = {}
    if not div and (data["pc"] or data.get("pseg")):
        anr = T.O.anr_curve(sim["e"], data["d"][:len(sim["e"])])
        times = (63 + 64 * np.arange(len(anr))) / T.FS
        pairs = ((15, 17),) if data["pc"] else ((10, 12), (20, 22))
        for start, end in pairs:
            if sim["nb"] >= end * bps:
                windows[start] = float(anr[(times >= start) & (times < end)].mean())

    return dict(
        L1_whole=whole, L1_first5=first5, L1_last10=last10,
        ex_0_1s=dbm(ex[:bps]) if not div else None,
        ex_0_5s=dbm(ex[:5 * bps]) if not div else None,
        ex_last10=dbm(ex[-max(1, len(ex) // 10):]) if not div else None,
        L1_pc15_17=windows.get(15), L1_pcb10_12=windows.get(10), L1_pcb20_22=windows.get(20),
        div_num=int(div), amp=int(not div and last10 > 0), false_rs=None, missed_rs=None,
    )


def rule_name(kind, mu):
    return f"{kind} mu{mu:g}"


def check():
    dd, references, stats = T.prep("G_s10", 24)
    w0 = 0.2 * T.wM
    errors = []

    def capture(i, w, mu, dv, ctx, rule, mf, **state):
        end = (i + 1) * B
        lo = max(0, end - M)
        direct = T.frozen_error(dd, w, lo, end)
        expected = T.frame_from(direct, lo, i * B)
        np.testing.assert_allclose(ctx.E, expected, rtol=1e-10, atol=1e-9)
        observed = T.ML.frame_E(state["e_all"], i * B, M)
        observed_dir = T.direction("SF", ctx.R, observed, ctx.Srr, ctx.Pbar, ctx.eps)
        if i == 0 or mu == 0:
            # At startup only one block can contribute. For the frozen run,
            # the actual secondary-path tail also used the current w.
            np.testing.assert_allclose(dv, observed_dir, rtol=1e-10, atol=1e-12)
        errors.append(np.linalg.norm(dv - observed_dir))

    for mu in (0.0, 0.3):
        errors.clear()
        rebuilt = ReconstructedError(dd["r"])
        T.simulate(dd, references, stats, T.Fixed("SF", mu), nb=24, w0=w0,
                   hook=capture, error_frame=rebuilt)
        if mu:
            assert max(errors[1:]) > 1e-6
            print(f"one-block history: direction delta norm {errors[0]:.3e}")
        else:
            print(f"constant coefficients: maximum direction delta norm {max(errors):.3e}")

    base = T.simulate(dd, references, stats, T.Fixed("SF", 0.3), nb=24)
    test = simulate(dd, references, stats, "SF", 0.3, 0.53, 24)
    for key in ("e", "w", "ex"):
        np.testing.assert_array_equal(base[key], test[key])
    print("SF baseline bitwise identical; SF-ND verified against the direct frozen convolution")


def out_path(tag, nb_limit):
    folder = "nodelay" if nb_limit is None else f"nodelay_nb{nb_limit}"
    return HERE / "out" / folder / f"{tag}.json"


def load_rows(tag, nb_limit, mu_e):
    path = out_path(tag, nb_limit)
    if path.exists():
        data = json.loads(path.read_text())
        if data["mu_e"] == mu_e:
            return data["rows"]
    return {}


def run_tag(args):
    cls, seed, nb_limit, mu_e = args
    tag = f"{cls}_s{seed}"
    path = out_path(tag, nb_limit)
    rows = load_rows(tag, nb_limit, mu_e)
    pending = [(kind, mu) for kind, mu in product(RULES, MUS) if rule_name(kind, mu) not in rows]
    if not pending:
        return tag
    dd, references, stats = prep3(tag, nb_limit)
    nb = len(dd["d"]) // B if nb_limit is None else nb_limit
    start = perf_counter()
    for kind, mu in pending:
        name = rule_name(kind, mu)
        sim = simulate(dd, references, stats, kind, mu, mu_e, nb)
        rows[name] = dict(tag=tag, **{"class": cls}, seed=seed, rule=name, **measure(sim, dd))
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(mu_e=mu_e, rows=rows)))
        tmp.replace(path)
    print(f"{tag}: {len(rows)} runs, {perf_counter() - start:.1f} s", flush=True)
    return tag


def sweep(smoke, mu_e):
    nb_limit, seeds = (1250, SEEDS[:1]) if smoke else (None, SEEDS)
    out_path("placeholder", nb_limit).parent.mkdir(parents=True, exist_ok=True)
    jobs = [(cls, seed, nb_limit, mu_e) for cls, seed in product(CLASSES, seeds)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for _ in pool.map(run_tag, jobs):
            pass
    rows = [row for cls, seed in product(CLASSES, seeds)
            for row in load_rows(f"{cls}_s{seed}", nb_limit, mu_e).values()]
    assert len(rows) == len(jobs) * len(RULES) * len(MUS)
    for kind in RULES:
        subset = [row for row in rows if row["rule"].rsplit(" ", 1)[0] == kind]
        print(f"{kind}: {sum(r['div_num'] for r in subset)} div, "
              f"{sum(r['amp'] for r in subset)} amp / {len(subset)} runs", flush=True)


def benchmark(mu_e):
    dd, references, stats = T.prep("G_s10")
    nb = len(dd["d"]) // B
    kinds = ("SF", "SF-ND")
    for kind in kinds:
        simulate(dd, references, stats, kind, 0.1, mu_e, 100)
    times = {kind: [] for kind in kinds}
    for trial in range(5):
        for kind in (kinds if trial % 2 == 0 else kinds[::-1]):
            start = perf_counter()
            sim = simulate(dd, references, stats, kind, 0.1, mu_e, nb)
            times[kind].append(perf_counter() - start)
            assert not sim["div"]
    return times


def class_average(rows, score):
    """Mean of the eight class means, omitting numerically failed runs."""
    per_class = []
    for cls in CLASSES:
        values = [r[score] for r in rows if r["class"] == cls]
        valid = [v for v in values if v is not None and np.isfinite(v)]
        if valid:
            per_class.append(np.mean(valid))
    return float(np.mean(per_class)) if per_class else float("nan")


def paired(candidate, baseline, score):
    """Average within each seed over the same eight input classes."""
    cand = {(r["class"], r["seed"]): r for r in candidate}
    base = {(r["class"], r["seed"]): r for r in baseline}
    diffs = []
    for seed in SEEDS:
        pairs = [(cand[(cls, seed)][score], base[(cls, seed)][score]) for cls in CLASSES]
        if all(a is not None and b is not None and np.isfinite([a, b]).all() for a, b in pairs):
            diffs.append(float(np.mean([a - b for a, b in pairs])))
    if not diffs:
        return "—"
    return f"{np.mean(diffs):+.3f} [{min(diffs):+.3f}, {max(diffs):+.3f}] ({len(diffs)}/5)"


def aggregate(index):
    table = {}
    for kind, mu in product(RULES, MUS):
        rows = index[rule_name(kind, mu)]
        scores = tuple(class_average(rows, score) for score in SCORES)
        counts = (sum(r["div_num"] for r in rows), sum(r["amp"] for r in rows))
        table[kind, mu] = scores, counts
    return table


def best_rows(table, index):
    lines = []
    ablations = []
    for i, score in enumerate(SCORES):
        best = {}
        for kind in RULES:
            stable = [mu for mu in MUS if table[kind, mu][1] == (0, 0)]
            if stable:
                best[kind] = min(stable, key=lambda mu: table[kind, mu][0][i])
        for kind in RULES:
            if kind not in best:
                lines.append(f"| {score} | {kind} | なし | — | — | — | — |")
                continue
            mu = best[kind]
            rs = index[rule_name(kind, mu)]
            same_mu = paired(rs, index[rule_name("SF", mu)], score)
            sf_mu, own_best = "—", "—"
            if "SF" in best:
                sf_mu = f"{best['SF']:g}"
                own_best = paired(rs, index[rule_name("SF", best["SF"])], score)
            lines.append(f"| {score} | {kind} | {mu:g} | {table[kind, mu][0][i]:.3f} | "
                         f"{same_mu} | {sf_mu} | {own_best} |")
        for kind, baseline in (("SF-ND+g", "SF+g"), ("SF-ND+g+early", "SF-ND+g")):
            if kind in best and baseline in best:
                mu, base_mu = best[kind], best[baseline]
                diff = paired(index[rule_name(kind, mu)], index[rule_name(baseline, base_mu)], score)
                ablations.append(f"| {score} | {rule_name(kind, mu)} | {rule_name(baseline, base_mu)} | {diff} |")
    return lines, ablations


def stable_limit(table, kind, include_amp):
    allowed_counts = (0, 0) if include_amp else (0,)
    stable = [mu for mu in MUS if table[kind, mu][1][:len(allowed_counts)] == allowed_counts]
    if not stable:
        return "なし"
    top = max(stable)
    holes = [mu for mu in MUS if mu < top and mu not in stable]
    upper = f"≥ {top:g}" if top == MUS[-1] else f"{top:g}"
    return upper + (f"（下側の格子に失敗あり {', '.join(map(str, holes))}）" if holes else "")


def report(mu_e):
    index = {rule_name(kind, mu): [] for kind, mu in product(RULES, MUS)}
    rows = []
    for cls, seed in product(CLASSES, SEEDS):
        tag_rows = load_rows(f"{cls}_s{seed}", None, mu_e)
        if set(tag_rows) != set(index):
            raise ValueError(f"Incomplete full run for {cls}_s{seed}")
        for name in index:
            row = tag_rows[name]
            index[name].append(row)
            rows.append(row)
    times = benchmark(mu_e)
    with (HERE / "RESULTS_nodelay.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if isinstance(value, float) and np.isnan(value) else value
                             for key, value in row.items()})

    table = aggregate(index)
    best, ablations = best_rows(table, index)
    c = np.asarray(C_DEF) / sum(C_DEF)
    sf_boundary = brentq(lambda a: rho_delay(a, c) - 1, 0.5, 2.0)
    median = {k: np.median(v) for k, v in times.items()}
    lines = [
        "# SF と誤差フレームを再構成した SF-ND の比較",
        "",
        "## 条件と評価",
        "",
        f"- 基準コミット `baa85203f59e5a8dddd65d5d34399281a2034dac`。"
        f"{', '.join(CLASSES)} の8クラス、seed 10–14、各30秒（3,750ブロック）。"
        "全40入力に50規則を適用した。seed 10 の8入力で1,250ブロックの短縮版を先に実行した。",
        f"- M={M}、N={N}、B={B}、サンプリング周波数 {T.FS} Hz。"
        "SF と SF-ND は同じ正則化、スペクトル因子、更新係数 2B/M を使う。"
        "実誤差から実際の二次経路出力を引いて外乱を求め、現在の係数による二次経路出力を足す。"
        "L1 はシミュレータの実誤差で評価する。",
        f"- +g は各ブロックの μ に 1/max(g, 0.25) を掛ける。"
        f"+early は初回のみ先頭 {EARLY_BLOCKS} ブロックの補正前 μ を **{mu_e:.3f}** とする。"
        "protoC.mu_early_value の既定模型（h_eff=0.2375、予測8ブロック、K=full）の c_k を (1, 0, 0, 0, 0) にした値。",
        "- 数値発散は非有限または Φ 超過比 > 10⁸。増幅は数値発散しなかった試行で L1 last10 > 0 dB。"
        "数値発散は div_num=1、amp=0 とし、指標は空欄。対象外の経路変化窓と再始動の2列も空欄。",
        "- L1 は既存 ANR の時間平均。ex は各試行の区間内の線形 Φ 超過比を平均してから dB 化する。"
        "以下の数値は各クラス内の有効 seed の dB 平均をクラス間で等重み平均した値。"
        "数値発散した試行は平均から除外し、有効試行がないクラスはその行の平均から除外する。増幅試行は平均に含む。",
        "",
        "## μ の掃引",
        "",
        "| 規則 | μ | L1 whole (dB) | L1 first5 (dB) | ex 0–1 s (dB) | 数値発散 / 40 | 増幅 / 40 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (kind, mu), (scores, counts) in table.items():
        values = " | ".join("—" if np.isnan(v) else f"{v:.3f}" for v in scores)
        lines.append(f"| {kind} | {mu:g} | {values} | {counts[0]} | {counts[1]} |")
    lines.extend([
        "",
        "## 指標を最小にする μ と対応差",
        "",
        "全40試行で数値発散・増幅ともに0件の μ が対象。差は候補 − SF（dB）。"
        "各 seed で8クラスの対応差を平均し、seed 間の平均 [最小, 最大] と集計できた seed 数を示す。"
        "いずれかのクラスで指標が欠ける seed はその差の集計から除外する。負値は候補の改善を示す。",
        "",
        "| 指標 | 規則 | 最良 μ | 平均 dB | 同じ μ の SF との差 | SF の最良 μ | 最良 μ の SF との差 |",
        "|---|---|---:|---:|---:|---:|---:|",
        *best,
        "",
        "gain 補正ありで再構成を追加した場合と、SF-ND+g に初期ステップを追加した場合の対応差。"
        "各規則で安定な格子点から選んだ指標別の最良 μ を使う。",
        "",
        "| 指標 | 候補 | 比較元 | 候補 − 比較元 (dB) |",
        "|---|---|---|---:|",
        *ablations,
        "",
        "## 安定な μ の上限",
        "",
        "| 規則 | 数値発散0件の最大 μ | 数値発散・増幅とも0件の最大 μ |",
        "|---|---:|---:|",
        *(f"| {kind} | {stable_limit(table, kind, False)} | {stable_limit(table, kind, True)} |" for kind in RULES),
        "",
        f"平均モデルの根の単位円交差は、SF が μh={sf_boundary:.3f}"
        f"（h=0.25 で μ≈{sf_boundary / T.H:.2f}）、SF-ND が μh=2（μ=8）。"
        "今回の μ 格子は最大3（μh=0.75）で、固定 μ の平均モデルの境界より低い。"
        "実験の上限は有限時間の数値発散と実誤差の増幅で決まり、平均の安定境界を直接推定する量とは異なる。"
        "+g では g に応じて実際の μ が変動する。"
        "出典は `THEORY_final.md` §3.1、`THEORY_vss2.md` §9.2、§10.1 と `protoC.rho_delay`。",
        "",
        "## 追加計算量と実測時間",
        "",
        "誤差スペクトルの FFT 1回と SF 方向計算の変換5回は共通。"
        "参照スペクトルの生成を含めると、SF は N 点変換7回、SF-ND は10回。"
        "SF-ND は過去 2M−1 サンプルの r=S*x と現在の w のそれぞれに N 点 rFFT を掛け、"
        "積の N 点 irFFT の有効 M サンプルを使う。追加は **N 点変換3回**"
        "（rFFT 2、irFFT 1）、**N/2+1=513 点のスペクトル乗算**"
        "（511 複素乗算と DC/Nyquist の実数乗算2回相当）、定常時の実数加減算 2M=1,024 回。"
        "畳み込みのための追加の直接 FIR 積和は0回。"
        "実測ではデータが持つ r と、シミュレータが物理誤差の生成に使う二次経路出力を共用した。"
        "ストリーミング実装で r が未生成なら、直接 FIR では別途 B×128=16,384 実数積和/ブロック。"
        "実機で二次経路出力のモデルを新たに動かす場合も、直接 FIR なら同数の積和を別途要する。",
        "",
        f"G_s10、固定 μ=0.1、30秒。1プロセス、BLAS 1スレッド、"
        f"Python {platform.python_version()}、NumPy {np.__version__}、SciPy {scipy.__version__}、{platform.machine()}。"
        "短いウォームアップの後、実行順を交互にし各5回測定した。"
        "計測区間はシミュレーション全体（実誤差生成・係数更新・Φ 評価を含む）。"
        "共通の前処理と実行後の指標集計を除いた。",
        "",
        "| 規則 | 中央値 s/30秒 | 範囲 s | 中央値 μs/ブロック | SF 比 |",
        "|---|---:|---:|---:|---:|",
    ])
    for kind, samples in times.items():
        lines.append(f"| {kind} | {median[kind]:.3f} | {min(samples):.3f}–{max(samples):.3f} | "
                     f"{median[kind] * 1e6 / T.NB_FULL:.1f} | {median[kind] / median['SF']:.3f}× |")
    lines.extend([
        "",
        "## 再現手順",
        "",
        "```sh",
        "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
        "python nodelay.py check",
        "python nodelay.py smoke",
        "python nodelay.py run",
        "python nodelay.py report",
        "```",
        "",
        "check では1ブロック分だけが非ゼロの初回フレームと、非ゼロの係数を固定した複数ブロックで、"
        "再構成方向と元の SF の方向の一致を検証する。係数が動くときは直接畳み込みで求めた"
        "固定係数の誤差スペクトルと照合する。掃引と短縮版は out/ の別ディレクトリに条件ごとに保存する。",
    ])
    (HERE / "RESULTS_nodelay.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(rows)} rows to RESULTS_nodelay.csv and RESULTS_nodelay.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "smoke", "run", "report"))
    command = parser.parse_args().command
    if command == "check":
        check()
        return
    mu_e = mu_early_value(Par(ck=NO_DELAY))
    print(f"mu_e = {mu_e:.3f}", flush=True)
    if command == "report":
        report(mu_e)
    else:
        sweep(command == "smoke", mu_e)


if __name__ == "__main__":
    main()
