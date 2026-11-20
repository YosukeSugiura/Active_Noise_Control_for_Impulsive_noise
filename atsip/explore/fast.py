"""Efficient realisation of the frozen method (final.P5Cap) with the same output (A1–A7), and its operation count.

Nothing in the rule changes; only how the quantities it needs are computed:
  A1  filtered reference r = s_hat * x: newest B samples from irfft(R_n) (R_n = S_hat X_n is computed anyway).
      Exact because len(s_hat) <= N - B + 1, so the last B samples of the circular product carry no wrap-around.
  A2  model output s_hat * y: newest B samples from irfft(S_hat rfft(y frame)) (same argument).
  A3  control output w * x: first B taps directly, the remaining M - B taps as three B-tap partitions, each a
      2B-point circular product with the spectrum of a past 2B-sample input window.  One input spectrum is
      computed per block and reused for the next two blocks (no delay is added: the tail needs x up to the
      previous block).
  A4  one FFT of the reference window and one spectral factor F per block, shared by the main frame, the trial
      frames and both directions.
  A5  real cepstrum by irfft instead of a complex ifft.
  A6  s_hat * y only in blocks that rebuild a frame (early phase or trial): the last M samples come from one
      product, since N - len(s_hat) + 1 >= M.  Ordinary blocks only store y.
  A7  the rebuilt part for the foreground coefficients is computed once per block and shared by the main frame
      and the acceptance ratio.

Commands:
  python fast.py check     equivalence with final.simulate on synthetic inputs + operation counts -> RESULTS_fast.md
"""
import collections
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

import final as F
from numpy.lib.stride_tricks import sliding_window_view

p, np, T = F.p, F.np, F.T
B, M, N = F.B, F.M, F.N
TRUNC = p.SFm.TRUNC
NT = 2 * B                      # size of the circular products for the tail of w
NP = M // B - 1                 # tail partitions
assert M % B == 0


class Ops:
    """FFT calls by size and multiply-accumulate operations of the direct parts, per block."""

    def __init__(s):
        s.fft = collections.Counter(); s.mac = 0; s.rows = []

    def take(s):
        out = (dict(s.fft), s.mac); s.fft = collections.Counter(); s.mac = 0
        return out


OPS = Ops()
_rfft, _irfft = np.fft.rfft, np.fft.irfft


def rfft(a, n=None):
    OPS.fft[n if n is not None else len(a)] += 1
    return _rfft(a, n)


def irfft(a, n):
    OPS.fft[n] += 1
    return _irfft(a, n)


class Direction:
    """SF direction with the factor F cached for the block (A4) and a real cepstrum (A5)."""

    def __init__(s):
        s.key = None; s.Srr = None; s.F = None

    def new_block(s):
        s.key = None; s.Srr = None

    def __call__(s, R, E, Srr, Pbar, s_eps, eps_reg, trunc=TRUNC, ret_y=False):
        key = (float(Pbar), float(s_eps), float(eps_reg))
        if Srr is not s.Srr or key != s.key:
            Lam = np.maximum(Srr + s_eps + eps_reg * Pbar, 1e-300)
            c = irfft(0.5 * np.log(Lam), N)
            c[1:N // 2] *= 2.0
            c[N // 2 + 1:] = 0.0
            s.F = np.exp(rfft(c)); s.key = key; s.Srr = Srr
        x = irfft(np.conj(R) * E / np.conj(s.F), N)
        x[trunc:] = 0.0
        G_ = rfft(x) / s.F
        if ret_y:
            return irfft(G_, N)[:M], x, s.F
        return irfft(G_, N)[:M]


class FastFrame:
    """Error frames of prop2.Frame / prop4.Frame4 from quantities built block by block (A1, A2, A4)."""

    def __init__(s, rule, Sf):
        s.rule, s.Sf = rule, Sf
        s.rwin = np.zeros(N)                    # last 2M samples of the filtered reference
        s.ybuf = np.zeros(N)                    # last N samples of the control output
        s.energy, s.n_rebuilt, s.rw = 0.0, 0, 0.0
        s._ref = s._sec = None; s._frozen = {}
        assert len(Sf) == N // 2 + 1

    def push(s, R, yb):
        s.rwin[:-B] = s.rwin[B:]; s.rwin[-B:] = irfft(R, N)[-B:]                    # A1
        s.ybuf[:-B] = s.ybuf[B:]; s.ybuf[-B:] = yb
        s.rw = float(s.rwin @ s.rwin)           # reference energy over 2M samples for the detector
        s._ref = s._sec = None; s._frozen = {}

    @property
    def sec(s):
        if s._sec is None:
            s._sec = irfft(s.Sf * rfft(s.ybuf), N)[-M:]                              # A2, A6
        return s._sec

    def trial_frame(s, w):
        end = s.end; start = max(0, end - M); n = end - start
        if s._ref is None:
            s._ref = rfft(s.rwin[1:], N)                                             # A4
        if id(w) not in s._frozen:                                                   # A7
            s._frozen[id(w)] = (w, irfft(s._ref * rfft(w, N), N))
        frozen = s._frozen[id(w)][1]
        return (s.e_all[start:end] - s.sec[-n:]) + frozen[2 * M - 1 - n:2 * M - 1]

    def __call__(s, i, w, e_all, ys=None):
        s.end, s.e_all = (i + 1) * B, e_all
        seg = e_all[max(0, s.end - M):s.end]
        s.energy = float(seg @ seg)
        if not s.rule.in_early(i):
            frame = np.zeros(N); frame[-len(seg):] = seg
            return rfft(frame)
        s.n_rebuilt += 1
        rebuilt = s.trial_frame(w)
        frame = np.zeros(N); frame[-len(rebuilt):] = rebuilt
        return rfft(frame)


def simulate_fast(dd, st, rule, nb=None):
    """final.simulate for a single plant and a single model, with A1–A7.  Returns the same fields plus the
    operation log (one entry per block: phase, FFT calls by size, direct multiply-accumulates)."""
    assert len(dd["hS_seq"]) == 1 and len(dd["hSm_seq"]) == 1
    assert dd.get("rmode", "fd") == "fd"
    assert 2 * M - 1 <= N
    d, xq = dd["d"], dd["xq"]
    nbmax = len(d) // B; nb = nbmax if nb is None else min(nb, nbmax)
    Sf = _rfft(dd["hSm_seq"][0][1], N)
    assert N - len(dd["hSm_seq"][0][1]) + 1 >= M
    hS = dd["hS_seq"][0][1]
    frame = FastFrame(rule, Sf); rule.frame = frame
    if getattr(rule, "det", None) is not None:
        rule.rw = np.zeros(nb)
    direction = Direction()
    swB = sliding_window_view(xq, B)
    xwin = [np.zeros(NT // 2 + 1, complex) for _ in range(NP)]      # spectra of the 2B-sample windows ending at s0, s0-B, ...
    w = np.zeros(M); e_all = np.zeros(nb * B); y_all = np.zeros_like(e_all)
    zi = np.zeros(len(hS) - 1)
    pref = dd.get("primary_power", float(np.mean(d ** 2)))
    div = False; n_done = nb; log = []
    saved = (p.SFm.sf_dir, np.fft.rfft, np.fft.irfft)
    p.SFm.sf_dir, np.fft.rfft, np.fft.irfft = direction, rfft, irfft
    OPS.take()
    try:
        for i in range(nb):
            s0 = i * B
            direction.new_block()
            # A3: control output.  Head: taps 0..B-1 directly.  Tail: taps B..M-1 on the previous 4B input samples.
            a = N + s0 - B + 1
            yb = swB[a:a + B][:, ::-1] @ w[:B]
            OPS.mac += B * B
            xwin = [rfft(xq[N + s0 - NT:N + s0], NT)] + xwin[:-1]
            acc = 0
            for q in range(NP):
                acc = acc + xwin[q] * rfft(w[(q + 1) * B:(q + 2) * B], NT)
            yb = yb + irfft(acc, NT)[B:]
            y_all[s0:s0 + B] = yb
            ys, zi = p.signal.lfilter(hS, [1.0], yb, zi=zi)          # the acoustic path (not part of the controller)
            eb = d[s0:s0 + B] + ys
            e_all[s0:s0 + B] = eb
            R = Sf * rfft(xq[s0 + B:s0 + B + N])
            frame.push(R, yb)
            E = frame(i, w, e_all)
            if getattr(rule, "det", None) is not None:
                rule.rw[i] = frame.rw
            c = T.Ctx(i, R, E, st, w)
            rule.mu_formula(i)
            early = rule.in_early(i); trial = getattr(rule, "bg", None) is not None
            mu, dv = rule.block(i, c)
            trial = trial or getattr(rule, "bg", None) is not None
            w = c.w_override if c.w_override is not None else w - 2 * mu * B / M * dv
            fft, mac = OPS.take()
            log.append((("early" if early else "late") + ("+trial" if trial else ""), fft, mac))
            div = not np.isfinite(w).all() or F.diverged(eb, pref)
            if div:
                e_all[s0:] = np.nan; n_done = i
                break
    finally:
        p.SFm.sf_dir, np.fft.rfft, np.fft.irfft = saved
    return dict(e=e_all, y=y_all, w=w, div=div, nb=nb, n_done=n_done, tr=rule.traces(), ops=log)


def fft_cost(n):
    """Real multiplications of an n-point real FFT (radix-2 complex FFT of n/2 points + pre/post twiddles)."""
    h = n // 2
    return 4 * (h // 2) * int(np.log2(h)) + 2 * n


def cost(fft, mac):
    return mac + sum(k * fft_cost(n) for n, k in fft.items())


def reference_cost(real_ffts, complex_ffts, Ls):
    """Direct FIRs plus real and complex FFTs of the reference implementation."""
    complex_cost = 4 * (N // 2) * int(np.log2(N))
    return B * (M + 2 * Ls) + real_ffts * fft_cost(N) + complex_ffts * complex_cost


def make_rule(dd, nb):
    X4 = p.parse(["c5", "--preset", "final"]).X4
    return F.P5Cap(F.q5.config(early_ok=(4, 0.25)), dd, nb, X4)


def count_reference(dd, Rall, st):
    """FFT calls per block of the reference simulator (plus one for R_n, which it takes precomputed)."""
    calls = [0, 0]
    saved = (np.fft.rfft, np.fft.irfft, np.fft.ifft)

    def wrap(f, kind):
        def g(*a, **k):
            calls[kind] += 1
            return f(*a, **k)
        return g

    rule = make_rule(dd, len(Rall))
    rule.frame = F.q4.Frame4(dd, rule)
    per = collections.defaultdict(list)
    inner = rule.block

    def block(i, c):
        early = rule.in_early(i); trial = rule.bg is not None
        out = inner(i, c)
        phase = ("early" if early else "late") + ("+trial" if trial or rule.bg is not None else "")
        per[phase].append((calls[0] + 1, calls[1]))  # R_n's precomputed reference FFT
        calls[:] = [0, 0]
        return out

    rule.block = block
    np.fft.rfft, np.fft.irfft, np.fft.ifft = (wrap(f, int(i == 2)) for i, f in enumerate(saved))
    try:
        sim = F.simulate(dd, Rall, st, rule, error_frame=rule.frame)
    finally:
        np.fft.rfft, np.fft.irfft, np.fft.ifft = saved
    assert sum(map(len, per.values())) == sim["n_done"] + int(sim["div"])
    return sim, rule, per


def check():
    cases = [("G", "stat", 10), ("a145", "stat", 10), ("II300", "stat", 11), ("a145", "RS-far", 10), ("G", "RS-far", 12),
             ("a120", "RS-near", 13), ("a145", "EN0", 10), ("G", "ENm10", 14)]
    Ls = F.LS_FINAL
    L = ["# 出力を変えない演算量の削減（A1–A7）", "",
         f"凍結した手法（final.P5Cap、Ŝ {Ls} タップ）を、同じ出力のまま少ない演算で実現する形（fast.py）と、基準の実装（final.simulate）の比較。",
         "表は自動生成。乗算は直接 FIR と FFT の概算。N 点の実 FFT は 4·(N/4)·log2(N/2) + 2N 回、複素 FFT は 4·(N/2)·log2(N) 回。スペクトルの点別演算、係数更新などは集計外。", "",
         "## 出力の一致", "", "| 入力 | 条件 | seed | 誤差の最大差（相対） | 出力の最大差（相対） | 試行の判定が一致 | 移した回数 |", "|---|---|---:|---:|---:|---|---:|"]
    ref_phase = collections.defaultdict(list); fast_phase = collections.defaultdict(list); worst = 0.0
    t_ref = t_fast = 0.0
    for cls, scen, seed in cases:
        dd, Rall, st = F.build(cls, scen, seed, Ls)
        t0 = time.perf_counter(); ref, rule0, per = count_reference(dd, Rall, st); t_ref += time.perf_counter() - t0
        for k, v in per.items():
            ref_phase[k] += v
        rule1 = make_rule(dd, len(Rall))
        t0 = time.perf_counter(); sim = simulate_fast(dd, st, rule1); t_fast += time.perf_counter() - t0
        assert not ref["div"] and not sim["div"], (cls, scen, seed)
        de = float(np.max(np.abs(sim["e"] - ref["e"])) / np.max(np.abs(ref["e"])))
        dy = float(np.max(np.abs(sim["y"] - ref["y"])) / max(np.max(np.abs(ref["y"])), 1e-300))
        key = lambda r: [(t["block"], t["status"], t["decide"]) for t in r.trials]
        same = key(rule0) == key(rule1)
        worst = max(worst, de, dy)
        L.append(f"| {cls} | {scen} | {seed} | {de:.1e} | {dy:.1e} | {'○' if same else '×'} | {len(rule1.restarts)} |")
        assert same and de < 1e-12 and dy < 1e-12, (cls, scen, seed, de, dy, same)
        for phase, fft, mac in sim["ops"]:
            fast_phase[phase].append((fft, mac))
    L += ["", f"最大の相対差は {worst:.1e}（丸め誤差の範囲）。試行の開始・判定・移行のブロックはすべて一致。", "",
          "## 1 ブロック（128 サンプル）あたりの演算", "",
          "| 状態 | ブロック数 | 基準：FFT 回数（実／複素） | 基準：乗算 | 削減後：FFT 回数（1024 点／256 点、最大） | 削減後：乗算（最大） | 比 |",
          "|---|---:|---:|---:|---|---:|---:|"]
    for phase in ("late", "early", "late+trial", "early+trial"):
        if phase not in fast_phase:
            continue
        real, complex_ = max(ref_phase[phase], key=lambda v: reference_cost(*v, Ls))
        rc = reference_cost(real, complex_, Ls)
        costs = [cost(f, m) for f, m in fast_phase[phase]]
        k = int(np.argmax(costs)); f = fast_phase[phase][k][0]
        L.append(f"| {phase} | {len(costs)} | {real}／{complex_} | {rc:,} | {f.get(N, 0)}／{f.get(NT, 0)} | {max(costs):,} | {max(costs) / rc:.2f} |")
    fx = B * (2 * M + Ls)
    L += ["", "状態：late＝後半の通常ブロック、early＝前半（作り直した誤差）、+trial＝裏の試行が動いているブロック。",
          "基準の乗算は、時間領域の FIR 3 本（w∗x、ŝ∗x、ŝ∗y）と FFT の合計。", "",
          "## 参考：時間領域の FxNLMS", "",
          f"1 サンプルあたり 出力 M + 濾波参照 Ls + 更新 M（正規化は逐次更新）として、1 ブロックあたり {fx:,} 回（M = {M}、Ls = {Ls}）。", "",
          "## 内訳（削減後、late）", ""]
    f, m = fast_phase["late"][len(fast_phase["late"]) // 2]
    L += [f"- 制御出力の先頭 {B} タップ（直接）：{m:,} 回", f"- FFT：1024 点 {f.get(N, 0)} 回、256 点 {f.get(NT, 0)} 回（1 回あたり {fft_cost(N):,}／{fft_cost(NT):,} 回）",
          "", f"計時（Python、参考）：基準 {t_ref:.1f} 秒、削減後 {t_fast:.1f} 秒（{len(cases)} 実行の合計。Python では FIR が BLAS で速く、演算数の比を反映しない）。"]
    (HERE / "RESULTS_fast.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "check":
        check()
    else:
        raise SystemExit(__doc__)
