"""Detector study on design seeds (10-49): log candidate statistics from runs without restart, then evaluate
detector rules offline on the logs.

With identical input and detector settings, the foreground, its spectra and the detector state
match a live run through the first detection. Rejected trials leave the foreground unchanged,
so the signal statistics still match until a confirmed transfer. Subsequent offline triggers
describe the no-restart trajectory only. The scoring windows do not remove that limitation.

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
WIN = 64                  # study window [change, change + 64), distinct from p.RS_TRUE_WIN
LOG_VERSION = 2


def run_log(job):
    cls, scen, seed = job
    assert seed in SEEDS
    f = OUT / f"{cls}_{scen}_s{seed}.npz"
    if f.exists():
        with np.load(f) as cached:
            if "version" in cached and int(cached["version"]) == LOG_VERSION:
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
        if not (np.isfinite(pr) and np.isfinite(pe) and pr > 0 and pe > 0):
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
    # Use exactly the same windowed dot products as prop5.P5 and protoC.rwin_energy.
    e = sim["e"]
    rr = p.rwin_energy(dd, nb, 2 * M)
    stop = sim["n_done"]
    ee = np.full(nb, np.nan)
    for i in range(stop):
        hi = (i + 1) * B
        window = e[max(0, hi - M):hi]
        ee[i] = window @ window
    z = np.divide(ee, rr, out=np.full(nb, np.nan), where=rr > 0)
    for stat in out.values():
        stat[stop:] = np.nan
    out["z"] = z
    for L in LAMS:
        out[f"q{L}"] = out[f"c{L}"] * z
    ref = np.where(rr > 0, rr, np.nan)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.stem + ".tmp.npz")
    np.savez(tmp, version=LOG_VERSION, div=sim["div"], changes=np.array(dd["changes"], int), events=np.array(dd["events"], int), rr=ref, **out)
    tmp.replace(f)
    print(f"{f.stem}: {time.monotonic() - t0:.0f} s", flush=True)


@F.fastbase.njit(cache=True)
def _triggers(series, ratios, thresholds, consecutive, beta, guard, freeze=False):
    n_part, n_block = series.shape
    base = np.full(n_part, np.nan)
    count = np.zeros(n_part, dtype=np.int64)
    last = -guard
    out = []
    for i in range(n_block):
        armed = i - last >= guard
        fire = False
        for j in range(n_part):
            v, ref = series[j, i], base[j]
            finite = np.isfinite(v)
            crossed = v > thresholds[j] * ref if ratios[j] else v - ref > thresholds[j]
            hit = armed and not np.isnan(ref) and finite and crossed
            count[j] = count[j] + 1 if hit else 0
            fire = fire or count[j] >= consecutive[j]
        if not freeze or (armed and not fire):
            for j in range(n_part):
                v, ref = series[j, i], base[j]
                if np.isfinite(v):
                    base[j] = v if np.isnan(ref) else (1 - beta) * ref + beta * v
        if fire:
            last = i
            out.append(i)
            count[:] = 0
    return out


def triggers(s, mode, k, d, beta=0.05, guard=64):
    """Single branch with the baseline and guard order of RSNDetector."""
    return triggers_or(((s, mode, k, d),), beta, guard)


def triggers_or(parts, beta=0.05, guard=64, freeze=False):
    """Independent previous EMAs and counts, with a shared guard and reset."""
    signals, modes, ks, ds = zip(*parts)
    return _triggers(np.asarray(signals), np.array([m == "ratio" for m in modes]),
                     np.asarray(ks, dtype=float), np.asarray(ds), beta, guard, freeze)


RULES = ([("z", "ratio", k, d) for k in (2, 3, 4, 6) for d in (1, 2, 3)]
         + [(f"q{L}", "ratio", k, d) for L in LAMS for k in (2, 3, 4, 6) for d in (1, 2, 3)]
         + [(f"c{L}", "diff", k, d) for L in LAMS for k in (0.05, 0.1, 0.2, 0.3) for d in (1, 2, 3)]
         + [("c0.25u", "diff", k, d) for k in (0.05, 0.1, 0.2, 0.3) for d in (1, 2, 3)]
         + [(("z", "ratio", 4, 2), (f"c{L}", "diff", k, d)) for L in LAMS for k in (0.1, 0.2, 0.3) for d in (2, 3)])


def evaluate():
    logs = {}
    for cls, scen, seed in itertools.product(CLASSES, SCEN, SEEDS):
        path = OUT / f"{cls}_{scen}_s{seed}.npz"
        with np.load(path) as entry:
            if int(entry["version"]) != LOG_VERSION:
                raise ValueError(f"incompatible detector log: {path}")
            lg = dict(entry)
        logs[cls, scen, seed] = lg
    L = ["# 検出器の比較（設計用 seed 10–49）", "",
         f"{len(logs)} ログ。切替ブロックから64ブロックの半開区間に検出がなければ見逃し。遅れは切替から区間内の最初の検出まで（ブロック）。",
         "誤反応は stat・外乱条件の全区間と切替条件の切替前を対象とし、開始2秒を除く。外乱への反応も誤反応と数える。",
         "再始動なしのログを使ったオフライン評価。実際の確定移行後に検出を繰り返す閉ループの結果とは区別する。発散ログは分母から除く。", ""]
    completed = {key: lg for key, lg in logs.items() if not bool(lg["div"])}
    rows = []
    for rule in RULES:
        combo = isinstance(rule[0], tuple)
        parts = rule if combo else (rule,)
        name = " または ".join(f"{s}・{m}・{k:g}・{d}" for s,m,k,d in parts)
        by_change, false = {}, {}
        for (cls, scen, seed), lg in completed.items():
            if combo:
                tr = triggers_or([(lg[s],m,k,d) for s,m,k,d in parts])
            else:
                s,m,k,d = rule
                tr = triggers(lg[s],m,k,d)
            ch = lg["changes"]
            end = int(ch[0]) if len(ch) else len(lg[parts[0][0]])
            false[cls,scen,seed] = (sum(WARM <= t < end for t in tr), max(0, end - WARM))
            if len(ch):
                by_change[cls,scen,seed] = next((t-end for t in tr if end <= t < end + WIN), None)
        values = lambda c: [v is None for (cl,_,_),v in by_change.items() if c==cl]
        mr = {c: float(np.mean(values(c))) for c in CLASSES if values(c)}
        far = {}
        for c in CLASSES:
            eligible = [v for (cl,_,_),v in false.items() if cl==c]
            duration = sum(b for _,b in eligible)
            if duration:
                far[c] = sum(n for n,_ in eligible)*BPS*60/duration
        delays = [d for d in by_change.values() if d is not None]
        rows.append(dict(rule=name, combo=combo, miss=float(np.mean(list(mr.values()))),
                         worst_miss=max(mr.values()), fa=float(np.mean(list(far.values()))), worst_fa=max(far.values()),
                         delay=float(np.median(delays)) if delays else None, change=by_change, false=false, mr=mr, far=far))
    rows.sort(key=lambda r: (r["miss"], r["fa"]))
    def table(headers, vals):
        for row in (headers, ["---"]*len(headers), *vals):
            L.append("| " + " | ".join(map(str,row)) + " |")
        L.append("")
    L += ["## 全規則（見逃し率、同率なら誤反応率の昇順）",""]
    table(("規則","見逃し（平均）","見逃し（最悪クラス）","誤反応/分（平均）","誤反応/分（最悪）","遅れ中央値"),
          [(r["rule"],f'{r["miss"]:.4f}',f'{r["worst_miss"]:.4f}',f'{r["fa"]:.2f}',
            f'{r["worst_fa"]:.2f}',F.fmt(r["delay"],1)) for r in rows])
    cur = next(r for r in rows if r["rule"]=="z・ratio・4・2")
    pareto = [r for r in rows if not any(o["miss"] <= r["miss"] and o["fa"] <= r["fa"]
             and (o["miss"] < r["miss"] or o["fa"] < r["fa"]) for o in rows)]
    L += ["## 見逃しと誤反応のパレート集合",""]
    table(("規則","見逃し率","誤反応/分"),[(r["rule"],f'{r["miss"]:.4f}',f'{r["fa"]:.2f}') for r in pareto])
    combos = [r for r in rows if r["combo"]][:5]
    L += ["## 所見","",f"ログ {len(logs)}/2000、発散 {len(logs)-len(completed)} 件。現在は {cur['rule']}。"
          "クラス内の見逃し率の95%区間は、seed ごとの近・遠を一緒に2,000回復元抽出する百分位区間。"
          "候補選定後の記述区間であり、別データでの性能を表す区間ではない。", ""]
    for label,r in [("現在",cur),("全候補中の最少",rows[0]),*((f"OR {i}",r) for i,r in enumerate(combos,1))]:
        L.extend([f"### {label} — {r['rule']}","",f"全クラスの平均見逃し率 {r['miss']:.2%}、誤反応 {r['fa']:.2f}/分、遅れ中央値 {F.fmt(r['delay'],1)} ブロック。",""])
        stats=[]
        for c in CLASSES:
            parts = []
            for scenes in (("RS-near","RS-far"),("RS-near",),("RS-far",)):
                per = {}
                dd = []
                for (cl,sc,sd),v in r["change"].items():
                    if cl!=c or sc not in scenes:
                        continue
                    per.setdefault(sd,[]).append(v is None)
                    if v is not None:
                        dd.append(v)
                if per:
                    blocks = np.array([(sum(v),len(v)) for _,v in sorted(per.items())])
                    ix = np.random.default_rng(0).integers(0,len(blocks),size=(2000,len(blocks)))
                    n,total = blocks[ix].sum(axis=1).T
                    lo,hi = np.quantile(n/total,[.025,.975])
                    misses, trials = blocks.sum(axis=0)
                    parts.extend((f"{misses}/{trials} ({misses/trials:.1%} [{lo:.1%}, {hi:.1%}])",F.fmt(float(np.median(dd)) if dd else None,1)))
                else:
                    parts.extend(("—","—"))
            n = sum(v[0] for key,v in r["false"].items() if key[0]==c)
            stats.append((c,*parts,n,F.fmt(r["far"].get(c),2)))
        table(("クラス","見逃し 全 [95%]","遅れ 全","見逃し near [95%]","遅れ near","見逃し far [95%]","遅れ far","誤反応件数","誤反応/分"),stats)
    L += ["### 最終評価で報告された傾向との照合",""]
    for cls in ("machinegun","volvo"):
        near = [v for (cl,sc,_),v in cur["change"].items() if (cl,sc)==(cls,"RS-near")]
        far = [v for (cl,sc,_),v in cur["change"].items() if (cl,sc)==(cls,"RS-far")]
        mn,mf = sum(v is None for v in near),sum(v is None for v in far)
        direction = ("near が多く、方向は一致" if mn/len(near) > mf/len(far)
                     else "near と far は同率" if mn/len(near) == mf/len(far)
                     else "far が多く、近遠の方向は一致しない")
        L.append(f"- {cls} の現在規則の見逃しは near {mn}/{len(near)}、far {mf}/{len(far)}。{direction}。")

    L += ["","近遠の見逃し件数の差を、最終評価で報告された傾向（machinegun・volvo の near が難しい）と対照した。"
          "最終評価は125ブロック内に始まる試行の確定移行を数え、ここでは64ブロック内の検出を数える。", "",
          "### 現在規則が見逃した実行の z と c0.1","",
          "各相対ブロックでの実行間中央値。括弧内は対象実行数。0 は切替後最初のブロック。", ""]
    series={}
    for c,sc in itertools.product(("machinegun","volvo"),("RS-near","RS-far")):
        selected = [completed[key] for key,v in cur["change"].items() if key[:2]==(c,sc) and v is None]
        if selected:
            vals = np.array([[lg[k][int(lg["changes"][0])-40:int(lg["changes"][0])+41] for k in ("z","c0.1")] for lg in selected])
            series[c,sc] = np.nanmedian(vals,axis=0)
        else:
            series[c,sc] = np.full((2,81),np.nan)
        L.append(f"- {c}・{sc}（{len(selected)}件）")
    L.append("")
    table(("クラス","条件","z 前40点","z 後40点","c0.1 前40点","c0.1 後40点"),
          [(c,sc,*(F.fmt(float(np.median(span)),4) for seq in a for span in (seq[:40],seq[40:80])))
           for (c,sc),a in series.items()])
    table(("相対ブロック",*(f"{c} {sc} {k}" for c,sc in series for k in ("z","c0.1"))),
          [(j-40,*(F.fmt(float(v),4) for a in series.values() for v in a[:,j])) for j in range(81)])
    (HERE/"RESULTS_det.md").write_text("\n".join(L).rstrip()+"\n",encoding="utf-8")
    print(f"wrote RESULTS_det.md {len(rows)} rules / {len(logs)} logs")


DET2_SCEN = ("stat", "RS-near", "RS-far")
DET2_GUARDS = (64, 32, 16, 8, 4, 0)


def trial_starts(triggers_, busy=16):
    """Count an accepted trigger and suppress the following busy blocks."""
    starts = []
    for t in triggers_:
        if not starts or t - starts[-1] > busy:
            starts.append(t)
    return starts


def evaluate_guard():
    logs = {}
    for cls, scen, seed in itertools.product(CLASSES, DET2_SCEN, SEEDS):
        with np.load(OUT / f"{cls}_{scen}_s{seed}.npz") as entry:
            if int(entry["version"]) != LOG_VERSION:
                raise ValueError(f"incompatible detector log: {cls}/{scen}/{seed}")
            if not bool(entry["div"]):
                change = entry["changes"]
                logs[cls, scen, seed] = (int(change[0]) if len(change) else None,
                                         {k:entry[k] for k in ("z","c0.1","c0.25")})
    current = ("z","ratio",4,2)
    rules = {"z": (current,), "OR c0.1": (current,("c0.1","diff",.1,2)),
             "OR c0.25": (current,("c0.25","diff",.1,2))}
    near = DET2_SCEN[1:]
    L = ["# 検出器の保護区間（設計用 seed 10–49）","",
         f"保存ログ1,200件を再評価。発散による除外 {1200-len(logs)} 件。stat、RS-near、RS-far を使用。",
         "見逃し窓は切替ブロック以上、切替+64未満。誤反応は stat の開始2秒以降と切替条件の開始2秒以降から切替直前まで。遅れは窓内最初の検出までのブロック数。",
         "更新停止は発火ブロックを含め、保護が解除されるブロックの直前まで。保護0でも発火ブロックの更新を省く。",
         "試行制約つきの列では、採用した検出の次の16ブロック（+1〜+16）に出た検出を数えない。抑制された検出で試行を延長せず、検出器自身の状態と保護は動かす。閉ループの移行は評価しない。",
         "見逃しの95%区間は各クラス・条件内で seed を2,000回復元抽出した百分位区間。", ""]
    def table(headers, values):
        for row in (headers, ["---"]*len(headers), *values):
            L.append("| " + " | ".join(map(str,row)) + " |")
        L.append("")

    def miss_cell(values):
        if not values:
            return "—"
        a = np.array([v is None for v in values])
        draws = np.random.default_rng(0).choice(a,size=(2000,len(a))).mean(axis=1)
        lo,hi = np.quantile(draws,[.025,.975])
        return f"{a.sum()}/{len(a)} ({a.mean():.1%} [{lo:.1%}, {hi:.1%}])"

    results = {}
    precedes = {}
    for name,rule in rules.items():
        for guard in DET2_GUARDS if name=="z" else (16,):
            for freeze in (False,True):
                detail={}
                for key,(cb,series) in logs.items():
                    hits=triggers_or([(series[s],mode,k,d) for s,mode,k,d in rule], guard=guard, freeze=freeze)
                    starts=trial_starts(hits)
                    end=cb if cb is not None else len(series["z"])
                    delays=tuple(next((i-cb for i in seq if cb<=i<cb+WIN),None) if cb is not None else None
                                 for seq in (hits,starts))
                    detail[key]=(delays,sum(WARM<=i<end for i in hits),end-WARM)
                    if name=="z" and guard==64 and not freeze and cb is not None and delays[0] is None:
                        precedes[key]=any(cb-64<=i<cb for i in hits)
                results[name,guard,freeze]=detail
    means={}
    miss_rates={}
    for (name,guard,freeze),detail in results.items():
        L += [f"## {name}・保護{guard}・{'更新停止' if freeze else '毎ブロック更新'}",""]
        values=[]; rates=[]
        for cls in CLASSES:
            row=[cls]
            for scen in near:
                delays=[d for (c,s,_),(d,_,_) in detail.items() if c==cls and s==scen]
                for v in zip(*delays):
                    row.extend((miss_cell(v),F.fmt(float(np.median([i for i in v if i is not None]))
                                                   if any(i is not None for i in v) else None,1)))
                miss_rates[name,guard,freeze,cls,scen]=sum(d[0] is None for d in delays)/len(delays)
            n=sum(v[1] for k,v in detail.items() if k[0]==cls)
            blocks=sum(v[2] for k,v in detail.items() if k[0]==cls)
            rate=n*60*BPS/blocks
            rates.append(rate)
            row.extend((f"{rate:.2f}",f"{rate*16/7500:.2%}"))
            values.append(row)
        means[name,guard,freeze]=(float(np.mean([d[0] is None for key,(d,_,_) in detail.items() if key[1] in near])),
                                  float(np.mean(rates)),
                                  float(np.mean([d[1] is None for key,(d,_,_) in detail.items() if key[1] in near])))
        table(("クラス","near 見逃し [95%]","near 遅れ","near 試行制約つき [95%]","near 制約つき遅れ",
               "far 見逃し [95%]","far 遅れ","far 試行制約つき [95%]","far 制約つき遅れ",
               "誤反応/分","試行稼働率の概算"),values)
    L += ["## 所見","","### 現行規則で見逃す直前の検出","",
          "切替の直前64ブロック（−64〜−1）に一回以上検出した件数／見逃し件数。直前検出の存在を数える。", ""]
    values=[]
    for c in CLASSES:
        row=[c]
        for scenes in (near,*((s,) for s in near)):
            v=[x for (cl,s,_),x in precedes.items() if cl==c and s in scenes]
            row.append(f"{sum(v)}/{len(v)} ({sum(v)/len(v):.1%})" if v else "— (見逃し0)")
        values.append(row)
    table(("クラス","近遠合計","near","far"),values)
    L += ["### 保護区間を短くしたときの見逃し","",
          "毎ブロック更新の結果。数値は保護64・32・16・8・4・0の順。増減は保護64→0の端点で分類。", ""]
    table(("クラス","near の見逃し率","near 増減","far の見逃し率","far 増減"),
          [(c,*(v for s in near for rates in [[miss_rates["z",g,False,c,s] for g in DET2_GUARDS]]
                for v in (" → ".join(f"{r:.1%}" for r in rates),
                          "減少" if rates[-1]<rates[0] else "増加" if rates[-1]>rates[0] else "同率"))) for c in CLASSES])
    L += ["### 誤反応と基準更新の差","",
          "稼働率は誤反応/分 × 16 ÷ 7,500。実際の試行長や重なりを反映しない概算。見逃し率は完走した近遠の全実行、誤反応率は等重みクラス平均。", ""]
    table(("規則","保護","毎回 見逃し","停止 見逃し","停止−毎回（ポイント）",
           "毎回 誤反応/分","停止 誤反応/分","停止−毎回 /分","毎回 稼働率","停止 稼働率",
           "制約つき 毎回の見逃し","制約つき 停止の見逃し"),
          [(name,g,f"{a:.2%}",f"{b:.2%}",f"{100*(b-a):+.2f}",f"{x:.2f}",f"{y:.2f}",f"{y-x:+.2f}",
            f"{x*16/7500:.2%}",f"{y*16/7500:.2%}",f"{u:.2%}",f"{v:.2%}")
           for (name,g,freeze),(a,x,u) in means.items() if not freeze for b,y,v in [means[name,g,True]]])
    (HERE/"RESULTS_det2.md").write_text("\n".join(L).rstrip()+"\n",encoding="utf-8")
    print(f"wrote RESULTS_det2.md: {len(logs)} usable logs, {len(results)} variants")


SCREEN_SCEN=("stat","RS-near","RS-far")
SCREEN_SEEDS=tuple(range(10,30))
def load_screen():
    logs={}
    for c,s,sd in itertools.product(CLASSES,SCREEN_SCEN,SCREEN_SEEDS):
        with np.load(OUT/f"{c}_{s}_s{sd}.npz") as f:
            assert int(f["version"])==LOG_VERSION and not bool(f["div"])
            logs[c,s,sd]={k:f[k] for k in ("z","c0.1","c0.25")}
    return logs

@F.fastbase.njit(cache=True)
def screen_clipped(z):
    base=np.nan
    last=-64
    count=0
    out=[]
    for i,v in enumerate(z):
        if not np.isfinite(v):
            count=0
            continue
        armed=i-last>=64
        count=count+1 if armed and np.isfinite(base) and v>4*base else 0
        if count==2:
            out.append(i)
            last=i
            count=0
        elif armed:
            base=v if not np.isfinite(base) else .95*base+.05*min(v,4*base)
    return out

SCREEN_VARIANTS={
    "F64":(64,None),
    "F32":(32,None),
    "Clip":(64,None),
    "C10":(64,"c0.1"),
    "C25":(64,"c0.25"),
}
def screen_candidate(logs,name):
    guard,coh=SCREEN_VARIANTS[name]
    misses={};false={}
    for (c,s,sd),data in logs.items():
        rule=[(data["z"],"ratio",4,2)]
        if coh:
            rule.append((data[coh],"diff",.1,2))
        hits=screen_clipped(data["z"]) if name=="Clip" else triggers_or(rule,guard=guard,freeze=True)
        starts=trial_starts(hits)
        end=CB if s!="stat" else len(data["z"])
        false[c,s,sd]=sum(WARM<=i<end for i in starts)
        if s!="stat":
            misses[c,s,sd]=next((i-CB for i in starts if CB<=i<CB+WIN),None)
    output={}
    for c in CLASSES:
        cells={}
        for s in SCREEN_SCEN[1:]:
            v=[i for (cc,ss,_),i in misses.items() if (cc,ss)==(c,s)]
            d=[i for i in v if i is not None]
            cells[s]=[len(v)-len(d),float(np.median(d)) if d else None]
        blocks=sum((CB if s!="stat" else len(logs[c,s,sd]["z"]))-WARM
                   for s,sd in itertools.product(SCREEN_SCEN,SCREEN_SEEDS))
        cells["fa"]=sum(v for (cc,_,_),v in false.items() if cc==c)*BPS*60/blocks
        output[c]=cells
    return output,misses

def screen_diagnostics(logs):
    _, ms=screen_candidate(logs,"F64")
    output={}
    for s in SCREEN_SCEN[1:]:
        traces=[logs[c,sc,sd] for (c,sc,sd),v in ms.items() if (c,sc)==("machinegun",s) and v is None]
        series={k:np.median([t[k][CB-40:CB+41] for t in traces],axis=0).tolist()
                for k in ("z","c0.1","c0.25")}
        output[s]={"n":len(traces),"series":series}
        print(s,len(traces),{k:[round(float(np.median(v[:40])),3),round(float(np.median(v[40:80])),3)] for k,v in series.items()})
    return output

def screen(name):
    start = time.monotonic()
    logs = load_screen()
    data = screen_diagnostics(logs) if name == "diagnostics" else screen_candidate(logs,name)[0]
    folder = HERE / "out" / "det4"
    folder.mkdir(parents=True, exist_ok=True)
    (folder/f"{name}.json").write_text(json.dumps(dict(seconds=time.monotonic()-start,data=data),indent=2)+"\n")


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=("log", "eval", "eval2", "screen")); a.add_argument("--procs", type=int, default=os.cpu_count() or 1)
    a.add_argument("--classes", default=",".join(CLASSES))
    a.add_argument("--scen", type=lambda s: s.split(","), default=SCEN)
    a.add_argument("--candidate", choices=("diagnostics", *SCREEN_VARIANTS), default="F64")
    o = a.parse_args()
    if any(s not in SCEN for s in o.scen):
        a.error("--scen expects a comma-separated list of known scenarios")
    if o.cmd == "log":
        jobs = [(c, s, sd) for c in o.classes.split(",") for s in o.scen for sd in SEEDS]
        with Pool(o.procs) as pool:
            list(pool.imap_unordered(run_log, jobs))
    elif o.cmd == "eval":
        evaluate()
    elif o.cmd == "eval2":
        evaluate_guard()
    else:
        screen(o.candidate)
