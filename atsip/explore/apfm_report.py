"""Compare APFM outputs with the frozen final2 results."""
from collections import defaultdict
import sys

import apfm_run as A
import coef as C
import final2
from paired_ci import paired

F, np = A.F, A.apfm.np
REFERENCES = ("P5-early-cap","nors-cap","Freq_FxNLMS_CohVSS")
LABELS = ("Proposed","Without trials","CohVSS-FxNLMS")
POWER = (.5,1,2)
SWITCH = ("RS-near","RS-far")
NOTES = """- Yin et al. (2023), DOI 10.1016/j.ymssp.2023.110274. 著者公開版 Mat_APFM の 74375829531fd62077217454da11f75a195138aa を参照。
- Parallel.m は説明用コードで、勾配 delta_w の本体を含まない。公開版の VSS 分母・初期ステップは論文式 (12) と異なるため、VSS・正規化・ゼロ選択時の初期化は論文式 (10)–(15) に従う。再構成誤差は論文 Table 1 の内部モデルを用い、実機の選択器に真の一次騒音を渡さない。
- 参照・制御出力・誤差を各1チャネルとし、候補は経路ごとに事前学習した1本・現行・ゼロの3本。学習には切替前のP、真のSと512タップのŜ、seed 901–908の30秒白色ガウス入力を用いた。制御をゼロから開始し、現行候補は各ブロック更新後の係数とした。
- 512係数・128サンプル更新。提案法側の設定および CohVSS と同様に1024点FFT、512サンプルの誤差履歴、逆FFT前半512タップへの制約を用いる。制御と二次伝搬には因果FIRと直前ブロックまでの履歴を使用。候補のゼロ出力は仮想経路全体でゼロとする。
- Eq. (10) の更新に使う E は観測誤差。Px・Pe には現在の128サンプルブロックのエネルギー（和）を使う。ゼロ選択時には前回の未スケール勾配をβ1倍し、正規化とレベル推定をリセットする。VSS単体も選択した同じステップを使う。"""


def empirical(case, built):
    path=A.target("apfm_reference_power",case)
    if path.exists():
        return
    dd,R,stats=built
    tune=A.read(A.HERE/"TUNE.json")["chosen"][REFERENCES[2]]
    x4=F.p.parse(["c5","--preset","final"]).X4
    out={}
    for method in REFERENCES:
        if method==REFERENCES[2]:
            sim=F.run_baseline(method,tune["mu"],tune["params"],dd)
        else:
            _,sim=F.run_rule(method,dd,R,stats,x4)
        metrics=F.measures(sim,dd)
        start=F.CB*F.B
        out[method]=dict(div=metrics["div"],L1_whole=metrics.get("L1_whole"),
              power_db={str(s):None if sim["div"] else F.ratio_db(sim["e"],dd["d"],start,start+int(s*F.FS))
                        for s in POWER})
    A.write(path,out)


def empirical_job(case):
    if A.target("apfm_reference_power",case).exists():
        return
    final2.configure()
    empirical(case,F.build(*case,F.LS_FINAL,final=True))


def gather():
    selected=A.read(A.SELECTED)["index"]
    groups=defaultdict(list)
    for case in A.cases(True):
        rows=A.read(A.target("final2_run",case))["rows"]
        is_switch=case[1] in SWITCH
        power=A.read(A.target("apfm_reference_power",case)) if is_switch else {}
        frozen={r["method"]:r["J"] for r in A.read(A.target("coef2_run",case))["rows"]} if is_switch else {}
        for method,label in zip(REFERENCES,LABELS,strict=True):
            match,=(r for r in rows if r["method"]==method and r["variant"]=="main")
            groups[label,case[1]].append(dict(match,J=frozen.get(method,{}),power_db=power.get(method,{}).get("power_db",{}),
                empirical_replay=power.get(method)))
        for method in A.METHODS:
            row=A.read(A.target(f"apfm_eval/{method}/g{selected}",case))
            if row["settings"]!=A.asdict(A.settings(selected)) or (
                row["cls"],row["scen"],row["seed"],row["method"])!=(*case,method):
                raise ValueError(f"wrong evaluation record: {method}/{case}")
            groups[method,case[1]].append(row)
    return groups


def values(rows,metric,c=None):
    result=[]
    if metric.startswith("J"):
        return [v for r in rows if not r["div"] and (v:=r.get("J",{}).get(str(c))) is not None]
    if metric=="power":
        return [v for r in rows if not r["div"] and (v:=r.get("power_db",{}).get(str(c))) is not None]
    for seed in F.FINAL_SEEDS:
        sample=[r[metric] for r in rows if r["seed"]==seed and not r["div"] and r.get(metric) is not None]
        if sample:
            result.append(float(np.mean(sample)))
    return result


def fmt(v,reduce=np.mean):
    return f"{reduce(v):.3f}" if len(v) else "—"


def report():
    groups=gather()
    selection=A.read(A.SELECTED)
    lines=["# 単一チャネル APFM", "", NOTES, "",
           f"パラメータの出典: {selection['source']['source']}", "", "## 選定", "",
           "| 倍率 | μ̃ | μ_max | 発散 / 試行 | 学習失敗による欠測 | 30秒 ANR [dB] | 選択 |",
           "|---:|---:|---:|---:|---:|---:|---|"]
    for row in selection["candidates"]:
        mu,cap=(row["settings"][k] for k in ("mu_tilde","mu_max"))
        anr=f"{row['ANR']:.4f}" if row["ANR"] is not None else "—"
        check="✓" if row["index"]==selection["index"] else ""
        lines.append(f"| {row['factor']:g} | {mu:.8g} | {cap:.8g} | {row['diverged']}/{row['n']} | {row['bank_failures']} | {anr} | {check} |")
    lines += ["", "## 評価", "",
        "回復はKaplan–Meier中央値。0.5秒と2秒までの回復数は全200試行を分母とした。ANRはseedごとにクラス平均した後に平均。経験パワー比は完走ラン平均、係数固定値は完走ラン中央値。既存3手法の経験パワー比は同じ設定で再実行した。",
        "", "| 条件 | 手法 | 発散 | 回復中央値 [s] | ≤0.5 s | ≤2 s | ANR 30 s | ANR 最後10 s | 事象後2 s ANR | 比0.5 s | 比1 s | 比2 s | J(16) | J(64) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for scen in A.SCEN:
        for method in (*LABELS,*A.METHODS):
            rows=groups[method,scen]
            recovered=[F.B/F.FS*r["rec_blocks"] for r in rows if not r["div"] and r.get("rec_blocks") is not None]
            km=F.recovery_summary(rows)[0]
            out=[f"{sum(r['div'] for r in rows)}/200",
                 f"{int(km)*F.B/F.FS:.3f}" if km.isdigit() else km,
                 *([f"{sum(v<=t for v in recovered)}/200" for t in (.5,2)] if scen in SWITCH else ["—","—"]),
                 *[fmt(values(rows,k)) for k in ("L1_whole","L1_last10","L1_ev")],
                 *[fmt(values(rows,"power",s)) for s in POWER],
                 *[fmt(values(rows,"J",k),np.median) for k in C.MAIN_KS]]
            lines.append("| "+" | ".join([scen,method,*out])+" |")
    lines += ["", "### 既存3手法の再実行照合", "",
              "経験パワー比の再実行と保存結果の発散フラグ・30秒 ANR の照合。", "",
              "| 手法 | 条件 | 発散フラグ不一致 | 30秒 ANR 最大絶対差 [dB] |",
              "|---|---|---:|---:|"]
    for method in LABELS:
        for scen in SWITCH:
            rows=groups[method,scen]
            different=sum(r['div']!=r['empirical_replay']['div'] for r in rows)
            diffs=[abs(r['L1_whole']-r['empirical_replay']['L1_whole']) for r in rows
                   if r.get('L1_whole') is not None and r['empirical_replay']['L1_whole'] is not None]
            largest=f"{max(diffs):.6g}" if diffs else "—"
            lines.append(f"| {method} | {scen} | {different} | {largest} |")
    lines += ["", "## 提案法 − APFM", "",
              "| 条件 | 指標 | 平均差 [95% CI] | 対数 |","|---|---|---:|---:|"]
    for scen in A.SCEN:
        metrics=[("30 s ANR",lambda r:r.get("L1_whole"))]
        if scen in SWITCH:
            metrics += [("回復時間 [s]",lambda r:None if r.get("rec_blocks") is None else F.B/F.FS*r["rec_blocks"])]
            metrics += [(f"J({k})",lambda r,k=k:r.get("J",{}).get(str(k))) for k in C.MAIN_KS]
        if scen.startswith("EN"):
            metrics += [("外乱後2 s ANR",lambda r:r.get("L1_ev"))]
        for label,metric in metrics:
            result=paired(groups["Proposed",scen],groups["APFM",scen],metric)
            result_text,n="—",0
            if result is not None:
                m,lo,hi,n=result
                result_text=f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            lines.append(f"| {scen} | {label} | {result_text} | {n} |")
    lines += ["", "## APFMの選択機構", "",
              "| 条件 | 学習 [%] | 現行 [%] | ゼロ [%] | 変化後2 sに現行以外を選択 | 候補切替回数（定常） |",
              "|---|---:|---:|---:|---:|---:|"]
    for scen in A.SCEN:
        rows=groups["APFM",scen]
        counts=np.sum([r["choice_counts"] for r in rows],axis=0)
        fractions=counts*100/counts.sum()
        noncurrent=sum(r["noncurrent_first_2s"] or 0 for r in rows) if scen in SWITCH else "—"
        switches=sum(r["switches"] for r in rows) if scen=="stat" else "—"
        lines.append(f"| {scen} | "+" | ".join(f"{v:.3f}" for v in fractions)+f" | {noncurrent} | {switches} |")
    (A.HERE/"RESULTS_apfm.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    if sys.argv[1:]==["report"]:
        report()
    else:
        from multiprocessing import Pool
        k,n=F.parse_shard(sys.argv[1])
        jobs=[c for c in A.cases(True) if c[1] in SWITCH]
        with Pool(4) as pool:
            for _ in pool.imap_unordered(empirical_job,F.shard_jobs(jobs,(k,n))):
                pass
