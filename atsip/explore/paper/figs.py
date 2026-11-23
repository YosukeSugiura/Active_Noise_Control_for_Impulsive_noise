"""Generate paper figures and tables from archived JSON, without running ANC.

Extract results/final2_run, coef2_run and cc_run (.tar.gz) into atsip/explore/out before running this script.
Use --refresh-reports to recompute the paper intervals and the change audit
before rendering. Without that option, verify against the committed reports.
"""
import argparse
import json
import subprocess
import sys
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT))
from paired_ci import paired
P, A, N = "P5-early-cap", "Freq_FxNLMS_CohVSS", "nors-cap"
C = "C_FXLMS"                                   # convex combination (Ferrer et al. 2013), results in out/cc_run and out/cc_coef
EXTERNAL_NAMES = {"Th_FxNLMS": "Th-FxNLMS", "EVSS_FxeLMS": "EVSS-FxeLMS", C: "C-FXLMS"}
NAMES = {P: "Proposed", A: "CohVSS-FxNLMS", N: "No restart",
         "P5-early-noG": "Proposed without 1/g", "P4-cap": "Immediate restart",
         "TP-always-cap": "Always-on background trial", "ND-always-cap": "Always reconstructed error",
         "SF-fix": "Fixed-step SF"}
CLASSES = ("G", "II300", "a145", "a120", "machinegun", "factory1",
           "destroyerengine", "volvo", "valve", "jackhammer")
SWITCHES = ("RS-near", "RS-far")
QUIET = ("stat", "EN0", "ENm10")
CLASS_NAMES = {"G": "Gaussian", "II300": "Isolated impulses", "a145": "α-stable (α = 1.45)", "a120": "α-stable (α = 1.20)",
               "machinegun": "Machine gun", "factory1": "Factory", "destroyerengine": "Destroyer engine",
               "volvo": "Car interior", "valve": "Valve", "jackhammer": "Jackhammer"}
SWITCH_TITLE = {"RS-near": "Switch to a close source", "RS-far": "Switch to a far source"}
SCEN = {"RS-near": "Close", "RS-far": "Far", "stat": "No change",
        "EN0": "Disturbance (0 dB)", "ENm10": "Disturbance (-10 dB)"}
BLOCK_SECONDS = 128 / 16000
KS = (16, 64)


def load(folder, scenarios):
    expected = {f"{c}_{s}_s{i}.json" for c in CLASSES for s in scenarios for i in range(100,120)}
    actual = {p.name for p in (ROOT / "out" / folder).glob("*.json")}
    if actual != expected:
        raise ValueError(f"{folder}: {len(expected - actual)} missing and {len(actual - expected)} extra cases")
    groups = defaultdict(list)
    for name in sorted(expected):
        rows = json.loads((ROOT / "out" / folder / name).read_text())["rows"]
        seen = set()
        for r in rows:
            assert name == f"{r['cls']}_{r['scen']}_s{r['seed']}.json"
            key = r["method"], r.get("variant", "main")
            assert key not in seen, (name, key)
            seen.add(key)
            if key[1] == "main":
                groups[r["scen"], r["method"]].append(r)
    assert all(len(rows) == 200 for rows in groups.values())
    return groups


def values(rows, metric):
    return [v for r in rows if not r["div"] and (v := metric(r)) is not None]


def j(k):
    return lambda r: r["J"][str(k)]


def l1(rows, key):
    by_seed = defaultdict(list)
    for r in rows:
        if not r["div"] and r.get(key) is not None:
            by_seed[r["seed"]].append(r[key])
    return np.mean([np.mean(v) for v in by_seed.values()]) if by_seed else None


def recovery(rows):
    times = Counter((r["rec_limit"] if r["rec_cens"] else r["rec_blocks"], r["rec_cens"])
                    for r in rows if not r["div"] and "rec_cens" in r)
    risk, survival = sum(times.values()), 1.
    for t in sorted({t for t,_ in times}):
        events, censored = times[t,0], times[t,1]
        survival *= (risk-events)/risk
        if survival <= .5 + 1e-12:
            return t
        risk -= events+censored
    return None


def md_rows(text):
    return [tuple(x.strip() for x in row.strip().strip("|").split("|"))
            for row in text.splitlines() if row.startswith("|")]


def section(text, title):
    return text.split(f"\n{title}\n", 1)[1].split("\n##", 1)[0]


def shown(x, digits=3):
    return "--" if x is None else f"{x:.{digits}f}"


def ci_text(stat, scale=1):
    a, lo, hi, n = stat
    return f"{a*scale:+.3f} [{lo*scale:+.3f}, {hi*scale:+.3f}] ({n})"


def verify(actual, expected, label):
    if actual != expected:
        raise ValueError(f"{label}: computed {actual!r}, reported {expected!r}")


def ci_verify(stat, expected, label, scale=1):
    actual = [float(x) for x in re.findall(r"[-+]?\d+\.\d+|\d+", ci_text(stat, scale))]
    verify(actual, [float(x) for x in re.findall(r"[-+]?\d+\.\d+|\d+", expected)], label)


def save(fig, name):
    for ext in ("pdf", "png"):
        options = {"metadata": {"CreationDate": None, "ModDate": None}} if ext == "pdf" else {"dpi": 450}
        fig.savefig(PAPER / "fig" / f"{name}.{ext}", bbox_inches="tight", **options)
    plt.close(fig)


def figure3(coef, report):
    methods = (P,A,N)
    styles = (("0.04","-","o"), ("0.32","--","s"), ("0.54","-.","^"))
    fig, axes = plt.subplots(1,2,figsize=(7.2,2.95),sharey=True,layout="constrained")
    for ax,s in zip(axes,SWITCHES):
        for m,(color,ls,mark) in zip(methods,styles):
            rows = coef[s,m]
            blocks = sorted(map(int, rows[0]["J"]))
            series = np.array([np.quantile(values(rows,j(k)),[.25,.5,.75]) for k in blocks])
            if m in (P,A,N):
                for k in KS:
                    key = (s,"全クラス",shown(k*BLOCK_SECONDS,2), A if m==P else m)
                    verify(shown(series[blocks.index(k),1],2), report[key][4 if m==P else 5], key)
            time = np.array(blocks)*BLOCK_SECONDS
            if m != N:
                ax.fill_between(time,series[:,0],series[:,2],color=color,alpha=.12,linewidth=0)
                for q in (0,2):
                    ax.plot(time,series[:,q],color=color,linestyle=ls,linewidth=.5)
            ax.plot(time,series[:,1],color=color,linestyle=ls,linewidth=1.65,marker=mark,
                    markersize=3.1,markevery=3,label=NAMES.get(m,m))
        for k in KS:
            ax.axvline(k*BLOCK_SECONDS,color="0.68",ls=":",lw=.95,zorder=0)
        ax.set_xscale("symlog",linthresh=.55)
        ax.set_xlim(0,15)
        ax.xaxis.set_major_locator(FixedLocator([0,.128,.512,2,8,15]))
        ax.xaxis.set_major_formatter(FixedFormatter(["0","0.13","0.51","2","8","15"]))
        subtitle = f"({'ab'[SWITCHES.index(s)]}) {SWITCH_TITLE[s]}"
        ax.set_xlabel(f"Time after switch [s]\n{subtitle}", linespacing=1.6)
        ax.grid(axis="y",color=".9",lw=.6)
    axes[0].set_ylabel("Noise reduction with\nfrozen coefficients [dB]")
    axes[1].legend(loc="upper right",fontsize=7,framealpha=.92)
    save(fig,"fig3")


def figure4(coef, report):
    columns = [(s,k) for s in SWITCHES for k in KS]
    vals = np.empty((len(CLASSES),len(columns)))
    cross = []
    for i,c in enumerate(CLASSES):
        for col,(s,k) in enumerate(columns):
            rows = [[r for r in coef[s,m] if r["cls"]==c] for m in (P,A,N)]
            for m,r in zip((A,N),rows[1:]):
                key=(s,c,shown(k*BLOCK_SECONDS,2),m)
                entry=report[key]
                stats=paired(rows[0],r,j(k))
                verify(ci_display(stats,2),entry[6],key)
                verify((shown(np.median(values(rows[0],j(k))),2),shown(np.median(values(r,j(k))),2)),entry[4:6],key)
                if m==A:
                    vals[i,col],lo,hi,_=stats
                    if lo<=0<=hi:
                        cross.append((i,col))
    fig,ax=plt.subplots(figsize=(5.7,4.15),layout="constrained")
    limit=np.ceil(np.abs(vals).max())
    im=ax.imshow(vals,cmap="gray_r",vmin=-limit,vmax=limit,aspect="auto")
    for (i,k),v in np.ndenumerate(vals):
        color="white" if im.norm(v)>.72 else "black"
        ax.text(k,i,f"{v:+.2f}"+(" †" if (i,k) in cross else ""),ha="center",va="center",color=color,fontsize=9)
    ax.set_yticks(range(len(CLASSES)),[CLASS_NAMES[c] for c in CLASSES])
    ax.set_xticks(range(4),[f"{SCEN[s]}\n{k*BLOCK_SECONDS:.2f} s" for s,k in columns])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.axvline(1.5,color="white",lw=2)
    fig.colorbar(im,ax=ax,label=f"Proposed − {NAMES[A]} [dB]")
    save(fig,"fig4")


def output_table(number, title, panels, notes):
    def esc(s):
        chars = {"&":r"\&","%":r"\%","_":r"\_","#":r"\#","−":"-", "μ":r"$\mu$", "<":r"$<$", ">":r"$>$"}
        return "".join(chars.get(x,x) for x in str(s))
    md = [f"# Table {number}. {title}",""]
    tex = [r"\begin{table*}[t]",r"\centering",r"\small",r"\setlength{\tabcolsep}{4pt}",r"\caption{"+esc(title)+"}",
           rf"\label{{tab:atsip-{number}}}"]
    for heading,header,rows in panels:
        md.extend([f"## {heading}",""] if heading else [])
        tex.extend(([r"\par\medskip",r"\textbf{"+esc(heading)+r"}\par\smallskip"] if heading else [r"\par\smallskip"])+[
                    r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
                    r"\begin{tabular}{l"+("l"*(len(header)-1))+"}",r"\toprule",
                    " & ".join(map(esc,header))+r" \\",r"\midrule"])
        for row in (header,["---"]*len(header),*rows):
            md.append("| "+" | ".join(map(str,row))+" |")
        md.append("")
        tex.extend(" & ".join(map(esc,row))+r" \\" for row in rows)
        tex.extend([r"\bottomrule",r"\end{tabular}}"])
    md.extend(notes)
    tex.extend([r"\par\medskip",r"\begin{minipage}{0.98\textwidth}",r"\footnotesize",
                *[esc(note)+r"\par" for note in notes],r"\end{minipage}",r"\end{table*}"])
    for ext,lines in (("md",md),("tex",tex)):
        (PAPER/"tab"/f"table{number}.{ext}").write_text("\n".join(lines).rstrip()+"\n")


def table1():
    tune=json.loads((ROOT/"TUNE.json").read_text())
    settings=[
        ("Sampling rate; input high-pass cutoff","16 kHz; 100 Hz"),
        ("Control filter M; FFT N; hop B","512; 1024; 128 samples (8 ms per block)"),
        ("Secondary-path model; error window","512 taps; 512 samples"),
        ("SF regularization; late-stage design target","0.01; 0.03"),
        ("Early / late step before gain normalization","1.0 / 0.0566"),
        ("Gain floor, early and background / late","0.55 / 0.25"),
        ("Cold-start phase; restart phase budget","32 blocks each; restart budget includes the trial"),
        ("Detection error / reference windows","512 / 1024 samples"),
        ("Detector EMA weight; ratio; consecutive blocks","0.05; 4; 2"),
        ("Guard after detection; EMA update","64 blocks; frozen from detection until guard release"),
        ("Trial abort ratio; earliest check",">4 or nonfinite; block 2"),
        ("Early transfer / final transfer","ratio <= 0.25 from block 4 / <= 0.5 at block 16"),
    ]
    baselines=[(NAMES.get(m,EXTERNAL_NAMES.get(m,m)),f"{r['mu']:g}",
                f"Error clipping to [{r['params']['c1']:g}, {r['params']['c2']:g}]" if m=="Th_FxNLMS" else "--")
               for m,r in tune["chosen"].items()]
    cc=json.loads((ROOT/"TUNE_cc.json").read_text())["chosen"][C]
    baselines.append((EXTERNAL_NAMES[C],f"{cc['mu']:g} (fast), {cc['mu']*cc['params']['ratio']:g} (slow)",
                      "Normalized component filters" if cc["params"]["norm"] else "LMS component filters"))
    output_table(1,"Evaluation settings and selected baseline steps",[
        ("Proposed method",("Setting","Value"),settings),
        ("Comparison algorithms",("Method","Selected step μ","Selected setting"),baselines)],
        ["The comparison algorithms use the common settings selected on design data (TUNE.json, TUNE_cc.json)."])


def table2(final,coef,cr,final_text,ccf,ccc):
    result=[]
    extra=[]
    for s in SWITCHES:
        reports={r[2]:r for r in md_rows(section(final_text,"### 騒音源切替の回復")) if len(r)==7 and r[:2]==(s,"全クラス")}
        main_rows = {r[0]:r for r in md_rows(section(final_text,f"## {s}")) if len(r)==15}
        methods=(P,A,N)
        recoveries=[recovery(final[s,m]) for m in methods]
        cells=[shown(x*BLOCK_SECONDS) for x in recoveries]
        intervals=[]
        for idx,m in enumerate(methods[1:],1):
            ref=reports[m]
            verify(cells[0],ref[3].split("（")[0],(s,P))
            verify(cells[idx],ref[4].split("（")[0],(s,m))
            v=paired(final[s,P],final[s,m],lambda r:r.get("rec_blocks"))
            ci_verify(v,ref[5],(s,m),BLOCK_SECONDS)
            expected = ci_display(v,2,count=True).replace(" (","（").replace(")","）")
            verify(expected,main_rows[m][14],(s,m,"recovery in blocks"))
            intervals.append(ci_text(v,BLOCK_SECONDS))
        result.append((SCEN[s],"Recovery time [s]",*cells,*intervals))
        v=paired(final[s,P],ccf[s,C],lambda r:r.get("rec_blocks"))
        extra.append((shown(recovery(ccf[s,C])*BLOCK_SECONDS),ci_text(v,BLOCK_SECONDS)))
        for k in KS:
            t=shown(k*BLOCK_SECONDS,2)
            cells=[shown(np.median(values(coef[s,m],j(k))),2) for m in methods]
            intervals=[]
            for idx,m in enumerate(methods[1:],1):
                r=cr[s,"全クラス",t,m]
                verify((cells[0],cells[idx]),r[4:6],(s,k,m))
                v=paired(coef[s,P],coef[s,m],j(k))
                verify(ci_display(v,2),r[6],(s,k,m))
                intervals.append(ci_display(v,2,count=True))
            result.append((SCEN[s],f"NR at {t} s [dB]",*cells,*intervals))
            extra.append((shown(np.median(values(ccc[s,C],j(k))),2),ci_display(paired(coef[s,P],ccc[s,C],j(k)),2,count=True)))
    # Keep the customary paper table compact by separating levels and paired differences.
    output_table(2,"Reconvergence after a source switch",[
        ("Median levels",("Switch","Metric",NAMES[P],NAMES[A],EXTERNAL_NAMES[C],NAMES[N]),
         [(*r[:4],x[0],r[4]) for r,x in zip(result,extra)]),
        ("Paired differences",("Switch","Metric",f"Proposed - {NAMES[A]}",f"Proposed - {EXTERNAL_NAMES[C]}","Proposed - No restart"),
         [(*r[:2],r[5],x[1],r[6]) for r,x in zip(result,extra)])],
        ["Recovery uses each method's post-change reference level +3 dB and a Kaplan-Meier median.",
         "NR at 0.13 s and 0.51 s is the noise reduction with the coefficients frozen at that time after the switch (residual-to-primary power ratio, first 256 ms excluded).",
         "Differences are paired means with seed-bootstrap 95% intervals and pair counts in parentheses. Recovery differences use pairs with recovery observed in both runs."])


def table3(final, final_text, fast, ccf):
    statements=[]
    internal=defaultdict(list)
    names=tuple(NAMES)
    for s in (*QUIET,*SWITCHES):
        published={r[0]:r for r in md_rows(section(final_text,f"## {s}")) if len(r)==15}
        if s in QUIET:
            for m in (P,A,N):
                rows=final[s,m]
                vals=[l1(rows,k) for k in ("L1_whole","L1_last10","L1_ev")]
                verify(tuple(shown(v,2) if v is not None else "—" for v in vals), published[m][3:6],(s,m))
                same=[r["same_as_nors"] for r in rows if "same_as_nors" in r]
                match=f"{sum(same)}/{len(same)}" if same else "--"
                verify(match if same else "—", published[m][10], (s,m,"identity"))
                transfers="--" if m==A else sum(r["n_confirm"] for r in rows)
                if m!=A:
                    verify(str(transfers),published[m][8],(s,m,"transfers"))
                statements.append((SCEN[s],NAMES[m],*(shown(v,2) for v in vals),transfers,match))
                if m==A:
                    statements.append((SCEN[s],EXTERNAL_NAMES[C],*(shown(l1(ccf[s,C],k),2) for k in ("L1_whole","L1_last10","L1_ev")),"--","--"))
        for m in names:
            if m==A:
                continue
            rows=final[s,m]
            div=sum(r["div"] for r in rows)
            false=sum(r["false_confirm"] for r in rows)
            verify((str(div),str(false)),(published[m][2],published[m][8]),(s,m))
            internal[m].append(f'{div}/{sum(r["n_confirm"] for r in rows)}/{false}')
    differences = []
    for s,m,key,title in l1_comparisons():
        saved = final_l1_rows(final_text)[s,m,key]
        v = paired(final[s,P],final[s,m],lambda row: row.get(key))
        ci_verify(v,saved[5],(s,m,key))
        differences.append((SCEN[s],NAMES[m],title,ci_text(v)))
        if m==A:
            differences.append((SCEN[s],EXTERNAL_NAMES[C],title,ci_text(paired(final[s,P],ccf[s,C],lambda row: row.get(key)))))
    FXNLMS_MULT=196608                                   # RESULTS_fast.md: time-domain FxNLMS, M = Ls = 512, per 128 samples
    assert "196,608" in fast
    costs=[]; tot=0; blocks=0
    for phase,label in (("late","Late phase"),("early","Early phase"),("late+trial","Late phase with trial"),("early+trial","Early phase with trial")):
        r=next(r for r in md_rows(fast) if r[0]==phase)
        mult=int(r[5].replace(",","")); n=int(r[1]); tot+=mult*n; blocks+=n
        costs.append((label,f"{mult:,}",f"{mult/FXNLMS_MULT:.2f}",f"{n:,}"))
    costs.append(("Average over the measured blocks",f"{round(tot/blocks):,}",f"{tot/blocks/FXNLMS_MULT:.2f}",f"{blocks:,}"))
    costs.append(("FxNLMS (time domain)",f"{FXNLMS_MULT:,}","1.00","--"))
    output_table(3,"Noise reduction with a fixed primary path",[
        ("Levels [dB]",
         ("Condition","Method","30 s","Last 10 s","2 s after disturbance","Transfers","Identical runs"),statements),
        ("Paired differences (proposed minus the other algorithm) [dB]",
         ("Condition","Other algorithm","Window","Difference [95% CI]"),
         [(c,m,{"30 s":"30 s","Last 10 s":"Last 10 s","Post-event 2 s":"2 s after disturbance"}.get(w,w),d) for c,m,w,d in differences])],
        ["Transfers count events in 200 runs. Identical runs are those whose error and control signals equal those of the algorithm without the trials exactly."])
    output_table(4,"Variants of the proposed algorithm (200 runs per condition)",[
        ("Diverged runs / transfers / runs differing from the algorithm without the trials",
         ("Method",*(SCEN[s] for s in (*QUIET,*SWITCHES))),
         [(NAMES[m],*counts) for m,counts in internal.items()])],[])
    output_table(5,"Real multiplications per block of 128 samples",[
        ("",("State","Multiplications","Relative to FxNLMS","Blocks"),costs)],
        ["Multiplications are the maximum over the blocks of each state and include the sample-by-sample filtering for the control signal. Blocks are those of the eight runs used for the count."])


COEF_TITLE = "## 主表：変化から 0.13 秒後と 0.51 秒後の消音量（評価点は seed 100–119 の実行前に固定。2026-11-21）"
REC_TITLE = "### 騒音源切替の回復"
L1_TITLE = "### 変化なし・外乱の消音量"
DIRECTIONS = ("提案", "比較", "0を含む", "欠測")


def direction(v):
    if v is None:
        return "欠測"
    return "提案" if v[2] < 0 else "比較" if v[1] > 0 else "0を含む"


def ci_display(v, digits, scale=1, count=False):
    if v is None:
        return "—"
    mean, lo, hi, n = v
    result = f"{mean*scale:+.{digits}f} [{lo*scale:+.{digits}f}, {hi*scale:+.{digits}f}]"
    return f"{result} ({n})" if count else result


def coef_main_rows(text):
    result = {}
    for s, cls, time, *rest in md_rows(section(text,COEF_TITLE)):
        if s in SWITCHES:
            prop, apsipa, no_restart, da, dn = rest
            time = time.split()[0]
            for m, median, diff in ((A,apsipa,da),(N,no_restart,dn)):
                key = (s,cls,time,m)
                result[key] = (*key,prop,median,diff)
    return result


def final_l1_rows(text):
    return {r[:3]:r for r in md_rows(section(text,L1_TITLE)) if r[0] in QUIET}


def l1_comparisons():
    return [
        ("stat",A,"L1_whole","30 s"),
        ("stat",A,"L1_last10","Last 10 s"),
        *((s,m,"L1_ev","Post-event 2 s") for s in QUIET[1:] for m in (A,N)),
    ]


def select(rows, cls):
    return [r for r in rows if cls == "全クラス" or r["cls"] == cls]


def refresh_reports(final, coef):
    def replace_rows(text, title, update):
        body = section(text,title)
        lines = body.splitlines(keepends=True)
        for i,line in enumerate(lines):
            if line.startswith("|"):
                row, = md_rows(line)
                new = update(list(row))
                if new is not None:
                    lines[i] = "| " + " | ".join(new) + " |\n"
        return text.replace(body,"".join(lines),1)

    def contrast(groups, s, m, metric, cls="全クラス"):
        return paired(select(groups[s,P],cls),select(groups[s,m],cls),metric)

    def update_coef(row):
        s, cls, time = row[:3]
        if s in SWITCHES:
            k = round(float(time.split()[0])/BLOCK_SECONDS)
            row[6:8] = [ci_display(contrast(coef,s,m,j(k),cls),2) for m in (A,N)]
            return row

    def findings(row, metric, scale):
        if row[0] in (*QUIET,*SWITCHES):
            s, c, m = row[:3] if metric == "rec_blocks" else (row[0],"全クラス",row[1])
            key = metric or row[2]
            v = contrast(final,s,m,lambda r: r.get(key),c)
            row[5] = ci_display(v,3,scale,count=True).replace(" (","（").replace(")","）")
            row[6] = direction(v)
            return row

    def final_main(row, s):
        m = row[0]
        if m == P or (s,m) not in final:
            return
        for col,key in ((12,"L1_whole"),(13,"L1_ev"),(14,"rec_blocks")):
            if (s,m,key) in paper_l1 or key == "rec_blocks":
                v = contrast(final,s,m,lambda r: r.get(key))
                row[col] = ci_display(v,2,count=True).replace(" (","（").replace(")","）")
        return row

    paper_l1 = {(s,m,key) for s,m,key,_ in l1_comparisons()}
    cp, fp = (ROOT/name for name in ("RESULTS_coef2.md","RESULTS_final2.md"))
    ct = replace_rows(cp.read_text(),COEF_TITLE,update_coef)
    ct = ct.replace("### 0.13秒・0.51秒の比較\n\n",
                    "### 0.13秒・0.51秒の比較\n\n以下は区間統一前の記録。論文では主表の再計算値を使用する。\n\n",1) if "以下は区間統一前の記録" not in ct else ct
    ft = fp.read_text()
    for s in (*SWITCHES,*QUIET):
        ft = replace_rows(ft,f"## {s}",lambda row: final_main(row,s))
    ft = replace_rows(ft,REC_TITLE,lambda row: findings(row,"rec_blocks",BLOCK_SECONDS))
    ft = replace_rows(ft,L1_TITLE,lambda row: findings(row,None,1))
    cp.write_text(ct)
    fp.write_text(ft)
    audit(final,coef)


def audit(final,coef):
    def previous(name):
        return subprocess.check_output(
            ["git","show",f"e1d9ddd:atsip/explore/{name}"],cwd=ROOT,text=True,
        )

    lines = [
        "# 論文用 95% 区間の統一","",
        "保存済み JSON の完走対を使用。回復時間は両手法で回復が観測された対を使用。",
        "seed ごとの平均を 2,000 回再標本化し、2.5・97.5 パーセンタイルを計算した。",
        "比較ごとに乱数生成器を初期値 20261121 で作り直した。",
        "旧値は e1d9ddd の報告書の所見。判定には丸め前の端点を使う。差は提案法 − 比較手法。","",
        "| 項目 | 条件 | 入力 | 指標 | 比較 | 旧値［95%］ | 新値［95%］ | 旧判定 | 新判定 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    counts = defaultdict(lambda: (Counter(),Counter()))
    changes = []
    def add(group, s, cls, metric, m, old, label, v, scale=1):
        new = direction(v)
        counts[group][0][label] += 1
        counts[group][1][new] += 1
        if label != new:
            changes.append(f"- {group}) {s} / {cls} / {metric} / {NAMES[m]} — {label} → {new}。")
        lines.append(f"| {group} | {s} | {cls} | {metric} | {NAMES[m]} | {old} | {ci_display(v,3,scale,count=True)} | {label} | {new} |")

    for row in md_rows(section(previous("RESULTS_coef2.md"),"### 0.13秒・0.51秒の比較")):
        if row[0] not in SWITCHES or row[3] != A:
            continue
        s,cls,time,m = row[:4]
        k = round(float(time)/BLOCK_SECONDS)
        v = paired(select(coef[s,P],cls),select(coef[s,m],cls),j(k))
        add("b" if cls == "全クラス" else "a",s,cls,f"J {time} s (dB)",m,row[6],row[7],v)

    old_final = previous("RESULTS_final2.md")
    for row in md_rows(section(old_final,REC_TITLE)):
        if row[0] not in SWITCHES or row[1] == "全クラス":
            continue
        s,cls,m = row[:3]
        v = paired(select(final[s,P],cls),select(final[s,m],cls),lambda r:r.get("rec_blocks"))
        add("c",s,cls,"回復 (s)",m,row[5],row[6],v,BLOCK_SECONDS)

    old_l1 = final_l1_rows(old_final)
    for s,m,key,_ in l1_comparisons():
        v = paired(final[s,P],final[s,m],lambda r:r.get(key))
        row = old_l1[s,m,key]
        add("d" if s == "stat" else "e",s,"全クラス",key+" (dB)",m,row[5],row[6],v)

    summary = ["## 判定の件数","",
               "順序は「提案法が良い／比較手法が良い／0 を含む／欠測」。","",
               "| 項目 | 旧 | 新 |","|---|---|---|"]
    for group in "abcde":
        old,new = ("/".join(str(c[d]) for d in DIRECTIONS) for c in counts[group])
        summary.append(f"| {group} | {old} | {new} |")
    (ROOT/"RESULTS_ci_unify.md").write_text(
        "\n".join(lines[:6]+[""]+summary+["","## 判定が変わった比較",""]+
                  (changes or ["なし。"])+["","## 新旧の値",""]+lines[7:])+"\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-reports",action="store_true")
    args = parser.parse_args()
    for path in ("fig","tab"):
        (PAPER/path).mkdir(exist_ok=True)
    times_families = ("Times New Roman", "Liberation Serif", "Nimbus Roman", "TeX Gyre Termes")
    installed = {font.name for font in font_manager.fontManager.ttflist}
    family = next((name for name in times_families if name in installed), None)
    if family is None:
        raise RuntimeError("Install a Times-compatible font: " + ", ".join(times_families))
    plt.rcParams.update({"font.family":family,"font.size":9,"pdf.fonttype":42,
                         "axes.spines.top":False,"axes.spines.right":False})
    final=load("final2_run",(*SWITCHES,*QUIET))
    coef=load("coef2_run",SWITCHES)
    if args.refresh_reports:
        refresh_reports(final,coef)
    text=(ROOT/"RESULTS_final2.md").read_text()
    ct=(ROOT/"RESULTS_coef2.md").read_text()
    report=coef_main_rows(ct)
    figure3(coef,report)
    figure4(coef,report)
    table1()
    ccf=load("cc_run",(*SWITCHES,*QUIET)); ccc=load("cc_coef",SWITCHES)
    table2(final,coef,report,text,ccf,ccc)
    table3(final,text,(ROOT/"RESULTS_fast.md").read_text(),ccf)
    print("Verified figure and table values against the regenerated reports.")


if __name__=="__main__":
    main()
