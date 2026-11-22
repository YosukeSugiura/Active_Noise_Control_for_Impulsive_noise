"""Generate paper figures and tables from archived JSON, without running ANC.

Extract the two archives into atsip/explore/out before running this script.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
P, A, N = "P5-early-cap", "Freq_FxNLMS_CohVSS", "nors-cap"
NAMES = {P: "Proposed", A: "APSIPA 2026 [our prior work]", N: "No restart",
         "P5-early-noG": "Proposed without 1/g", "P4-cap": "Immediate restart",
         "TP-always-cap": "Always-on background trial", "ND-always-cap": "Always reconstructed error",
         "SF-fix": "Fixed-step SF"}
CLASSES = ("G", "II300", "a145", "a120", "machinegun", "factory1",
           "destroyerengine", "volvo", "valve", "jackhammer")
SWITCHES = ("RS-near", "RS-far")
QUIET = ("stat", "EN0", "ENm10")
SCEN = {"RS-near": "Near", "RS-far": "Far", "stat": "No change",
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


def paired(left, right, metric):
    keyed = {(r["cls"], r["seed"]): metric(r) for r in right if not r["div"]}
    by_seed = defaultdict(list)
    for r in sorted(left, key=lambda r: (r["cls"], r["seed"])):
        a, b = metric(r), keyed.get((r["cls"], r["seed"]))
        if not r["div"] and a is not None and b is not None:
            by_seed[r["seed"]].append(a-b)
    v = np.array([np.mean(by_seed[s]) for s in sorted(by_seed)])
    if not len(v):
        return None
    draws = np.random.default_rng(0).choice(v, (2000, len(v))).mean(axis=1)
    return (v.mean(), *np.quantile(draws, [.025, .975]), sum(map(len, by_seed.values())))


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
    wanted = (P,A,N,"FxNLMS")
    available = [m for m in wanted if all((s,m) in coef for s in SWITCHES)]
    styles = (("0.04","-","o"), ("0.32","--","s"), ("0.54","-.","^"), ("0.70",":","D"))
    fig, axes = plt.subplots(1,2,figsize=(7.2,2.95),sharey=True,layout="constrained")
    for ax,s in zip(axes,SWITCHES):
        for m,(color,ls,mark) in zip(available,styles):
            rows = coef[s,m]
            blocks = sorted(map(int, rows[0]["J"]))
            series = np.array([np.quantile(values(rows,j(k)),[.25,.5,.75]) for k in blocks])
            if m in (P,A,N):
                for k in KS:
                    key = (s,"全クラス",shown(k*BLOCK_SECONDS,2), A if m==P else m)
                    verify(shown(series[blocks.index(k),1]), report[key][4 if m==P else 5], key)
            time = np.array(blocks)*BLOCK_SECONDS
            ax.fill_between(time,series[:,0],series[:,2],color=color,alpha=.12,linewidth=0)
            ax.plot(time,series[:,1],color=color,linestyle=ls,linewidth=1.65,marker=mark,
                    markersize=3.1,markevery=3,label=NAMES.get(m,m))
        for k in KS:
            ax.axvline(k*BLOCK_SECONDS,color="0.68",ls=":",lw=.95,zorder=0)
        ax.set_xscale("symlog",linthresh=.55)
        ax.set_xlim(0,15)
        ax.xaxis.set_major_locator(FixedLocator([0,.128,.512,2,8,15]))
        ax.xaxis.set_major_formatter(FixedFormatter(["0","0.13","0.51","2","8","15"]))
        ax.set_title(f"({'ab'[SWITCHES.index(s)]}) {SCEN[s]} source switch",loc="left",fontsize=10)
        ax.set_xlabel("Time after switch (s)")
        ax.grid(axis="y",color=".9",lw=.6)
    axes[0].set_ylabel("Frozen-filter residual / primary (dB)")
    axes[1].legend(loc="upper right",fontsize=7,framealpha=.92)
    missing = ", ".join(m for m in wanted if m not in available)
    fig.supxlabel("Bands: interquartile range. Time axis: linear to 0.55 s, logarithmic thereafter."
                  + (f"\n{missing}: fixed-filter data unavailable." if missing else ""),fontsize=8)
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
                ci_verify(stats,entry[6],key)
                verify((shown(np.median(values(rows[0],j(k)))),shown(np.median(values(r,j(k))))),entry[4:6],key)
                if m==A:
                    vals[i,col],lo,hi,_=stats
                    if lo<=0<=hi:
                        cross.append((i,col))
    verify(len(cross),3,"zero-containing intervals")
    fig,ax=plt.subplots(figsize=(5.7,4.15),layout="constrained")
    limit=np.ceil(np.abs(vals).max())
    im=ax.imshow(vals,cmap="gray_r",vmin=-limit,vmax=limit,aspect="auto")
    for (i,k),v in np.ndenumerate(vals):
        color="white" if im.norm(v)>.72 else "black"
        ax.text(k,i,f"{v:+.2f}"+(" †" if (i,k) in cross else ""),ha="center",va="center",color=color,fontsize=9)
    ax.set_yticks(range(len(CLASSES)),CLASSES)
    ax.set_xticks(range(4),[f"{SCEN[s]}\n{k*BLOCK_SECONDS:.2f} s" for s,k in columns])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.axvline(1.5,color="white",lw=2)
    fig.colorbar(im,ax=ax,label="Proposed − APSIPA 2026 [our prior work] (dB)")
    ax.set_xlabel("Negative favors Proposed.  †  95% interval includes zero.",labelpad=10)
    save(fig,"fig4")


def output_table(number, title, panels, notes):
    def esc(s):
        chars = {"&":r"\&","%":r"\%","_":r"\_","#":r"\#","−":"-", "μ":r"$\mu$", "<":r"$<$", ">":r"$>$"}
        return "".join(chars.get(x,x) for x in str(s))
    md = [f"# Table {number}. {title}",""]
    tex = [r"\begin{table*}[t]",r"\centering",r"\small",r"\setlength{	abcolsep}{4pt}",r"\caption{"+esc(title)+"}",
           rf"\label{{tab:atsip-{number}}}"]
    for heading,header,rows in panels:
        md.extend([f"## {heading}",""])
        tex.extend([r"\par\medskip",r"\textbf{"+esc(heading)+r"}\par\smallskip",
                    r"\begin{tabular}{l"+("l"*(len(header)-1))+"}",r"\toprule",
                    " & ".join(map(esc,header))+r" \\",r"\midrule"])
        for row in (header,["---"]*len(header),*rows):
            md.append("| "+" | ".join(map(str,row))+" |")
        md.append("")
        tex.extend(" & ".join(map(esc,row))+r" \\" for row in rows)
        tex.extend([r"\bottomrule",r"\end{tabular}"])
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
    baselines=[(NAMES.get(m,m),f"{r['mu']:g}",
                f"Error clipping to [{r['params']['c1']:g}, {r['params']['c2']:g}]" if m=="Th_FxNLMS" else "--")
               for m,r in tune["chosen"].items()]
    output_table(1,"Evaluation settings and selected baseline steps",[
        ("Proposed method",("Setting","Value"),settings),
        ("External baselines",("Method","Selected step μ","Selected clipping"),baselines)],
        ["The external methods use the common settings selected on design data (TUNE.json)."])


def table2(final,coef,cr,final_text):
    result=[]
    for s in SWITCHES:
        reports={r[2]:r for r in md_rows(section(final_text,"### 騒音源切替の回復")) if len(r)==7 and r[:2]==(s,"全クラス")}
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
            intervals.append(ci_text(v,BLOCK_SECONDS))
        result.append((SCEN[s],"Recovery (s)",*cells,*intervals))
        for k in KS:
            t=shown(k*BLOCK_SECONDS,2)
            cells=[shown(np.median(values(coef[s,m],j(k)))) for m in methods]
            intervals=[]
            for idx,m in enumerate(methods[1:],1):
                r=cr[s,"全クラス",t,m]
                verify((cells[0],cells[idx]),r[4:6],(s,k,m))
                v=paired(coef[s,P],coef[s,m],j(k))
                ci_verify(v,r[6],(s,k,m))
                intervals.append(ci_text(v))
            result.append((SCEN[s],f"Frozen-filter J at {t} s (dB)",*cells,*intervals))
    # Keep the customary paper table compact by separating levels and paired differences.
    output_table(2,"Reconvergence after a source switch",[
        ("Median levels",("Switch","Metric",NAMES[P],NAMES[A],NAMES[N]),[r[:5] for r in result]),
        ("Paired differences",("Switch","Metric","Proposed - APSIPA 2026","Proposed - No restart"),
         [(*r[:2],*r[5:]) for r in result])],
        ["Recovery uses each method's post-change reference level +3 dB and a Kaplan-Meier median.",
         "Frozen-filter J is the residual-to-primary power ratio with the first 256 ms excluded.",
         "Differences are paired means with seed-bootstrap 95% intervals and pair counts in parentheses. Recovery differences use pairs with recovery observed in both runs."])


def table3(final, final_text, fast):
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
        for m in names:
            if m==A:
                continue
            rows=final[s,m]
            div=sum(r["div"] for r in rows)
            false=sum(r["false_confirm"] for r in rows)
            verify((str(div),str(false)),(published[m][2],published[m][8]),(s,m))
            internal[m].append(f'{div}/{sum(r["n_confirm"] for r in rows)}/{false}')
    costs=[]
    for phase,label in (("late","Ordinary block"),("late+trial","Ordinary block with trial")):
        r=next(r for r in md_rows(fast) if r[0]==phase)
        costs.append((label,r[2].replace("／","/"),r[3],r[4].replace("／","/"),r[5],r[6]))
    output_table(3,"Cost of the restart mechanism",[
        ("Stationary path and error-side disturbances",
         ("Condition","Method","30 s L1","Last 10 s","Post-event 2 s","Transfers","Matches"),statements),
        ("Internal comparisons (200 runs per condition)",
         ("Method",*(SCEN[s] for s in (*QUIET,*SWITCHES))),
         [(NAMES[m],*counts) for m,counts in internal.items()]),
        ("Approximate operations per block",
         ("State","Ref. FFT","Ref. mult.","Fast FFT","Fast mult.","Ratio"),costs)],
        ["L1 values are in dB. Transfers count events. Output matches require exact numerical equality for both error and control output against No restart.",
         "Internal comparisons show divergences / all transfers / unmatched transfers.",
         "Ref. FFT counts real/complex transforms. Fast FFT counts 1024-/256-point transforms. The benchmark counts FIR and FFT multiplications."])


def verify_coef_main(coef, text):
    primary = section(text, "## 主表：変化から 0.13 秒後と 0.51 秒後の消音量（評価点は seed 100–119 の実行前に固定。2026-11-21）")
    lines = [
        "# Coefficient report cross-check", "",
        "Figures and tables use the seed-bootstrap intervals in the findings section of RESULTS_coef2.md.",
        "The entries below differ from its autogenerated main table at two-decimal precision.",
        "The main table and findings use different bootstrap random seeds.", "",
        "| Switch | Input | Time | Baseline | Autogenerated mean [95% CI] | Recomputed mean [95% CI] |",
        "|---|---|---|---|---|---|",
    ]
    for switch, cls, time, *row in md_rows(primary):
        if switch not in SWITCHES:
            continue
        k = round(float(time.split()[0]) / BLOCK_SECONDS)
        left = [r for r in coef[switch,P] if cls in ("全クラス",r["cls"])]
        for i, m in enumerate((A,N)):
            right = [r for r in coef[switch,m] if cls in ("全クラス",r["cls"])]
            x, lo, hi, _ = paired(left,right,j(k))
            computed = f"{x:+.2f} [{lo:+.2f}, {hi:+.2f}]"
            original = row[3+i]
            numbers = lambda t: [float(n) for n in re.findall(r"[-+]?\d+\.\d+",t)]
            if numbers(computed) != numbers(original):
                label = "All inputs" if cls == "全クラス" else cls
                lines.append(f"| {SCEN[switch]} | {label} | {time.split()[0]} s | {NAMES[m]} | {original} | {computed} |")
    (PAPER/"tab"/"verification.md").write_text("\n".join(lines)+"\n")
    return len(lines)-8


def main():
    for path in ("fig","tab"):
        (PAPER/path).mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,
                         "axes.spines.top":False,"axes.spines.right":False})
    final=load("final2_run",(*SWITCHES,*QUIET))
    coef=load("coef2_run",SWITCHES)
    text=(ROOT/"RESULTS_final2.md").read_text()
    ct=(ROOT/"RESULTS_coef2.md").read_text()
    report={(*r[:4],):r for r in md_rows(section(ct,"### 0.13秒・0.51秒の比較")) if len(r)==8}
    figure3(coef,report)
    figure4(coef,report)
    table1()
    table2(final,coef,report,text)
    table3(final,text,(ROOT/"RESULTS_fast.md").read_text())
    differences = verify_coef_main(coef,ct)
    print(f"Verified published findings. {differences} interval rows differ from the autogenerated coefficient table. Fig. 3: FxNLMS fixed-filter data unavailable.")


if __name__=="__main__":
    main()
