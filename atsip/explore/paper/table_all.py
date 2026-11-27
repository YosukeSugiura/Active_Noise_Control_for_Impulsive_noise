"""Table 10: summary of all comparison algorithms with the common setting.
Reads out/final2_run, coef2_run, cc_run, cc_coef, mfx_run, mfx_coef, thx_run, thx_coef, frozen5_coef."""
from pathlib import Path
import figs as F

SC = ("stat", "RS-near", "RS-far", "EN0", "ENm10")
ROWS = [("Proposed", "final2_run", "P5-early-cap", "coef2_run"),
        ("Without trials", "final2_run", "nors-cap", "coef2_run"),
        ("FxNLMS", "final2_run", "FxNLMS", "frozen5_coef"),
        ("Th-FxNLMS", "thx_run", "Th2_FxNLMS", "thx_coef"),
        ("FxlogNLMS", "final2_run", "FxlogNLMS", "frozen5_coef"),
        ("MFxLMM", "final2_run", "MFxLMM", "frozen5_coef"),
        ("FxNFLOC", "final2_run", "FxNFLOC", "frozen5_coef"),
        ("EVSS-FxeLMS", "final2_run", "EVSS_FxeLMS", "frozen5_coef"),
        ("FreqFxLMS", "final2_run", "FreqFxLMS", "coef2_run"),
        ("CohVSS-FxNLMS", "final2_run", "Freq_FxNLMS_CohVSS", "coef2_run"),
        ("C-FXLMS", "cc_run", "C_FXLMS", "cc_coef"),
        ("MFxNLMS", "mfx_run", "MFxNLMS", "mfx_coef"),
        ("APFM", "apfm_eval:APFM_2", "APFM", "apfm_eval:APFM_2")]

cache = {}
def grp(folder, scen):
    key = folder, scen
    if key not in cache:
        g = {}
        for c in F.CLASSES:
            for i in range(100, 120):
                if ":" in folder:
                    d, pre = folder.split(":")
                    f = F.ROOT / "out" / d / f"{pre}_{c}_{scen}_s{i}.json"
                    rows = [F.json.loads(f.read_text())]
                else:
                    f = F.ROOT / "out" / folder / f"{c}_{scen}_s{i}.json"
                    rows = F.json.loads(f.read_text())["rows"]
                for r in rows:
                    if r.get("variant", "main") == "main":
                        g.setdefault((scen, r["method"]), []).append(r)
        assert all(len(v) == 200 for v in g.values()), (folder, scen)
        cache[key] = g
    return cache[key]

def fmt(x, d):
    return "--" if x is None else (f"{x:.{d}f}".lstrip("-") if float(f"{x:.{d}f}") == 0 else f"{x:.{d}f}").replace("-", "$-$")

lines = []
frozen_complete = True
for name, run, m, coef in ROWS:
    div = sum(sum(bool(r["div"]) for r in grp(run, s)[s, m]) for s in SC)
    rec = []
    for s in ("RS-near", "RS-far"):
        t = F.recovery(grp(run, s)[s, m])
        rec.append(None if t is None else 0.008 * t)
    fz = []
    for k in F.KS:
        for s in ("RS-near", "RS-far"):
            values = F.values(grp(coef, s)[s, m], F.j(k)) if coef else []
            fz.append(float(F.np.median(values)) if values else None)
    frozen_complete &= all(x is not None for x in fz)
    whole = F.l1(grp(run, "stat")["stat", m], "L1_whole")
    dist = F.l1(grp(run, "EN0")["EN0", m], "L1_ev")
    lines.append(" & ".join([name, str(div), fmt(rec[0], 2), fmt(rec[1], 2), *(fmt(x, 1) for x in fz),
                             fmt(whole, 2), fmt(dist, 2)]) + r" \\")

footnote = "" if frozen_complete else ' "--": frozen coefficients were not evaluated for this algorithm.'

tex = r"""\begin{sidewaystable}[tbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{All algorithms with the common setting}
\label{tab:atsip-10}
\par\smallskip
\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%
\begin{tabular}{llllllllll}
\toprule
 & Diverged & \multicolumn{2}{l}{Median recovery time [s]} & \multicolumn{2}{l}{Frozen at 0.13 s [dB]} & \multicolumn{2}{l}{Frozen at 0.51 s [dB]} & No change & 2 s after 0 dB \\
Algorithm & runs & Close & Far & Close & Far & Close & Far & Whole run [dB] & disturbance [dB] \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\par\medskip
\begin{minipage}{0.98\textwidth}
\footnotesize
Diverged runs are counted over the 1000 runs of the five conditions. The noise reduction values are averaged over the runs without divergence. The frozen values are medians over the runs. APFM uses one filter trained for the primary path before the switch.""" + footnote + r"""\par
\end{minipage}
\end{sidewaystable}
"""
(Path(__file__).resolve().parent / "tab" / "table10.tex").write_text(tex)
print("\n".join(lines))
