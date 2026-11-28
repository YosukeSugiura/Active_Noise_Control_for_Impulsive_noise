"""Build tex/main.tex from draft_v4.md (author notes removed) and compile it.

  python paper/build_tex.py      -> paper/tex/main.tex, paper/tex/main.pdf

The class is now-journal (option sip) of Now Publishers; its files are in paper/tex.  Bibliography: biblatex with biber.
"""
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "tex"
TITLE = "Fast Reconvergence in Active Impulsive Noise Control Using a Background Trial Filter"
CITE = {"Shao and Nikias 1993": "shao1993", "Kuo and Morgan 1996": "kuo1996", "Sun 2006": "sun2006", "Akhtar 2008": "akhtar2008",
        "Akhtar 2009": "akhtar2009", "Wu 2011": "wu2011", "Thanigai 2007": "thanigai2007", "Wu and Qiu 2013": "wu2013",
        "Sun et al. 2015": "sun2015", "Feng et al. 2024": "feng2024", "Cheng 2023": "cheng2023", "APSIPA 2026": "sugiura2026",
        "Ferrer 2013": "ferrer2013", "Cheng 2022": "cheng2022", "Yu 2020": "yu2020", "Ochiai et al. 1977": "ochiai1977",
        "Das 2007": "das2007", "Harman 2022": "harman2022", "Lindstrom 2007": "lindstrom2007", "Iqbal 2007": "iqbal2007", "Jung 2014": "jung2014", "Yin 2023": "yin2023", "Zhang 2022": "zhang2022", "Wang 2025": "wang2025", "Elliott and Rafaely 1997": "elliott1997", "Elliott and Rafaely 2000": "elliott2000",
        "Bjarnason 1992": "bjarnason1992", "Yang 2018 review": "yang2018", "MeshRIR": "meshrir", "NOISEX": "noisex",
        "MIMII": "mimii", "SONYC": "sonyc"}
FLOATS = {  # inserted after the heading that starts with the key
    "3.1 ": r"\begin{figure}[tbp]\centering\includegraphics[width=0.8\linewidth]{../fig/fig1.pdf}\caption{Single-channel feedforward ANC system.}\label{fig:system}\end{figure}",
    "4.1 ": r"\begin{figure}[tbp]\centering\includegraphics[width=0.8\linewidth]{../fig/fig2.pdf}\caption{Structure of the proposed algorithm. The foreground filter generates the control signal. The background filter is adapted only during a trial, and its coefficients are transferred to the foreground filter or discarded according to the reconstructed errors of the two filters.}\label{fig:structure}\end{figure}",
    "4.6 ": r"\input{../tab/alg1.tex}",
    "5.3 ": r"\input{../tab/table1.tex}",
    "5.4 ": r"\input{../tab/table11.tex}",
    "6.1 ": r"\begin{figure}[tbp]\centering\includegraphics[width=\linewidth]{../fig/fig_diff.pdf}\caption{Paired differences, proposed algorithm minus each algorithm, after a source switch (mean over runs with seed-bootstrap 95\% intervals). Panel (a) is in seconds, and panels (b) and (c) are in decibels. Negative values favor the proposed algorithm. Recovery differences use the runs in which both algorithms recovered.}\label{fig:diff}\end{figure}" "\n" r"\begin{figure}[tbp]\centering\includegraphics[width=\linewidth]{../fig/fig_anr_common.pdf}\caption{Time course of the noise reduction for the $\alpha$-stable signal with $\alpha = 1.45$ (median over 20 seeds of the ratio of the error power to the primary-noise power, both smoothed by an exponential average with a forgetting factor of 0.9995). The noise source switches at 15 s. The figure shows eight algorithms with the common setting, and Table 3 lists all algorithms.}\label{fig:anr}\end{figure}" "\n" r"\begin{figure}[tbp]\centering\includegraphics[width=\linewidth]{../fig/fig3.pdf}\caption{Noise reduction with the coefficients frozen at each time after the switch of the noise source (median over 200 runs). The bands show the interquartile range for the proposed algorithm and CohVSS-FxNLMS. The time axis is linear up to 0.55 s and logarithmic thereafter. The vertical dotted lines mark 0.13 s and 0.51 s.}\label{fig:frozen}\end{figure}" "\n" r"\input{../tab/table10.tex}" "\n" r"\begin{figure}[tbp]\centering\includegraphics[width=\linewidth]{../fig/fig_stepsweep.pdf}\caption{Median recovery time and percentage of diverged runs of two comparison algorithms with step sizes from 0.5 to 16 times the common setting (200 runs each). The median is computed over the runs without divergence. CohVSS-FxNLMS diverged in all runs at 16 times. The stars mark the proposed algorithm with 0.5, 1, and 1.5 times $\sigma_{\mathrm E}$, enlarged in the inset of each panel.}\label{fig:stepsweep}\end{figure}" "\n" r"\input{../tab/table8.tex}",
    "6.2 ": r"\begin{figure}[tbp]\centering\includegraphics[width=0.75\linewidth]{../fig/fig4.pdf}\caption{Difference in the noise reduction with frozen coefficients between the proposed algorithm and CohVSS-FxNLMS for each noise type (mean of the paired differences over 20 seeds). Negative values indicate a larger noise reduction for the proposed algorithm. A dagger marks a 95\% confidence interval that includes zero. The intervals are not adjusted for the 40 comparisons, and the per-type results are exploratory.}\label{fig:classes}\end{figure}" "\n" r"\input{../tab/table12.tex}",
    "6.4 ": r"\input{../tab/table3.tex}",
    "7.1 ": r"\input{../tab/table4.tex}",
    "7.2 ": r"\input{../tab/table5.tex}",
    "7.3 ": r"\input{../tab/table6.tex}",
    "8.1 ": r"\input{../tab/table7.tex}",
    "8.2 ": r"\input{../tab/table9.tex}",
}


def clean(md):
    md = md[md.index("## Abstract"):]
    out, skip = [], False
    for line in md.splitlines():
        if re.match(r"^\[[^\]]*(メモ|語)[^\]]*\]?", line):          # author notes
            skip = line.endswith("メモ]")
            continue
        if skip:
            if line.startswith("- ") or not line.strip():
                continue
            skip = False
        out.append(line)
    return "\n".join(out)



def normalize_tables():
    """Apply the table style of now-journal.cls to copies of tab/*.tex: table font footnotesize,
    note font scriptsize, sub-headings in italics, no scaling of tabulars."""
    import re as _re
    out = HERE / "tex" / "tabn"; out.mkdir(exist_ok=True)
    for f in (HERE / "tab").glob("*.tex"):
        t = f.read_text()
        if "\\begin{table" in t or "\\begin{sidewaystable" in t:
            t = _re.sub(r"\\(small|footnotesize)\n", "", t, count=1)
            t = _re.sub(r"\\setlength\{\\tabcolsep\}\{[0-9.]+pt\}\n", "", t)
            t = t.replace("\\centering\n", "\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n", 1)
            t = t.replace("\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%\n", "")
            t = t.replace("\\end{tabular}}", "\\end{tabular}")
            if f.stem not in ("table1", "table11"):
                t = _re.sub(r"\\begin\{tabular\}\{(l+)\}", lambda m: "\\begin{tabular}{l" + "r" * (len(m.group(1)) - 1) + "}", t)
                t = t.replace("\\multicolumn{2}{l}{", "\\multicolumn{2}{c}{").replace("\\multicolumn{4}{l}{", "\\multicolumn{4}{c}{")
                t = t.replace("& \\makecell[l]{", "& \\makecell[r]{")
            def _sub(m):
                n, txt = m.group(1), m.group(2)
                return "\\multicolumn{%s}{c}{\\textit{%s}} \\\\\n\\cmidrule(lr){1-%s}" % (n, txt, n)
            t = _re.sub(r"\\multicolumn\{(\d+)\}\{[lc]\}\{\\textit\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\} \\\\", _sub, t)
            t = _re.sub(r"(?<!\\midrule)\n(\\multicolumn\{\d+\}\{c\}\{\\textit)", r"\n\\midrule\n\1", t)
            t = _re.sub(r"\\textbf\{([^{}]*)\}\\par\\smallskip", r"\\textit{\1}\\par\\smallskip", t)
            t = t.replace("\\begin{minipage}{0.98\\textwidth}\n\\footnotesize", "\\begin{minipage}{0.98\\textwidth}\n\\scriptsize")
        import re as _re
        t = _re.sub(r"\\end\{(table\*?|sidewaystable)\}", lambda m: "\\par\\smallskip{\\footnotesize\\raggedright Source: Author\'s own work\\par}\n\\end{" + m.group(1) + "}", t)
        (out / f.name).write_text(t)

def join_after_display(md):
    """A paragraph that continues the sentence of a display equation ("where ...", "for ...") is not indented."""
    return re.sub(r"(\$\$[^\n]*\$\$)\n\n(?=[a-z])", r"\1\n", md)


def cites(md):
    def one(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        if all(k in CITE for k in keys):
            return "\\cite{" + ",".join(CITE[k] for k in keys) + "}"
        return m.group(0)
    return re.sub(r"\[([^\[\]]+)\]", one, md)


FIG = {"1": "fig:system", "2": "fig:structure", "3": "fig:anr", "4": "fig:frozen", "5": "fig:stepsweep", "6": "fig:classes", "7": "fig:diff"}


def xref(tex, stats):
    """Replace hand-typed cross references in the prose by \\ref / \\eqref and number the equations."""
    def eq(m):
        stats["eq-def"] += 1
        return "\\begin{equation}\\label{eq:" + m.group(2) + "}" + m.group(1).rstrip() + "\\end{equation}"
    tex = re.sub(r"\\\[((?:(?!\\\]).)*?)\\tag\{(\d+)\}\s*\\\]", eq, tex, flags=re.S)
    out = []
    for line in tex.split("\n"):
        if line.startswith(("\\begin{figure}", "\\input", "\\begin{equation}")) or "\\caption" in line:
            out.append(line); continue
        def sec(m):
            stats["sec"] += 1
            r = lambda x: "\\ref{sec:" + x + "}"
            s = m.group(1) + "~" + r(m.group(2))
            return s + (" " + m.group(3) + "~" + r(m.group(4)) if m.group(3) else "")
        line = re.sub(r"(Sections?) (\d+(?:\.\d+)?)(?: (to|and) (\d+(?:\.\d+)?))?", sec, line)
        line = re.sub(r"Fig\. (\d)", lambda m: stats.update(fig=stats["fig"] + 1) or "Fig.~\\ref{" + FIG[m.group(1)] + "}", line)
        line = line.replace("Algorithm 1", "Algorithm~\\ref{alg:block}")
        tr = lambda x: "\\ref{tab:atsip-" + x + "}"
        line = re.sub(r"Tables (\d+) and (\d+)", lambda m: "Tables~" + tr(m.group(1)) + " and~" + tr(m.group(2)), line)
        line = re.sub(r"Table (\d+)", lambda m: stats.update(tab=stats["tab"] + 1) or "Table~" + tr(m.group(1)), line)
        line = re.sub(r"(?<![\w\\{])\((\d{1,2})\)", lambda m: stats.update(eq=stats["eq"] + 1) or "\\eqref{eq:" + m.group(1) + "}", line)
        out.append(line)
    return "\n".join(out)


def main():
    md = join_after_display(cites(clean((HERE / "draft_v4.md").read_text())))
    abstract, body = md.split("## 1. Introduction", 1)
    abstract = abstract.replace("## Abstract", "").strip()
    body = "## 1. Introduction" + body
    for key, tex in FLOATS.items():                              # floats as raw LaTeX after the heading
        body, n = re.subn(rf"(^### {re.escape(key)}.*$)", lambda m: m.group(1) + "\n\n```{=latex}\n" + tex + "\n```\n", body, flags=re.M)
        assert n == 1, key
    body = re.sub(r"^(#+) (\d+(?:\.\d+)*)\.? (.*)$", r"\1 \3 {#sec:\2}", body, flags=re.M)   # numbers come from LaTeX; label keeps the typed number
    body = re.sub(r"^## ", "# ", body, flags=re.M).replace("\n### ", "\n## ")
    conv = lambda t: subprocess.run(["pandoc", "-f", "markdown+raw_attribute", "-t", "latex", "--wrap=none"], input=t, text=True,
                                    capture_output=True, check=True).stdout
    tex = r"""\documentclass[sip,biber]{now-journal}
\usepackage[utf8]{inputenc}
\usepackage{amssymb,amsmath,newunicodechar}
\usepackage{float}
\usepackage{algorithm}
\usepackage{makecell}
\usepackage{rotating}
\usepackage[noend]{algpseudocode}
\newunicodechar{α}{\ensuremath{\alpha}}\newunicodechar{−}{\ensuremath{-}}\newunicodechar{≤}{\ensuremath{\le}}\newunicodechar{≥}{\ensuremath{\ge}}
\newunicodechar{μ}{\ensuremath{\mu}}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\raggedbottom\emergencystretch=2em
\renewcommand{\topfraction}{0.9}\renewcommand{\bottomfraction}{0.8}\renewcommand{\textfraction}{0.07}\renewcommand{\floatpagefraction}{0.75}\renewcommand{\dbltopfraction}{0.9}\renewcommand{\dblfloatpagefraction}{0.75}\setcounter{topnumber}{3}\setcounter{bottomnumber}{2}\setcounter{totalnumber}{5}\setcounter{dbltopnumber}{3}
\title{""" + TITLE + r"""}
\author[1*]{Sugiura,Yosuke}
\affil[1]{School of Science and Technology, Meiji University, Kawasaki, Japan}
\ExecuteBibliographyOptions{sorting=none}
\addbibresource{refs.bib}
\issuevolumeyear{2027}
\issuevolumenumber{xx}
\issueelocation{e}
\articletype{Original Paper}
\articledatabox{Received xx xxxxx 2026; Revised xx xxxxx 2027\\[2pt]
ISSN 2048-7703; DOI 10.1561/116.xxxxxxxx\\
\copyright\ 2027 Y. Sugiura}
\OAstatement{This is an Open Access article, distributed under the terms of the Creative Commons Attribution licence (\url{http://creativecommons.org/licenses/by-nc/4.0/}), which permits unrestricted re-use, distribution, and reproduction in any medium, for non-commercial use, provided the original work is properly cited.}
\keywords{Active noise control, impulsive noise, filtered-x algorithm, frequency-domain adaptive filter, primary path change}
\creditline*{Corresponding author: Yosuke Sugiura, ysugiura@meiji.ac.jp.}
\begin{document}
\begin{abstract}
""" + conv(abstract) + r"""\end{abstract}
""" + conv(body) + r"""
\clearpage
\printbibliography
\end{document}
"""
    stats = {"sec": 0, "fig": 0, "tab": 0, "eq": 0, "eq-def": 0}
    tex = xref(tex, stats)
    print("converted:", stats)
    tex = tex.replace(r"$\blacksquare$", r"\hfill$\blacksquare$")
    TEX.mkdir(exist_ok=True)
    normalize_tables()
    tex = tex.replace(r"\input{../tab/", r"\input{tabn/")
    tex = tex.replace("\\end{figure}", "\\par\\smallskip{\\footnotesize\\raggedright Source: Author\'s own work\\par}\\end{figure}")
    tex = tex.replace("\\section{Acknowledgements}", "\\section*{Acknowledgements}")
    (TEX / "main.tex").write_text(tex)
    for cmd in (["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main"], ["biber", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main"], ["pdflatex", "-interaction=nonstopmode", "main"]):
        r = subprocess.run(cmd, cwd=TEX, capture_output=True, text=True)
        if r.returncode and cmd[0] == "pdflatex":
            print(r.stdout[-1500:]); raise SystemExit(1)
    log = (TEX / "main.log").read_text(errors="ignore")
    print("pages:", re.search(r"Output written on main.pdf \((\d+) pages", log).group(1),
          "| undefined:", len(re.findall(r"undefined", log)), "| overfull:", len(re.findall(r"Overfull", log)))


if __name__ == "__main__":
    main()
