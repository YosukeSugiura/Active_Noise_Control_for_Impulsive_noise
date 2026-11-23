"""Build tex/main.tex from draft_v4.md (author notes removed) and compile it.

  python paper/build_tex.py      -> paper/tex/main.tex, paper/tex/main.pdf

The class is the standard article class for now; the journal class replaces it once its style files are added.
"""
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "tex"
TITLE = "Trial Adaptation with a Background Filter for Fast Reconvergence in Active Control of Impulsive Noise"
CITE = {"Shao and Nikias 1993": "shao1993", "Kuo and Morgan 1996": "kuo1996", "Sun 2006": "sun2006", "Akhtar 2008": "akhtar2008",
        "Akhtar 2009": "akhtar2009", "Wu 2011": "wu2011", "Thanigai 2007": "thanigai2007", "Wu and Qiu 2013": "wu2013",
        "Sun et al. 2015": "sun2015", "Feng et al. 2024": "feng2024", "Cheng 2023": "cheng2023", "APSIPA 2026": "sugiura2026",
        "Ferrer 2013": "ferrer2013", "Cheng 2022": "cheng2022", "Yu 2020": "yu2020", "Ochiai et al. 1977": "ochiai1977",
        "Kuo 2005": "kuo2005", "Elliott and Rafaely 1997": "elliott1997", "Elliott and Rafaely 2000": "elliott2000",
        "Bjarnason 1992": "bjarnason1992", "Yang 2018 review": "yang2018", "MeshRIR": "meshrir", "NOISEX": "noisex",
        "MIMII": "mimii", "SONYC": "sonyc"}
FLOATS = {  # inserted after the heading that starts with the key
    "3.1 ": r"\begin{figure}[t]\centering\includegraphics[width=0.8\linewidth]{../fig/fig1.pdf}\caption{Single-channel feedforward ANC system.}\label{fig:system}\end{figure}",
    "4.1 ": r"\begin{figure}[t]\centering\includegraphics[width=0.8\linewidth]{../fig/fig2.pdf}\caption{Structure of the proposed algorithm. The foreground filter generates the control signal. The background filter is adapted only during a trial, and its coefficients are transferred to the foreground filter or discarded according to the reconstructed errors of the two filters.}\label{fig:structure}\end{figure}",
    "5.3 ": r"\input{../tab/table1.tex}",
    "6.1 ": r"\input{../tab/table2.tex}" "\n" r"\begin{figure}[t]\centering\includegraphics[width=\linewidth]{../fig/fig_anr_common.pdf}\caption{Time course of the noise reduction for the $\alpha$-stable signal with $\alpha = 1.45$ (median over 20 seeds). The noise source switches at 15 s. Each algorithm uses its common setting.}\label{fig:anr}\end{figure}" "\n" r"\begin{figure}[t]\centering\includegraphics[width=\linewidth]{../fig/fig3.pdf}\caption{Noise reduction with the coefficients frozen at each time after the switch of the noise source (median over 200 runs). The bands show the interquartile range for the proposed algorithm and CohVSS-FxNLMS. The time axis is linear up to 0.55 s and logarithmic thereafter. The vertical dotted lines mark 0.13 s and 0.51 s.}\label{fig:frozen}\end{figure}",
    "6.2 ": r"\begin{figure}[t]\centering\includegraphics[width=0.75\linewidth]{../fig/fig4.pdf}\caption{Difference in the noise reduction with frozen coefficients between the proposed algorithm and CohVSS-FxNLMS for each noise type (mean of the paired differences over 20 seeds). Negative values indicate a larger noise reduction for the proposed algorithm. A dagger marks a 95\% confidence interval that includes zero.}\label{fig:classes}\end{figure}",
    "7.1 ": r"\input{../tab/table3.tex}",
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


def cites(md):
    def one(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        if all(k in CITE for k in keys):
            return "\\cite{" + ",".join(CITE[k] for k in keys) + "}"
        return m.group(0)
    return re.sub(r"\[([^\[\]]+)\]", one, md)


def main():
    md = cites(clean((HERE / "draft_v4.md").read_text()))
    abstract, body = md.split("## 1. Introduction", 1)
    abstract = abstract.replace("## Abstract", "").strip()
    body = "## 1. Introduction" + body
    for key, tex in FLOATS.items():                              # floats as raw LaTeX after the heading
        body, n = re.subn(rf"(^### {re.escape(key)}.*$)", lambda m: m.group(1) + "\n\n```{=latex}\n" + tex + "\n```\n", body, flags=re.M)
        assert n == 1, key
    body = re.sub(r"^(#+) \d+(\.\d+)*\.? ", r"\1 ", body, flags=re.M)      # numbers come from LaTeX
    body = re.sub(r"^## ", "# ", body, flags=re.M).replace("\n### ", "\n## ")
    conv = lambda t: subprocess.run(["pandoc", "-f", "markdown+raw_attribute", "-t", "latex", "--wrap=none"], input=t, text=True,
                                    capture_output=True, check=True).stdout
    tex = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=25mm]{geometry}
\usepackage{amsmath,amssymb,amsthm,graphicx,booktabs,newunicodechar}
\usepackage[hidelinks]{hyperref}
\newunicodechar{α}{\ensuremath{\alpha}}\newunicodechar{−}{\ensuremath{-}}\newunicodechar{≤}{\ensuremath{\le}}\newunicodechar{≥}{\ensuremath{\ge}}
\newunicodechar{μ}{\ensuremath{\mu}}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\title{""" + TITLE + r"""}
\author{Yosuke Sugiura\\Meiji University, Japan}
\date{}
\begin{document}
\maketitle
\begin{abstract}
""" + conv(abstract) + r"""\end{abstract}
""" + conv(body) + r"""
\bibliographystyle{ieeetr}
\bibliography{refs}
\end{document}
"""
    tex = tex.replace(r"$\blacksquare$", r"\hfill$\blacksquare$")
    TEX.mkdir(exist_ok=True)
    (TEX / "main.tex").write_text(tex)
    for cmd in (["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main"], ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main"], ["pdflatex", "-interaction=nonstopmode", "main"]):
        r = subprocess.run(cmd, cwd=TEX, capture_output=True, text=True)
        if r.returncode and cmd[0] == "pdflatex":
            print(r.stdout[-1500:]); raise SystemExit(1)
    log = (TEX / "main.log").read_text(errors="ignore")
    print("pages:", re.search(r"Output written on main.pdf \((\d+) pages", log).group(1),
          "| undefined:", len(re.findall(r"undefined", log)), "| overfull:", len(re.findall(r"Overfull", log)))


if __name__ == "__main__":
    main()
