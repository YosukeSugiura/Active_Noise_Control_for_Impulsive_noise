"""Fig. stepsweep: diverged runs and median recovery time of two comparison
algorithms for step sizes 0.5 to 16 times the common setting."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
S = json.loads((HERE.parent / "out/stepsweep_summary.json").read_text())
names = {"FxNLMS": "FxNLMS", "Freq_FxNLMS_CohVSS": "CohVSS-FxNLMS"}
style = {"FxNLMS": ("#000000", "s"), "Freq_FxNLMS_CohVSS": ("#0072B2", "o")}
prop = {"RS-near": 0.360, "RS-far": 0.400}
fam = next((f.name for f in font_manager.fontManager.ttflist
            if f.name in ("Times New Roman", "TeX Gyre Termes", "Nimbus Roman", "Liberation Serif")), None)
plt.rcParams.update({"font.family": fam, "font.size": 9, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
for ax, (sc, sub) in zip(axes, (("RS-near", "(a) Close source"), ("RS-far", "(b) Far source"))):
    for m, lab in names.items():
        g = sorted((x for x in S["groups"] if x["method"] == m and x["condition"] == sc), key=lambda x: x["factor"])
        pts = [(100 * x["divergence_count"] / x["n"], x["recovery_without_divergence"]["median"], x["factor"])
               for x in g if x["recovery_without_divergence"]["median"] is not None and x["divergence_count"] < x["n"]]
        c, mk = style[m]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, marker=mk, ms=4, lw=1, label=lab)
        for x, y, f in pts:
            ax.annotate(f"×{f:g}", (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7, color=c)
    ax.plot([0], [prop[sc]], marker="*", ms=10, color="#D55E00", ls="none", label="Proposed")
    ax.set_yscale("log"); ax.set_xlim(-4, 104)
    ax.set_yticks([0.3, 0.5, 1, 2, 5]); ax.set_yticklabels(["0.3", "0.5", "1", "2", "5"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(f"Diverged runs [%]\n{sub}", linespacing=1.6)
    ax.grid(color=".9", lw=.6)
axes[0].set_ylabel("Median recovery time [s]")
axes[0].legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(HERE / "fig" / f"fig_stepsweep.{ext}", bbox_inches="tight", dpi=200)
