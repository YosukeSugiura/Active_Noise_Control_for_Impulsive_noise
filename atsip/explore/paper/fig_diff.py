"""Figure replacing Table 2: paired differences (proposed minus other) after a source switch, with seed-bootstrap 95% intervals.
Also redraws Fig. 4 (fig3.pdf) via figs.figure3."""
import figs as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
fam = next(n for n in ("Times New Roman", "Liberation Serif", "Nimbus Roman", "TeX Gyre Termes") if n in {f.name for f in font_manager.fontManager.ttflist})
plt.rcParams.update({"font.family": fam, "font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
final = F.load("final2_run", (*F.SWITCHES, *F.QUIET)); coef = F.load("coef2_run", F.SWITCHES)
ccf = F.load("cc_run", (*F.SWITCHES, *F.QUIET)); ccc = F.load("cc_coef", F.SWITCHES)
mf = F.load("mfx_run", (*F.SWITCHES, *F.QUIET)); mc = F.load("mfx_coef", F.SWITCHES)
report = F.coef_main_rows((F.ROOT / "RESULTS_coef2.md").read_text())
F.figure3(coef, report)
others = [("CohVSS-FxNLMS", final, coef, F.A), ("C-FXLMS", ccf, ccc, F.C), ("MFxNLMS", mf, mc, F.M), ("Without trials", final, coef, F.N)]
panels = [("Recovery time [s]", None), ("Noise reduction at 0.13 s [dB]", 16), ("Noise reduction at 0.51 s [dB]", 64)]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True, layout="constrained")
mark = {"RS-near": ("o", "#D55E00", "Close source"), "RS-far": ("s", "#0072B2", "Far source")}
out = {}
for col, (ax, (lab, k)) in enumerate(zip(axes, panels)):
    for row, (name, run, cf, m) in enumerate(others):
        for d, s in ((-0.13, "RS-near"), (0.13, "RS-far")):
            if k is None:
                v = F.paired(final[s, F.P], run[s, m], lambda r: r.get("rec_blocks")); sc = F.BLOCK_SECONDS
            else:
                v = F.paired(coef[s, F.P], cf[s, m], F.j(k)); sc = 1
            a, lo, hi, n = v; a, lo, hi = a * sc, lo * sc, hi * sc
            out[name, s, lab] = (round(a, 3), round(lo, 3), round(hi, 3), n)
            mk, c, l = mark[s]
            ax.errorbar(a, row + d, xerr=[[a - lo], [hi - a]], fmt=mk, color=c, mfc=c if s == "RS-near" else "white",
                        ms=4.5, capsize=2, lw=1, label=l if (row == 0 and col == 0) else None)
    ax.axvline(0, color="0.5", lw=.7, ls=":")
    ax.set_xlabel(f"Proposed − other\n({'abc'[col]}) {lab}", linespacing=1.6)
    ax.grid(axis="x", color=".9", lw=.6)
axes[0].set_yticks(range(len(others)), [o[0] for o in others]); axes[0].invert_yaxis()
fig.legend(loc="outside upper center", ncol=2, frameon=False, fontsize=8)
for ext in ("pdf", "png"):
    fig.savefig(F.PAPER / "fig" / f"fig_diff.{ext}", bbox_inches="tight", dpi=300)
for k, v in out.items(): print(k, v)
