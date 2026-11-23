"""Time course of the noise reduction (ANR) for the paper: one synthetic noise type, source switch at 15 s.

  python paper/anr_curves.py            -> paper/fig/fig_anr.{pdf,png}, out/anr_curves.npz

ANR(k) = 10 log10(Pe(k) / Pd(k)), Pe(k) = lam Pe(k-1) + (1 - lam) e(k)^2 (same for d), lam = 0.9995; the curves are the
median in dB over the evaluation seeds 100-119.  Settings are those of the final evaluation (final2.configure,
TUNE.json, TUNE_cc.json).  Only synthetic inputs are generated here (the recordings are not stored in the repository).
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT))
import final as F
import final2
import cc

CLS, SCENS, SEEDS, LAM = "a145", ("RS-near", "RS-far"), range(100, 120), 0.9995
METHODS = ("P5-early-cap", "Freq_FxNLMS_CohVSS", "C_FXLMS", "FxNLMS", "FxlogNLMS", "MFxLMM", "FxNFLOC", "EVSS_FxeLMS")


def smooth(v):
    from scipy.signal import lfilter
    return lfilter([1 - LAM], [1, -LAM], v * v)


def job(a):
    scen, seed = a
    final2.configure(); cc.install(F)
    tune = dict(json.loads((ROOT / "TUNE.json").read_text())["chosen"], **json.loads((ROOT / "TUNE_cc.json").read_text())["chosen"])
    dd, Rall, st = F.build(CLS, scen, seed, F.LS_FINAL, final=True)
    X4 = F.p.parse(["c5", "--preset", "final"]).X4
    out = {}
    for m in METHODS:
        sim = F.run_rule(m, dd, Rall, st, X4)[1] if m == "P5-early-cap" else F.run_baseline(m, tune[m]["mu"], tune[m]["params"], dd)
        assert not sim["div"], (m, scen, seed)
        n = len(sim["e"])
        out[m] = smooth(np.asarray(sim["e"]))[::160].astype(np.float32)
    out["d"] = smooth(dd["d"][:n])[::160].astype(np.float32)
    return scen, out


def main():
    f = ROOT / "out" / "anr_curves.npz"
    if not f.exists():
        with Pool() as pool:
            res = pool.map(job, [(s, k) for s in SCENS for k in SEEDS])
        acc = {}
        for scen, out in res:
            for m, v in out.items():
                acc.setdefault((scen, m), []).append(v)
        np.savez(f, **{f"{s}|{m}": np.array(v) for (s, m), v in acc.items()})
    z = np.load(f)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names = {x.name for x in font_manager.fontManager.ttflist}
    fam = next(n for n in ("Times New Roman", "Liberation Serif", "Nimbus Roman", "TeX Gyre Termes") if n in names)
    plt.rcParams.update({"font.family": fam, "font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    label = {"P5-early-cap": "Proposed", "Freq_FxNLMS_CohVSS": "CohVSS-FxNLMS", "C_FXLMS": "C-FXLMS", "FxNLMS": "FxNLMS",
             "FxlogNLMS": "FxlogNLMS", "MFxLMM": "MFxLMM", "FxNFLOC": "FxNFLOC", "EVSS_FxeLMS": "EVSS-FxeLMS"}
    style = {"P5-early-cap": ("0", "-", 1.8), "Freq_FxNLMS_CohVSS": ("0.25", "--", 1.3), "C_FXLMS": ("0.25", "-.", 1.1),
             "FxNLMS": ("0.5", "-", 0.9), "FxlogNLMS": ("0.5", ":", 1.2), "MFxLMM": ("0.6", "--", 0.9),
             "FxNFLOC": ("0.4", (0, (5, 1, 1, 1, 1, 1)), 0.9), "EVSS_FxeLMS": ("0.7", "-", 0.9)}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), layout="constrained", sharey=True, gridspec_kw={"width_ratios": [1.6, 1]})
    t = np.arange(z[f"{SCENS[0]}|d"].shape[1]) * 160 / 16000
    for row, scen in enumerate(SCENS):
        for col, (a, b) in enumerate(((0, 30), (14.8, 17.5))):
            ax = axes[row, col]; k = (t >= a) & (t <= b)
            for m in METHODS[::-1]:
                c, ls, lw = style[m]
                ax.plot(t[k], np.median(10 * np.log10(z[f"{scen}|{m}"][:, k] / z[f"{scen}|d"][:, k]), axis=0), color=c, ls=ls, lw=lw, label=label[m])
            ax.axvline(15, color="0.7", lw=0.6, ls=":", zorder=0); ax.set_xlim(a, b); ax.grid(axis="y", color=".9", lw=.6)
            sub = "nearby" if scen == "RS-near" else "distant"
            ax.set_xlabel(f"Time (s)\n({'abcd'[2 * row + col]}) Switch to a {sub} source" + (", around the switch" if col else ""), linespacing=1.6)
        axes[row, 0].set_ylabel("ANR (dB)")
    axes[0, 0].set_ylim(-17, 3)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="outside upper center", ncol=4, fontsize=7.5, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(PAPER / "fig" / f"fig_anr.{ext}", **({"dpi": 300} if ext == "png" else {"metadata": {"CreationDate": None, "ModDate": None}}))


if __name__ == "__main__":
    main()
