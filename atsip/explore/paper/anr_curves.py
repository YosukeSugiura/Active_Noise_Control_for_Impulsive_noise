"""Time course of the noise reduction (ANR) for the paper: one synthetic noise type, source switch at 15 s.

  python paper/anr_curves.py [--common]  -> paper/fig/fig_anr_class.{pdf,png} (or fig_anr_common), out/anr_curves_*.npz

ANR(k) = 10 log10(Pe(k) / Pd(k)), Pe(k) = lam Pe(k-1) + (1 - lam) e(k)^2 (same for d), lam = 0.9995; the curves are the
median in dB over the evaluation seeds 100-119.  The proposed algorithm uses its common setting.  The comparison algorithms use the setting that is best for this noise
type (TUNE.json "per_class"; for C-FXLMS the same selection is made here on the design seeds 10-14 and stored in
out/anr_cc_class.json): fewest diverged runs, then the largest noise reduction over 30 s, on no change and far switch.  Only synthetic inputs are generated here (the recordings are not stored in the repository).
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


def cc_point(a):
    key, mu, par = a
    final2.configure(); cc.install(F)
    div, l1 = 0, []
    for scen in F.TUNE_SCEN:
        for seed in F.TUNE_SEEDS:
            dd = F.build(CLS, scen, seed, F.LS_FINAL)[0]
            sim = F.run_baseline(cc.NAME, mu, par, dd)
            div += int(sim["div"])
            if not sim["div"]:
                l1.append(F.measures(sim, dd)["L1_whole"])
    return dict(key=key, mu=mu, params=par, div=div, L1_whole=float(np.mean(l1)) if l1 else None)


def class_settings(pool):
    tune = {m: v[CLS] for m, v in json.loads((ROOT / "TUNE.json").read_text())["per_class"].items()}
    f = ROOT / "out" / "anr_cc_class.json"
    if not f.exists():
        rows = pool.map(cc_point, cc.tune_grid(F))
        best = min((r for r in rows if r["L1_whole"] is not None), key=lambda r: (r["div"], r["L1_whole"]))
        f.write_text(json.dumps(dict(cls=CLS, chosen=best, grid=rows), indent=1))
    tune[cc.NAME] = json.loads(f.read_text())["chosen"]
    return tune


def job(a):
    scen, seed, tune = a
    final2.configure(); cc.install(F)
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
    common = "--common" in sys.argv          # comparison algorithms with the common settings of the tables
    tag = "_common" if common else "_class"
    f = ROOT / "out" / f"anr_curves{tag}.npz"
    if not f.exists():
        with Pool() as pool:
            tune = class_settings(pool)
            if common:
                tune = dict(json.loads((ROOT / "TUNE.json").read_text())["chosen"], **json.loads((ROOT / "TUNE_cc.json").read_text())["chosen"])
            print({m: (tune[m]["key"], tune[m]["mu"]) for m in METHODS[1:]})
            res = pool.map(job, [(s, k, tune) for s in SCENS for k in SEEDS])
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
            ax.set_xlabel(f"Time [s]\n({'abcd'[2 * row + col]}) Switch to a {sub} source" + (", around the switch" if col else ""), linespacing=1.6)
        axes[row, 0].set_ylabel("ANR [dB]")
    axes[0, 0].set_ylim(-17, 3)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="outside upper center", ncol=4, fontsize=7.5, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(PAPER / "fig" / f"fig_anr{tag}.{ext}", **({"dpi": 300} if ext == "png" else {"metadata": {"CreationDate": None, "ModDate": None}}))


if __name__ == "__main__":
    main()
