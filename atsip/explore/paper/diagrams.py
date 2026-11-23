"""Block diagrams of the paper: Fig. 1 (ANC system) and Fig. 2 (foreground and background filters).

  python paper/diagrams.py        -> paper/fig/fig1.{pdf,png}, paper/fig/fig2.{pdf,png}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent / "fig"
LW = 0.9


def setup():
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = next(n for n in ("Times New Roman", "Liberation Serif", "Nimbus Roman", "TeX Gyre Termes") if n in names)
    plt.rcParams.update({"font.family": family, "font.size": 9, "pdf.fonttype": 42, "mathtext.fontset": "stix"})


def box(ax, x, y, w, h, text, dashed=False, size=9):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fill=False, lw=LW, ls="--" if dashed else "-"))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, linespacing=1.15)


def line(ax, pts, arrow=True, dashed=False):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="k", lw=LW, ls="--" if dashed else "-", solid_capstyle="butt")
    a, b = pts[-2], pts[-1]
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>" if arrow else "-", mutation_scale=8, lw=LW, color="k",
                                 ls="--" if dashed else "-", shrinkA=0, shrinkB=0))


def dot(ax, x, y):
    ax.add_patch(Circle((x, y), 0.035, color="k"))


def summer(ax, x, y, r=0.13):
    ax.add_patch(Circle((x, y), r, fill=False, lw=LW))
    ax.text(x, y, "+", ha="center", va="center", fontsize=9)


def finish(fig, ax, name, xlim, ylim):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off")
    for ext in ("pdf", "png"):
        opt = {"metadata": {"CreationDate": None, "ModDate": None}} if ext == "pdf" else {"dpi": 450}
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.03, **opt)
    plt.close(fig)


def figure1():
    fig, ax = plt.subplots(figsize=(5.6, 2.75))
    ax.text(-0.05, 1.08, "$x(k)$", ha="left", va="bottom")
    line(ax, [(-0.05, 1.0), (0.4, 1.0)], arrow=False); dot(ax, 0.4, 1.0)
    line(ax, [(0.4, 1.0), (0.4, 2.0), (1.6, 2.0)])                       # to primary path
    box(ax, 2.1, 2.0, 1.0, 0.5, "Primary path\n$p(k)$")
    line(ax, [(2.6, 2.0), (4.3, 2.0), (4.3, 1.13)]); ax.text(3.5, 2.08, "$d(k)$", ha="center", va="bottom")
    line(ax, [(0.4, 1.0), (0.85, 1.0)])
    box(ax, 1.3, 1.0, 0.9, 0.5, "Control filter\n$\\mathbf{w}(k)$", size=8)
    line(ax, [(1.75, 1.0), (2.45, 1.0)]); ax.text(2.1, 1.08, "$y(k)$", ha="center", va="bottom")
    box(ax, 2.95, 1.0, 1.0, 0.5, "Secondary path\n$s(k)$", size=8)
    line(ax, [(3.45, 1.0), (4.17, 1.0)])
    summer(ax, 4.3, 1.0)
    line(ax, [(5.0, 1.55), (4.39, 1.09)]); ax.text(5.02, 1.57, "$v(k)$", ha="left", va="bottom")
    line(ax, [(4.43, 1.0), (5.2, 1.0)]); ax.text(5.2, 1.08, "$e(k)$", ha="right", va="bottom")
    dot(ax, 4.8, 1.0)
    line(ax, [(0.4, 1.0), (0.4, 0.0), (0.85, 0.0)])                      # filtered reference
    box(ax, 1.3, 0.0, 0.9, 0.5, "Model\n$\\hat{s}(k)$")
    line(ax, [(1.75, 0.0), (2.45, 0.0)]); ax.text(2.1, 0.08, "$r(k)$", ha="center", va="bottom")
    box(ax, 2.95, 0.0, 1.0, 0.5, "Adaptive\nalgorithm")
    line(ax, [(4.8, 1.0), (4.8, 0.0), (3.45, 0.0)])
    line(ax, [(2.55, 0.25), (0.9, 1.42)])                   # adaptation arrow through the control filter
    finish(fig, ax, "fig1", (-0.1, 5.45), (-0.35, 2.35))


def figure2():
    fig, ax = plt.subplots(figsize=(5.6, 4.05))
    ax.text(-0.05, 2.08, "$x(k)$", ha="left", va="bottom")
    line(ax, [(-0.05, 2.0), (0.35, 2.0)], arrow=False); dot(ax, 0.35, 2.0)
    line(ax, [(0.35, 2.0), (0.8, 2.0)])
    box(ax, 1.4, 2.0, 1.2, 0.55, "Foreground filter\n$\\mathbf{w}_{\\mathrm{f}}$")
    line(ax, [(2.0, 2.0), (2.7, 2.0)]); ax.text(2.35, 2.08, "$y(k)$", ha="center", va="bottom")
    box(ax, 3.15, 2.0, 0.9, 0.55, "Secondary\npath $s(k)$")
    line(ax, [(3.6, 2.0), (4.17, 2.0)])
    summer(ax, 4.3, 2.0)
    line(ax, [(4.3, 2.75), (4.3, 2.13)]); ax.text(4.3, 2.78, "$d(k)+v(k)$", ha="center", va="bottom")
    line(ax, [(4.43, 2.0), (5.2, 2.0)]); ax.text(5.2, 2.08, "$e(k)$", ha="right", va="bottom")
    dot(ax, 4.8, 2.0)
    line(ax, [(0.35, 2.0), (0.35, 0.55), (0.8, 0.55)])
    box(ax, 1.4, 0.55, 1.2, 0.55, "Background filter\n$\\mathbf{w}_{\\mathrm{b}}$ (trial)", dashed=True)
    line(ax, [(1.15, 1.725), (1.15, 0.825)]); ax.text(1.1, 1.275, "copy at\nstart of trial", ha="right", va="center", fontsize=7.5)
    line(ax, [(1.65, 0.825), (1.65, 1.725)], dashed=True); ax.text(1.7, 1.38, "transfer\nif confirmed", ha="left", va="center", fontsize=7.5)
    box(ax, 3.55, 1.05, 1.5, 0.5, "Detector: error power /\nfiltered-reference power", size=8)
    line(ax, [(4.8, 2.0), (4.8, 1.05), (4.3, 1.05)])
    line(ax, [(2.8, 1.05), (2.45, 1.05), (2.45, 0.7), (2.0, 0.7)]); ax.text(2.4, 0.87, "start", ha="right", va="center", fontsize=7.5)
    box(ax, 3.55, 0.05, 1.5, 0.6, "Errors of $\\mathbf{w}_{\\mathrm{f}}$ and $\\mathbf{w}_{\\mathrm{b}}$\nreconstructed with $\\hat{s}(k)$", size=8)
    line(ax, [(4.8, 1.05), (4.8, 0.05), (4.3, 0.05)]); dot(ax, 4.8, 1.05)
    line(ax, [(2.0, 0.4), (2.45, 0.4), (2.45, 0.05), (2.8, 0.05)])
    line(ax, [(3.55, -0.25), (3.55, -0.6), (1.4, -0.6), (1.4, 0.275)], dashed=True)
    ax.text(2.5, -0.68, "transfer or discard", ha="center", va="top", fontsize=7.5)
    finish(fig, ax, "fig2", (-0.1, 5.45), (-0.95, 3.05))


if __name__ == "__main__":
    setup(); figure1(); figure2()
