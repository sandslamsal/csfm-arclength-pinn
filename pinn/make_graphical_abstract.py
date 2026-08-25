"""Graphical abstract for EAAI.

EAAI requires one and reads it before the abstract, so it has to carry the
same split the abstract is organised around: what the contribution in
artificial intelligence is, and what the application in engineering is.

The design is a left-to-right narrative rather than three unrelated panels.
A load-controlled solver dies at the limit point (left); re-parametrising by
position along the path makes the fold an ordinary interior state, provided
four training failures are defeated (centre); and the payoff is one trained
network covering a whole deterioration family, which is where a learned path
beats an incremental solve (right). Arrows carry the eye across the three.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as F

HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / "figures"

INK, MUTED = "#101418", "#5A6672"
AI, APP = "#2962FF", "#00963E"
FAIL = "#FF4E11"

fam = json.load(open(HERE.parent / "oracle" / "deepbeam_family_newton.json"))
net = json.load(open(HERE / "runs" / "parametric_rho"
                     / "parametric_curves_anchored_newton.json"))["curves"]

fig = plt.figure(figsize=(13.6, 5.1))
gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.16, 1.02],
                      left=0.048, right=0.985, top=0.685, bottom=0.175,
                      wspace=0.30)

# banner naming the two halves EAAI asks to be distinguishable
fig.text(0.048, 0.955, "CONTRIBUTION IN AI", color=AI, fontsize=10.5,
         weight="bold", va="center")
fig.text(0.048, 0.900, "a path parametrisation and four training ingredients",
         color=MUTED, fontsize=8.8, va="top")
fig.text(0.700, 0.955, "APPLICATION IN ENGINEERING", color=APP,
         fontsize=10.5, weight="bold", va="center")
fig.text(0.700, 0.900, "post-peak response of corroding concrete D-regions",
         color=MUTED, fontsize=8.8, va="top")

c0 = fam["curves"]["0.00"]
d0, l0 = np.array(c0["delta"]), np.array(c0["lam"])
ipk = int(np.argmax(l0))

# ---------------- panel a: why the load parametrisation dies ------------
ax = fig.add_subplot(gs[0, 0])
ax.plot(d0[:ipk + 1], l0[:ipk + 1], color=INK, lw=3.0, solid_capstyle="round")
ax.plot(d0[ipk:], l0[ipk:], color=INK, lw=3.0, alpha=0.18,
        solid_capstyle="round")
ax.axhline(l0[ipk], color=FAIL, lw=1.0, ls=(0, (4, 3)), alpha=0.85)
ax.plot(d0[ipk], l0[ipk], "o", ms=9, mfc="white", mec=FAIL, mew=2.2, zorder=5)
ax.annotate("load control\nstops here", xy=(d0[ipk], l0[ipk]),
            xytext=(d0[ipk] + 1.15, l0[ipk] * 0.60), fontsize=8.8,
            color=FAIL, weight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color=FAIL, lw=1.3,
                            connectionstyle="arc3,rad=-0.25"))
ax.set_xlabel(r"deflection $\delta$  (mm)", fontsize=9)
ax.set_ylabel(r"load factor $\lambda$", fontsize=9)
ax.set_title("the limit point", fontsize=10.5, weight="bold",
             loc="left", color=INK, pad=8)
F.clean(ax)

# ---------------- panel b: the parametrisation, as equations ---------
ax = fig.add_subplot(gs[0, 1])
ax.axis("off")

ax.add_patch(FancyBboxPatch((0.02, 0.60), 0.96, 0.30,
                            boxstyle="round,pad=0.015,rounding_size=0.04",
                            transform=ax.transAxes, facecolor="none",
                            edgecolor="#8A929B", lw=1.0, zorder=1))
ax.text(0.5, 0.795, r"$(x,\, y,\, s)\;\longmapsto\;"
        r"\left(\mathbf{u}(x,y,s),\ \lambda(s)\right)$",
        transform=ax.transAxes, ha="center", va="center", fontsize=15.5,
        color=INK, zorder=2)
ax.text(0.5, 0.668, r"$\lambda$ is an output, not an input",
        transform=ax.transAxes, ha="center", va="center", fontsize=10.0,
        color=INK, zorder=2)

ax.text(0.5, 0.475,
        r"$\left\Vert \partial \mathbf{u} / \partial s \right\Vert^{2} = "
        r"S_{\max}^{2}$",
        transform=ax.transAxes, ha="center", va="center", fontsize=14.0,
        color=INK, zorder=2)
ax.text(0.5, 0.363, "imposed pointwise on the loaded patch",
        transform=ax.transAxes, ha="center", va="center", fontsize=9.0,
        color=MUTED, zorder=2)

chips = ["elastic pre-training", "pointwise arc loss",
         "fixed arc weight", r"$C^1$ constitutive"]
for k, name in enumerate(chips):
    cx = 0.255 if k % 2 == 0 else 0.745
    cy = 0.185 if k < 2 else 0.055
    ax.add_patch(FancyBboxPatch((cx - 0.235, cy - 0.045), 0.47, 0.09,
                                boxstyle="round,pad=0.006,rounding_size=0.03",
                                transform=ax.transAxes, facecolor="none",
                                edgecolor="#8A929B", lw=0.9, zorder=1))
    ax.text(cx, cy, name, transform=ax.transAxes, ha="center", va="center",
            fontsize=8.9, color=INK, zorder=2)

ax.set_title("parametrise by path position", fontsize=10.5, weight="bold",
             loc="left", color=INK, pad=8)

# ---------------- panel c: the payoff -----------------------------------
ax = fig.add_subplot(gs[0, 2])
levels = ["0.00", "0.10", "0.20", "0.30"]
cols = plt.cm.viridis(np.linspace(0.10, 0.78, len(levels)))
for k, lv in enumerate(levels):
    r = fam["curves"][lv]
    rd, rl = np.array(r["delta"]), np.array(r["lam"])
    ax.plot(rd, rl, color=cols[k], lw=1.0, alpha=0.45, zorder=2)
    n = net[lv]
    nd, nl = np.array(n["delta"]), np.array(n["lam"])
    m = nd <= rd.max()
    ax.plot(nd[m], nl[m], color=cols[k], lw=2.8, zorder=3,
            solid_capstyle="round")
    ax.text(rd.max() * 1.01, rl[-1], f"{float(lv)*100:.0f}%",
            color=cols[k], fontsize=8.2, va="center", weight="bold")
ax.text(0.035, 0.055, "thin: reference   thick: network",
        transform=ax.transAxes, fontsize=8.2, color=MUTED, va="bottom")
ax.set_xlabel(r"deflection $\delta$  (mm)", fontsize=9)
ax.set_ylabel(r"load factor $\lambda$", fontsize=9)
ax.set_xlim(0, d0.max() * 1.14)
ax.set_title("one network, the whole family", fontsize=10.5, weight="bold",
             loc="left", color=INK, pad=8)
F.clean(ax)

# ---------------- connective arrows + the headline number ---------------
# gutters only: the second arrow must clear panel c's y-label
for x0, x1 in ((0.353, 0.379), (0.672, 0.698)):
    fig.patches.append(FancyArrowPatch(
        (x0, 0.42), (x1, 0.42), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=17, lw=1.7, color="#B3BFCC",
        shrinkA=0, shrinkB=0, zorder=0))

fig.text(0.700, 0.040, "held-out states within 0.5%",
         fontsize=9.0, color=INK)

out = FIGDIR / "graphical_abstract.png"
fig.savefig(out, dpi=300, facecolor="white")
print(f"  wrote {out}")
