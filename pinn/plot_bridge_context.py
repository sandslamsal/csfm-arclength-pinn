"""Bridge-context locator panel prepended to the deep-beam geometry
figure. One elevation of a girder bridge shows where the three
benchmark archetypes live in a real structure: the deep beam as the
pier-cap transfer member, the corbel as the bearing bracket carrying a
drop-in span at the halving joint, and the squat wall pier as the
substructure (the Bimschas test units are literally bridge piers).
Panel b is the deep-beam benchmark unchanged.

Writes ../figures/geometry.png (+ .pdf), replacing the single-panel
version so the manuscript include is unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as F                                                        # noqa: E402
import matplotlib.pyplot as plt                                             # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "P2" / "pinn"))
from problem import DeepBeam                                                # noqa: E402

F.apply()
HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / "figures"

C_DEEP = F.VERM      # deep beam accent
C_CORB = F.SKY       # corbel accent
C_PIER = F.GREEN     # wall pier accent
GREY = "0.55"


def draw_bridge(ax):
    """Simple girder-bridge elevation with the three archetypes called
    out. Proportions are schematic, not to scale."""
    # ground
    ax.axhline(0, color="0.3", lw=1.2, zorder=1)
    for x in range(-2, 96, 4):
        ax.plot([x, x - 2], [0, -2.2], color="0.55", lw=0.6, zorder=1)

    # left abutment
    ax.add_patch(Polygon([(2, 0), (10, 0), (10, 30), (7, 30), (2, 8)],
                         closed=True, facecolor="0.94", edgecolor="0.35",
                         lw=1.0, zorder=2))
    # the wall pier (highlighted) and a plain companion pier
    ax.add_patch(Rectangle((34, 0), 8, 30, facecolor="#E8F5E9",
                           edgecolor=C_PIER, lw=2.0, zorder=2))
    ax.add_patch(Rectangle((72, 0), 8, 37.4, facecolor="0.94",
                           edgecolor="0.35", lw=1.0, zorder=2))

    # pier cap on the wall pier: the deep beam
    ax.add_patch(Rectangle((29, 30), 18, 7.4, facecolor="#FFF3E0",
                           edgecolor=C_DEEP, lw=2.0, zorder=3))

    # left girder from the abutment onto the cap, ending in a corbel
    # bracket; the right girder is a drop-in span seated on it
    ax.add_patch(Rectangle((8, 37.4), 41, 3.4, facecolor="0.90",
                           edgecolor="0.35", lw=1.0, zorder=3))
    ax.add_patch(Polygon([(49.0, 37.4), (55.5, 37.4), (55.5, 39.0),
                          (49.0, 41.2)],
                         closed=True, facecolor="#E1F5FE",
                         edgecolor=C_CORB, lw=2.0, zorder=5))
    ax.add_patch(Rectangle((54.0, 39.4), 26.0, 3.4, facecolor="0.90",
                           edgecolor="0.35", lw=1.0, zorder=4))
    # deck line
    ax.plot([6, 82], [42.9, 42.9], color="0.3", lw=1.6, zorder=5)

    # traffic load arrows
    for x in (18, 44, 66):
        ax.add_patch(FancyArrowPatch((x, 52), (x, 44.3),
                                     arrowstyle="-|>", mutation_scale=10,
                                     color="0.45", lw=1.2, zorder=5))

    def callout(text, xy, xytext, colour, ha="center"):
        ax.annotate(text, xy=xy, xytext=xytext,
                    fontsize=F.FS_ANNOT, color=colour, ha=ha,
                    va="center", fontweight="bold", zorder=7,
                    arrowprops=dict(arrowstyle="-", color=colour, lw=1.1,
                                    shrinkA=4, shrinkB=2))

    callout("deep beam\n(pier cap)", (30.5, 33.7), (12, 24), C_DEEP)
    callout("corbel\n(bearing bracket)", (52.3, 40.0), (66, 54), C_CORB)
    callout("wall pier\n(the tested unit)", (42, 14), (64, 14), C_PIER,
            ha="left")

    ax.set_xlim(-2, 92)
    ax.set_ylim(-4, 60)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_deepbeam(ax):
    """The deep-beam benchmark, redrawn from make_figures.fig_geometry
    into a provided axes, dimensions included."""
    prob = DeepBeam()
    L, H = prob.L, prob.H
    half = prob.bearing / 2.0
    a = 250.0
    nx, ny = 40, 20
    dx, dy = L / nx, H / ny
    for i in range(nx + 1):
        ax.plot([i * dx, i * dx], [0, H], color="0.92", lw=0.3, zorder=0)
    for j in range(ny + 1):
        ax.plot([0, L], [j * dy, j * dy], color="0.92", lw=0.3, zorder=0)
    ax.plot([0, L, L, 0, 0], [0, 0, H, H, 0], color="black", lw=1.4)

    band = 150.0
    ax.fill_between([0, L], [0, 0], [band, band], color=F.BLUE,
                    alpha=0.16, lw=0)
    ax.plot([a, L - a], [95, 95], color=F.BLUE, lw=3.0, zorder=4,
            solid_capstyle="round")
    ax.text(L / 2, 205, r"bottom tie band, $\rho_x = 1.2$ %",
            color=F.BLUE, fontsize=F.FS_LABEL, ha="center",
            va="bottom", fontweight="bold")

    for x0 in (a, L - a):
        ax.plot([x0, L / 2], [95, H - 45], color="0.6", lw=7.0,
                alpha=0.40, zorder=2, solid_capstyle="round")
    ax.text(L * 0.32, H * 0.60, "strut", color="0.4", rotation=38,
            fontsize=F.FS_ANNOT, ha="center")
    for x0, yy in ((a, 95), (L - a, 95), (L / 2, H - 45)):
        ax.plot(x0, yy, marker="o", ms=7, mfc="white", mec="black",
                mew=1.2, zorder=6, ls="none")

    sup_y = -8
    for x0, name, side in ((a, "pin", -1), (L - a, "roller", 1)):
        ax.add_patch(Polygon([(x0 - 45, sup_y - 62), (x0 + 45, sup_y - 62),
                              (x0, sup_y)],
                             closed=True, facecolor="0.25",
                             edgecolor="0.25", zorder=5))
        ax.text(x0, sup_y - 100, name, ha="center", va="top",
                fontsize=F.FS_ANNOT, color="0.35")

    ax.plot([L / 2 - half, L / 2 + half], [H + 8, H + 8], color=F.VERM,
            lw=4.0, zorder=5)
    ax.add_patch(FancyArrowPatch((L / 2, H + 165), (L / 2, H + 16),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color=F.VERM, lw=2.0, zorder=6))
    ax.text(L / 2 + 60, H + 80, r"$\lambda\,P_{\mathrm{ref}}$",
            color=F.VERM, fontsize=F.FS_LABEL, fontweight="bold",
            ha="left")
    ax.text(L / 2, H + 230,
            r"$P_{\mathrm{ref}} = 800$ kN, bearing 200 mm",
            color=F.VERM, fontsize=F.FS_LABEL, ha="center")

    # dimensions
    ax.annotate("", xy=(0, -210), xytext=(L, -210),
                arrowprops=dict(arrowstyle="<->", color="0.45", lw=1.0))
    ax.text(L / 2, -240, r"$L = 2000$ mm", ha="center", va="top",
            fontsize=F.FS_ANNOT, color="0.35")
    ax.annotate("", xy=(L + 90, 0), xytext=(L + 90, H),
                arrowprops=dict(arrowstyle="<->", color="0.45", lw=1.0))
    ax.text(L + 150, H / 2, r"$H = 1000$ mm", rotation=90,
            va="center", ha="center", fontsize=F.FS_ANNOT, color="0.35")
    ax.text(60, H - 175, r"$t = 300$ mm, mesh $40 \times 20$",
            ha="left", fontsize=F.FS_ANNOT, color="0.45")

    ax.set_xlim(-140, L + 230)
    ax.set_ylim(-300, H + 300)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    fig = plt.figure(figsize=(F.FIG_W, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.94, 1.30], wspace=0.10)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    draw_bridge(ax_a)
    draw_deepbeam(ax_b)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.canvas.draw()
    F.fig_panel(fig, ax_a, 'a', 'where the benchmarks live', y=0.965)
    F.fig_panel(fig, ax_b, 'b', 'the deep-beam benchmark', y=0.965)
    out = FIGDIR / "geometry.png"
    F.save(fig, out)
    plt.close(fig)


if __name__ == "__main__":
    main()
