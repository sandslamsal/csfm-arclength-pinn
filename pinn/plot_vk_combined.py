"""Combined wall-pier figure: geometry, both specimens, and sensitivity.

Replaces the separate vk1_full and vk_pair figures, which duplicated the
VK1 measured-versus-reference comparison. Because VK1 and VK3 are the
same pier (they differ only in longitudinal reinforcement ratio), one
schematic serves both, and the three panels then answer three distinct
questions without repeating a curve:

  (a) what the specimen and its actions are
  (b) how the network, the reference and the measurement compare, for
      both reinforcement ratios, with the deviation from measurement
      carried in an inset, since the agreement is 1 to 3 per cent from
      15 mm onward but 13 to 16 per cent below 10 mm and a single
      overlay hides which of the two the reader is looking at
  (c) whether the modelled sensitivity to reinforcement matches the
      measured one, separated from the absolute level, and read against
      a measured band rather than a single line, since the reported
      ultimate and the digitised peak differ by one to two per cent

Encoding: colour identifies the specimen (VK1 dark, VK3 amber), stroke
identifies the source (open markers measured, solid line reference,
dash-dot the network).

Writes ../figures/vk_combined.png (+ .pdf)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ArclengthPINN                                            # noqa: E402
from wallpier_vk1 import WallPierVK1                                       # noqa: E402
import figstyle as F                                                       # noqa: E402
import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.patches import ConnectionPatch                             # noqa: E402
from plot_vk1_unified import draw_vk1_geometry                             # noqa: E402

F.apply()

HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / "figures"
ORACLE = HERE.parent / "oracle"
EXP = HERE.parent / "experimental"
CKPT = HERE / "runs" / "vk1_newton_t350" / "vk1_pinn.pt"

# Reported ultimates of the two units, against which the digitised
# first-cycle peaks are the second basis. Quoting both is what lets
# panel c show a band rather than assert one measured number.
V_ULT = {"VK1": 725.0, "VK3": 876.0}

UNITS = {
    "VK1": dict(ref="vk1_newton_plotschema.json", meas="vk1_backbone.csv",
                rho=0.82, colour=F.BLACK, mk="o"),
    "VK3": dict(ref="vk3_newton_plotschema.json", meas="vk3_backbone.csv",
                rho=1.23, colour=F.ORANGE, mk="s"),
}


def reference(cfg):
    r = json.loads((ORACLE / cfg["ref"]).read_text())
    d = np.array([p["delta_x"] for p in r["curve"]])
    V = np.array([p["lam"] for p in r["curve"]]) * r["V_ref"] / 1e3
    cv = np.array([p.get("converged", True) for p in r["curve"]], bool)
    return d[cv], V[cv], r


def pinn_curve():
    """The strain-anchored VK1 trace, or None if its checkpoint predates
    the corrected reference. Drawing a trace trained against the
    t = 200 mm pier beside a t = 350 mm reference would misreport it."""
    if not CKPT.exists():
        return None
    if CKPT.stat().st_mtime < (ORACLE / "vk1_newton.json").stat().st_mtime:
        print(f"  [skip] {CKPT.name} predates the corrected reference; "
              f"panel b omits the network trace until it is retrained")
        return None
    U0 = 1.0e-3
    prob = WallPierVK1(include_N=True)
    net = ArclengthPINN(width=96, depth=6)
    net.load_state_dict(torch.load(CKPT))
    net.eval()
    with torch.no_grad():
        s = torch.linspace(0, 1, 200).unsqueeze(-1)
        x = torch.full_like(s, 0.0)
        y = torch.full_like(s, prob.h_eff)
        out, lam = net(torch.cat([x / prob.L, y / prob.H], -1), s)
        ux = out[:, 0:1] * U0 * prob.L
    return ux.squeeze().numpy(), lam.squeeze().numpy() * prob.P / 1e3


def draw_backbones(ax, inset=False):
    """Both measured backbones against both references. Returns the
    per-unit (rho, measured peak, model peak) rows."""
    rows = []
    for u, cfg in UNITS.items():
        d, V, r = reference(cfg)
        m = np.loadtxt(EXP / cfg["meas"], delimiter=",", skiprows=1)
        # faint join so the measured points read as a backbone, not a
        # scatter, without competing with the reference stroke
        ax.plot(m[:, 0], m[:, 1], ls=(0, (2.2, 2.0)), lw=1.0,
                color=cfg["colour"], alpha=0.40, zorder=2)
        ax.plot(m[:, 0], m[:, 1], ls="none", marker=cfg["mk"], ms=5.0,
                color=cfg["colour"], mfc="white", mew=1.4, zorder=5,
                label=None if inset else f"{u} measured")
        ax.plot(d, V, ls="-", color=cfg["colour"], lw=2.2, zorder=3,
                label=None if inset else f"{u} reference")
        # the located maxima, which is what the comparison is about
        k = int(np.argmax(V))
        km = int(np.argmax(m[:, 1]))
        ax.plot(d[k], V[k], marker="*", ms=11.0, color=cfg["colour"],
                mec="white", mew=0.8, zorder=7, ls="none")
        ax.plot(m[km, 0], m[km, 1], marker="*", ms=11.0, mfc="white",
                mec=cfg["colour"], mew=1.5, zorder=7, ls="none")
        rows.append((u, cfg["rho"], float(m[:, 1].max()), float(V[k]),
                     float(d[k]), float(m[km, 0])))
    return rows


def main() -> None:
    fig = plt.figure(figsize=(F.FIG_W, 5.1))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.56, 1.62],
                          height_ratios=[1.42, 1.0], wspace=0.52,
                          hspace=0.66)
    ax_g = fig.add_subplot(gs[:, 0])
    ax = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[1, 1])

    draw_vk1_geometry(ax_g)

    rows = draw_backbones(ax)

    pc = pinn_curve()
    if pc is not None:
        ax.plot(pc[0], pc[1], **F.style('anchored', marker='none',
                                        label=False),
                lw=2.0, zorder=4, label="VK1 arc-length PINN")

    ax.set_ylabel(r"horizontal force $V$  (kN)")
    ax.set_xlabel(r"deflection at $h_{\mathrm{eff}}$  (mm)")
    ax.set_xlim(0, 65)
    ax.set_ylim(0, 1010)
    F.clean(ax)
    # Both legend and inset go in the band below the backbones, which is
    # empty; upper left would sit inside the zoom rectangle.
    ax.legend(loc="lower left", fontsize=F.FS_ANNOT, ncol=1,
              columnspacing=1.0, handlelength=1.9, borderaxespad=0.4,
              labelspacing=0.30, frameon=False)

    # Inset: deviation from the measured backbone. A zoom on the peak
    # would show the agreement there and conceal the early branch, which
    # is where the reference and the specimens actually differ.
    axi = ax.inset_axes([0.505, 0.085, 0.470, 0.380])
    axi.axhspan(-5, 5, color="0.88", lw=0, zorder=0)
    axi.axhline(0, color="0.45", lw=0.8, zorder=1)
    for u, cfg in UNITS.items():
        d, V, _ = reference(cfg)
        m = np.loadtxt(EXP / cfg["meas"], delimiter=",", skiprows=1)
        k = m[:, 0] <= d.max()
        err = (np.interp(m[k, 0], d, V) / m[k, 1] - 1) * 100.0
        axi.plot(m[k, 0], err, ls="-", marker=cfg["mk"], ms=4.0,
                 color=cfg["colour"], mfc="white", mew=1.1, lw=1.4,
                 zorder=3)
    axi.set_xlim(0, 46)
    axi.set_ylim(-20, 13)      # headroom inside for the title
    axi.set_yticks([-15, -10, -5, 0, 5])
    axi.set_xticks([0, 15, 30, 45])
    axi.tick_params(labelsize=F.FS_SMALL, length=2.2, pad=1.5)
    for sp in ("top", "right"):
        axi.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axi.spines[sp].set_linewidth(0.7)
    axi.text(0.03, 0.97, "deviation from measured  (%)",
             transform=axi.transAxes, fontsize=F.FS_SMALL, color="0.35",
             ha="left", va="top")
    axi.text(0.97, 0.05, "grey band: within 5 %", transform=axi.transAxes,
             fontsize=F.FS_SMALL, color="0.45", ha="right", va="bottom")
    # opaque and above the backbones: the VK3 measured branch drops
    # steeply across this corner of the panel
    axi.set_facecolor("white")
    axi.patch.set_alpha(1.0)
    axi.set_zorder(6)

    # ---- panel c: level against sensitivity -----------------------
    rho = np.array([r[1] for r in rows])
    meas = np.array([r[2] for r in rows])              # digitised peaks
    ult = np.array([V_ULT[r[0]] for r in rows])        # reported ultimates
    mod = np.array([r[3] for r in rows])

    lo, hi = np.minimum(meas, ult), np.maximum(meas, ult)
    ax_s.fill_between(rho, lo, hi, color=F.ORANGE, alpha=0.25, lw=0,
                      zorder=1)
    ax_s.plot(rho, meas, "-o", color=F.ORANGE, lw=2.0, ms=6.0,
              mec="white", mew=0.7, zorder=4, label="measured, digitised peak")
    ax_s.plot(rho, ult, ls=(0, (3, 2)), marker="v", color=F.ORANGE,
              lw=1.3, ms=5.0, mfc="white", mew=1.2, alpha=0.85, zorder=3,
              label="measured, reported ultimate")
    ax_s.plot(rho, mod, "-s", color=F.BLACK, lw=2.0, ms=6.0,
              mec="white", mew=0.7, zorder=5, label="reference")

    # The left annotation sits to the left of its point: to the right it
    # runs into the statistics box in the lower corner.
    for i, (x, a, b) in enumerate(zip(rho, meas, mod)):
        ax_s.annotate("", xy=(x, b), xytext=(x, a),
                      arrowprops=dict(arrowstyle="<->", color="0.55",
                                      lw=0.9, shrinkA=3.2, shrinkB=3.2))
        if i == 0:                       # above: the box holds the floor
            ax_s.text(x, max(a, b) + 12, f"{(b/a-1)*100:+.1f} %",
                      fontsize=F.FS_ANNOT, color="0.35", va="bottom",
                      ha="center")
        else:
            ax_s.text(x + 0.016, 0.5 * (a + b), f"{(b/a-1)*100:+.1f} %",
                      fontsize=F.FS_ANNOT, color="0.35", va="center",
                      ha="left")

    d_meas = (meas[1] / meas[0] - 1) * 100
    d_ult = (ult[1] / ult[0] - 1) * 100
    d_mod = (mod[1] / mod[0] - 1) * 100
    ax_s.text(0.985, 0.055,
              f"sensitivity to reinforcement\n"
              f"measured {d_meas:+.1f} %, model {d_mod:+.1f} %\n"
              f"{d_mod / d_meas * 100:.0f} % recovered",
              transform=ax_s.transAxes, ha="right", va="bottom",
              fontsize=F.FS_ANNOT, color="0.3", linespacing=1.35,
              bbox=dict(boxstyle="round,pad=0.32", fc="white",
                        ec="0.85", lw=0.6))
    ax_s.set_xlabel(r"longitudinal reinforcement $\rho_\ell$  (%)")
    ax_s.set_ylabel(r"peak $V$  (kN)")
    ax_s.set_xlim(0.72, 1.36)
    ax_s.set_ylim(660, 990)
    ax_s.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    F.clean(ax_s)
    ax_s.legend(loc="upper left", fontsize=F.FS_ANNOT, frameon=False,
                handlelength=1.9, borderaxespad=0.3, labelspacing=0.28)

    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.canvas.draw()
    bb, bbs = ax.get_position(), ax_s.get_position()
    Y_FOOT = bbs.y0 + 0.135        # room beneath the pier for its data
    F.fit_schematic(fig, ax_g, Y_FOOT, bb.y1, x0=0.030)
    # The schematic's width follows its aspect, so the gap to panel b is
    # whatever is left over. Take it from b and c instead of leaving it
    # to chance.
    GAP = 0.115
    ga = ax_g.get_position()
    x_new = ga.x0 + ga.width + GAP
    for a in (ax, ax_s):
        q = a.get_position()
        a.set_position((x_new, q.y0, q.x1 - x_new, q.height))
    fig.canvas.draw()
    bb, bbs = ax.get_position(), ax_s.get_position()
    F.fig_panel(fig, ax_g, 'a', 'wall pier geometry')
    F.fig_panel(fig, ax, 'b', 'response at both reinforcement ratios')
    y_c = bbs.y1 + 0.022          # just above panel c, clear of b's label
    fig.text(bbs.x0 - 0.006, y_c, 'c', fontsize=F.FS_PANEL,
             fontweight='bold', va='bottom', ha='left')
    fig.text(bbs.x0 - 0.006 + 0.034, y_c, 'level versus sensitivity',
             fontsize=F.FS_LABEL, fontweight='bold', va='bottom',
             ha='left', color='0.15')

    # Specimen data under the schematic. The section thickness is stated
    # because a 200 mm value, 57 % of the true section, was carried for
    # some time before being checked against the source.
    gb = ax_g.get_position()
    fig.text(gb.x0 + gb.width / 2, gb.y0 - 0.030,
             "section $1500 \\times 350$ mm\n"
             "$h_{\\mathrm{eff}} = 3300$ mm,  $N = 1370$ kN\n"
             "$f_c = 35$ MPa,  $f_y = 515$ MPa\n"
             "VK1 $\\rho_\\ell = 0.82$ %,  VK3 $1.23$ %",
             fontsize=F.FS_SMALL, color="0.30", ha="center", va="top",
             linespacing=1.5)

    # the gap the schematic leaves against panel b, reported so it is
    # checked rather than eyeballed
    ga = ax_g.get_position()
    print(f"  column gap: {bb.x0 - (ga.x0 + ga.width):.3f} of figure width")

    out = FIGDIR / "vk_combined.png"
    F.save(fig, out)
    plt.close(fig)

    print(f"\n{'unit':6}{'rho_l':>7}{'meas':>8}{'ult':>7}{'model':>8}"
          f"{'vs meas':>9}{'vs ult':>8}")
    for (u, r, a, b, dp, dm), v in zip(rows, ult):
        print(f"{u:6}{r:7.2f}{a:8.0f}{v:7.0f}{b:8.0f}"
              f"{(b/a-1)*100:+8.1f}%{(b/v-1)*100:+7.1f}%")
    print(f"sensitivity: measured {d_meas:+.1f} % (ultimates {d_ult:+.1f} %), "
          f"reference {d_mod:+.1f} %  ->  {d_mod/d_meas*100:.0f} % recovered")


if __name__ == "__main__":
    main()
