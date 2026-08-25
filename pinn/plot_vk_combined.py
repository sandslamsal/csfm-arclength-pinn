"""Combined wall-pier figure: geometry, both specimens, and sensitivity.

Replaces the separate vk1_full and vk_pair figures, which duplicated the
VK1 measured-versus-reference comparison. Because VK1 and VK3 are the
same pier (they differ only in longitudinal reinforcement ratio), one
schematic serves both, and the three panels then answer three distinct
questions without repeating a curve:

  (a) what the specimen and its actions are
  (b) how the network, the reference and the measurement compare, for
      both reinforcement ratios
  (c) whether the modelled sensitivity to reinforcement matches the
      measured one, separated from the absolute level

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
from plot_vk1_unified import draw_vk1_geometry                             # noqa: E402

F.apply()

HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / "figures"
ORACLE = HERE.parent / "oracle"
EXP = HERE.parent / "experimental"
CKPT = HERE / "runs" / "vk1_newton" / "vk1_pinn.pt"

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


def main() -> None:
    fig = plt.figure(figsize=(F.FIG_W, 4.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.80, 1.50],
                          height_ratios=[1.50, 1.0], wspace=0.34,
                          hspace=0.70)
    ax_g = fig.add_subplot(gs[:, 0])
    ax = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[1, 1])

    draw_vk1_geometry(ax_g)

    rows = []
    for u, cfg in UNITS.items():
        d, V, r = reference(cfg)
        m = np.loadtxt(EXP / cfg["meas"], delimiter=",", skiprows=1)
        ax.plot(m[:, 0], m[:, 1], ls="none", marker=cfg["mk"], ms=5.0,
                color=cfg["colour"], mfc="white", mew=1.4, zorder=5,
                label=f"{u} measured")
        ax.plot(d, V, ls="-", color=cfg["colour"], lw=2.2, zorder=3,
                label=f"{u} reference")
        rows.append((u, cfg["rho"], float(m[:, 1].max()),
                     float(r["peak"]["V_kN"])))

    dp, Vp = pinn_curve()
    ax.plot(dp, Vp, **F.style('anchored', marker='none', label=False),
            lw=2.0, zorder=4, label="VK1 arc-length PINN")

    ax.set_ylabel(r"horizontal force $V$  (kN)")
    ax.set_xlabel(r"deflection at $h_{\mathrm{eff}}$  (mm)")
    ax.set_xlim(0, 65)
    ax.set_ylim(0, 1000)
    F.clean(ax)
    ax.legend(loc="lower right", fontsize=F.FS_ANNOT, ncol=2,
              columnspacing=1.0)

    rho = np.array([r[1] for r in rows])
    meas = np.array([r[2] for r in rows])
    mod = np.array([r[3] for r in rows])
    ax_s.plot(rho, meas, "-o", color=F.ORANGE, lw=2.0, ms=6.5,
              mec="white", mew=0.7, label="measured")
    ax_s.plot(rho, mod, "-s", color=F.BLACK, lw=2.0, ms=6.0,
              mec="white", mew=0.7, label="reference")
    for x, a, b in zip(rho, meas, mod):
        ax_s.annotate("", xy=(x, b), xytext=(x, a),
                      arrowprops=dict(arrowstyle="<->", color="0.6",
                                      lw=0.9))
        ax_s.text(x + 0.018, 0.5 * (a + b), f"{(b/a-1)*100:+.0f} %",
                  fontsize=F.FS_ANNOT, color="0.4", va="center")
    d_meas = (meas[1] / meas[0] - 1) * 100
    d_mod = (mod[1] / mod[0] - 1) * 100
    ax_s.text(0.97, 0.10,
              f"slope: measured {d_meas:+.1f} %,\nreference {d_mod:+.1f} %",
              transform=ax_s.transAxes, ha="right", va="bottom",
              fontsize=F.FS_ANNOT, color="0.35")
    ax_s.set_xlabel(r"longitudinal reinforcement $\rho_\ell$  (%)")
    ax_s.set_ylabel(r"peak $V$  (kN)")
    ax_s.set_xlim(0.70, 1.42)
    ax_s.set_ylim(520, 990)
    F.clean(ax_s)
    ax_s.legend(loc="upper left", fontsize=F.FS_ANNOT)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.canvas.draw()
    bb, bbs = ax.get_position(), ax_s.get_position()
    F.fit_schematic(fig, ax_g, bbs.y0, bb.y1)
    F.fig_panel(fig, ax_g, 'a', 'wall pier geometry')
    F.fig_panel(fig, ax, 'b', 'response at both reinforcement ratios')
    y_c = bbs.y1 + 0.022          # just above panel c, clear of b's label
    fig.text(bbs.x0 - 0.006, y_c, 'c', fontsize=F.FS_PANEL,
             fontweight='bold', va='bottom', ha='left')
    fig.text(bbs.x0 - 0.006 + 0.034, y_c, 'level versus sensitivity',
             fontsize=F.FS_LABEL, fontweight='bold', va='bottom',
             ha='left', color='0.15')
    out = FIGDIR / "vk_combined.png"
    F.save(fig, out)
    plt.close(fig)

    print(f"{'unit':6}{'rho_l':>7}{'measured':>10}{'model':>8}{'err':>8}")
    for u, r, a, b in rows:
        print(f"{u:6}{r:7.2f}{a:10.0f}{b:8.0f}{(b/a-1)*100:+7.1f}%")
    print(f"sensitivity: measured {d_meas:+.1f} %, reference {d_mod:+.1f} %")


if __name__ == "__main__":
    main()
