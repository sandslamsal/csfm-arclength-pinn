"""Publication-quality corbel figure: geometry + anchored trace, one
two-panel figure (corbel_combined.png). The standalone pieces remain
available for slides."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as F                                                        # noqa: E402
import matplotlib.pyplot as plt                                             # noqa: E402

F.apply()

HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / "figures"
JSON_PATH = HERE.parent / "oracle" / "corbel_newton_plotschema.json"

L, H, T = 500.0, 400.0, 300.0
BEARING = 100.0
BAND = 80.0
NX, NY = 30, 24


def draw_corbel_geometry(ax):
    """Compact corbel schematic: mesh, tie band, clamped face, load,
    and the strut-and-tie load path. Numbers live in the caption."""
    dx, dy = L / NX, H / NY
    for i in range(NX + 1):
        ax.plot([i * dx, i * dx], [0, H], color="0.90", lw=0.3, zorder=0)
    for j in range(NY + 1):
        ax.plot([0, L], [j * dy, j * dy], color="0.90", lw=0.3, zorder=0)
    ax.plot([0, L, L, 0, 0], [0, 0, H, H, 0], color="black", lw=1.4)

    # top tie band
    ax.fill_between([0, L], [H - BAND, H - BAND], [H, H],
                    color=F.BLUE, alpha=0.16, lw=0)

    # clamped left face
    ax.add_patch(plt.Rectangle((-28, 0), 28, H, hatch="//",
                               facecolor="0.93", edgecolor="black",
                               lw=0.6))
    ax.text(-46, H / 2, "clamped", ha="right", va="center",
            fontsize=F.FS_ANNOT, color="0.35", rotation=90)

    # load patch + arrow
    half = BEARING / 2.0
    x_load = L - half
    ax.add_patch(plt.Rectangle((x_load - half, H), BEARING, 8,
                               color=F.VERM, alpha=0.9))
    arrow_len = 85
    ax.annotate("", xy=(x_load, H + 5),
                xytext=(x_load, H + 5 + arrow_len),
                arrowprops=dict(arrowstyle="-|>", color=F.VERM, lw=2.0))
    ax.text(x_load, H + 5 + arrow_len + 8, r"$\lambda \, P_{\rm ref}$",
            color=F.VERM, fontsize=F.FS_LABEL, fontweight="bold",
            ha="center", va="bottom")

    # strut-and-tie load path: inclined strut into the column base,
    # tie along the top band back to the column. Thin curved flow lines
    # fan around the strut axis: the bottle-shaped compression field.
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    p0, p1 = (x_load, H - BAND / 2), (30.0, 55.0)
    mx, my = 0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])
    import numpy as _np
    tvec = _np.array([p1[0] - p0[0], p1[1] - p0[1]])
    nvec = _np.array([-tvec[1], tvec[0]])
    nvec = nvec / _np.hypot(*nvec)
    for off in (-95.0, -50.0, 50.0, 95.0):
        ctrl = (mx + off * nvec[0], my + off * nvec[1])
        path = mpath.Path([p0, ctrl, p1],
                          [mpath.Path.MOVETO, mpath.Path.CURVE3,
                           mpath.Path.CURVE3])
        ax.add_patch(mpatches.PathPatch(
            path, fill=False, edgecolor="0.62", lw=0.8,
            ls=(0, (4, 2)), zorder=1))
    ax.plot([x_load, 30.0], [H - BAND / 2, 55.0], color="0.45", lw=8,
            alpha=0.35, solid_capstyle="round", zorder=1)
    ax.text(0.52 * (x_load + 30) + 25, 0.5 * (H - 25 + 55) - 12,
            "strut", color="0.35", fontsize=F.FS_ANNOT,
            rotation=np.degrees(np.arctan2(H - 80, x_load - 30)),
            ha="center", va="center")
    # truss nodes at the true intersections: the loaded node where the
    # tie meets the strut head under the bearing, the back node at the
    # clamped face, and the base node at the strut foot. The tie is
    # drawn between the node centres so no chord end pokes out.
    n_load = (x_load, H - BAND / 2)
    n_back = (18.0, H - BAND / 2)
    n_foot = (30.0, 55.0)
    ax.plot([n_back[0], n_load[0]], [n_back[1], n_load[1]],
            color=F.BLUE, lw=3.0, zorder=2, solid_capstyle="butt")
    for cx, cy in (n_load, n_back, n_foot):
        ax.add_patch(plt.Circle((cx, cy), 14, facecolor="white",
                                edgecolor="black", lw=1.2, zorder=4))

    # coloured callouts, reference-paper style
    ax.annotate("top tie band",
                xy=(120, H - 18), xytext=(-95, H + 118),
                color=F.BLUE, fontsize=F.FS_LABEL, ha="left",
                va="bottom",
                arrowprops=dict(arrowstyle="->", color=F.BLUE, lw=1.2,
                                connectionstyle="arc3,rad=-0.25"))
    ax.annotate("bottle-shaped field",
                xy=(mx + 100 * nvec[0] + 40, my + 100 * nvec[1] + 30),
                xytext=(322, 74), color="0.45",
                fontsize=F.FS_ANNOT, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color="0.55", lw=0.9,
                                connectionstyle="arc3,rad=0.18"))

    # dimensions
    ax.annotate("", xy=(L, -34), xytext=(0, -34),
                arrowprops=dict(arrowstyle="<->", color="0.45", lw=0.9))
    ax.text(L / 2, -48, rf"$L = {int(L)}$ mm", ha="center", va="top",
            fontsize=F.FS_ANNOT)
    ax.annotate("", xy=(L + 28, H), xytext=(L + 28, 0),
                arrowprops=dict(arrowstyle="<->", color="0.45", lw=0.9))
    ax.text(L + 42, H / 2, rf"$H = {int(H)}$ mm", ha="left",
            va="center", rotation=90, fontsize=F.FS_ANNOT)

    ax.set_xlim(-120, L + 115)
    ax.set_ylim(-95, H + arrow_len + 85)
    ax.set_aspect("equal")
    ax.axis("off")


def envelope(lams):
    out = list(lams)
    i_peak = int(np.argmax(out))
    for i in range(i_peak + 1, len(out)):
        out[i] = min(out[i], out[i - 1])
    return np.array(out)


def draw_corbel_overlay(ax):
    """Anchored PINN trace vs displacement-controlled reference."""
    import torch
    from model import ArclengthPINN
    sys.path.insert(0, str(HERE.parents[1] / "P2" / "pinn"))
    from problem import Corbel

    ckpt = HERE / "runs" / "corbel_anchor_newton" / "pinn.pt"
    U0 = 1.0e-3
    prob = Corbel()
    net = ArclengthPINN(width=96, depth=6)
    net.load_state_dict(torch.load(ckpt))
    net.eval()
    with torch.no_grad():
        n = 400
        s = torch.linspace(0.0, 1.0, n).unsqueeze(-1)
        x = torch.full_like(s, prob.x_load)
        y = torch.full_like(s, prob.H)
        xy_n = torch.cat([x / prob.L, y / prob.H], dim=-1)
        out, lam = net(xy_n, s)
        uy = out[:, 1:2] * U0 * prob.H
        delta_p = (-uy).squeeze().numpy()
        lam_p = lam.squeeze().numpy()
    i_peak = int(np.argmax(lam_p))

    ref = json.load(open(JSON_PATH))
    ref_delta = np.array([p["delta"] for p in ref["curve"]])
    ref_lam = np.array([p["lam"] for p in ref["curve"]])
    ref_env = envelope(ref_lam)
    pk = ref["peak"]
    csfm_at_pinn_peak = float(np.interp(delta_p[i_peak], ref_delta,
                                        ref_env))
    in_window_err = (lam_p[i_peak] - csfm_at_pinn_peak) \
        / csfm_at_pinn_peak * 100.0

    ax.plot(ref_delta, ref_lam, ls='none', marker='.', color='0.72',
            ms=3.0, label="reference, raw steps")
    ax.plot(ref_delta, ref_env,
            **F.style('reference', marker='none', label=False), lw=2.4,
            label="reference, envelope")
    ax.plot(pk["delta"], pk["lam"], "*", color=F.BLACK, ms=11,
            mec="white", mew=0.6, zorder=6)
    ax.plot(delta_p, lam_p,
            **F.style('anchored', marker='none', label=False), lw=2.4,
            label=r"arc-length PINN, $\lambda$-anchored")
    ax.plot(delta_p[i_peak], lam_p[i_peak], "*", color=F.SKY, ms=11,
            mec="white", mew=0.6, zorder=6)

    ax.text(0.030, 0.985,
            rf"PINN peak $\lambda = {lam_p[i_peak]:.2f}$"
            rf" at $\delta = {delta_p[i_peak]:.1f}$ mm" + "\n"
            rf"reference there: {csfm_at_pinn_peak:.2f}"
            rf" ({in_window_err:+.1f} %)" + "\n"
            rf"reference peak {pk['lam']:.2f} at {pk['delta']:.1f} mm" + "\n"
            rf"steep branch: $\leq 0.30$ mm apart in $\delta$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=F.FS_ANNOT, color="0.32", linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="0.85",
                      lw=0.6), zorder=8)

    # Beyond the reference's converged range the comparison has nothing
    # to compare against, so the panel says so rather than leaving the
    # PINN trace to run on alone.
    d_end = float(ref_delta.max())
    ax.axvspan(d_end, max(delta_p.max(), ref_delta.max()) * 1.02,
               color="0.93", lw=0, zorder=0)
    ax.text(d_end + 0.72, 0.675, "reference ends",
            transform=ax.get_xaxis_transform(), fontsize=F.FS_SMALL,
            color="0.5", ha="center", va="bottom", rotation=90)

    ax.set_ylabel(r"load factor $\lambda$")
    ax.set_xlim(0, max(delta_p.max(), ref_delta.max()) * 1.02)
    ax.set_ylim(0, max(lam_p.max(), ref_lam.max()) * 1.34)
    F.clean(ax)
    ax.legend(loc="lower right", fontsize=F.FS_ANNOT, frameon=False,
              handlelength=1.9, labelspacing=0.30, borderaxespad=0.4)

    # Inset: the trace against the reference in per cent. At full scale
    # the two curves lie within a stroke width of each other over most
    # of the window, so the overlay alone cannot show where they part.
    axi = ax.inset_axes([0.455, 0.335, 0.510, 0.300])
    m = (delta_p >= float(ref_delta.min())) & (delta_p <= d_end)
    dev = (lam_p[m] / np.interp(delta_p[m], ref_delta, ref_env) - 1) * 100.0
    axi.axhspan(-5, 5, color="0.88", lw=0, zorder=0)
    axi.axhline(0, color="0.45", lw=0.8, zorder=1)
    axi.plot(delta_p[m], dev, color=F.SKY, lw=1.6, zorder=3)
    axi.set_xlim(0, d_end)
    axi.set_ylim(-8, 8)
    axi.set_yticks([-5, 0, 5])
    axi.set_xticks([0, 2, 4, 6, 8])
    axi.tick_params(labelsize=F.FS_SMALL, length=2.2, pad=1.5)
    for sp in ("top", "right"):
        axi.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axi.spines[sp].set_linewidth(0.7)
    axi.text(0.97, 0.97, "PINN minus reference  (%)",
             transform=axi.transAxes, fontsize=F.FS_SMALL, color="0.35",
             ha="right", va="top")
    # Below about 1 mm the branch is near-vertical, so a difference read
    # off at fixed deflection is dominated by a horizontal offset and
    # leaves the frame. The offset itself is the meaningful measure
    # there, and it is quoted rather than plotted.
    axi.annotate("", xy=(0.055, 1.02), xytext=(0.055, 0.86),
                 xycoords="axes fraction", textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color=F.SKY, lw=1.2))

    axi.set_facecolor("white")
    axi.patch.set_alpha(1.0)
    axi.set_zorder(6)

    # The numbers the text quotes, written from the same arrays the
    # figure is drawn from, so the two cannot drift apart. The path
    # error is a root-mean-square in lambda, sampled on the network's
    # own trace over the reference's converged span.
    m_all = (delta_p >= float(ref_delta.min())) & (delta_p <= d_end)
    rms = float(np.sqrt(np.mean(
        (lam_p[m_all] - np.interp(delta_p[m_all], ref_delta, ref_env)) ** 2)))
    json.dump({"pinn_peak_lam": float(lam_p[i_peak]),
               "pinn_peak_delta": float(delta_p[i_peak]),
               "ref_at_pinn_peak": csfm_at_pinn_peak,
               "err_at_pinn_peak_pct": in_window_err,
               "ref_peak_lam": float(pk["lam"]),
               "ref_peak_delta": float(pk["delta"]),
               "err_vs_ref_peak_pct":
                   (float(lam_p[i_peak]) / float(pk["lam"]) - 1) * 100.0,
               "path_rmse": rms,
               "path_rmse_pct_of_peak": rms / float(pk["lam"]) * 100.0},
              open(HERE.parent / "oracle" / "corbel_comparison.json", "w"),
              indent=1)
    return delta_p, lam_p, ref_delta, ref_env


def fig_corbel_combined():
    fig = plt.figure(figsize=(F.FIG_W, 3.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 1.05], wspace=0.30)
    ax_g = fig.add_subplot(gs[0])
    ax_c = fig.add_subplot(gs[1])
    draw_corbel_geometry(ax_g)
    draw_corbel_overlay(ax_c)
    ax_c.set_xlabel(r"loaded-patch deflection $\delta$  (mm)")

    # lay out the data panel first, then place the schematic exactly
    # against it: its box spans the data panel's height, so every edge
    # in the figure aligns with another edge
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.canvas.draw()
    bb_c = ax_c.get_position()
    F.fit_schematic(fig, ax_g, bb_c.y0 + 0.185, bb_c.y1, x0=0.030)
    # The schematic's width follows its aspect, so the gap to the data
    # panel is whatever is left over. Set it instead of accepting it.
    GAP = 0.115
    ga = ax_g.get_position()
    x_new = ga.x0 + ga.width + GAP
    q = ax_c.get_position()
    ax_c.set_position((x_new, q.y0, q.x1 - x_new, q.height))
    fig.canvas.draw()
    F.fig_panel(fig, ax_g, 'a', 'the corbel and its load path')
    F.fig_panel(fig, ax_c, 'b', 'anchored trace vs reference')
    gb = ax_g.get_position()
    fig.text(gb.x0 + gb.width / 2, gb.y0 - 0.045,
             "section $500 \\times 400 \\times 300$ mm,  "
             "$f_c = 30$ MPa,  $f_y = 500$ MPa\n"
             "top tie band $\\rho_x = 1.2$ % over 80 mm,  "
             "stirrups $\\rho_y = 0.25$ %",
             fontsize=F.FS_SMALL, color="0.30", ha="center", va="top",
             linespacing=1.5)
    out = FIGDIR / "corbel_combined.png"
    F.save(fig, out)
    plt.close(fig)


def main() -> None:
    fig_corbel_combined()


if __name__ == "__main__":
    main()
