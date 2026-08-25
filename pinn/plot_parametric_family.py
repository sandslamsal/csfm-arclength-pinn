"""Centrepiece figure for the parametric deterioration family.

Panel (a): the PINN's equilibrium paths for the whole tie-loss family
against the displacement-controlled reference envelopes, coloured by
section loss. Held-out levels (5 % and 25 %), which no training term
ever saw, are drawn with open markers so the interpolation test is
visible rather than asserted.

Panel (b): capacity and fold location against section loss, which is
the engineering read of the family: the peak load factor plateaus while
the deflection at which it is reached migrates outward.

Reads : runs/parametric_rho/parametric_curves_anchored.json
        oracle/deepbeam_rho_family_clean.json
Writes: ../figures/parametric_family.png (+ .pdf)
"""
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
PINN_JSON = HERE / "runs" / "parametric_rho" / "parametric_curves_anchored_newton.json"
REF_JSON = HERE.parent / "oracle" / "deepbeam_family_newton_plotschema.json"
HELD_OUT = (0.05, 0.25)


def pinn_peak(delta, lam, d_min=2.0):
    """First crest of the network's curve, matching the rule used on the
    reference family (oracle/analyze_rho_family.py)."""
    d, l = np.asarray(delta), np.asarray(lam)
    i = int(np.argmax(np.where(d >= d_min, l, -np.inf)))
    return float(l[i]), float(d[i])


def main() -> None:
    pin = json.loads(PINN_JSON.read_text())
    ref = json.loads(REF_JSON.read_text())
    ref_by_loss = {round(c["loss"], 2): c for c in ref["curves"]}

    fig = plt.figure(figsize=(F.FIG_W, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.32)
    ax = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1])

    rows = []
    for key, cur in sorted(pin["curves"].items(), key=lambda kv: float(kv[0])):
        loss = float(key)
        col = F.family_color(loss)
        held = any(abs(loss - h) < 1e-9 for h in HELD_OUT)
        r = ref_by_loss.get(round(loss, 2))
        if r is not None:
            ax.plot(r["delta"], r["lam_env"], color=col, lw=1.1,
                    alpha=0.55, zorder=2)
        ax.plot(cur["delta"], cur["lam"], color=col,
                ls=(0, (5, 2)) if held else "-", lw=2.0, zorder=3)

        lp, dp = pinn_peak(cur["delta"], cur["lam"])
        if r is not None:
            rows.append((loss, held, lp, dp, r["peak_lam"], r["peak_delta"]))
            mk = dict(mfc="white", mec=col, mew=1.6) if held \
                else dict(color=col, mec="white", mew=0.6)
            ax_p.plot(loss * 100, lp, "o", ms=7, zorder=3, **mk)
            ax_p.plot(loss * 100, r["peak_lam"], "_", ms=11,
                      color=col, mew=2.0, zorder=2)

    ax.set_xlabel(r"midspan deflection $\delta$  (mm)")
    ax.set_ylabel(r"load factor $\lambda$")
    # The equilibrium-converged reference spans 7 mm, and the anchor is
    # masked beyond it, so the network's sweep continues unconstrained
    # past that point. Plot only the range the reference supports rather
    # than showing an extrapolation the fold test shows is unreliable.
    ax.set_xlim(0, max(np.max(r["delta"]) for r in ref["curves"]) * 1.05)
    ax.set_ylim(bottom=0)
    F.clean(ax)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color="0.35", lw=2.0, label="PINN, trained level"),
        Line2D([], [], color="0.35", lw=2.0, ls=(0, (5, 2)),
               label="PINN, held-out level"),
        Line2D([], [], color="0.55", lw=1.1, label="reference envelope"),
    ], loc="lower right", fontsize=F.FS_ANNOT)

    ax_p.set_xlabel("tie section loss  (%)")
    ax_p.set_ylabel(r"peak load factor $\lambda_{\rm peak}$")
    F.clean(ax_p)
    ax_p.legend(handles=[
        Line2D([], [], ls="none", marker="o", color="0.35",
               mec="white", ms=7, label="PINN"),
        Line2D([], [], ls="none", marker="_", color="0.35", mew=2.0,
               ms=11, label="reference"),
        Line2D([], [], ls="none", marker="o", mfc="white", mec="0.35",
               mew=1.6, ms=7, label="held out"),
    ], loc="best", fontsize=F.FS_ANNOT)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    F.fig_panel(fig, ax, 'a', 'one network, the whole family')
    F.fig_panel(fig, ax_p, 'b', 'capacity against section loss')
    out = FIGDIR / "parametric_family.png"
    F.save(fig, out)
    plt.close(fig)

    print(f"\n{'loss':>5} {'held':>5} {'PINN pk':>8} {'@mm':>6} "
          f"{'ref pk':>7} {'@mm':>6} {'err%':>6}")
    for loss, held, lp, dp, rl, rd in rows:
        print(f"{loss:5.2f} {str(held):>5} {lp:8.3f} {dp:6.2f} "
              f"{rl:7.3f} {rd:6.2f} {(lp - rl) / rl * 100:+6.1f}")


if __name__ == "__main__":
    main()
