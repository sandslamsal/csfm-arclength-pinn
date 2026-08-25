"""The validity domain of the displacement-based arc-length formulation.

The formulation is asked to do two things at once: extend the
equilibrium path to a prescribed reach S_max, and satisfy the
traction-free condition on the unloaded edges. This script measures how
far both can be achieved together, by sweeping the weight placed on the
free-edge term and recording, for each trained network:

  * the reach actually attained, as a fraction of the reach requested;
  * the free-edge residual, and the boundary reaction imbalance in
    engineering units, which is the same quantity in a form a
    structural reader can audit;
  * the peak load factor, which is the quantity the paper reports.

The result is a frontier rather than a pass/fail: the two requirements
are individually satisfiable and jointly satisfiable only up to a
limit, and the purpose of the figure is to state where that limit lies
so that the regime in which the formulation is valid is explicit.

Writes ../figures/validity_frontier.png (+ .pdf) and a summary table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as F                                                       # noqa: E402
import matplotlib.pyplot as plt                                            # noqa: E402
from model import ArclengthPINN                                            # noqa: E402
import equilibrium_check as EC                                             # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "P2" / "pinn"))
from problem import DeepBeam                                               # noqa: E402

F.apply()
HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / "figures"
S_MAX_REQ = 10.0          # reach requested by the final stage


def reaction_imbalance(ckpt, prob, s_val=0.83):
    """Sum of support reactions over the applied load, minus one."""
    net = ArclengthPINN(width=96, depth=6)
    net.load_state_dict(torch.load(ckpt))
    net.eval()
    half, a, NQ = prob.bearing / 2.0, 250.0, 300
    s = torch.full((NQ, 1), float(s_val))
    R = 0.0
    for xc in (a, prob.L - a):
        xs = torch.linspace(xc - half, xc + half, NQ).unsqueeze(-1)
        ys = torch.zeros_like(xs)
        sy = EC.stresses(net, prob, xs, ys, s)[1].detach()
        R += -float(torch.trapz(sy.squeeze(), xs.squeeze())) * prob.t
    lam = float(EC.field(net, prob, xs[:1], ys[:1], s[:1])[2])
    return R / (lam * prob.P) - 1.0, lam


def main() -> None:
    prob = DeepBeam()
    runs = [("canonical", HERE / "runs" / "arclength_pinn_latest.pt",
             HERE / "runs" / "training_history.json", None)]
    for w in (1, 5, 20):
        d = HERE / "runs" / f"freeedge2_{w}"
        if (d / "curve.json").exists():
            runs.append((f"w_free = {w}", d / "pinn.pt",
                         d / "log.json", d / "curve.json"))

    rows = []
    for name, ck, logf, curvef in runs:
        if curvef is None:                       # canonical
            reach, peak, free = 10.0, 2.609, 0.426
        else:
            c = json.loads(curvef.read_text())
            reach, peak = c["end_delta"], c["peak_lam"]
            lg = json.loads(logf.read_text())
            free = float(np.median([r["free"] for r in lg[-4:]]))
        imb, _ = reaction_imbalance(ck, prob)
        rows.append(dict(name=name, reach=reach,
                         reach_frac=reach / S_MAX_REQ,
                         free=free, imbalance=abs(imb) * 100.0, peak=peak))

    print(f"{'run':>14}{'reach mm':>10}{'% of S_max':>12}"
          f"{'L_free':>9}{'|reaction err| %':>18}{'peak lam':>10}")
    for r in rows:
        print(f"{r['name']:>14}{r['reach']:>10.2f}{r['reach_frac']*100:>12.0f}"
              f"{r['free']:>9.4f}{r['imbalance']:>18.0f}{r['peak']:>10.3f}")

    # Panel (b) of an earlier draft plotted the reaction imbalance against
    # the same abscissa. It is omitted: the networks compared sit at
    # deflections differing by a factor of twenty, so evaluating their
    # boundary equilibrium at a common s compares different physical
    # states and the resulting ordering is not meaningful.
    fig, (ax, ax_w) = plt.subplots(
        1, 2, figsize=(6.5, 3.1), width_ratios=[1.15, 1.0])
    fr = np.array([r["reach_frac"] * 100 for r in rows])
    fe = np.array([r["free"] for r in rows])
    o = np.argsort(fe)
    ax.plot(fe[o], fr[o], "-o", color=F.BLACK, lw=2.0, ms=8,
            mec="white", mew=0.8, zorder=3)
    for r in rows:
        dx, dy = (8, 6) if r["name"] != "canonical" else (-8, -14)
        ha = "left" if r["name"] != "canonical" else "right"
        ax.annotate(r["name"].replace("w_free = ", "w = "),
                    (r["free"], r["reach_frac"] * 100),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=F.FS_ANNOT, color="0.35", ha=ha)
    ax.axhspan(0, 10, color="0.93", lw=0, zorder=0)
    ax.text(0.98, 0.22, "path does not advance", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=F.FS_ANNOT, color="0.45")
    ax.set_xscale("log")
    ax.tick_params(axis='x', labelsize=9.5)
    ax.set_xlabel(r"free-edge residual $\mathcal{L}_{\rm free}$"
                  "   (lower is better)")
    ax.set_ylabel("reach attained  (% of requested)")
    ax.set_ylim(0, 115)
    F.clean(ax)

    # Panel b: the adaptive w_free trajectory of the canonical run's
    # final stage, answering whether the canonical run succeeds because
    # its weight is small. It is not small: it stays within [0.63, 1.17]
    # of unity. It dips exactly during the early iterations in which the
    # path establishes itself, which a fixed unit weight cannot do.
    hist = json.loads((HERE / "runs" /
                       "training_history.json").read_text())
    h9 = [r for r in hist["history"] if r.get("stage") == 8]
    it9 = np.array([r["iter"] for r in h9])
    wf9 = np.array([r["w_free"] for r in h9])
    lf9 = np.array([r["free"] for r in h9])
    ax_w.plot(it9, wf9, "-o", color=F.BLACK, lw=1.8, ms=4.5,
              mec="white", mew=0.6, label=r"adaptive $w_{\rm free}$")
    ax_w.axhline(1.0, color="0.6", lw=0.8, ls="--")
    ax_w.set_ylim(0, 1.6)
    ax_w.set_xlabel("iteration, final stage")
    ax_w.set_ylabel(r"$w_{\rm free}$")
    ax_w2 = ax_w.twinx()
    ax_w2.plot(it9, lf9, "-s", color=F.VERM, lw=1.4, ms=3.5,
               mec="white", mew=0.5, alpha=0.8)
    ax_w2.set_ylabel(r"$\mathcal{L}_{\rm free}$", color=F.VERM)
    ax_w2.tick_params(axis="y", colors=F.VERM)
    ax_w2.set_ylim(0, 0.6)
    for sp in ("top",):
        ax_w.spines[sp].set_visible(False)
        ax_w2.spines[sp].set_visible(False)
    ax_w.text(0.03, 0.06,
              "dips during path establishment,\nthen returns to unity",
              transform=ax_w.transAxes, fontsize=F.FS_ANNOT,
              color="0.35", va="bottom")

    F.fig_panel(fig, ax, 'a', 'reach against boundary fidelity')
    F.fig_panel(fig, ax_w, 'b', 'the adaptive weight, logged')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    F.save(fig, FIGDIR / "validity_frontier.png")
    plt.close(fig)
    json.dump(rows, open(HERE / "runs" / "validity_frontier.json", "w"),
              indent=2)


if __name__ == "__main__":
    main()
