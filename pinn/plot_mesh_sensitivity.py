"""Publication-quality mesh-sensitivity figure from `mesh_sensitivity.json`.

Three panels in one figure:
  (a) lambda-delta envelopes for all mesh resolutions, with peak stars and
      a shaded ribbon between the coarsest and finest envelopes;
  (b) bar chart of peak load factor vs element size h, showing that
      peak lambda is approximately mesh-independent;
  (c) bar chart of peak deflection vs element size h, showing the
      localisation-driven shift in the predicted peak location.
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

# resolution family: colour + dash + marker, greyscale-safe like the
# entity registry
MESH_STYLES = [
    dict(color=F.BLACK, ls='-', marker='o'),
    dict(color=F.VERM, ls=(0, (4, 2)), marker='s'),
    dict(color=F.SKY, ls=(0, (1.4, 1.3)), marker='^'),
    dict(color=F.GREEN, ls=(0, (6, 2, 1.4, 2)), marker='D'),
]

HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / "figures"
JSON_PATH = HERE.parent / "oracle" / "mesh_sensitivity.json"


def envelope(lams):
    """Monotone-decreasing envelope past peak (smooths basin-jump noise)."""
    out = list(lams)
    i_peak = int(np.argmax(out))
    for i in range(i_peak + 1, len(out)):
        out[i] = min(out[i], out[i - 1])
    return np.array(out)


def main() -> None:
    data = json.load(open(JSON_PATH))["resolutions"]

    fig = plt.figure(figsize=(F.FIG_W, 5.0), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.10, w_pad=0.04,
                                    hspace=0.14, wspace=0.08)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.7, 1.0])

    # ---- (a) lambda-delta envelopes --------------------------------------
    ax = fig.add_subplot(gs[0, :])
    envs = []
    for r, st in zip(data, MESH_STYLES):
        d = np.array(r["delta"])
        l = np.array(r["lam"])
        env = envelope(l)
        envs.append((d, env))
        ax.plot(d, env, color=st['color'], ls=st['ls'], marker='none',
                lw=1.8,
                label=f"{r['nx']}×{r['ny']}, h = {r['h_mm']:.0f} mm")
        ax.plot(r["peak_delta"], r["peak_lam"], "*", color=st['color'],
                ms=11, mec="white", mew=0.5, zorder=5)

    # shaded ribbon between coarsest and finest envelopes (common grid)
    d_common = np.linspace(0.5, max(d.max() for d, _ in envs), 80)
    coarsest = np.interp(d_common, envs[0][0], envs[0][1])
    finest = np.interp(d_common, envs[-1][0], envs[-1][1])
    lo = np.minimum(coarsest, finest)
    hi = np.maximum(coarsest, finest)
    ax.fill_between(d_common, lo, hi, alpha=0.12, color="0.35", lw=0,
                    label="coarsest to finest spread")

    ax.set_xlabel(r"loaded-patch deflection $\delta$  (mm)")
    ax.set_ylabel(r"load factor $\lambda$")
    # follow the data: the equilibrium-converged traces run to the
    # family window rather than the secant sweep's 20 mm
    ax.set_xlim(0, max(d.max() for d, _ in envs) * 1.05)
    ax.set_ylim(bottom=0.0)
    F.clean(ax)
    # curves plateau at the top, so the lower right corner is free
    ax.legend(loc="lower right", ncol=1, fontsize=F.FS_ANNOT)
    F.panel(ax, 'a', 'the envelopes nearly coincide')

    # ---- (b) peak lambda vs h --------------------------------------------
    ax_b = fig.add_subplot(gs[1, 0])
    hs = [r["h_mm"] for r in data]
    peak_l = [r["peak_lam"] for r in data]
    cols = [st['color'] for st in MESH_STYLES]
    bars_b = ax_b.bar(range(len(data)), peak_l, color=cols, alpha=0.85,
                      edgecolor="black", lw=0.5)
    ax_b.set_xticks(range(len(data)))
    ax_b.set_xticklabels([f"{h:.0f}" for h in hs])
    ax_b.set_xlabel("element size h  (mm)")
    ax_b.set_ylabel(r"peak $\lambda$")
    ax_b.set_ylim(0.9 * min(peak_l), 1.18 * max(peak_l))
    for bar, v in zip(bars_b, peak_l):
        ax_b.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                  f"{v:.2f}", ha="center", va="bottom",
                  fontsize=F.FS_ANNOT)
    F.clean(ax_b)
    F.panel(ax_b, 'b', 'capacity is mesh-stable')

    # ---- (c) peak delta vs h ---------------------------------------------
    ax_c = fig.add_subplot(gs[1, 1])
    peak_d = [r["peak_delta"] for r in data]
    bars_c = ax_c.bar(range(len(data)), peak_d, color=cols, alpha=0.85,
                      edgecolor="black", lw=0.5)
    ax_c.set_xticks(range(len(data)))
    ax_c.set_xticklabels([f"{h:.0f}" for h in hs])
    ax_c.set_xlabel("element size h  (mm)")
    ax_c.set_ylabel(r"peak $\delta$  (mm)")
    ax_c.set_ylim(0, 1.22 * max(peak_d))
    for bar, v in zip(bars_c, peak_d):
        ax_c.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                  f"{v:.1f}", ha="center", va="bottom",
                  fontsize=F.FS_ANNOT)
    F.clean(ax_c)
    F.panel(ax_c, 'c', 'the fold location shifts')

    out = FIGDIR / "mesh_sensitivity.png"
    F.save(fig, out)
    plt.close(fig)


if __name__ == "__main__":
    main()
