"""Generate publication-quality PNG figures for the P3 manuscript.

Outputs (all in ../figures/, all PNG at 300 dpi):
  geometry.png            beam geometry, reinforcement layout, mesh, BCs
  parametrisation.png      delta(s) and lambda(s) showing arc-length tracing
  equilibrium_path.png    PINN curve vs displacement-controlled reference
  speed_profile.png        |du/ds|^2 across s (Cauchy-Schwarz health)
  loss_history.png         per-stage loss components from training history
  stress_field.png         sigma field at pre-peak / peak / post-peak s values
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ArclengthPINN                                             # noqa: E402
from pretrain_elastic import elastic_fe                                     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "P2" / "pinn"))
from problem import DeepBeam                                                # noqa: E402

import figstyle as F                                                        # noqa: E402

# House figure style shared with the rest of the series (Arial at true
# printed size, Okabe-Ito entity colours, despined axes, bold headings).
F.apply()

U0 = 1.0e-3

# Legacy aliases kept for the functions not yet migrated (stress_field,
# matplotlib architecture); every included figure uses figstyle entities.
CSFD_BLUE = F.BLUE
CSFD_RED = F.VERM
CSFD_GREY = "0.45"

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
FIGDIR = HERE.parent / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def load_pinn():
    net = ArclengthPINN(width=96, depth=6)
    # Canonical-seed (20260522) network reported throughout the text: this
    # checkpoint produces peak lambda = 2.61 at delta = 8.3 mm, end (2.53, 10.0),
    # matching runs/seed_sweep_both/seed20260522.json. (NB: the older
    # validation_metrics.json on disk still lists a superseded 2.378@1.99 net;
    # it is stale and should be regenerated from this checkpoint.)
    net.load_state_dict(torch.load(RUNS / "arclength_pinn_latest.pt"))
    net.eval()
    return net


def pinn_curve_with_grads(net: ArclengthPINN, prob: DeepBeam, n: int = 200):
    """Return (s, delta, lam, dux_ds, duy_ds) sampled on the loaded-patch
    centre. delta = -uy. Computed WITH autograd so we can pull du/ds."""
    s = torch.linspace(0.0, 1.0, n).unsqueeze(-1).requires_grad_(True)
    x = torch.full_like(s, prob.x_load)
    y = torch.full_like(s, prob.H)
    xy_n = torch.cat([x / prob.L, y / prob.H], dim=-1)
    out, lam = net(xy_n, s)
    ux = out[:, 0:1] * U0 * prob.L
    uy = out[:, 1:2] * U0 * prob.H
    dux = torch.autograd.grad(ux, s, grad_outputs=torch.ones_like(ux),
                              create_graph=True)[0]
    duy = torch.autograd.grad(uy, s, grad_outputs=torch.ones_like(uy),
                              create_graph=True)[0]
    return (s.detach().numpy().squeeze(),
            (-uy).detach().numpy().squeeze(),
            lam.detach().numpy().squeeze(),
            dux.detach().numpy().squeeze(),
            duy.detach().numpy().squeeze())


# --------------------------------------------------------------------------- #
# Figure 1: equilibrium path overlay
# --------------------------------------------------------------------------- #


def fig_equilibrium_path():
    """The fold, against the equilibrium-converged reference.

    Rebuilt for the equilibrium-converged reference. The earlier version
    plotted the secant-Picard curve as THE reference and drew a "capacity
    bracket" between it and the consistent-tangent follower. That framing
    is withdrawn: the two solvers do not bracket a quantity, one satisfies
    equilibrium and the other does not. The two consistent-tangent solvers
    agree with each other to 1.4 % (1.3628 and 1.3818) while the secant
    sits 66 % above both, so they are drawn as reference and independent
    confirmation, and the secant curve is kept only as a faint superseded
    trace so the size of the error stays visible.
    """
    prob = DeepBeam()
    net = load_pinn()
    s, delta, lam, _, _ = pinn_curve_with_grads(net, prob, 400)
    i_peak = int(np.argmax(lam))

    # primary reference: equilibrium-converged Newton, intact level
    fam = json.load(open(HERE.parent / "oracle"
                         / "deepbeam_family_newton.json"))["curves"]["0.00"]
    ref_delta = np.array(fam["delta"])
    ref_lam = np.array(fam["lam"])
    ref_pk_lam, ref_pk_delta = fam["lam_max"], fam["delta_peak"]

    # independent consistent-tangent follower: corroboration, not a branch
    cris = json.load(open(HERE.parent / "oracle" / "deepbeam_crisfield.json"))
    cris_delta = np.array([p["delta"] for p in cris["curve"]])
    cris_lam = np.array([p["lam"] for p in cris["curve"]])

    # superseded secant reference, shown faintly
    sec = json.load(open(HERE.parent / "oracle" / "deepbeam_oracle.json"))
    sec_delta = np.array([p["delta"] for p in sec["curve"]])
    sec_lam = np.array([p["lam"] for p in sec["curve"]])

    xy_fe, u_fe, info = elastic_fe(prob, nx=40, ny=20)
    delta_fe = float(-u_fe[info["load_nodes"], 1].mean())
    slope_elastic = 1.0 / delta_fe

    fig, ax = plt.subplots(figsize=(F.FIG_W, 3.7))
    delta_el = np.linspace(0.0, delta.max() * 1.05, 40)
    ax.plot(delta_el, delta_el * slope_elastic,
            **F.style('elastic', marker='none', lw=1.4), alpha=0.8)

    # superseded secant curve, faint
    ax.plot(sec_delta, sec_lam, color='0.80', lw=1.4, ls=(0, (1, 2)),
            zorder=1, label='secant fixed point (superseded)')

    # the reference
    ax.plot(ref_delta, ref_lam,
            **F.style('reference', marker='none', label=False), lw=2.6,
            label='equilibrium-converged reference')
    ax.plot(ref_pk_delta, ref_pk_lam, "*", color=F.BLACK, ms=11,
            mec="white", mew=0.6, zorder=6)

    # independent confirmation
    ax.plot(cris_delta, cris_lam,
            **F.style('crackband', marker='none', label=False), lw=1.8,
            label='consistent-tangent follower (independent)')

    # the un-anchored network
    ax.plot(delta, lam, **F.style('pinn', marker='none', label=False),
            lw=2.6, label='arc-length PINN, un-anchored')
    ax.plot(delta[i_peak], lam[i_peak], "*", color=F.GREEN, ms=11,
            mec="white", mew=0.6, zorder=6)

    # the closed-form admissibility bound
    LAM_BOUND = 1.48
    ax.axhline(LAM_BOUND, color='0.45', lw=0.9, ls='--', alpha=0.9, zorder=2)
    ax.text(delta.max() * 0.99, LAM_BOUND * 1.02,
            r"closed-form bound, tie at yield  $\lambda = 1.48$",
            ha='right', va='bottom', fontsize=F.FS_ANNOT, color='0.45')

    # placed right of the elastic warm-start line, which is steep and
    # otherwise runs through a top-left note
    F.note(ax, 0.30, 0.97,
           rf"PINN peak $\lambda = {lam[i_peak]:.2f}$"
           rf" at $\delta = {delta[i_peak]:.1f}$ mm" + "\n"
           rf"reference peak $\lambda = {ref_pk_lam:.3f}$"
           rf" at $\delta = {ref_pk_delta:.2f}$ mm",
           ha="left", va="top", color="0.35")

    ax.set_xlabel(r"loaded-patch deflection $\delta$  (mm)")
    ax.set_ylabel(r"load factor $\lambda$")
    ax.set_xlim(0, delta.max() * 1.02)
    ax.set_ylim(0, max(lam.max(), sec_lam.max()) * 1.12)
    F.clean(ax)
    F.heading(ax, 'the fold, traced in one forward pass')
    F.place_legend(ax, fontsize=F.FS_LEGEND)
    out = FIGDIR / "equilibrium_path.png"
    F.save(fig, out)
    plt.close(fig)


def fig_speed_profile():
    prob = DeepBeam()
    net = load_pinn()
    s, delta, lam, dux, duy = pinn_curve_with_grads(net, prob, 200)
    speed2 = dux ** 2 + duy ** 2

    # Decompose speed into loaded-direction (-y) and transverse (x).
    # The reported deep-beam net uses the directional arc loss, which
    # constrains the loaded-direction (duy/ds)^2 to S_max^2; the loaded
    # direction sits on the band and the transverse ux drift is small and
    # unconstrained.
    S_max = 10.0   # canonical-seed loaded-direction arc-length target
    target2 = S_max ** 2
    speed2_y = duy ** 2     # loaded-direction (y) speed^2
    speed2_x = dux ** 2     # transverse (x) speed^2

    fig, ax = plt.subplots(figsize=(4.5, 3.1))
    ax.axhspan(target2 * 0.95, target2 * 1.05, color='0.92', lw=0,
               label="\u00b15 % band")
    ax.axhline(target2, color='0.55', ls=(0, (5, 2)), lw=1.2,
               label=f"target {int(target2)} mm\u00b2")
    ax.plot(s, speed2, color=F.BLACK, ls='-', lw=2.2,
            label="total speed")
    ax.plot(s, speed2_y, color=F.GREEN, ls=(0, (4, 2)), lw=2.2,
            label="loaded direction")
    ax.plot(s, speed2_x, color=F.VERM, ls=(0, (1.4, 1.3)), lw=2.2,
            label="transverse")
    ax.set_xlabel(r"arc-length coordinate $s$")
    ax.set_ylabel("squared speed  (mm\u00b2)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(speed2.max(), target2 * 1.20))
    F.clean(ax)
    F.heading(ax, 'the arc constraint holds')
    F.place_legend(ax, fontsize=F.FS_ANNOT)
    out = FIGDIR / "speed_profile.png"
    F.save(fig, out, target_w=4.5)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3: loss history across the 9 stages
# --------------------------------------------------------------------------- #


def fig_loss_history():
    """Loss history along the *effective* training trajectory {1,2,8,9}.

    The five rolled-back intermediate-alpha attempts (stages 3-7) are
    collapsed into a single shaded "bypassed" band -- they carry no
    information beyond non-convergence (Table 3) and otherwise dominate
    the axis with noise. The L-BFGS equilibrium-refinement phase is
    appended as a final shaded segment, showing the interior equilibrium
    residual drop ~7x. Key values are annotated."""
    data = json.load(open(RUNS / "training_history.json"))
    hist = data["history"]
    completed_idx = {c["stage"] for c in data["completed_stages"]
                     if c["completed"]}
    by_stage: dict[int, list[dict]] = {}
    for r in hist:
        by_stage.setdefault(r["stage"], []).append(r)
    stages = sorted(by_stage)
    a0 = {s for s in stages if abs(by_stage[s][0]["alpha"]) < 1e-9}
    elastic = [s for s in stages if s in completed_idx and s in a0]
    cracked = [s for s in stages if s in completed_idx and s not in a0]

    keys = ("eq", "supp", "load", "free", "arc")

    def seg(stage_list, x0):
        out = {k: [] for k in keys}
        out["x"] = []
        divs = []
        cum = x0
        for si in stage_list:
            rows = sorted(by_stage[si], key=lambda r: r["iter"])
            base = cum
            for r in rows:
                out["x"].append(base + r["iter"])
                for k in keys:
                    out[k].append(r[k])
            cum = base + max(r["iter"] for r in rows) + 1
            divs.append(cum)
        return out, cum, divs[:-1]   # internal dividers only

    el, x_el, el_divs = seg(elastic, 0.0)
    BAND = 0.18 * x_el
    cr, x_cr, cr_divs = seg(cracked, x_el + BAND)

    rm = json.load(open(RUNS / "refine_metrics.json"))
    traj = rm["trajectory"]
    GAP = 0.03 * x_cr
    rx0 = x_cr + GAP
    rmax = max(t["iter"] for t in traj) or 1
    RW = 0.32 * x_el
    r_x = [rx0 + t["iter"] / rmax * RW for t in traj]
    r_eq = [t["l_eq_holdout"] for t in traj]   # held-out (generalizable) residual
    x_end = rx0 + RW

    fig, (ax_phys, ax_arc) = plt.subplots(
        2, 1, figsize=(F.FIG_W, 4.7), sharex=True,
        gridspec_kw=dict(height_ratios=[2.0, 1.0], hspace=0.22))

    def gsmooth(y, w=7):
        """Geometric (log-space) centred moving average -- appropriate for
        log-scaled loss curves; tames the per-batch sampling noise without
        letting spikes dominate the trend."""
        y = np.clip(np.asarray(y, float), 1e-12, None)
        n = len(y)
        if n < 3:
            return y
        w = min(w, n if n % 2 else n - 1)
        w = max(3, w - 1 if w % 2 == 0 else w)
        lp = np.pad(np.log(y), w // 2, mode="edge")
        return np.exp(np.convolve(lp, np.ones(w) / w, mode="valid"))

    styles = [("eq", 'l_eq', "equilibrium"),
              ("supp", 'l_supp', "support"),
              ("load", 'l_load', "loaded patch"),
              ("free", 'l_free', "free edge")]
    for k, ent, lab in styles:
        col = F.color(ent)
        ls = F.ENTITY[ent]['ls']
        lw = 2.2 if k == "eq" else 1.5            # emphasise the hero curve
        z = 6 if k == "eq" else 4
        for j, sg in enumerate((el, cr)):
            ax_phys.semilogy(sg["x"], sg[k], "-", color=col, lw=0.5,
                             alpha=0.13)
            ax_phys.semilogy(sg["x"], gsmooth(sg[k]), ls=ls, color=col,
                             lw=lw, alpha=0.95, zorder=z,
                             label=lab if j == 0 else None)
    # refinement: equilibrium residual continuing to drop (markers)
    ax_phys.semilogy(r_x, r_eq, "o-", color=F.color('l_eq'), lw=1.2,
                     ms=2.8, mfc="white", mec=F.color('l_eq'), alpha=0.95)

    # phase shading on both panels: alternating neutral greys, darker for
    # the bypassed band, light tint for the refinement segment
    for ax in (ax_phys, ax_arc):
        ax.axvspan(x_el, x_el + BAND, color="0.85", lw=0, zorder=0)
        ax.axvspan(rx0 - GAP / 2, x_end, color="0.94", lw=0, zorder=0)
        for d in el_divs + cr_divs:
            ax.axvline(d, color="0.75", lw=0.4, ls=":")
        ax.axvline(x_el + BAND, color="0.6", lw=0.6)
        ax.axvline(rx0 - GAP / 2, color="0.6", lw=0.6)

    ax_phys.set_ylabel("physics loss")
    ax_phys.tick_params(axis='y', labelsize=9.5)
    ax_phys.legend(loc="upper right", ncol=2, fontsize=F.FS_ANNOT)
    ax_phys.set_ylim(min(min(cr["eq"]), min(r_eq)) * 0.3, 5e2)

    # annotate the equilibrium plateau and the refined value
    ax_phys.annotate("equilibrium plateaus at 0.55",
                     xy=(cr["x"][-1], cr["eq"][-1]),
                     xytext=(cr["x"][-1] - 0.42 * x_el, 8.0),
                     fontsize=F.FS_ANNOT, color="0.35",
                     arrowprops=dict(arrowstyle="->", color="0.45", lw=0.7))
    ax_phys.annotate("refined to 0.013, about 40x",
                     xy=(r_x[-1], r_eq[-1]),
                     xytext=(r_x[-1] - 0.40 * x_el, r_eq[-1] * 0.16),
                     fontsize=F.FS_ANNOT, color="0.35",
                     arrowprops=dict(arrowstyle="->", color="0.45", lw=0.7))

    for j, sg in enumerate((el, cr)):
        ax_arc.semilogy(sg["x"], sg["arc"], "-", color=F.color('l_arc'),
                        lw=0.5, alpha=0.15)
        ax_arc.semilogy(sg["x"], gsmooth(sg["arc"]), "-",
                        color=F.color('l_arc'), lw=1.6, alpha=0.95,
                        label=("arc length, fixed weight 20")
                        if j == 0 else None)
    ax_arc.set_ylabel("arc loss")
    ax_arc.tick_params(axis='y', labelsize=9.5)
    ax_arc.legend(loc="upper right", fontsize=F.FS_ANNOT)
    ax_arc.set_xlim(-0.01 * x_el, x_end + 0.01 * x_el)
    ax_arc.set_xticks([])
    ax_arc.set_xlabel("training progression  "
                      r"(Adam within each phase $\to$ L-BFGS refinement)")

    # phase names inside the bottom of the arc panel, out of the data's way
    def plabel(x, txt, rot=0):
        ax_arc.text(x, 0.06, txt, transform=ax_arc.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=F.FS_ANNOT,
                    color="0.35", rotation=rot)
    plabel(x_el / 2, r"elastic, $\alpha = 0$")
    plabel(x_el + BAND / 2, "bypassed", rot=90)
    plabel((x_el + BAND + x_cr) / 2, r"full CSFM, $\alpha = 1$")
    plabel((rx0 + x_end) / 2, "refine", rot=90)

    for ax in (ax_phys, ax_arc):
        F.clean(ax)
    F.panel(ax_phys, 'a', 'physics losses through the curriculum')
    F.panel(ax_arc, 'b', 'the arc constraint stays pinned')
    fig.tight_layout()
    out = FIGDIR / "loss_history.png"
    F.save(fig, out)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 4: parametrisation diagram - delta(s) and lambda(s) side by side
# --------------------------------------------------------------------------- #


def fig_parametrisation():
    prob = DeepBeam()
    net = load_pinn()
    s, delta, lam, _, _ = pinn_curve_with_grads(net, prob, 400)
    i_peak = int(np.argmax(lam))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(F.FIG_W, 2.7),
                                   sharex=True)
    ax1.plot(s, delta, color=F.GREEN, ls='-', lw=2.4)
    ax1.plot(s[i_peak], delta[i_peak], "*", color=F.GREEN, ms=11,
             mec="white", mew=0.6, zorder=6)
    ax1.axvline(s[i_peak], color='0.6', ls=(0, (1.4, 1.3)), lw=0.9)
    ax1.set_xlabel(r"arc-length coordinate $s$")
    ax1.set_ylabel(r"deflection $\delta$  (mm)")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, delta.max() * 1.08)
    ax1.text(s[i_peak] - 0.04, delta[i_peak] + delta.max() * 0.015,
             "limit point", color='0.35', fontsize=F.FS_ANNOT,
             va="center", ha="right")

    ax2.plot(s, lam, color=F.GREEN, ls='-', lw=2.4)
    ax2.plot(s[i_peak], lam[i_peak], "*", color=F.GREEN, ms=11,
             mec="white", mew=0.6, zorder=6)
    ax2.axvline(s[i_peak], color='0.6', ls=(0, (1.4, 1.3)), lw=0.9)
    ax2.set_xlabel(r"arc-length coordinate $s$")
    ax2.set_ylabel(r"load factor $\lambda$")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, lam.max() * 1.15)
    ax2.text(0.5, 0.06,
             rf"$d\lambda/ds = 0$ at $s = {s[i_peak]:.2f}$",
             color='0.35', fontsize=F.FS_ANNOT, va="bottom",
             ha="center", transform=ax2.transAxes)

    for ax in (ax1, ax2):
        F.clean(ax)
    F.panel(ax1, 'a', 'the deflection sweeps the window')
    F.panel(ax2, 'b', 'the load factor folds')
    fig.tight_layout()
    out = FIGDIR / "parametrisation.png"
    F.save(fig, out)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 5: beam geometry with mesh, reinforcement, BCs
# --------------------------------------------------------------------------- #


def fig_geometry():
    """Deep-beam geometry plot, matching the corbel-style polish:
    clean fine-grid background (no diagonal CST overlay), hatched
    support blocks instead of small pin/roller triangles detached
    from the beam, single arrow + textbox for the load."""
    prob = DeepBeam()
    L, H = prob.L, prob.H
    bearing = prob.bearing
    half = bearing / 2.0
    a = 250.0  # support inset (centre)
    nx, ny = 40, 20

    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.set_aspect("equal")

    # fine mesh grid (light) — matches corbel_geometry style
    dx, dy = L / nx, H / ny
    for i in range(nx + 1):
        ax.plot([i * dx, i * dx], [0, H], color="0.88", lw=0.3)
    for j in range(ny + 1):
        ax.plot([0, L], [j * dy, j * dy], color="0.88", lw=0.3)

    # outer beam rectangle
    ax.plot([0, L, L, 0, 0], [0, 0, H, H, 0], color="black", lw=1.4)

    # bottom tie band (the one highlighted entity, house blue)
    band_h = 150.0
    ax.fill_between([0, L], [0, 0], [band_h, band_h],
                    color=F.BLUE, alpha=0.18, lw=0)
    ax.text(L * 0.5, 38, r"bottom tie band, $\rho_x = 1.2$ %",
            ha="center", va="center", color=F.BLUE,
            fontsize=F.FS_LABEL, fontweight='bold')

    # strut-and-tie load path: two struts fan from the loaded patch to
    # the supports, the tie chord closes the truss along the soffit band
    for xs_ in (a, L - a):
        ax.plot([L / 2, xs_], [H - 45, 95], color="0.45", lw=8,
                alpha=0.32, solid_capstyle="round", zorder=1)
    # Label the left strut along its own axis. The axes are equal-aspect,
    # so the data-space angle is the display angle; the sign must follow the
    # strut, which rises from the support towards the loaded patch.
    ang = float(np.degrees(np.arctan2((H - 45) - 95, (L / 2) - a)))
    xm, ym = 0.5 * (a + L / 2), 0.5 * (95 + (H - 45))
    nx_off, ny_off = -np.sin(np.radians(ang)), np.cos(np.radians(ang))
    ax.text(xm + 78 * nx_off, ym + 78 * ny_off, "strut",
            color="0.35", fontsize=F.FS_ANNOT, rotation=ang,
            rotation_mode="anchor", ha="center", va="center")
    ax.plot([a, L - a], [95, 95], color=F.BLUE, lw=2.6, zorder=2)
    ax.text(L / 2 + 230, 122, "tie", color=F.BLUE,
            fontsize=F.FS_ANNOT, fontweight="bold", ha="center")
    for nx_, ny_ in ((a, 95), (L - a, 95), (L / 2, H - 45)):
        ax.add_patch(plt.Circle((nx_, ny_), 16, facecolor="white",
                                edgecolor="black", lw=1.0, zorder=3))

    # supports: triangle markers (pin = filled triangle, roller = pin
    # + open circle on top), with a bearing strip on the beam soffit.
    # This restores the earlier symbol convention; the hatched-block
    # form was harder to read at a glance.
    sup_size = 60
    sup_y = -10
    sup_h_tri = 60
    for xs, kind in [(a, "pin"), (L - a, "roller")]:
        # bearing strip (the bottom of the beam reacts on the support)
        ax.add_patch(plt.Rectangle((xs - half, sup_y), bearing, 10,
                                   color="black", alpha=0.85))
        # filled triangle support
        tri = plt.Polygon([[xs, sup_y],
                           [xs - sup_size / 2, sup_y - sup_h_tri],
                           [xs + sup_size / 2, sup_y - sup_h_tri]],
                          closed=True, color="black", alpha=0.85)
        ax.add_patch(tri)
        if kind == "pin":
            ax.text(xs - 55, sup_y - sup_h_tri / 2, "pin",
                    ha="right", va="center", fontsize=F.FS_ANNOT,
                    color="0.35")
        else:
            # roller: pin triangle + open circle (the rollable contact)
            ax.add_patch(plt.Circle(
                (xs, sup_y - sup_h_tri - 10), 8,
                facecolor="white", edgecolor="black", lw=0.8, zorder=3))
            ax.text(xs + 55, sup_y - sup_h_tri / 2, "roller",
                    ha="left", va="center", fontsize=F.FS_ANNOT,
                    color="0.35")
    # update support height bookkeeping for the dimension-arrow placement
    sup_h = sup_h_tri + 28

    # load patch + arrow (vermilion: the action)
    ax.add_patch(plt.Rectangle((L / 2 - half, H), bearing, 8,
                               color=F.VERM, alpha=0.9))
    arr_len = 90
    ax.annotate("", xy=(L / 2, H + 5),
                xytext=(L / 2, H + 5 + arr_len),
                arrowprops=dict(arrowstyle="-|>",
                                color=F.VERM, lw=2.0))
    ax.text(L / 2 + 50, H + 5 + arr_len / 2,
            r"$\lambda \, P_{\rm ref}$",
            color=F.VERM, fontsize=F.FS_LABEL, fontweight='bold',
            va="center")
    ax.text(L / 2, H + 5 + arr_len + 18,
            rf"$P_{{\rm ref}} = {int(prob.P / 1000)}$ kN, "
            rf"bearing {int(bearing)} mm",
            ha="center", va="bottom", color=F.VERM,
            fontsize=F.FS_LABEL)

    # dimension labels (outside the geometry, grey like the reference)
    ax.annotate("", xy=(L, -sup_h - 50), xytext=(0, -sup_h - 50),
                arrowprops=dict(arrowstyle="<->", color="0.45", lw=0.9))
    ax.text(L / 2, -sup_h - 65, rf"$L = {int(L)}$ mm",
            ha="center", va="top", fontsize=F.FS_ANNOT)
    ax.annotate("", xy=(L + 60, H), xytext=(L + 60, 0),
                arrowprops=dict(arrowstyle="<->", color="0.45", lw=0.9))
    ax.text(L + 78, H / 2, rf"$H = {int(H)}$ mm",
            ha="left", va="center", fontsize=F.FS_ANNOT, rotation=90)

    # mesh + thickness annotations
    ax.text(L / 2, 265,
            rf"CST mesh  $n_x \!\times\! n_y = {nx}\!\times\!{ny}$",
            ha="center", va="center", fontsize=F.FS_LABEL, color="0.4",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75))
    ax.text(L * 0.97, H * 0.93,
            rf"thickness $t = {int(prob.t)}$ mm",
            ha="right", va="top", fontsize=F.FS_ANNOT, color="0.4",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75))

    ax.set_xlim(-80, L + 160)
    ax.set_ylim(-sup_h - 90, H + arr_len + 50)
    ax.axis("off")
    out = FIGDIR / "geometry.png"
    F.save(fig, out, target_w=5.5)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 6: stress field at three characteristic load levels
# --------------------------------------------------------------------------- #


def fig_stress_field():
    """Stress field at six s values along the equilibrium path. Each
    panel shows the most-compressive principal stress sigma_2 as a
    coloured contour (the canonical CSFM compression-flow view) AND
    the principal compression direction as quiver arrows (the
    strut-and-tie pattern). The progression from elastic loading
    through the limit point and onto the post-peak branch is visible
    as the colour intensity rises then relaxes."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "P2" / "pinn"))
    from csfm_constitutive import membrane                                  # noqa
    from pinn_arclength import displacements                                # noqa

    prob = DeepBeam()
    net = load_pinn()

    # find limit-point s from a fine sweep
    s_dense = torch.linspace(0.01, 1.0, 200).unsqueeze(-1)
    x_c = torch.full_like(s_dense, prob.x_load)
    y_c = torch.full_like(s_dense, prob.H)
    with torch.no_grad():
        _ux, uy, lam = displacements(net, prob, x_c, y_c, s_dense)
    lam_np = lam.squeeze().numpy()
    delta_np = (-uy).squeeze().numpy()
    i_peak = int(np.argmax(lam_np))
    s_peak = float(s_dense[i_peak])

    # Snapshots spanning the FULL path: two pre-peak states (ascending
    # branch), the limit point itself, and three post-peak states
    # (descending branch). The post-peak fields are the whole point of
    # the study, so the figure must show them, not only the ascending
    # branch.
    s_samples = [s_peak]
    # pre-peak: first s reaching given fractions of the peak load factor
    for frac in (0.45, 0.85):
        cand = np.where(lam_np[: i_peak + 1] >= frac * lam_np[i_peak])[0]
        if len(cand):
            s_samples.append(float(s_dense[cand[0]]))
    # post-peak: evenly spaced in s on the descending branch (last = s = 1)
    for k in (1, 2, 3):
        s_samples.append(float(s_peak + k * (1.0 - s_peak) / 3.0))
    s_samples = sorted(set(s_samples))

    nx_eval, ny_eval = 60, 30
    xs = np.linspace(0, prob.L, nx_eval)
    ys = np.linspace(0, prob.H, ny_eval)
    XX, YY = np.meshgrid(xs, ys)

    # evaluate sigma + principal directions at each s
    panels = []
    for s_val in s_samples:
        x_t = torch.tensor(XX.flatten(), dtype=torch.float32).unsqueeze(-1)
        y_t = torch.tensor(YY.flatten(), dtype=torch.float32).unsqueeze(-1)
        s_t = torch.full_like(x_t, float(s_val))
        x_t.requires_grad_(True)
        y_t.requires_grad_(True)
        ux, uy_p, _ = displacements(net, prob, x_t, y_t, s_t)
        ex = torch.autograd.grad(ux, x_t,
                                 grad_outputs=torch.ones_like(ux),
                                 create_graph=False, retain_graph=True)[0]
        ey = torch.autograd.grad(uy_p, y_t,
                                 grad_outputs=torch.ones_like(uy_p),
                                 create_graph=False, retain_graph=True)[0]
        gxy = (torch.autograd.grad(ux, y_t,
                                   grad_outputs=torch.ones_like(ux),
                                   create_graph=False, retain_graph=True)[0]
               + torch.autograd.grad(uy_p, x_t,
                                     grad_outputs=torch.ones_like(uy_p),
                                     create_graph=False)[0])
        rho_x = prob.rho_x(x_t, y_t)
        rho_y = prob.rho_y(x_t, y_t)
        sig = membrane(ex.detach(), ey.detach(), gxy.detach(),
                       rho_x, rho_y, prob.mat)
        sx = sig["sigma_x"].squeeze().numpy().reshape(ny_eval, nx_eval)
        sy = sig["sigma_y"].squeeze().numpy().reshape(ny_eval, nx_eval)
        txy = sig["tau_xy"].squeeze().numpy().reshape(ny_eval, nx_eval)
        av = 0.5 * (sx + sy)
        r = np.hypot(0.5 * (sx - sy), txy)
        sig_min = av - r
        # principal compression direction (angle of sigma_2)
        theta = 0.5 * np.arctan2(2.0 * txy, sx - sy) + np.pi / 2
        u_arrow = np.cos(theta)
        v_arrow = np.sin(theta)
        # interp delta at this s
        delta_here = float(np.interp(s_val, s_dense.squeeze().numpy(), delta_np))
        lam_here = float(np.interp(s_val, s_dense.squeeze().numpy(), lam_np))
        panels.append((s_val, delta_here, lam_here, sig_min,
                       u_arrow, v_arrow))

    # shared colour scale
    vmin = min(p[3].min() for p in panels)
    vmax = 0.0

    # Grid layout: choose rows/cols to fit n panels in a nearly-square
    # block; common cases for n=11 or n=12 land at 3x4.
    n = len(panels)
    if n <= 6:
        nrows, ncols = 2, 3
    elif n <= 8:
        nrows, ncols = 2, 4
    elif n == 9:
        nrows, ncols = 3, 3
    elif n <= 10:
        nrows, ncols = 2, 5
    elif n <= 12:
        nrows, ncols = 3, 4
    else:
        nrows = int(np.ceil(n / 4))
        ncols = 4
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.0, nrows * 1.5),
                             sharex=True, sharey=True)
    # Hide any unused axes
    axes_flat = list(axes.flat)
    for ax in axes_flat[n:]:
        ax.axis("off")
    axes_use = axes_flat[:n]
    for ax, (s_val, d, l, fld, ua, va) in zip(axes_use, panels):
        cs = ax.contourf(XX, YY, fld, levels=14,
                         cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.contour(XX, YY, fld, levels=7, colors="black",
                   linewidths=0.3, alpha=0.6)
        # principal compression direction quiver (subsample for clarity)
        step = 5
        ax.quiver(XX[::3, ::step], YY[::3, ::step],
                  ua.reshape(ny_eval, nx_eval)[::3, ::step],
                  va.reshape(ny_eval, nx_eval)[::3, ::step],
                  pivot="middle", headwidth=0, headlength=0,
                  headaxislength=0, color="black", alpha=0.55,
                  scale=22, width=0.0025)
        ax.set_aspect("equal")
        is_peak = abs(s_val - s_peak) < 1e-6
        marker = r" $\bigstar$" if is_peak else ""
        # Two-line title: s on top, delta+lambda below. Shorter per
        # line keeps each title inside its own panel width even on
        # the dense 3x4 grid.
        ax.set_title(
            rf"$s={s_val:.2f}${marker}" + "\n"
            rf"$\delta={d:.1f}\,\mathrm{{mm}}$, $\lambda={l:.2f}$",
            fontsize=7.0, pad=2.5)
        ax.set_xticks([0, prob.L / 2, prob.L])
        ax.set_xticklabels(["$0$", "$L/2$", "$L$"], fontsize=6.5)
        ax.set_yticks([0, prob.H])
        ax.set_yticklabels(["$0$", "$H$"], fontsize=6.5)

    # Add vertical breathing room so the two-line titles don't bump
    # into the panel above.
    fig.subplots_adjust(hspace=0.55, wspace=0.10)
    cbar = fig.colorbar(cs, ax=axes, fraction=0.020, pad=0.02,
                        shrink=0.85)
    cbar.set_label(r"principal compression $\sigma_2$ (MPa)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    out = FIGDIR / "stress_field.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


# --------------------------------------------------------------------------- #
# Figure 7: network architecture (boxes-and-arrows diagram)
# --------------------------------------------------------------------------- #


def fig_architecture():
    """ArclengthPINN: (x_n, y_n, s) -> (u, lambda) via shared trunk +
    two heads + s * tilde(.) ansatz on each output."""
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.set_aspect("equal")
    ax.axis("off")

    def box(x, y, w, h, label, fc="white", ec="black", fs=8, lw=0.9):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc,
                                    edgecolor=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs)

    def arrow(x0, y0, x1, y1, color="black"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->",
                                    color=color, lw=0.9))

    # input nodes
    inp_x = 0.0
    box(inp_x, 3.8, 1.2, 0.5, r"$x_n = x / L$", fs=7.5)
    box(inp_x, 3.0, 1.2, 0.5, r"$y_n = y / H$", fs=7.5)
    box(inp_x, 2.2, 1.2, 0.5, r"$s$", fs=8)

    # shared trunk (rectangle representing 6-layer SiLU MLP)
    trunk_x = 2.0
    trunk_w, trunk_h = 1.7, 2.6
    box(trunk_x, 2.0, trunk_w, trunk_h,
        "field trunk\n(SiLU MLP)\n$6 \\times 96$", fc="#E6EFF8")

    # lambda head trunk
    lam_trunk_x = 2.0
    box(lam_trunk_x, 0.8, trunk_w, 1.0,
        "$\\lambda$ head\n(SiLU MLP) $3 \\times 32$",
        fc="#F8E6E6", fs=7.5)

    # routes from inputs into trunks
    arrow(inp_x + 1.2, 4.05, trunk_x, 4.30)  # x -> field trunk
    arrow(inp_x + 1.2, 3.25, trunk_x, 3.50)  # y -> field trunk
    arrow(inp_x + 1.2, 2.45, trunk_x, 2.75)  # s -> field trunk
    arrow(inp_x + 1.2, 2.45, lam_trunk_x, 1.30)  # s -> lambda head only

    # head outputs (tildes)
    head_x = 4.2
    box(head_x, 3.8, 1.4, 0.5, r"$\tilde N_{u_x}(x, y, s)$", fs=7.5)
    box(head_x, 3.0, 1.4, 0.5, r"$\tilde N_{u_y}(x, y, s)$", fs=7.5)
    box(head_x, 1.0, 1.4, 0.5, r"$\tilde N_{\lambda}(s)$", fs=7.5)
    arrow(trunk_x + trunk_w, 4.05, head_x, 4.05)
    arrow(trunk_x + trunk_w, 3.25, head_x, 3.25)
    arrow(lam_trunk_x + trunk_w, 1.30, head_x, 1.25)

    # ansatz multiplication (s . tilde)
    ans_x = 6.2
    box(ans_x, 3.8, 1.6, 0.5,
        r"$u_x = s \cdot \tilde N_{u_x} \cdot U_0 L$",
        fc="#FFF7E6", fs=7.5)
    box(ans_x, 3.0, 1.6, 0.5,
        r"$u_y = s \cdot \tilde N_{u_y} \cdot U_0 H$",
        fc="#FFF7E6", fs=7.5)
    box(ans_x, 1.0, 1.6, 0.5,
        r"$\lambda = s \cdot \tilde N_{\lambda}$",
        fc="#FFF7E6", fs=7.5)
    arrow(head_x + 1.4, 4.05, ans_x, 4.05)
    arrow(head_x + 1.4, 3.25, ans_x, 3.25)
    arrow(head_x + 1.4, 1.25, ans_x, 1.25)

    # final outputs labels
    out_x = 8.0
    ax.text(out_x + 0.05, 3.55, r"$\mathbf{u}(x, y, s)$", fontsize=10,
            va="center", color=CSFD_BLUE)
    ax.text(out_x + 0.05, 1.25, r"$\lambda(s)$", fontsize=10,
            va="center", color=CSFD_RED)

    # hard-IC note
    ax.annotate(
        r"hard IC at $s = 0$: $\mathbf{u} = \mathbf{0},\; \lambda = 0$",
        xy=(ans_x + 0.8, 0.7), fontsize=8, ha="center", style="italic",
        color="0.35")

    ax.set_xlim(-0.2, 10.0)
    ax.set_ylim(0.3, 4.7)
    fig.tight_layout()
    out = FIGDIR / "architecture.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


def main() -> None:
    # the five raster figures the manuscript includes; the architecture
    # figure is TikZ (figures/architecture.tex) and stress_field is not
    # currently included
    fig_geometry()
    fig_parametrisation()
    fig_equilibrium_path()
    fig_speed_profile()
    fig_loss_history()


if __name__ == "__main__":
    main()
