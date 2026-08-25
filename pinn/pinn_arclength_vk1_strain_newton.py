"""VK1 strain-anchored training, re-anchored to equilibrium-converged states.

Identical to pinn_arclength_vk1_strain.py except that both the strain
target and the lambda curve come from vk1_newton_full.json
(newton_displacement_control_vk1, converged states only)
instead of the secant fixed points. Warm-starts from the
secant-anchored checkpoint and writes to runs/vk1_newton/.

Original docstring:
VK1 with-N PINN with STRAIN-LEVEL anchor against the CSFM
reference. Replaces the lambda-only anchor used in v_withN/v_lbfgs
with a direct match of the network's full strain field to the CSFM
strain field at the same physical state. Strain match implies stress
match implies lambda match (no degree of freedom left for the
strain field to absorb the displacement budget into modes that don't
contribute to V).

Warm-starts from the L-BFGS-polished checkpoint
(`runs/vk1_withN/vk1_pinn.pt` at the time of writing, V=541 kN at
delta=51, -10%). Goal: close the gap below -5%.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_loss import CSFMCurveTarget                                     # noqa: E402
from compute_losses_vk1_withN import compute_losses_vk1_withN                # noqa: E402
from model import ArclengthPINN                                             # noqa: E402
from pinn_arclength import (                                                # noqa: E402
    FIXED_ARC_WEIGHT, N_INT, RELOBRALO_EVERY, RELOBRALO_LOSSES, SEED,
    ReLoBraLo, displacements, strains as autograd_strains,
)
from pinn_arclength_vk1_v3 import sanitise_grads_                            # noqa: E402
from vk1_background import bg_strain_at, get_background                      # noqa: E402
from vk1_strain_target import (                                              # noqa: E402
    get_strain_target, strain_at_state,
)
from wallpier_vk1 import WallPierVK1                                         # noqa: E402

U0 = 1.0e-3
# Re-derived from the corrected t = 350 mm trace: 44 of 45 states
# converge over 1 to 45 mm, with the peak at 33 mm, so the sweep
# covers the whole converged range.
S_MAX = 45.0
# The floor is rate*s and must stay well below the reference peak,
# which is now 2.435. 1.45 is 0.6 of it. A floor above the peak
# stops selecting a basin and starts forcing the solution upward.
FLOOR_RATE = 1.45
LAMBDA_ANCHOR_WEIGHT = 500.0
STRAIN_ANCHOR_WEIGHT = 5000.0    # large because strain values are O(1e-3)
DEAD_WEIGHT = 100.0
FLOOR_WEIGHT = 100.0
LR = 3e-5
GRAD_CLIP = 0.3
N_ITER = 6000
N_STRAIN_SAMPLES = 1024


def main() -> None:
    here = Path(__file__).resolve().parent
    prob = WallPierVK1(include_N=True)
    out_dir = here / "runs" / "vk1_newton_t350"
    out_dir.mkdir(parents=True, exist_ok=True)
    warm = here / "runs" / "vk1_withN" / "vk1_pinn.pt"   # secant-era start
    if not warm.exists():
        raise FileNotFoundError(warm)

    sys.path.insert(0, str(here.parents[1] / "P1" / "oracle"))
    from arclength_oracle_vk1 import vk1_default
    bg = get_background(vk1_default())
    strain_target = get_strain_target(
        here.parent / "oracle" / "vk1_newton_full.json")

    gen = torch.Generator().manual_seed(SEED + 41)
    torch.manual_seed(SEED + 41)
    net = ArclengthPINN(width=96, depth=6)
    net.load_state_dict(torch.load(warm))
    print(f"warm from {warm.name}")
    print(f"lambda_anchor_w={LAMBDA_ANCHOR_WEIGHT}  "
          f"strain_anchor_w={STRAIN_ANCHOR_WEIGHT}  lr={LR}  "
          f"n_iter={N_ITER}")

    lam_target_curve = CSFMCurveTarget.from_json(
        here.parent / "oracle" / "vk1_newton_full.json",
        delta_key="delta_x", lam_key="lam")

    opt = torch.optim.Adam(net.parameters(), lr=LR)
    weighter = ReLoBraLo(list(RELOBRALO_LOSSES))
    last_good = {k: v.clone() for k, v in net.state_dict().items()}
    history: list[dict] = []
    skipped = 0
    t0 = time.time()
    for it in range(N_ITER):
        opt.zero_grad()
        # core physics + dead + load (sigma-vs-lambda) + arc + floor
        losses = compute_losses_vk1_withN(
            net, prob, alpha=1.0, gen=gen, bg=bg,
            S_max_mm=S_MAX, lambda_floor_rate=FLOOR_RATE,
        )
        if any(torch.isnan(v).any() for v in losses.values()):
            skipped += 1
            continue

        # lambda anchor at V-patch
        xa = torch.zeros(128, 1)
        ya = torch.full_like(xa, prob.h_eff)
        sa = torch.rand(128, 1, generator=gen)
        xy_n = torch.cat([xa / prob.L, ya / prob.H], dim=-1)
        out, lam_a = net(xy_n, sa)
        ux_a = out[:, 0:1] * U0 * prob.L
        lam_target = lam_target_curve.interp(ux_a.detach())
        l_lam_anchor = ((lam_a - lam_target) ** 2).mean()

        # NEW: strain anchor at interior collocation points
        xi = torch.rand(N_STRAIN_SAMPLES, 1, generator=gen) * prob.L
        yi = torch.rand(N_STRAIN_SAMPLES, 1, generator=gen) * prob.H
        si = torch.rand(N_STRAIN_SAMPLES, 1, generator=gen)
        xi_g = xi.clone().requires_grad_(True)
        yi_g = yi.clone().requires_grad_(True)
        ux_i, uy_i, _ = displacements(net, prob, xi_g, yi_g, si)
        ex_pinn, ey_pinn, gxy_pinn = autograd_strains(
            ux_i, uy_i, xi_g, yi_g)
        ex_bg, ey_bg, gxy_bg = bg_strain_at(bg, xi_g, yi_g)
        ex_total = ex_pinn + ex_bg
        ey_total = ey_pinn + ey_bg
        gxy_total = gxy_pinn + gxy_bg
        # target strain at (x, y, delta = si * S_max)
        delta_eff = si * S_MAX
        ex_t, ey_t, gxy_t = strain_at_state(strain_target, xi, yi,
                                            delta_eff)
        l_strain_anchor = (((ex_total - ex_t) ** 2
                            + (ey_total - ey_t) ** 2
                            + (gxy_total - gxy_t) ** 2).mean())

        physics = {n: losses[n] for n in RELOBRALO_LOSSES}
        if it % RELOBRALO_EVERY == 0:
            weighter.update(physics)
        total = (weighter.weighted_sum(physics)
                 + FIXED_ARC_WEIGHT * losses["arc"]
                 + DEAD_WEIGHT * losses["dead"]
                 + LAMBDA_ANCHOR_WEIGHT * l_lam_anchor
                 + STRAIN_ANCHOR_WEIGHT * l_strain_anchor
                 + FLOOR_WEIGHT * losses["lam_floor"])
        total.backward()
        sanitise_grads_(net)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=GRAD_CLIP)
        opt.step()
        if any(not torch.isfinite(p).all() for p in net.parameters()):
            print(f"  [PARAM NaN at iter {it}] rolling back")
            net.load_state_dict(last_good)
            break
        if it % 500 == 0 or it == N_ITER - 1:
            with torch.no_grad():
                sp = torch.linspace(0, 1, 6).unsqueeze(-1)
                xp = torch.zeros_like(sp)
                yp = torch.full_like(sp, prob.h_eff)
                xyn = torch.cat([xp / prob.L, yp / prob.H], dim=-1)
                op, lp = net(xyn, sp)
                uxp = op[:, 0:1] * U0 * prob.L
            history.append({"iter": it, "lam_at_s1": float(lp[-1]),
                            "delta_at_s1": float(uxp[-1]),
                            "lam_anchor": float(l_lam_anchor.detach()),
                            "strain_anchor": float(l_strain_anchor.detach())})
            print(f"  it={it:5d}  V@s=1={float(lp[-1]) * prob.P / 1e3:.1f} kN"
                  f" @ delta={float(uxp[-1]):.1f} mm  "
                  f"strain_anc={float(l_strain_anchor):.3e}  "
                  f"lam_anc={float(l_lam_anchor):.3e}  skipped={skipped}")
            last_good = {k: v.clone() for k, v in net.state_dict().items()}

    torch.save(net.state_dict(), out_dir / "vk1_pinn.pt")
    wall = time.time() - t0
    with open(out_dir / "training_history_strain.json", "w") as f:
        json.dump({"history": history, "wall_s": wall, "skipped": skipped,
                   "lambda_anchor_w": LAMBDA_ANCHOR_WEIGHT,
                   "strain_anchor_w": STRAIN_ANCHOR_WEIGHT,
                   "n_iter": N_ITER, "lr": LR}, f)
    print(f"\nwall={wall:.1f}s skipped={skipped}")


if __name__ == "__main__":
    main()
