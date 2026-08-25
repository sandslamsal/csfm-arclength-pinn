"""VK1-specific PINN loss formulation.

The deep-beam compute_losses in `pinn_arclength.py` assumes the live
load is applied on the *top* edge in the *-y* direction, so the
load BC is σ_yy = -λ·p with σ_xy = 0. The VK1 wall pier has:

  * live (arc-parametrised) V on the *left* face in the *+x*
    direction, requiring σ_xx = -λ·p_V with σ_xy = 0;
  * dead (constant) N on a *centred top bearing plate* in the *-y*
    direction, requiring σ_yy = -p_N with σ_xy = 0.

The traction-free edges, supports, equilibrium and arc-length losses
share the same form as the deep-beam case (with directional arc-
length loss since VK1 is asymmetric, like the corbel). The pointwise
λ-floor of `pinn_arclength.compute_losses` is reused. The dead-load
term is new.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ArclengthPINN                                             # noqa: E402
from pinn_arclength import (                                                # noqa: E402
    N_BC, N_INT, U0, displacements, equilibrium_residual,
    grad, strains, stresses_homotopy,
)


def compute_losses_vk1(net: ArclengthPINN, prob, alpha: float,
                       gen: torch.Generator,
                       S_max_mm: float = 1.0,
                       lambda_floor_rate: float | None = None,
                       ) -> dict[str, Tensor]:
    """All-in-one loss for the VK1 wall pier with horizontal V (live)
    on the left face and downward N (dead) on a centred top plate.
    Arc-length is directional with direction (+1, 0) -- the +x
    deflection at the V-patch grows at rate S_max per unit s.
    """
    fc = prob.mat.fc

    # ---- interior equilibrium ----------------------------------------
    xi, yi = prob.interior(N_INT, gen)
    si = torch.rand(N_INT, 1, generator=gen)
    xi = xi.clone().requires_grad_(True)
    yi = yi.clone().requires_grad_(True)
    ux_i, uy_i, _ = displacements(net, prob, xi, yi, si)
    ex, ey, gxy = strains(ux_i, uy_i, xi, yi)
    rho_x = prob.rho_x(xi, yi)
    rho_y = prob.rho_y(xi, yi)
    sx, sy, txy = stresses_homotopy(ex, ey, gxy, rho_x, rho_y, prob, alpha)
    rx, ry = equilibrium_residual(sx, sy, txy, xi, yi)
    l_eq = ((rx * prob.L / fc) ** 2 + (ry * prob.L / fc) ** 2).mean()

    # ---- support BC (u = 0 on clamped base) ---------------------------
    xs, ys = prob.supports(N_BC, gen)
    ss = torch.rand(N_BC, 1, generator=gen)
    ux_s, uy_s, _ = displacements(net, prob, xs, ys, ss)
    l_supp = prob.support_residual(ux_s / U0, uy_s / U0, xs)

    # ---- live load V on left face: sigma_xx = -lambda * pressure -----
    xl, yl = prob.loaded_patch(N_BC, gen)
    sl = torch.rand(N_BC, 1, generator=gen)
    xl = xl.clone().requires_grad_(True)
    yl = yl.clone().requires_grad_(True)
    ux_l, uy_l, lam_l = displacements(net, prob, xl, yl, sl)
    ex_l, ey_l, gxy_l = strains(ux_l, uy_l, xl, yl)
    rho_x_l = prob.rho_x(xl, yl)
    rho_y_l = prob.rho_y(xl, yl)
    sx_l, sy_l, txy_l = stresses_homotopy(
        ex_l, ey_l, gxy_l, rho_x_l, rho_y_l, prob, alpha)
    # left face: normal (-1, 0); applied V in +x; sigma_xx = -lambda·p
    target_xx = -lam_l * prob.pressure
    l_load = (((sx_l - target_xx) / fc) ** 2 + (txy_l / fc) ** 2).mean()

    # ---- dead load N on centred top plate: sigma_yy = -p_N -----------
    # The dead BC requires non-zero strain at the top plate even at
    # s = 0, which conflicts with the network's hard IC u(s=0) = 0.
    # If `prob.include_N` is False we skip this term and treat the top
    # edge as fully traction-free (handled by prob.free_edges).
    if getattr(prob, "include_N", True):
        xn, yn = prob.loaded_patch_N(N_BC, gen)
        sn = torch.rand(N_BC, 1, generator=gen)
        xn = xn.clone().requires_grad_(True)
        yn = yn.clone().requires_grad_(True)
        ux_n, uy_n, _ = displacements(net, prob, xn, yn, sn)
        ex_n, ey_n, gxy_n = strains(ux_n, uy_n, xn, yn)
        rho_x_n = prob.rho_x(xn, yn)
        rho_y_n = prob.rho_y(xn, yn)
        sx_n, sy_n, txy_n = stresses_homotopy(
            ex_n, ey_n, gxy_n, rho_x_n, rho_y_n, prob, alpha)
        # top face: normal (0, +1); applied N in -y; sigma_yy = -p_N
        target_yy = -prob.pressure_N
        l_dead = (((sy_n - target_yy) / fc) ** 2
                  + (txy_n / fc) ** 2).mean()
    else:
        l_dead = torch.tensor(0.0)

    # ---- traction-free edges -----------------------------------------
    xf, yf, nf = prob.free_edges(N_BC, gen)
    sf = torch.rand(xf.shape[0], 1, generator=gen)
    xf = xf.clone().requires_grad_(True)
    yf = yf.clone().requires_grad_(True)
    ux_f, uy_f, _ = displacements(net, prob, xf, yf, sf)
    ex_f, ey_f, gxy_f = strains(ux_f, uy_f, xf, yf)
    rho_x_f = prob.rho_x(xf, yf)
    rho_y_f = prob.rho_y(xf, yf)
    sx_f, sy_f, txy_f = stresses_homotopy(
        ex_f, ey_f, gxy_f, rho_x_f, rho_y_f, prob, alpha)
    nx, ny = nf[:, 0:1], nf[:, 1:2]
    tx = sx_f * nx + txy_f * ny
    ty = txy_f * nx + sy_f * ny
    l_free = ((tx / fc) ** 2 + (ty / fc) ** 2).mean()

    # ---- directional arc-length: u_x at V-patch grows at S_max/s -----
    xa, ya = prob.loaded_patch(N_BC, gen)
    sa = torch.rand(N_BC, 1, generator=gen).requires_grad_(True)
    ux_a, uy_a, _ = displacements(net, prob, xa, ya, sa)
    dux_ds = grad(ux_a, sa)
    duy_ds = grad(uy_a, sa)
    # arc_direction = (+1, 0): speed_in_load = dux_ds; want = +S_max
    speed_in_load = dux_ds
    l_arc = (((speed_in_load - S_max_mm) / S_max_mm) ** 2).mean()

    out = {"eq": l_eq, "supp": l_supp, "load": l_load,
           "free": l_free, "arc": l_arc, "dead": l_dead}

    # ---- soft lambda-floor (blocks the stress-trivial attractor) ----
    if lambda_floor_rate is not None:
        target_floor = lambda_floor_rate * sl
        l_lam_floor = (torch.relu(target_floor - lam_l) ** 2).mean()
        out["lam_floor"] = l_lam_floor

    return out
