"""C^1-regularised cracked rotating-membrane constitutive for P3.

Mirrors `Research/P2/pinn/csfm_constitutive.py::membrane()` and
`::membrane_homotopy()`, but replaces every `torch.where` branch and
every `torch.clamp` boundary with a sigmoid- or softplus-based smooth
blend so that autograd produces well-defined, bounded gradients
everywhere, including arbitrarily close to the e1 = 0 / e2 = 0 / steel
yield / parabola-plateau lines that the hard branches would otherwise
turn into derivative discontinuities.

The motivation: P2's `membrane()` is C^0 at those branch lines, which is
fine for the inverse-PINN training of [P2] in which the equilibrium
residual is treated as a regulariser rather than a sole training
objective. P3's forward arc-length PINN, in contrast, drives the
equilibrium residual to zero through Adam on the second derivatives of
the network output; the C^0 derivatives at the branch lines make that
optimisation return NaN on the very first backward pass at any
constitutive-homotopy level alpha > 0. Sharp sigmoid blends solve the
gradient pathology while preserving the original constitutive's values
to within the tolerance reported by `test_smooth_membrane.py`.

The sharpness parameter `SHARP` controls the width of the smooth
transition. The default `SHARP = 5e3` gives a transition width of
~2e-4 strain units, which is two orders of magnitude smaller than the
cracking-onset strain (~5e-5 to 5e-3 typical) and one order smaller
than the steel-yield strain (~2.5e-3), so the smooth version is
indistinguishable from the hard version away from the branch lines.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor

# Reuse P2's CsfmMaterial dataclass + elastic_plane_stress unchanged
# (they are smooth already; we only smooth the cracked-membrane branches).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "P2" / "pinn"))
from csfm_constitutive import CsfmMaterial, elastic_plane_stress              # noqa: E402

_EPS = 1e-12
# Sharpness sets the smooth-transition width at ~1/SHARP in strain units.
# Must be << typical PINN training strains (~1e-5 to 1e-2 on the deep beam)
# so the smooth blends agree with the original hard branches everywhere the
# constitutive is non-trivial. Empirically, SHARP = 1e5 gives <1% deviation
# from the P2 hard version on the cross-check states of
# test_smooth_membrane.py while keeping the per-branch gradient bounded at
# ~SHARP/4 = 2.5e4 (handled cleanly by Adam with gradient clipping).
SHARP = 1.0e5


def soft_step(x: Tensor, sharp: float = SHARP) -> Tensor:
    """Smooth indicator of x > 0. ~1 when x > 1/sharp, ~0 when x < -1/sharp."""
    return torch.sigmoid(sharp * x)


def soft_max0(x: Tensor, sharp: float = SHARP) -> Tensor:
    """Smooth max(x, 0) via scaled softplus. Equals x for x >> 1/sharp,
    0 for x << -1/sharp, log(2)/sharp at x = 0."""
    return torch.nn.functional.softplus(sharp * x) / sharp


def soft_min(a: Tensor, b: float, sharp: float = SHARP) -> Tensor:
    """Smooth min(a, b) for scalar b."""
    return b - soft_max0(b - a, sharp)


def softening_kc2_smooth(eps1: Tensor, soften: bool = True,
                         sharp: float = SHARP) -> Tensor:
    """C^1 version of P2's `softening_kc2`. For eps1 < 0 returns ~1; for
    eps1 > 0 returns min(1, 1/(0.8 + 140 eps1)) with smooth caps."""
    if not soften:
        return torch.ones_like(eps1)
    e_p = soft_max0(eps1, sharp)
    decayed = 1.0 / (0.8 + 140.0 * e_p + _EPS)
    kc2 = soft_min(decayed, 1.0, sharp)
    # blend with 1 for eps1 < 0
    pos = soft_step(eps1, sharp)
    return pos * kc2 + (1.0 - pos) * torch.ones_like(eps1)


def comp_mag_smooth(eps_mag: Tensor, fce: Tensor | float, eps_c2: float,
                    sharp: float = SHARP) -> Tensor:
    """C^1 parabola-rectangle compressive stress magnitude. Equals
    fce*(1-(1-e/eps_c2)^2) for e in [0, eps_c2], plateau at fce for
    e >= eps_c2, soft-clamped at 0 for e < 0."""
    e = soft_max0(eps_mag, sharp)
    parab = fce * (1.0 - (1.0 - e / eps_c2) ** 2)
    plateau = (fce if isinstance(fce, Tensor) else float(fce)) \
        * torch.ones_like(e)
    plateau_indicator = soft_step(e - eps_c2, sharp)
    return plateau_indicator * plateau + (1.0 - plateau_indicator) * parab


def steel_stress_smooth(eps: Tensor, mat: CsfmMaterial,
                        sharp: float = SHARP) -> Tensor:
    """Signed bilinear bare-bar steel stress, C^1 at yield."""
    # smooth |eps|: sqrt(eps^2 + EPS) is already C^1 everywhere
    s_abs = torch.sqrt(eps * eps + _EPS)
    ey = mat.eps_y
    esh = (mat.ft - mat.fy) / (mat.eps_u - ey)
    elastic = mat.Es * s_abs
    harden_raw = mat.fy + esh * (s_abs - ey)
    # soft cap at f_t
    harden = mat.ft - soft_max0(mat.ft - harden_raw, sharp)
    # blend at yield s = ey
    yielded = soft_step(s_abs - ey, sharp)
    mag = yielded * harden + (1.0 - yielded) * elastic
    # smooth sign(eps): eps / sqrt(eps^2 + EPS) is C^1 at 0
    sgn = eps / s_abs
    return sgn * mag


def membrane_smooth(
    eps_x: Tensor, eps_y: Tensor, gam_xy: Tensor,
    rho_x: Tensor | float, rho_y: Tensor | float,
    mat: CsfmMaterial, soften: bool = True, sharp: float = SHARP,
) -> dict[str, Tensor]:
    """C^1 cracked rotating-compression-field response.

    Drop-in replacement for `csfm_constitutive.membrane`. Returns the same
    dict keys."""
    # ---- principal strains ---------------------------------------------
    eav = 0.5 * (eps_x + eps_y)
    rad = torch.sqrt(
        (0.5 * (eps_x - eps_y)) ** 2 + (0.5 * gam_xy) ** 2 + _EPS
    )
    e1 = eav + rad
    e2 = eav - rad
    theta = 0.5 * torch.atan2(gam_xy, eps_x - eps_y)
    c, s = torch.cos(theta), torch.sin(theta)

    # ---- smooth case indicators ---------------------------------------
    e1_neg = soft_step(-e1, sharp)              # ~1 when e1 < 0
    e2_neg = soft_step(-e2, sharp)              # ~1 when e2 < 0
    biax = e1_neg * e2_neg                      # ~1 when both negative

    # ---- concrete principal stresses ----------------------------------
    eta_fc = mat.eta * mat.fc
    kc2 = softening_kc2_smooth(e1, soften, sharp)
    fce_soft = mat.eta * kc2 * mat.fc

    sc1_biax = -comp_mag_smooth(-e1, eta_fc, mat.eps_c2, sharp)
    sc2_biax = -comp_mag_smooth(-e2, eta_fc, mat.eps_c2, sharp)
    sc2_crack = -comp_mag_smooth(-e2, fce_soft, mat.eps_c2, sharp)

    # smooth blends:
    #   sc1 = biax ? sc1_biax : 0
    sc1 = biax * sc1_biax
    #   sc2 = biax ? sc2_biax : (e2_neg ? sc2_crack : 0)
    sc2 = biax * sc2_biax + (1.0 - biax) * e2_neg * sc2_crack

    # kc2_eff: 1 when biax OR e2 >= 0, else kc2
    e2_pos = 1.0 - e2_neg
    one_when = biax + (1.0 - biax) * e2_pos
    kc2_eff = one_when * torch.ones_like(kc2) + (1.0 - one_when) * kc2

    # small tensile regulariser (matches P2's hyperparameter)
    e_t = 0.002 * mat.Ec0
    sc1 = sc1 + e_t * soft_max0(e1, sharp)
    sc2 = sc2 + e_t * soft_max0(e2, sharp)

    # ---- rotate principal -> global ----------------------------------
    c2, s2, cs = c * c, s * s, c * s
    sgx = c2 * sc1 + s2 * sc2
    sgy = s2 * sc1 + c2 * sc2
    sgxy = cs * sc1 - cs * sc2

    # ---- smeared reinforcement ---------------------------------------
    ssx = steel_stress_smooth(eps_x, mat, sharp)
    ssy = steel_stress_smooth(eps_y, mat, sharp)
    sigma_x = sgx + rho_x * ssx
    sigma_y = sgy + rho_y * ssy
    tau_xy = sgxy

    return {
        "sigma_x": sigma_x, "sigma_y": sigma_y, "tau_xy": tau_xy,
        "e1": e1, "e2": e2, "kc2": kc2_eff, "theta": theta, "sc2": sc2,
    }


def membrane_homotopy_smooth(
    eps_x: Tensor, eps_y: Tensor, gam_xy: Tensor,
    rho_x: Tensor | float, rho_y: Tensor | float,
    mat: CsfmMaterial, alpha: float, soften: bool = True,
    sharp: float = SHARP,
) -> dict[str, Tensor]:
    """Constitutive homotopy (1 - alpha) * elastic + alpha * cracked-CSFM,
    using the C^1-smoothed cracked-membrane response."""
    el = elastic_plane_stress(eps_x, eps_y, gam_xy, rho_x, rho_y, mat)
    if alpha <= 0.0:
        return el
    cs = membrane_smooth(eps_x, eps_y, gam_xy, rho_x, rho_y, mat,
                         soften=soften, sharp=sharp)
    if alpha >= 1.0:
        return cs
    return {
        "sigma_x": (1 - alpha) * el["sigma_x"] + alpha * cs["sigma_x"],
        "sigma_y": (1 - alpha) * el["sigma_y"] + alpha * cs["sigma_y"],
        "tau_xy": (1 - alpha) * el["tau_xy"] + alpha * cs["tau_xy"],
    }
