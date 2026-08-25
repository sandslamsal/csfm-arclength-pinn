"""Differentiable port of the CSFM cracked-membrane constitutive map.

Mirrors `membrane()` in `src/core/csfm/continuum.ts`: given the in-plane strain
(eps_x, eps_y, gamma_xy) and the smeared reinforcement ratios (rho_x, rho_y),
it returns the global stress (sigma_x, sigma_y, tau_xy) of the cracked rotating
compression field. Concrete tensile strength is neglected; the effective
compressive strength is reduced by the compression-softening factor k_c2(eps_1).

Everything is batched over collocation points and fully autograd-traceable, so
the equilibrium residual div(sigma) + b can be formed by automatic
differentiation in the PINN. SI units: stress in MPa, strain dimensionless.

Faithfulness note: `continuum.ts` hardcodes the parabola exponent n = 2 in its
`compMag`, so this port does too (it must match the oracle, not EN 1992 in
general). Verify numerically against the TypeScript solver before relying on it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

_EPS = 1e-12


@dataclass(frozen=True)
class CsfmMaterial:
    """Material constants for one design (scalars; broadcast over points)."""
    fc: float                 # concrete cylinder strength (MPa)
    lam: float = 1.0          # lightweight factor lambda
    fy: float = 500.0         # steel yield strength (MPa)
    ft: float = 550.0         # steel ultimate strength (MPa)
    Es: float = 200000.0      # steel elastic modulus (MPa)
    eps_u: float = 0.05       # steel strain at ultimate

    # ---- derived constants (match src/core/materials.ts) -----------------
    @property
    def eta(self) -> float:                       # etaFc = min(1, cbrt(30/fc))
        return min(1.0, (30.0 / self.fc) ** (1.0 / 3.0))

    @property
    def Ec0(self) -> float:                       # concreteEc = 4700*lam*sqrt(fc)
        return 4700.0 * self.lam * (self.fc ** 0.5)

    @property
    def eps_c2(self) -> float:                    # parabolaRectParams(fc).epsC2
        if self.fc <= 50.0:
            return 0.0020
        return (2.0 + 0.085 * (self.fc - 50.0) ** 0.53) / 1000.0

    @property
    def eps_y(self) -> float:                     # steel yield strain
        return self.fy / self.Es


def softening_kc2(eps1: Tensor) -> Tensor:
    """Compression-softening factor k_c2(eps_1) — constitutive.ts::softeningKc2.

    k_c2 = min(1, 1 / (0.8 + 140 eps_1)) for eps_1 > 0, else 1. Smooth
    (hyperbolic) apart from the cap at eps_1 ~ 0.00143.
    """
    decayed = 1.0 / (0.8 + 140.0 * torch.clamp(eps1, min=0.0))
    kc2 = torch.clamp(decayed, max=1.0)
    return torch.where(eps1 > 0.0, kc2, torch.ones_like(eps1))


def comp_mag(eps_mag: Tensor, fce: Tensor | float, eps_c2: float) -> Tensor:
    """Parabola-rectangle compressive stress magnitude (MPa), n = 2.

    eps_mag is the compressive strain magnitude (>= 0 used). Mirrors
    `compMag` in continuum.ts: parabola up to eps_c2, plateau at fce after.
    """
    e = torch.clamp(eps_mag, min=0.0)
    parab = fce * (1.0 - (1.0 - e / eps_c2) ** 2)
    return torch.where(e >= eps_c2, fce * torch.ones_like(e), parab)


def steel_stress(eps: Tensor, mat: CsfmMaterial) -> Tensor:
    """Signed bilinear bare-bar steel stress (MPa) — continuum.ts::steelStress."""
    s = torch.abs(eps)
    ey = mat.eps_y
    esh = (mat.ft - mat.fy) / (mat.eps_u - ey)
    elastic = mat.Es * s
    harden = torch.clamp(mat.fy + esh * (s - ey), max=mat.ft)
    mag = torch.where(s <= ey, elastic, harden)
    return torch.sign(eps) * mag


def membrane(
    eps_x: Tensor,
    eps_y: Tensor,
    gam_xy: Tensor,
    rho_x: Tensor | float,
    rho_y: Tensor | float,
    mat: CsfmMaterial,
    soften: bool = True,
) -> dict[str, Tensor]:
    """Cracked rotating compression field — port of continuum.ts::membrane().

    soften=False forces k_c2 = 1 (no compression softening) -- the well-posed
    cracked / strut-and-tie regime targeted by the parametric P2 PINN.

    Returns a dict with the global stress components and diagnostics:
      sigma_x, sigma_y, tau_xy : global stress (MPa)
      e1, e2                   : principal strains (e1 most tensile)
      kc2                      : effective compression-softening factor
      theta                    : inclination of the principal frame (rad)
    """
    # ---- principal strains ----------------------------------------------
    eav = 0.5 * (eps_x + eps_y)
    rad = torch.sqrt((0.5 * (eps_x - eps_y)) ** 2 + (0.5 * gam_xy) ** 2 + _EPS)
    e1 = eav + rad                       # most tensile
    e2 = eav - rad                       # most compressive
    theta = 0.5 * torch.atan2(gam_xy, eps_x - eps_y)
    c, s = torch.cos(theta), torch.sin(theta)

    # ---- concrete principal stresses (tension neglected) ----------------
    eta_fc = mat.eta * mat.fc
    biax = (e1 < 0.0) & (e2 < 0.0)       # biaxial compression: no softening
    comp2 = e2 < 0.0

    kc2 = softening_kc2(e1) if soften else torch.ones_like(e1)
    fce_soft = mat.eta * kc2 * mat.fc

    sc1_biax = -comp_mag(-e1, eta_fc, mat.eps_c2)
    sc2_biax = -comp_mag(-e2, eta_fc, mat.eps_c2)
    sc2_crack = -comp_mag(-e2, fce_soft, mat.eps_c2)

    zero = torch.zeros_like(e1)
    sc1 = torch.where(biax, sc1_biax, zero)
    sc2 = torch.where(biax, sc2_biax, torch.where(comp2, sc2_crack, zero))
    kc2_eff = torch.where(biax | ~comp2, torch.ones_like(kc2), kc2)

    # Small concrete tension stiffness (regularization). The CSFM neglects
    # concrete tension for strength, but the continuum solver keeps a secant
    # stiffness floor Emin = 0.002 Ec so the equilibrium problem stays
    # well-posed; the PINN needs the same in stress form, otherwise tension
    # zones (sigma = 0 for any u) leave the strong-form residual a null space.
    # At eps ~ 1e-3 this adds ~0.05 MPa -- negligible against f_c.
    e_t = 0.002 * mat.Ec0
    sc1 = sc1 + e_t * torch.clamp(e1, min=0.0)
    sc2 = sc2 + e_t * torch.clamp(e2, min=0.0)

    # ---- rotate principal -> global:  sigma_g = Te^T [sc1, sc2, 0] ------
    c2, s2, cs = c * c, s * s, c * s
    sgx = c2 * sc1 + s2 * sc2
    sgy = s2 * sc1 + c2 * sc2
    sgxy = cs * sc1 - cs * sc2

    # ---- smeared reinforcement (global x / y bars) ----------------------
    ssx = steel_stress(eps_x, mat)
    ssy = steel_stress(eps_y, mat)
    sigma_x = sgx + rho_x * ssx
    sigma_y = sgy + rho_y * ssy
    tau_xy = sgxy

    return {
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "tau_xy": tau_xy,
        "e1": e1,
        "e2": e2,
        "kc2": kc2_eff,
        "theta": theta,
        "sc2": sc2,
    }


def elastic_plane_stress(
    eps_x: Tensor, eps_y: Tensor, gam_xy: Tensor,
    rho_x: Tensor | float, rho_y: Tensor | float, mat: CsfmMaterial,
) -> dict[str, Tensor]:
    """Isotropic linear plane-stress concrete + smeared linear steel.

    The alpha = 0 endpoint of the constitutive homotopy: a unique, well-posed
    boundary value problem whose stress trajectories already route the load to
    the supports. Poisson ratio 0.2.
    """
    nu = 0.2
    coef = mat.Ec0 / (1.0 - nu * nu)
    sx = coef * (eps_x + nu * eps_y) + rho_x * mat.Es * eps_x
    sy = coef * (nu * eps_x + eps_y) + rho_y * mat.Es * eps_y
    txy = coef * (0.5 * (1.0 - nu)) * gam_xy
    return {"sigma_x": sx, "sigma_y": sy, "tau_xy": txy}


def _w_parab(e: Tensor, fce: Tensor | float, eps_c2: float) -> Tensor:
    """Strain-energy density of the parabola-rectangle law: integral of the
    compressive stress from 0 to the compressive strain magnitude e >= 0."""
    e = torch.clamp(e, min=0.0)
    r = 1.0 - e / eps_c2
    w_par = fce * e + fce * eps_c2 / 3.0 * (r ** 3 - 1.0)        # e <= eps_c2
    w_pl = fce * eps_c2 * (2.0 / 3.0) + fce * (e - eps_c2)        # e  > eps_c2
    return torch.where(e <= eps_c2, w_par, w_pl)


def _w_steel(eps: Tensor, rho: Tensor | float, mat: CsfmMaterial) -> Tensor:
    """Smeared-reinforcement strain-energy density (bilinear steel)."""
    s = torch.abs(eps)
    ey = mat.eps_y
    esh = (mat.ft - mat.fy) / (mat.eps_u - ey)
    w_el = 0.5 * mat.Es * s ** 2
    w_pl = (0.5 * mat.Es * ey ** 2 + mat.fy * (s - ey)
            + 0.5 * esh * (s - ey) ** 2)
    return rho * torch.where(s <= ey, w_el, w_pl)


def strain_energy_density(
    eps_x: Tensor, eps_y: Tensor, gam_xy: Tensor,
    rho_x: Tensor | float, rho_y: Tensor | float, mat: CsfmMaterial,
    alpha: float = 1.0, soften: bool = True,
) -> Tensor:
    """Strain-energy density W(eps) for the Deep Energy Method.

    alpha = 1 : the CSFM cracked-membrane energy (parabola-rectangle concrete
                in the compressive principal directions, with compression
                softening, plus smeared-steel energy, plus a small tensile
                regularization). alpha = 0 : linear plane-stress elastic
                energy. Intermediate alpha blends the two (homotopy).
    """
    # ---- CSFM cracked-membrane energy -----------------------------------
    eav = 0.5 * (eps_x + eps_y)
    rad = torch.sqrt((0.5 * (eps_x - eps_y)) ** 2 + (0.5 * gam_xy) ** 2 + _EPS)
    e1 = eav + rad
    e2 = eav - rad
    eta_fc = mat.eta * mat.fc
    biax = (e1 < 0.0) & (e2 < 0.0)
    kc2 = softening_kc2(e1) if soften else torch.ones_like(e1)
    fce2 = torch.where(biax, torch.full_like(e1, eta_fc),
                       mat.eta * kc2 * mat.fc)
    w_c2 = torch.where(e2 < 0.0, _w_parab(-e2, fce2, mat.eps_c2),
                       torch.zeros_like(e2))
    w_c1 = torch.where(e1 < 0.0, _w_parab(-e1, eta_fc, mat.eps_c2),
                       torch.zeros_like(e1))
    e_t = 0.002 * mat.Ec0
    w_ten = 0.5 * e_t * (torch.clamp(e1, min=0.0) ** 2
                         + torch.clamp(e2, min=0.0) ** 2)
    w_steel = (_w_steel(eps_x, rho_x, mat) + _w_steel(eps_y, rho_y, mat))
    w_csfm = w_c1 + w_c2 + w_ten + w_steel

    if alpha >= 1.0:
        return w_csfm

    # ---- linear plane-stress elastic energy  W = 1/2 sigma:eps ----------
    el = elastic_plane_stress(eps_x, eps_y, gam_xy, rho_x, rho_y, mat)
    w_el = 0.5 * (el["sigma_x"] * eps_x + el["sigma_y"] * eps_y
                  + el["tau_xy"] * gam_xy)
    if alpha <= 0.0:
        return w_el
    return (1.0 - alpha) * w_el + alpha * w_csfm


def membrane_homotopy(
    eps_x: Tensor, eps_y: Tensor, gam_xy: Tensor,
    rho_x: Tensor | float, rho_y: Tensor | float, mat: CsfmMaterial,
    alpha: float, soften: bool = True,
) -> dict[str, Tensor]:
    """Constitutive homotopy: (1 - alpha) * elastic  +  alpha * CSFM.

    alpha = 0 is linear plane-stress elasticity (easy, unique); alpha = 1 is
    the exact CSFM cracked-membrane map. The PINN is continuated along alpha,
    warm-started, so it stays in the basin of the true stress field instead of
    collapsing to the diffuse-fan local minimum.
    """
    el = elastic_plane_stress(eps_x, eps_y, gam_xy, rho_x, rho_y, mat)
    if alpha <= 0.0:
        return el
    cs = membrane(eps_x, eps_y, gam_xy, rho_x, rho_y, mat, soften=soften)
    if alpha >= 1.0:
        return cs
    return {
        "sigma_x": (1 - alpha) * el["sigma_x"] + alpha * cs["sigma_x"],
        "sigma_y": (1 - alpha) * el["sigma_y"] + alpha * cs["sigma_y"],
        "tau_xy": (1 - alpha) * el["tau_xy"] + alpha * cs["tau_xy"],
    }


if __name__ == "__main__":
    # smoke test: uniaxial compression should give sigma ~ -comp_mag along x
    mat = CsfmMaterial(fc=30.0)
    ex = torch.linspace(-0.004, 0.001, 11)
    zeros = torch.zeros_like(ex)
    out = membrane(ex, zeros, zeros, 0.0, 0.0, mat)
    for e, sx, e1, kc2 in zip(ex, out["sigma_x"], out["e1"], out["kc2"]):
        print(f"eps_x={e:+.4f}  sigma_x={sx:+8.3f} MPa  "
              f"e1={e1:+.4f}  kc2={kc2:.3f}")
