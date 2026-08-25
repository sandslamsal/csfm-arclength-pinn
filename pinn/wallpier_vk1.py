"""VK1 wall-pier problem for the P3 arc-length PINN.

A torch-side mirror of `oracle/arclength_oracle_vk1.py::VK1Problem`,
with the BC-sampling and reinforcement-field interface that
`pinn_arclength.compute_losses` already consumes from `DeepBeam`,
extended with two new fields for the dead axial load:

  * `loaded_patch_N(n, gen)` -- centred bearing-plate samples on the
    top edge where the dead-load BC is enforced
  * `pressure_N` -- the static stress magnitude (in MPa) under that
    bearing plate, equal to N / (2 * bearing_N_half * thickness)

The live (arc-length-parametrised) load is the horizontal V on the
left face at y = h_eff. `loaded_patch`, `pressure`, `x_load` therefore
refer to the V-patch (so the existing PINN loss machinery routes the
arc-length parametrised BC to V automatically, matching deep-beam
and corbel).

Reinforcement layout follows Bimschas (2010) VK1 / Kaufmann et al.
(2020) Section 6.3:
  * vertical (flexural) rho_l = 0.82%, uniform
  * horizontal (shear) rho_t = 0.08%, densified to 0.21% over the top
    300 mm
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "P2" / "pinn"))
from csfm_constitutive import CsfmMaterial                                  # noqa: E402


@dataclass
class WallPierVK1:
    L: float = 1500.0
    H: float = 3700.0
    t: float = 350.0               # Bimschas (2010) Tab. 5.1. A 200 mm
                                   # value gives 57 % of the section and
                                   # 57 % of the flexural steel, and must
                                   # match the oracle's VK1Problem.
    h_eff: float = 3300.0
    bearing_V: float = 200.0       # half-height of the V-load patch (mm)
    bearing_N_half: float = 200.0  # half-width of the centred N plate (mm)
    P: float = 300.0e3             # V_ref (N) -- live arc-parametrised load
    N: float = 1370.0e3            # dead axial load (N), constant
    include_N: bool = True         # apply the dead N BC + exclude N
                                   # patch from free_edges
    mat: CsfmMaterial = field(
        default_factory=lambda: CsfmMaterial(fc=35.0, fy=515.0))
    rho_l: float = 0.0082
    rho_t: float = 0.0008
    rho_min: float = 0.0010

    # ---- live-load pressure (used by compute_losses for the load BC) ----
    @property
    def pressure(self) -> float:
        """V-patch live-load pressure (MPa). Compatible with the
        DeepBeam interface that compute_losses expects."""
        return self.P / (2.0 * self.bearing_V * self.t)

    @property
    def pressure_N(self) -> float:
        """Centred top bearing-plate dead-load pressure (MPa)."""
        return self.N / (2.0 * self.bearing_N_half * self.t)

    @property
    def x_load(self) -> float:
        """Used by plot_curve to probe the trained network at the
        load patch. The V-patch sits on x = 0 (left face) at
        y = h_eff. Return 0.0 so xy=(x_load, H) probes the top-left
        corner -- not what we want here, see plot_curve override in
        pinn_arclength_vk1.py."""
        return 0.0

    @property
    def x_supp(self) -> tuple[float, ...]:
        return (self.L / 2.0,)

    # ---- domain inclusion -------------------------------------------------
    def inside(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.ones_like(x)

    # ---- reinforcement field ----------------------------------------------
    def rho_x(self, x: Tensor, y: Tensor) -> Tensor:
        densified = (y > self.H - 300.0).float()
        rho_top = max(self.rho_t * 200.0 / 75.0, self.rho_min)
        return (self.rho_min
                + (self.rho_t - self.rho_min) * (1 - densified)
                + (rho_top - self.rho_min) * densified)

    def rho_y(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.full_like(x, max(self.rho_l, self.rho_min))

    # ---- support residual -------------------------------------------------
    def support_residual(self, ux: Tensor, uy: Tensor, x: Tensor) -> Tensor:
        """Base fully clamped: u_x = u_y = 0."""
        return (ux ** 2).mean() + (uy ** 2).mean()

    # ---- interior samples -------------------------------------------------
    def interior(self, n: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        x = torch.rand(n, 1, generator=gen) * self.L
        y = torch.rand(n, 1, generator=gen) * self.H
        return x, y

    def _edge(self, n: int, gen: torch.Generator,
              x0: float, x1: float, y0: float, y1: float
              ) -> tuple[Tensor, Tensor]:
        s = torch.rand(n, 1, generator=gen)
        return x0 + (x1 - x0) * s, y0 + (y1 - y0) * s

    def supports(self, n: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        return self._edge(n, gen, 0.0, self.L, 0.0, 0.0)

    # ---- live load patch (V on left face at y = h_eff) -------------------
    def loaded_patch(self, n: int, gen: torch.Generator
                     ) -> tuple[Tensor, Tensor]:
        return self._edge(n, gen, 0.0, 0.0,
                          self.h_eff - self.bearing_V,
                          self.h_eff + self.bearing_V)

    # ---- dead load patch (N on centred top plate) -----------------------
    def loaded_patch_N(self, n: int, gen: torch.Generator
                       ) -> tuple[Tensor, Tensor]:
        return self._edge(n, gen,
                          self.L / 2 - self.bearing_N_half,
                          self.L / 2 + self.bearing_N_half,
                          self.H, self.H)

    # ---- traction-free edges (all except the two loaded patches) --------
    def free_edges(self, n: int, gen: torch.Generator
                   ) -> tuple[Tensor, Tensor, Tensor]:
        if self.include_N:
            segs: list[tuple[float, float, float, float, float, float]] = [
                # top: left of N-patch
                (0.0, self.L / 2 - self.bearing_N_half, self.H, self.H, 0.0, 1.0),
                # top: right of N-patch
                (self.L / 2 + self.bearing_N_half, self.L, self.H, self.H, 0.0, 1.0),
                # right face full height
                (self.L, self.L, 0.0, self.H, 1.0, 0.0),
                # left face below V patch
                (0.0, 0.0, 0.0, self.h_eff - self.bearing_V, -1.0, 0.0),
                # left face above V patch
                (0.0, 0.0, self.h_eff + self.bearing_V, self.H, -1.0, 0.0),
            ]
        else:
            # no dead N: entire top edge is traction-free
            segs = [
                (0.0, self.L, self.H, self.H, 0.0, 1.0),
                (self.L, self.L, 0.0, self.H, 1.0, 0.0),
                (0.0, 0.0, 0.0, self.h_eff - self.bearing_V, -1.0, 0.0),
                (0.0, 0.0, self.h_eff + self.bearing_V, self.H, -1.0, 0.0),
            ]
        per = max(1, n // len(segs))
        xs, ys, nm = [], [], []
        for (x0, x1, y0, y1, nx, ny) in segs:
            x, y = self._edge(per, gen, x0, x1, y0, y1)
            xs.append(x); ys.append(y)
            nm.append(torch.tensor([[nx, ny]]).repeat(per, 1))
        return torch.cat(xs), torch.cat(ys), torch.cat(nm)
