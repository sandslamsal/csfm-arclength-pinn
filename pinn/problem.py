"""D-region problem definitions for the continuum CSFM PINN.

A problem fixes the rectangular domain, the boundary conditions (supports and
applied tractions), the smeared reinforcement field rho_x(x,y), rho_y(x,y),
and the material. The first problem is a simply supported deep beam: a single
top load spreading through an inclined concrete strut to two supports, tied
along the bottom.

Coordinates are physical (mm); x in [0, L], y in [0, H], y up.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from csfm_constitutive import CsfmMaterial


@dataclass
class DeepBeam:
    L: float = 2000.0          # span / width (mm)
    H: float = 1000.0          # height (mm)
    t: float = 300.0           # out-of-plane thickness (mm)
    a: float = 250.0           # support center inset from each end (mm)
    bearing: float = 200.0     # support / load bearing width (mm)
    support_bearing: float | None = None  # support plate width if it differs
                                          # from the load plate (falls back to
                                          # `bearing` when None)
    P: float = 800.0e3         # total applied load (N), downward, top center
    mat: CsfmMaterial = field(default_factory=lambda: CsfmMaterial(fc=30.0))
    rho_tie: float = 0.012     # bottom horizontal band reinforcement ratio
    rho_stirrup: float = 0.0015  # distributed vertical ratio
    rho_min: float = 0.0010    # mesh-reinforcement floor (regularises tension)
    band: float = 150.0        # bottom tie-band thickness (mm)

    # ---- derived ---------------------------------------------------------
    @property
    def pressure(self) -> float:
        """Bearing pressure under the load patch (MPa), positive magnitude."""
        return self.P / (self.bearing * self.t)

    @property
    def x_load(self) -> float:
        return self.L / 2.0

    @property
    def x_supp(self) -> tuple[float, float]:
        return self.a, self.L - self.a

    # ---- domain inclusion test -----------------------------------------
    def inside(self, x: Tensor, y: Tensor) -> Tensor:
        """The rectangle is solid -- every point in [0, L] x [0, H] is in."""
        return torch.ones_like(x)

    # ---- reinforcement field --------------------------------------------
    def rho_x(self, x: Tensor, y: Tensor) -> Tensor:
        """Smeared horizontal ratio: tie band near the soffit + mesh floor."""
        in_band = (y < self.band).float()
        return self.rho_min + (self.rho_tie - self.rho_min) * in_band

    def rho_y(self, x: Tensor, y: Tensor) -> Tensor:
        """Smeared vertical (stirrup) ratio + mesh floor."""
        return torch.full_like(x, self.rho_min + self.rho_stirrup)

    # ---- support displacement BC residual -------------------------------
    def support_residual(self, ux: Tensor, uy: Tensor, x: Tensor) -> Tensor:
        """Pin (left) + roller (right): u_y = 0 at both supports; u_x = 0
        only at the left (pinned) support."""
        left = x < self.L / 2.0
        return (uy ** 2).mean() + (ux[left] ** 2).mean()

    # ---- interior-cut helpers for force recovery -----------------------
    def cut_xrange(self, y_cut: float) -> tuple[float, float]:
        """Horizontal extent over which sigma_y is integrated to recover the
        load at height y_cut. Defaults to the full width of the solid
        rectangle. When the measurements cover only part of the width (e.g.
        a fiber grid inset from the edges), set `self.cut_x = (xlo, xhi)` to
        the load-bearing span that brackets the supports: integrating over
        the full width while averaging a stress resolved only on the inset
        strip over-counts the load by (L / strip-width)."""
        if getattr(self, "cut_x", None) is not None:
            return self.cut_x
        return 0.0, self.L

    def default_cuts(self) -> tuple[float, ...]:
        """A handful of interior heights at which integrating sigma_y
        recovers the applied load magnitude. Picked to avoid the
        bearing-zone stress concentrations at y=0 and y=H."""
        return (0.5 * self.H, 0.7 * self.H, 0.85 * self.H)

    # ---- collocation / boundary samplers --------------------------------
    def interior(self, n: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        x = torch.rand(n, 1, generator=gen) * self.L
        y = torch.rand(n, 1, generator=gen) * self.H
        return x, y

    def near_features(self, n: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        """Collocation clustered at the load and the two supports -- the
        high-gradient bearing zones the uniform sample under-resolves."""
        centers = [(self.x_load, self.H),
                   (self.x_supp[0], 0.0), (self.x_supp[1], 0.0)]
        per = max(1, n // len(centers))
        spread = self.bearing
        xs, ys = [], []
        for cx, cy in centers:
            x = (cx + spread * torch.randn(per, 1, generator=gen)).clamp(0, self.L)
            y = (cy + spread * torch.randn(per, 1, generator=gen)).clamp(0, self.H)
            xs.append(x); ys.append(y)
        return torch.cat(xs), torch.cat(ys)

    def _edge(self, n, gen, x0, x1, y0, y1):
        s = torch.rand(n, 1, generator=gen)
        return x0 + (x1 - x0) * s, y0 + (y1 - y0) * s

    def supports(self, n: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        """Points under the two soffit supports (y = 0)."""
        half = (self.support_bearing or self.bearing) / 2.0
        pts = []
        for xc in self.x_supp:
            x, y = self._edge(n // 2, gen, xc - half, xc + half, 0.0, 0.0)
            pts.append((x, y))
        x = torch.cat([p[0] for p in pts])
        y = torch.cat([p[1] for p in pts])
        return x, y

    def loaded_patch(self, n: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        """Points under the top load patch (y = H)."""
        half = self.bearing / 2.0
        return self._edge(n, gen, self.x_load - half, self.x_load + half,
                          self.H, self.H)

    def free_edges(self, n: int, gen: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
        """Traction-free boundary points; returns (x, y, normal) with normal
        the outward unit normal stacked as [nx, ny]."""
        half = self.bearing / 2.0
        segs = []  # (x0,x1,y0,y1, nx,ny)
        # bottom soffit, excluding the two support patches
        for (xa, xb) in (
            (0.0, self.x_supp[0] - half),
            (self.x_supp[0] + half, self.x_supp[1] - half),
            (self.x_supp[1] + half, self.L),
        ):
            if xb > xa:
                segs.append((xa, xb, 0.0, 0.0, 0.0, -1.0))
        # top, excluding the load patch
        for (xa, xb) in (
            (0.0, self.x_load - half),
            (self.x_load + half, self.L),
        ):
            if xb > xa:
                segs.append((xa, xb, self.H, self.H, 0.0, 1.0))
        # left and right edges
        segs.append((0.0, 0.0, 0.0, self.H, -1.0, 0.0))
        segs.append((self.L, self.L, 0.0, self.H, 1.0, 0.0))

        per = max(1, n // len(segs))
        xs, ys, nm = [], [], []
        for (x0, x1, y0, y1, nx, ny) in segs:
            x, y = self._edge(per, gen, x0, x1, y0, y1)
            xs.append(x); ys.append(y)
            nm.append(torch.tensor([[nx, ny]]).repeat(per, 1))
        return torch.cat(xs), torch.cat(ys), torch.cat(nm)


@dataclass
class Corbel:
    """A cantilever-bracket idealization of a precast corbel: a
    rectangular D-region fully clamped along its left face (the proxy
    for the column-to-corbel interface) and loaded vertically over a
    short bearing patch at the free end. The compression flow is a
    single inclined strut from the loaded patch down to the bottom of
    the clamped face, balanced by a horizontal tie along the top of
    the bracket. The standard 2D textbook abstraction of a corbel
    (Kaufmann & Marti; ACI 318 Sec. 16.5): the column itself is outside
    the modeled domain, its rigid attachment captured by the clamped
    boundary.

    Geometry (default dimensions, mm):
        bounding box: L x H = 500 x 400
        load patch:   x in [L - bearing, L] at y = H
        clamped face: x = 0 for all y in [0, H]
    """
    L: float = 500.0           # projection from the clamped face (mm)
    H: float = 400.0           # depth of the bracket (mm)
    t: float = 300.0           # out-of-plane thickness (mm)
    bearing: float = 100.0     # load bearing width on the top edge (mm)
    P: float = 90.0e3          # total applied load (N), failureLF ~ 1.3
    mat: CsfmMaterial = field(default_factory=lambda: CsfmMaterial(fc=30.0))
    rho_tie: float = 0.012     # top horizontal-tie reinforcement ratio
    rho_stirrup: float = 0.0015
    rho_min: float = 0.0010
    band: float = 80.0         # top tie-band thickness (mm)

    @property
    def pressure(self) -> float:
        return self.P / (self.bearing * self.t)

    @property
    def x_load(self) -> float:
        return self.L - self.bearing / 2.0

    @property
    def x_supp(self) -> tuple[float, ...]:
        # for plotting: midpoint of the clamped left face
        return (0.0,)

    # ---- domain inclusion test (rectangle: always inside) --------------
    def inside(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.ones_like(x)

    # ---- reinforcement field --------------------------------------------
    def rho_x(self, x: Tensor, y: Tensor) -> Tensor:
        """Horizontal ratio: top-tie band over the full bracket length;
        mesh floor elsewhere."""
        in_band = (y > self.H - self.band).float()
        return self.rho_min + (self.rho_tie - self.rho_min) * in_band

    def rho_y(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.full_like(x, self.rho_min + self.rho_stirrup)

    # ---- support displacement BC residual -------------------------------
    def support_residual(self, ux: Tensor, uy: Tensor, x: Tensor) -> Tensor:
        """Left face fully clamped: u_x = u_y = 0."""
        return (uy ** 2).mean() + (ux ** 2).mean()

    # ---- interior-cut helpers for force recovery -----------------------
    def cut_xrange(self, y_cut: float) -> tuple[float, float]:
        """Cut always spans the full bracket depth (rectangle)."""
        return 0.0, self.L

    def default_cuts(self) -> tuple[float, ...]:
        """Three vertical cuts, well away from the clamped face and the
        load patch, that integrate sigma_y to the applied load magnitude.
        For a cantilever a *vertical* cut is more natural (the strut runs
        diagonally and crosses vertical sections cleanly), but we use the
        existing sigma_y / horizontal-cut convention; the diagonal strut
        still crosses these horizontal lines."""
        return (0.25 * self.H, 0.5 * self.H, 0.75 * self.H)

    # ---- collocation / boundary samplers --------------------------------
    def interior(self, n: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        x = torch.rand(n, 1, generator=gen) * self.L
        y = torch.rand(n, 1, generator=gen) * self.H
        return x, y

    def _edge(self, n, gen, x0, x1, y0, y1):
        s = torch.rand(n, 1, generator=gen)
        return x0 + (x1 - x0) * s, y0 + (y1 - y0) * s

    def supports(self, n: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        """Points on the clamped left face (x = 0, all y)."""
        return self._edge(n, gen, 0.0, 0.0, 0.0, self.H)

    def loaded_patch(self, n: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        """Points on the top load patch (y = H, x in [L - bearing, L])."""
        return self._edge(n, gen, self.L - self.bearing, self.L,
                          self.H, self.H)

    def free_edges(self, n: int, gen: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
        """Traction-free boundary: top (excluding the load patch), right
        face and bottom (soffit) face."""
        segs = [
            (0.0, self.L - self.bearing, self.H, self.H, 0.0, 1.0),
            (self.L, self.L, 0.0, self.H, 1.0, 0.0),
            (0.0, self.L, 0.0, 0.0, 0.0, -1.0),
        ]
        per = max(1, n // len(segs))
        xs, ys, nm = [], [], []
        for (x0, x1, y0, y1, nx, ny) in segs:
            x, y = self._edge(per, gen, x0, x1, y0, y1)
            xs.append(x); ys.append(y)
            nm.append(torch.tensor([[nx, ny]]).repeat(per, 1))
        return torch.cat(xs), torch.cat(ys), torch.cat(nm)


@dataclass
class WallPier:
    """The cantilever wall-type bridge pier VK1 of Bimschas (2010),
    included in the experimental validation suite of the CSFM
    (Kaufmann et al. 2020, Section 6.3). A 1500 mm x 3700 mm reinforced-
    concrete wall, 200 mm thick, fixed at its base and loaded by a
    constant vertical compression N at the top and a horizontal force V
    applied near the top edge. The horizontal force is the test's
    primary variable; its experimentally measured ultimate is
    V_u,exp = 725 kN (concrete crushing + flexural yield).

    The flexural reinforcement is vertical (rho_y) and matches the test
    specimen (Phi 14 at 130 mm, rho_y,geo = 0.82 %); the shear
    reinforcement is horizontal (rho_x), at the very low 0.08 % of the
    test (Phi 6 at 200 mm hoops).
    """
    L: float = 1500.0          # wall width (mm)
    H: float = 3700.0          # wall height (mm)
    t: float = 200.0           # wall thickness (mm)
    h_eff: float = 3300.0      # height where V is applied (mm)
    bearing_top: float = 200.0 # half-width of the N-load patch (mm)
    bearing_V: float = 200.0   # half-height of the V-load patch (mm)
    N: float = 1370.0e3        # axial compression, applied uniformly at y=H
    P: float = 300.0e3         # horizontal load V at the analysis level
                               # (= 0.41 V_u,exp; service-load regime where
                               # our continuum solver converges cleanly).
    mat: CsfmMaterial = field(default_factory=lambda: CsfmMaterial(fc=35.0))
    rho_l: float = 0.0082      # vertical / flexural reinforcement ratio
    rho_t: float = 0.0008      # horizontal / shear reinforcement ratio
    rho_min: float = 0.0010    # mesh-reinforcement floor

    @property
    def pressure(self) -> float:
        """Pressure under the horizontal load patch (MPa)."""
        return self.P / (2.0 * self.bearing_V * self.t)

    @property
    def x_load(self) -> float:
        # for plotting only; the V load patch center on the left face
        return 0.0

    @property
    def x_supp(self) -> tuple[float, ...]:
        # for plotting only; midpoint of the clamped base
        return (self.L / 2.0,)

    # ---- domain inclusion (rectangle: always inside) -------------------
    def inside(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.ones_like(x)

    # ---- reinforcement field --------------------------------------------
    def rho_x(self, x: Tensor, y: Tensor) -> Tensor:
        """Horizontal (shear) reinforcement: 0.08 % across the full
        height, with the doubled density near the top (s_t = 75 mm
        instead of 200 mm over the top 300 mm of the wall)."""
        densified = (y > self.H - 300.0).float()
        rho_top = max(self.rho_t * 200.0 / 75.0, self.rho_min)
        return self.rho_min + (self.rho_t - self.rho_min) * (1 - densified) \
            + (rho_top - self.rho_min) * densified

    def rho_y(self, x: Tensor, y: Tensor) -> Tensor:
        """Vertical (flexural) reinforcement: 0.82 % uniform."""
        return torch.full_like(x, self.rho_l)

    # ---- support displacement BC residual ------------------------------
    def support_residual(self, ux: Tensor, uy: Tensor, x: Tensor) -> Tensor:
        """Base fully clamped: u_x = u_y = 0."""
        return (uy ** 2).mean() + (ux ** 2).mean()

    # ---- interior-cut helpers (both N via sigma_y and V via tau_xy) ----
    def cut_xrange(self, y_cut: float) -> tuple[float, float]:
        return 0.0, self.L

    def default_cuts(self) -> tuple[float, ...]:
        """Three cuts in the lower half of the wall (away from the
        H_eff patch where V is applied and from the foundation), where
        the interior gauges densely constrain the field."""
        return (0.25 * self.h_eff, 0.45 * self.h_eff, 0.70 * self.h_eff)

    # ---- collocation / boundary samplers --------------------------------
    def interior(self, n: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        x = torch.rand(n, 1, generator=gen) * self.L
        y = torch.rand(n, 1, generator=gen) * self.H
        return x, y

    def _edge(self, n, gen, x0, x1, y0, y1):
        s = torch.rand(n, 1, generator=gen)
        return x0 + (x1 - x0) * s, y0 + (y1 - y0) * s

    def supports(self, n: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        """Points on the clamped base (y = 0, all x)."""
        return self._edge(n, gen, 0.0, self.L, 0.0, 0.0)

    def loaded_patch(self, n: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        """Points on the V-load patch (left face, y in [h_eff - bearing_V,
        h_eff + bearing_V])."""
        return self._edge(n, gen, 0.0, 0.0,
                          self.h_eff - self.bearing_V,
                          self.h_eff + self.bearing_V)

    def free_edges(self, n: int, gen: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
        """Traction-free boundary: top (excluding the N-load patch), right
        face (full height), and the left face above & below the V patch."""
        segs: list[tuple[float, float, float, float, float, float]] = []
        # top: full width is loaded by N (uniform), so it is NOT traction-free
        # right face: x = L, full height
        segs.append((self.L, self.L, 0.0, self.H, 1.0, 0.0))
        # left face below V patch
        segs.append((0.0, 0.0, 0.0, self.h_eff - self.bearing_V, -1.0, 0.0))
        # left face above V patch
        segs.append((0.0, 0.0, self.h_eff + self.bearing_V, self.H, -1.0, 0.0))
        per = max(1, n // len(segs))
        xs, ys, nm = [], [], []
        for (x0, x1, y0, y1, nx, ny) in segs:
            x, y = self._edge(per, gen, x0, x1, y0, y1)
            xs.append(x); ys.append(y)
            nm.append(torch.tensor([[nx, ny]]).repeat(per, 1))
        return torch.cat(xs), torch.cat(ys), torch.cat(nm)
