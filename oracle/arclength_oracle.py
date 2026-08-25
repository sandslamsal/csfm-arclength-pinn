"""Post-peak CSFM oracle for P3 — displacement-controlled Picard driver.

The production solver `src/core/csfm/continuum.ts` is load-controlled: it
advances lambda on a fixed schedule and Picard-iterates the secant
`membrane()` stiffness. At the limit point the secant Jacobian goes
singular, Picard stalls, and the solver returns `Capacity reached at load
factor ≈ ...` — the post-peak softening branch is unreachable because
lambda is the wrong parameter past the peak.

This module is the methods reference for P3. It uses the same cracked
rotating-compression-field constitutive map as `continuum.ts::membrane()`
but drives the solve in *displacement* control: the y-displacement at the
load patch is prescribed to delta_n, the corresponding reaction R is read
off the internal force, and lambda_n = -R / P_ref. Past the peak, delta
keeps growing while lambda drops — that descending branch is the P3
reference curve.

NumPy-only on purpose: it is a *third* independent constitutive
implementation alongside `continuum.ts` (TypeScript) and
`csfm_constitutive.py` (PyTorch). Two independent implementations agreeing
is stronger evidence than one validated against itself.

Known limitations of the current driver:
  - The curve is noisy past the peak (Picard at fixed delta wanders between
    solution basins). Smooth post-hoc before using as a PINN reference.
  - Strain softening localises into one element row under the load (mesh-
    dependent). Crack-band scaling would cure it; not implemented.

Math sources (line-anchored, 2026-05-22):
  - membrane map           : src/core/csfm/continuum.ts L208-272
  - parabola-rectangle law : src/core/csfm/constitutive.ts (parabolaRectParams)
  - softening k_c2         : src/core/csfm/constitutive.ts (softeningKc2)
  - secant D + clamps      : src/core/csfm/continuum.ts L229-269
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Constitutive primitives
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Material:
    """CSFM material constants. Mirrors `CsfmMaterial` in
    Research/P2/pinn/csfm_constitutive.py."""

    fc: float = 30.0
    lam: float = 1.0
    fy: float = 500.0
    ft: float = 550.0
    Es: float = 200_000.0
    eps_u: float = 0.05

    @property
    def eta(self) -> float:
        return min(1.0, (30.0 / self.fc) ** (1.0 / 3.0))

    @property
    def Ec0(self) -> float:
        return 4700.0 * self.lam * self.fc ** 0.5

    @property
    def eps_c2(self) -> float:
        if self.fc <= 50.0:
            return 0.0020
        return (2.0 + 0.085 * (self.fc - 50.0) ** 0.53) / 1000.0

    @property
    def eps_y(self) -> float:
        return self.fy / self.Es


def softening_kc2(eps1: float, soften: bool = True) -> float:
    if not soften or eps1 <= 0.0:
        return 1.0
    return min(1.0, 1.0 / (0.8 + 140.0 * eps1))


def comp_mag(eps_mag: float, fce: float, eps_c2: float) -> float:
    e = max(0.0, eps_mag)
    if e >= eps_c2:
        return fce
    return fce * (1.0 - (1.0 - e / eps_c2) ** 2)


def steel_stress(eps: float, mat: Material) -> float:
    s = abs(eps)
    ey = mat.eps_y
    if s <= ey:
        return float(np.sign(eps)) * mat.Es * s
    esh = (mat.ft - mat.fy) / (mat.eps_u - ey)
    mag = min(mat.ft, mat.fy + esh * (s - ey))
    return float(np.sign(eps)) * mag


def membrane(
    ex: float, ey: float, gxy: float,
    rho_x: float, rho_y: float,
    mat: Material, soften: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Cracked rotating-compression-field stress + 3x3 secant D."""
    eav = 0.5 * (ex + ey)
    rad = float(np.hypot(0.5 * (ex - ey), 0.5 * gxy))
    e1 = eav + rad
    e2 = eav - rad
    theta = 0.5 * float(np.arctan2(gxy, ex - ey))
    c, s = float(np.cos(theta)), float(np.sin(theta))

    eta_fc = mat.eta * mat.fc
    sc1 = sc2 = 0.0
    kc2 = 1.0
    if e1 < 0.0 and e2 < 0.0:
        sc1 = -comp_mag(-e1, eta_fc, mat.eps_c2)
        sc2 = -comp_mag(-e2, eta_fc, mat.eps_c2)
    elif e2 < 0.0:
        kc2 = softening_kc2(max(0.0, e1), soften)
        sc2 = -comp_mag(-e2, eta_fc * kc2, mat.eps_c2)
    fcd_eff = eta_fc * kc2

    Ec0 = mat.Ec0
    Emin = 0.002 * Ec0
    E1 = float(np.clip(sc1 / e1, Emin, Ec0)) if abs(e1) > EPS else Emin
    E2 = float(np.clip(sc2 / e2, Emin, Ec0)) if abs(e2) > EPS else Emin
    if abs(e1 - e2) > 1e-9:
        G = float(np.clip((sc1 - sc2) / (2.0 * (e1 - e2)), Emin / 2.0, Ec0))
    else:
        G = Emin

    c2, s2, cs = c * c, s * s, c * s
    Te = np.array([
        [c2,      s2,      cs],
        [s2,      c2,     -cs],
        [-2 * cs, 2 * cs,  c2 - s2],
    ])
    Dp = np.diag([E1, E2, G])
    D = Te.T @ Dp @ Te

    scg = np.array([
        Te[0, 0] * sc1 + Te[1, 0] * sc2,
        Te[0, 1] * sc1 + Te[1, 1] * sc2,
        Te[0, 2] * sc1 + Te[1, 2] * sc2,
    ])

    ssx = steel_stress(ex, mat)
    ssy = steel_stress(ey, mat)
    sigma = np.array([
        scg[0] + rho_x * ssx,
        scg[1] + rho_y * ssy,
        scg[2],
    ])
    Esx = float(np.clip(ssx / ex, 0.0, mat.Es)) if abs(ex) > EPS else mat.Es
    Esy = float(np.clip(ssy / ey, 0.0, mat.Es)) if abs(ey) > EPS else mat.Es
    D[0, 0] += rho_x * Esx
    D[1, 1] += rho_y * Esy

    diag = {"e1": e1, "e2": e2, "kc2": kc2, "fcd_eff": fcd_eff,
            "sc1": sc1, "sc2": sc2, "ssx": ssx, "ssy": ssy, "theta": theta}
    return sigma, D, diag


# --------------------------------------------------------------------------- #
# Mesh + boundary application (CST triangles, identical layout to continuum.ts)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Problem:
    """Rectangular CSFM D-region problem. Coordinates in mm, y up.

    Supports the standard deep-beam configuration (pin/roller patches
    on the bottom edge) and a fully clamped left-face configuration
    (corbel-style cantilever bracket). For the latter set
    ``clamped_left = True`` and pass an empty ``supports`` tuple."""

    L: float
    H: float
    thickness: float
    nx: int
    ny: int
    rho_x: Callable[[float, float], float]
    rho_y: Callable[[float, float], float]
    x_load: float
    bearing: float
    P_ref: float
    supports: tuple[tuple[float, bool, bool], ...]
    mat: Material
    clamped_left: bool = False     # clamp every node on the x = 0 face


@dataclass
class Mesh:
    n_node: int
    ndof: int
    xy: np.ndarray
    tris: list[tuple[np.ndarray, float, float]]
    B: list[np.ndarray]
    area: list[float]
    load_dofs: list[int]               # y-DOFs of the load patch nodes
    fixed: np.ndarray
    F_ref: np.ndarray


def build_mesh(prob: Problem) -> Mesh:
    nx, ny = prob.nx, prob.ny
    nnx, nny = nx + 1, ny + 1
    dx, dy = prob.L / nx, prob.H / ny
    n_node = nnx * nny
    ndof = 2 * n_node

    def nid(i: int, j: int) -> int:
        return j * nnx + i

    xy = np.array([[i * dx, j * dy] for j in range(nny) for i in range(nnx)],
                  dtype=float)

    tris: list[tuple[np.ndarray, float, float]] = []
    for j in range(ny):
        for i in range(nx):
            a, b = nid(i, j), nid(i + 1, j)
            c, d = nid(i + 1, j + 1), nid(i, j + 1)
            xc, yc = (i + 0.5) * dx, (j + 0.5) * dy
            rx = prob.rho_x(xc, yc)
            ry = prob.rho_y(xc, yc)
            tris.append((np.array([a, b, c]), rx, ry))
            tris.append((np.array([a, c, d]), rx, ry))

    B_list, area_list = [], []
    for nodes, _, _ in tris:
        p = xy[nodes]
        b1, b2, b3 = p[1, 1] - p[2, 1], p[2, 1] - p[0, 1], p[0, 1] - p[1, 1]
        c1, c2, c3 = p[2, 0] - p[1, 0], p[0, 0] - p[2, 0], p[1, 0] - p[0, 0]
        det = p[0, 0] * b1 + p[1, 0] * b2 + p[2, 0] * b3
        area_list.append(abs(det) / 2.0)
        inv = 1.0 / det
        B_list.append(np.array([
            [b1 * inv, 0, b2 * inv, 0, b3 * inv, 0],
            [0, c1 * inv, 0, c2 * inv, 0, c3 * inv],
            [c1 * inv, b1 * inv, c2 * inv, b2 * inv, c3 * inv, b3 * inv],
        ]))

    half = prob.bearing / 2.0
    # x_load may be a single position or several. Several is needed for
    # four-point bending, where the shear span is set by the load positions
    # rather than by the supports; for a symmetric specimen prescribing the
    # same deflection at both patches is equivalent to applying equal forces.
    x_loads = (prob.x_load if isinstance(prob.x_load, (tuple, list))
               else (prob.x_load,))
    load_node_ids = [nid(i, ny) for i in range(nnx)
                     if any(abs(i * dx - xl) <= half + 1e-6 for xl in x_loads)]

    F_ref = np.zeros(ndof)
    for n in load_node_ids:
        F_ref[2 * n + 1] -= prob.P_ref / len(load_node_ids)

    fixed = np.zeros(ndof, dtype=bool)
    for xc, fx, fy in prob.supports:
        for n in range(n_node):
            if xy[n, 1] < 1e-6 and abs(xy[n, 0] - xc) <= half + 1e-6:
                if fx:
                    fixed[2 * n] = True
                if fy:
                    fixed[2 * n + 1] = True
    if prob.clamped_left:
        for n in range(n_node):
            if xy[n, 0] < 1e-6:
                fixed[2 * n] = True
                fixed[2 * n + 1] = True

    return Mesh(n_node=n_node, ndof=ndof, xy=xy, tris=tris,
                B=B_list, area=area_list,
                load_dofs=load_node_ids, fixed=fixed, F_ref=F_ref)


# --------------------------------------------------------------------------- #
# Assembly + displacement-controlled Picard
# --------------------------------------------------------------------------- #


def assemble(u: np.ndarray, prob: Problem, mesh: Mesh,
             soften: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Assemble global secant K and internal force F_int from u."""
    ndof = mesh.ndof
    K = np.zeros((ndof, ndof))
    F_int = np.zeros(ndof)
    t = prob.thickness
    for e, (nodes, rx, ry) in enumerate(mesh.tris):
        B = mesh.B[e]
        a = mesh.area[e]
        dofs = np.array([2 * nodes[0], 2 * nodes[0] + 1,
                         2 * nodes[1], 2 * nodes[1] + 1,
                         2 * nodes[2], 2 * nodes[2] + 1])
        eps = B @ u[dofs]
        sigma, D, _ = membrane(eps[0], eps[1], eps[2], rx, ry, prob.mat,
                               soften=soften)
        vol = a * t
        K[np.ix_(dofs, dofs)] += (B.T @ D @ B) * vol
        F_int[dofs] += (B.T @ sigma) * vol
    return K, F_int


def picard_displacement_controlled(
    u: np.ndarray, delta: float, prob: Problem, mesh: Mesh,
    max_iter: int = 200, tol: float = 2e-3, relax: float = 0.30,
    stall_window: int = 20, soften: bool = True,
) -> tuple[bool, float, np.ndarray]:
    """One displacement-controlled Picard solve at prescribed load-patch
    deflection delta (positive magnitude; applied as -delta in y)."""
    ndof = mesh.ndof
    load_dofs = [2 * n + 1 for n in mesh.load_dofs]
    prescribed = np.zeros(ndof, dtype=bool)
    prescribed[mesh.fixed] = True
    for d in load_dofs:
        prescribed[d] = True
    free = ~prescribed

    u[mesh.fixed] = 0.0
    for d in load_dofs:
        u[d] = -delta

    best_resid = np.inf
    stall = 0
    converged = False
    for it in range(max_iter):
        K, _ = assemble(u, prob, mesh, soften=soften)
        rhs = -K[np.ix_(free, prescribed)] @ u[prescribed]
        u_new_free = np.linalg.solve(K[np.ix_(free, free)], rhs)

        d_max = float(np.max(np.abs(u_new_free - u[free])))
        u_max = max(1e-9, float(np.max(np.abs(u[free]))))
        u[free] = u[free] + relax * (u_new_free - u[free])

        resid = d_max / u_max
        if resid < tol:
            converged = True
            break
        if resid < best_resid * 0.999:
            best_resid = resid
            stall = 0
        else:
            stall += 1
            if stall >= stall_window:
                break

    K, _ = assemble(u, prob, mesh, soften=soften)
    R = float(np.sum((K @ u)[load_dofs]))
    lam = -R / abs(prob.P_ref)
    return converged, lam, u


# --------------------------------------------------------------------------- #
# Driver: sweep delta and trace the lambda-delta curve through the limit point
# --------------------------------------------------------------------------- #


@dataclass
class CurvePoint:
    delta: float
    lam: float
    converged: bool


def field_diagnostics(u: np.ndarray, prob: Problem, mesh: Mesh,
                      soften: bool = True) -> dict:
    """Per-element scalars used to diagnose failure progression."""
    max_e2_mag = 0.0
    min_kc2 = 1.0
    max_e1 = 0.0
    for e, (nodes, rx, ry) in enumerate(mesh.tris):
        B = mesh.B[e]
        dofs = np.array([2 * nodes[0], 2 * nodes[0] + 1,
                         2 * nodes[1], 2 * nodes[1] + 1,
                         2 * nodes[2], 2 * nodes[2] + 1])
        eps = B @ u[dofs]
        _, _, diag = membrane(eps[0], eps[1], eps[2], rx, ry, prob.mat,
                              soften=soften)
        max_e2_mag = max(max_e2_mag, abs(diag["e2"]))
        min_kc2 = min(min_kc2, diag["kc2"])
        max_e1 = max(max_e1, diag["e1"])
    return {"max_e2_mag": max_e2_mag, "min_kc2": min_kc2, "max_e1": max_e1}


def trace_curve(
    prob: Problem, mesh: Mesh, delta_max: float, n_steps: int = 50,
    soften: bool = True, verbose: bool = False,
) -> tuple[list[CurvePoint], np.ndarray, list[dict]]:
    """Sweep delta from 0 to delta_max in n_steps equal increments."""
    u = np.zeros(mesh.ndof)
    curve: list[CurvePoint] = []
    diags: list[dict] = []
    schedule = np.linspace(0.0, delta_max, n_steps + 1)[1:]
    for k, delta in enumerate(schedule):
        conv, lam, u = picard_displacement_controlled(
            u, float(delta), prob, mesh, soften=soften)
        curve.append(CurvePoint(float(delta), float(lam), bool(conv)))
        diag = field_diagnostics(u, prob, mesh, soften=soften)
        diags.append(diag)
        if verbose:
            tag = " " if conv else "*"
            print(f"  step {k + 1:>2}/{n_steps}  delta={delta:7.3f} mm  "
                  f"lambda={lam:+.4f}  "
                  f"|e2|_max={diag['max_e2_mag']:.4f}  "
                  f"kc2_min={diag['min_kc2']:.3f}  "
                  f"e1_max={diag['max_e1']:.4f} {tag}")
    return curve, u, diags
