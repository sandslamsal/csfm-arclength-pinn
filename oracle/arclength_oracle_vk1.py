"""Displacement-controlled CSFM oracle for the VK1 wall pier (Bimschas
2010).

The base `arclength_oracle.py` was written for the deepbeam topology —
a single vertical load on the top edge with two bottom-edge bearing
supports. VK1 is a *cantilever* wall pier with two simultaneous loads:

  * a constant axial compression  N = 1370 kN distributed uniformly along
    the top edge of the wall;
  * an arc-length-parametrised horizontal force V applied through a
    bearing patch on the left face at effective height h_eff = 3300 mm.

The wall is fully clamped along its bottom edge (y = 0). The
experimentally measured ultimate horizontal force is V_u,exp = 725 kN
(Bimschas 2010; concrete crushing + flexural yield). The corresponding
load factor on the V_ref = 300 kN unit-load definition used by P2 is
lambda_u,exp = V_u,exp / V_ref = 2.42.

The CSFM material map (`membrane`) and the CST primitives are
unchanged from `arclength_oracle.py`. What is new here:

  * `VK1Problem` dataclass with geometry, reinforcement, and the two
    load magnitudes (N held constant, V arc-parametrised).
  * `build_vk1_mesh` that produces (a) the load_dofs as *horizontal*
    DOFs of the left-face nodes inside the V-bearing patch, (b) the
    fixed-DOF mask covering all bottom-edge x and y DOFs, and (c) a
    *dead-load force vector* F_dead that distributes the axial N
    uniformly across the top-edge y-DOFs (lumped consistent).
  * `picard_displacement_controlled_vk1` that prescribes the horizontal
    displacement at the V-patch, accumulates F_dead into the RHS, and
    reports lambda_V = R_x_reaction / V_ref.

Validation target: the P2 load-controlled reference reaches its
load-control capacity at lambda ~ 1.95 (δ_x ~ 20.8 mm) before the
load-controlled Picard stalls. This driver should agree with P2's
pre-peak branch and continue past the limit point onto the descending
branch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from arclength_oracle import (Material, assemble as _assemble_unused,
                              field_diagnostics as _field_diag_unused,
                              membrane)

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Problem + mesh
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VK1Problem:
    """Cantilever wall-pier D-region with dual loading.

    Geometry from Bimschas (2010) Tab. 5.1, which states the section
    depth l_w = 1.5 m, the section width b_w = 0.35 m and the effective
    height L_v = 3.3 m:
      * L     : section depth l_w  (mm)
      * H     : total height       (mm)
      * t     : section width b_w  (mm)
      * h_eff : centre of the V-load patch on the left face (mm)

    The section width was 200 mm here until 2026-08-24, which is wrong.
    Kaufmann et al. (2020) Section 6.3 and Bimschas Tab. 5.1 both give
    350 mm. The error mattered three ways at once: the section area was
    43 per cent too small, the fixed 1370 kN axial load acted at 1.75
    times the correct stress, and rho_l = 0.82 per cent applied to the
    undersized area gave 2460 mm2 of flexural steel against the
    specimen's 28 x d14 = 4310 mm2, or 57 per cent of it. Capacity came
    out about a third below the measured value while a production CSFM
    predicts the same test to within 7 per cent.
    """

    L: float = 1500.0
    H: float = 3700.0
    thickness: float = 350.0
    nx: int = 30
    ny: int = 74
    rho_x: Callable[[float, float], float] = field(
        default=lambda x, y: 0.001
    )
    rho_y: Callable[[float, float], float] = field(
        default=lambda x, y: 0.001
    )
    h_eff: float = 3300.0
    bearing_V: float = 200.0       # half-height of the V-load patch (mm)
    V_ref: float = 300.0e3         # arc-length-parametrised reference (N)
    N: float = 1370.0e3            # constant axial compression (N)
    # Bimschas (2010) applied N through a stiff actuator bearing plate
    # at the centre of the top edge — not as a uniform pressure across
    # the full top width. The bearing_N_half below sets the half-width
    # of that plate; 200 mm matches the ~400 mm centred patch used in
    # the test rig and avoids the singular point-load case while staying
    # faithful to the experimental loading geometry.
    bearing_N_half: float = 200.0  # half-width of the N bearing plate (mm)
    mat: Material = field(
        default_factory=lambda: Material(fc=35.0, fy=515.0)
    )


@dataclass
class VK1Mesh:
    n_node: int
    ndof: int
    xy: np.ndarray
    tris: list[tuple[np.ndarray, float, float]]
    B: list[np.ndarray]
    area: list[float]
    load_dofs: list[int]          # x-DOFs of left-face V-patch nodes
    load_node_ids: list[int]      # node ids of those V-patch nodes
    top_node_ids: list[int]       # all top-edge node ids
    fixed: np.ndarray             # bottom-edge clamp
    F_dead: np.ndarray            # constant axial-N force vector (N)


def build_vk1_mesh(prob: VK1Problem) -> VK1Mesh:
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

    # V-load patch: left face (x = 0), y in [h_eff - bearing_V, h_eff + bearing_V]
    load_node_ids: list[int] = []
    for n in range(n_node):
        if xy[n, 0] < 1e-6 and abs(xy[n, 1] - prob.h_eff) <= prob.bearing_V + 1e-6:
            load_node_ids.append(n)
    load_dofs = [2 * n for n in load_node_ids]   # x-DOFs

    # Top-edge nodes inside the centred N-bearing plate (mirrors the
    # Bimschas 2010 actuator pad: a stiff bearing plate at the centre
    # of the top edge, not a uniform pressure across the full width).
    top_node_ids: list[int] = []
    x_centre = prob.L / 2.0
    for i in range(nnx):
        x_i = i * dx
        if abs(x_i - x_centre) <= prob.bearing_N_half + 1e-6:
            top_node_ids.append(nid(i, ny))

    # F_dead: lumped downward force on those centred-patch nodes,
    # tributary-weighted (end nodes of the patch get half the spacing).
    F_dead = np.zeros(ndof)
    n_top = len(top_node_ids)
    for k, n in enumerate(top_node_ids):
        trib = dx if 0 < k < n_top - 1 else dx / 2.0
        F_dead[2 * n + 1] -= prob.N * trib / (2.0 * prob.bearing_N_half)

    # Clamped-bottom: every node on y = 0 has u_x = u_y = 0
    fixed = np.zeros(ndof, dtype=bool)
    for n in range(n_node):
        if xy[n, 1] < 1e-6:
            fixed[2 * n] = True
            fixed[2 * n + 1] = True

    return VK1Mesh(n_node=n_node, ndof=ndof, xy=xy, tris=tris,
                   B=B_list, area=area_list,
                   load_dofs=load_dofs,
                   load_node_ids=load_node_ids,
                   top_node_ids=top_node_ids,
                   fixed=fixed, F_dead=F_dead)


# --------------------------------------------------------------------------- #
# Assembly + Picard
# --------------------------------------------------------------------------- #


def _assemble_to_coo(prob: VK1Problem, mesh: VK1Mesh,
                     D_per_elem, F_int: np.ndarray) -> csr_matrix:
    """Assemble a CSR stiffness matrix by collecting (i, j, v) triplets.
    `D_per_elem` is a callable e -> 3x3 D matrix; F_int (if not None) is
    written in place. Memory is O(elements * 36) instead of O(ndof^2)."""
    ndof = mesh.ndof
    t = prob.thickness
    n_e = len(mesh.tris)
    rows = np.empty(n_e * 36, dtype=np.int32)
    cols = np.empty(n_e * 36, dtype=np.int32)
    vals = np.empty(n_e * 36, dtype=np.float64)
    k = 0
    for e, (nodes, _, _) in enumerate(mesh.tris):
        B = mesh.B[e]
        a = mesh.area[e]
        D = D_per_elem(e)
        dofs = np.array([2 * nodes[0], 2 * nodes[0] + 1,
                         2 * nodes[1], 2 * nodes[1] + 1,
                         2 * nodes[2], 2 * nodes[2] + 1])
        Ke = (B.T @ D @ B) * a * t
        for i in range(6):
            for j in range(6):
                rows[k] = dofs[i]; cols[k] = dofs[j]; vals[k] = Ke[i, j]
                k += 1
    K = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    return K


def assemble_vk1_elastic(prob: VK1Problem, mesh: VK1Mesh,
                          nu: float = 0.2) -> csr_matrix:
    """Sparse linear-elastic global stiffness used to warm-start the
    cracked Picard. Returns a CSR matrix."""
    Ec = prob.mat.Ec0
    Es = prob.mat.Es
    coef = Ec / (1.0 - nu * nu)
    Dc = coef * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, 0.5 * (1.0 - nu)],
    ])

    def D_of(e: int) -> np.ndarray:
        _, rx, ry = mesh.tris[e]
        D = Dc.copy()
        D[0, 0] += rx * Es
        D[1, 1] += ry * Es
        return D

    return _assemble_to_coo(prob, mesh, D_of, F_int=None)


def elastic_warmstart_vk1(delta_x: float, prob: VK1Problem,
                          mesh: VK1Mesh) -> np.ndarray:
    """Sparse elastic warm-start: prescribes δ_x at the V-patch and
    applies the axial-N dead load; returns the elastic equilibrium
    displacement field. Used to seed the cracked Picard so the secant
    operates around a realistic strain state from iter 0."""
    K_e = assemble_vk1_elastic(prob, mesh)
    u = np.zeros(mesh.ndof)
    prescribed = np.zeros(mesh.ndof, dtype=bool)
    prescribed[mesh.fixed] = True
    for d in mesh.load_dofs:
        prescribed[d] = True
    free = ~prescribed
    u[mesh.fixed] = 0.0
    for d in mesh.load_dofs:
        u[d] = delta_x
    # K_e is CSR; slicing on bool index returns CSR; spsolve accepts it.
    rhs = (mesh.F_dead[free]
           - K_e[:, prescribed][free, :] @ u[prescribed])
    u[free] = spsolve(K_e[:, free][free, :], rhs)
    return u


def assemble_vk1(u: np.ndarray, prob: VK1Problem, mesh: VK1Mesh,
                 soften: bool = True) -> tuple[csr_matrix, np.ndarray]:
    """Sparse secant assemble for VK1. Returns (K_csr, F_int)."""
    ndof = mesh.ndof
    t = prob.thickness
    F_int = np.zeros(ndof)
    n_e = len(mesh.tris)
    rows = np.empty(n_e * 36, dtype=np.int32)
    cols = np.empty(n_e * 36, dtype=np.int32)
    vals = np.empty(n_e * 36, dtype=np.float64)
    k = 0
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
        Ke = (B.T @ D @ B) * vol
        for i in range(6):
            for j in range(6):
                rows[k] = dofs[i]; cols[k] = dofs[j]; vals[k] = Ke[i, j]
                k += 1
        F_int[dofs] += (B.T @ sigma) * vol
    K = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    return K, F_int


def picard_displacement_controlled_vk1(
    u: np.ndarray, delta_x: float,
    prob: VK1Problem, mesh: VK1Mesh,
    max_iter: int = 200, tol: float = 2e-3, relax: float = 0.30,
    stall_window: int = 20, soften: bool = True,
    use_elastic_warmstart: bool = False,
) -> tuple[bool, float, np.ndarray]:
    """One displacement-controlled Picard solve for the VK1 cantilever.

    The horizontal displacement at the V-load patch is prescribed to
    `delta_x` (positive = applied toward +x), and the axial N is
    accumulated as a constant force vector. The horizontal reaction at
    the V patch is read off and reported as lambda_V = R_x / V_ref.
    """
    ndof = mesh.ndof
    load_dofs = list(mesh.load_dofs)
    prescribed = np.zeros(ndof, dtype=bool)
    prescribed[mesh.fixed] = True
    for d in load_dofs:
        prescribed[d] = True
    free = ~prescribed

    if use_elastic_warmstart:
        u = elastic_warmstart_vk1(delta_x, prob, mesh)
    else:
        u[mesh.fixed] = 0.0
        for d in load_dofs:
            u[d] = delta_x

    best_resid = np.inf
    stall = 0
    converged = False
    for it in range(max_iter):
        K, _ = assemble_vk1(u, prob, mesh, soften=soften)
        # K_ff u_f = F_dead_f - K_fp u_p   (sparse: K is CSR; slice & spsolve)
        rhs = (mesh.F_dead[free]
               - K[:, prescribed][free, :] @ u[prescribed])
        u_new_free = spsolve(K[:, free][free, :], rhs)

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

    K, _ = assemble_vk1(u, prob, mesh, soften=soften)
    # V_applied: external horizontal load (at the V-patch) that would
    # produce the same δ_x under load control. For FE displacement
    # control with prescribed u and applied F_dead, statics gives
    #   V_applied = (K u - F_dead) at the V-patch x-DOFs
    # (F_dead is zero at those DOFs in our setup). With u[V-patch] =
    # +δ_x, this is positive for the expected push-direction response.
    V_applied = float(np.sum((K @ u - mesh.F_dead)[load_dofs]))
    lam = V_applied / abs(prob.V_ref)
    return converged, lam, u


def assemble_vk1_tangent(u: np.ndarray, prob: VK1Problem, mesh: VK1Mesh
                         ) -> tuple[csr_matrix, np.ndarray]:
    """Consistent tangent and internal force for VK1.

    The secant stiffness returned by `assemble_vk1` is a serviceable
    preconditioner but a poor search direction: driving the true residual with
    it converges linearly at about 0.955 per iteration on this problem, so
    reaching the tolerance takes of order a hundred iterations per step. The
    consistent tangent is obtained per element by central-differencing the
    three-component constitutive law, six extra stress evaluations per element,
    which costs little and restores the fast convergence the deep-beam
    reference enjoys. The tangent is deliberately not clipped to stay positive
    definite; a Levenberg-Marquardt shift handles the indefinite region near
    the limit point.
    """
    from arclength_oracle_crisfield import _tangent_fd

    ndof = mesh.ndof
    t = prob.thickness
    F_int = np.zeros(ndof)
    n_e = len(mesh.tris)
    rows = np.empty(n_e * 36, dtype=np.int32)
    cols = np.empty(n_e * 36, dtype=np.int32)
    vals = np.empty(n_e * 36, dtype=np.float64)
    k = 0
    for e, (nodes, rx, ry) in enumerate(mesh.tris):
        B = mesh.B[e]
        dofs = np.array([2 * nodes[0], 2 * nodes[0] + 1,
                         2 * nodes[1], 2 * nodes[1] + 1,
                         2 * nodes[2], 2 * nodes[2] + 1])
        eps = B @ u[dofs]
        sigma, D = _tangent_fd(eps[0], eps[1], eps[2], rx, ry, prob.mat)
        vol = mesh.area[e] * t
        Ke = (B.T @ D @ B) * vol
        for i in range(6):
            for j in range(6):
                rows[k] = dofs[i]; cols[k] = dofs[j]; vals[k] = Ke[i, j]
                k += 1
        F_int[dofs] += (B.T @ sigma) * vol
    K = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    return K, F_int


def newton_displacement_control_vk1(
    prob: VK1Problem, mesh: VK1Mesh,
    delta_max: float = 50.0, n_steps: int = 50,
    newton_tol_rel: float = 5e-4, newton_max_iter: int = 120,
    lm_tries: int = 25, soften: bool = True, verbose: bool = False,
):
    """Displacement-controlled trace converged on the TRUE force residual.

    This replaces `picard_displacement_controlled_vk1` for anything that is
    reported. That routine iterates u <- K(u)^-1 F_dead and stops when the
    displacement stops changing, so it converges to the fixed point of the
    secant map. The principal secant moduli are clipped to a floor to keep the
    stiffness positive definite while the constitutive stress is evaluated
    independently of that clip, so K*u and the assembled sum of B^T sigma are
    different quantities and a root of one is not a root of the other. On the
    deep-beam benchmark of this study the same defect overstated capacity by
    66 per cent.

    The equilibrium condition for this problem is not that the internal force
    vanishes. A constant axial compression N is carried as `mesh.F_dead`, so
    equilibrium on the free degrees of freedom reads

        R(u) = F_int(u) - F_dead = 0 ,

    and the applied lateral load is read from the same residual at the
    prescribed patch, V = sum(R) over the V-patch x-DOFs. Note that
    `assemble_vk1` already returns F_int and the Picard routine simply
    discards it.

    The search direction comes from the clipped secant, which remains a
    serviceable preconditioner, with a Levenberg-Marquardt shift raised until
    the step reduces ||R|| and lowered on success. The floor therefore
    influences the path taken and not the answer arrived at.
    """
    ndof = mesh.ndof
    load_dofs = list(mesh.load_dofs)
    prescribed = np.zeros(ndof, dtype=bool)
    prescribed[mesh.fixed] = True
    for d in load_dofs:
        prescribed[d] = True
    free = ~prescribed

    # The residual is an absolute 2-norm, which accumulates over free degrees
    # of freedom, so it is scaled by sqrt(DOF) against the 40x20 deep-beam mesh
    # the tolerance was set on. Without this a finer mesh is held to a stricter
    # criterion rather than an equal one.
    tol = newton_tol_rel * abs(prob.V_ref) * np.sqrt(ndof / (41 * 21 * 2))

    u = np.zeros(ndof)
    out = []
    for k, delta in enumerate(np.linspace(0.0, delta_max, n_steps + 1)[1:]):
        if k == 0:
            # Cold-starting the first step leaves the whole 1370 kN axial dead
            # load unbalanced, and the iteration crawls out of that from a
            # residual of 1109 kN. The elastic solution under the same axial
            # load and the same prescribed displacement starts it at 112 kN
            # instead, which is the reason the Picard routine warm-starts too.
            u = elastic_warmstart_vk1(float(delta), prob, mesh)
        u[mesh.fixed] = 0.0
        for d in load_dofs:
            u[d] = float(delta)
        mu = 1e-3

        _K, F_int = assemble_vk1(u, prob, mesh, soften=soften)
        R = F_int - mesh.F_dead
        rn = float(np.linalg.norm(R[free]))
        converged = rn < tol
        nit = 0
        for it in range(newton_max_iter):
            nit = it + 1
            if rn < tol:
                converged = True
                break
            K, F_int = assemble_vk1_tangent(u, prob, mesh)
            R = F_int - mesh.F_dead
            rn = float(np.linalg.norm(R[free]))
            if rn < tol:
                converged = True
                break
            Kff = K[:, free][free, :]
            dscale = float(np.abs(Kff.diagonal()).mean()) or 1.0
            accepted = False
            for _ in range(lm_tries):
                A = Kff + coo_matrix(
                    (mu * dscale * np.ones(Kff.shape[0]),
                     (np.arange(Kff.shape[0]), np.arange(Kff.shape[0]))),
                    shape=Kff.shape).tocsr()
                du = spsolve(A, -R[free])
                u_try = u.copy()
                u_try[free] = u[free] + du
                _Kt, F_try = assemble_vk1(u_try, prob, mesh, soften=soften)
                rn_try = float(np.linalg.norm((F_try - mesh.F_dead)[free]))
                if np.isfinite(rn_try) and rn_try < rn:
                    accepted = True
                    mu = max(mu * 0.5, 1e-8)
                    break
                mu *= 3.0
            if not accepted:
                break
            u, rn = u_try, rn_try

        _K, F_int = assemble_vk1(u, prob, mesh, soften=soften)
        V_applied = float(np.sum((F_int - mesh.F_dead)[load_dofs]))
        lam = V_applied / abs(prob.V_ref)
        out.append({"delta_x": float(delta), "lam": lam,
                    "converged": bool(converged), "resid": rn,
                    "iters": nit, "u": u.copy()})
        if verbose and (k % 5 == 0 or k == n_steps - 1):
            print(f"  step {k + 1:>3}/{n_steps}  delta={delta:6.2f} mm  "
                  f"lam={lam:+.4f}  V={lam * prob.V_ref / 1e3:+.1f} kN  "
                  f"r={rn:.2e}{'' if converged else '  [no conv]'}",
                  flush=True)
    return out


# --------------------------------------------------------------------------- #
# Driver: sweep horizontal delta past the limit point
# --------------------------------------------------------------------------- #


@dataclass
class VK1CurvePoint:
    delta_x: float
    lam: float
    converged: bool


def field_diagnostics_vk1(u: np.ndarray, prob: VK1Problem, mesh: VK1Mesh,
                          soften: bool = True) -> dict:
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


def trace_vk1_curve(
    prob: VK1Problem, mesh: VK1Mesh, delta_max_mm: float,
    n_steps: int = 60, soften: bool = True, verbose: bool = False,
) -> tuple[list[VK1CurvePoint], np.ndarray, list[dict]]:
    """Sweep δ_x from 0 to delta_max_mm in n_steps equal increments."""
    u = np.zeros(mesh.ndof)
    curve: list[VK1CurvePoint] = []
    diags: list[dict] = []
    schedule = np.linspace(0.0, delta_max_mm, n_steps + 1)[1:]
    for k, delta in enumerate(schedule):
        # Elastic warm-start only on the first step. Subsequent steps
        # continue from the previous converged cracked state (cheaper
        # and consistent with the cracked branch of the curve).
        warm = (k == 0)
        conv, lam, u = picard_displacement_controlled_vk1(
            u, float(delta), prob, mesh, soften=soften,
            use_elastic_warmstart=warm)
        curve.append(VK1CurvePoint(float(delta), float(lam), bool(conv)))
        diag = field_diagnostics_vk1(u, prob, mesh, soften=soften)
        diags.append(diag)
        if verbose:
            tag = " " if conv else "*"
            print(f"  step {k + 1:>2}/{n_steps}  delta={delta:7.3f} mm  "
                  f"lambda={lam:+.4f}  V={lam * prob.V_ref / 1e3:+.1f} kN  "
                  f"|e2|_max={diag['max_e2_mag']:.4f}  "
                  f"kc2_min={diag['min_kc2']:.3f}  "
                  f"e1_max={diag['max_e1']:.4f} {tag}")
    return curve, u, diags


# --------------------------------------------------------------------------- #
# VK1 problem factory matching the P2 reinforcement
# --------------------------------------------------------------------------- #


def vk1_default() -> VK1Problem:
    """VK1 with the reinforcement densities from P2 (rho_l = 0.82%
    flexural / vertical, rho_t = 0.08% shear / horizontal, with the
    top 300 mm of the wall having a denser hoop spacing)."""
    rho_l = 0.0082
    rho_t = 0.0008
    rho_min = 0.0010
    H = 3700.0

    def rho_x_field(x: float, y: float) -> float:
        # horizontal hoops: denser near the top per the test specimen
        densified = (y > H - 300.0)
        rho_top = max(rho_t * 200.0 / 75.0, rho_min)
        return rho_top if densified else max(rho_t, rho_min)

    def rho_y_field(x: float, y: float) -> float:
        return max(rho_l, rho_min)

    return VK1Problem(rho_x=rho_x_field, rho_y=rho_y_field)
