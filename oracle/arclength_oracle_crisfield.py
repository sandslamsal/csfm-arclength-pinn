"""Crisfield cylindrical arc-length path-follower for the deep beam.

Independent load-controlled continuation reference for the post-peak
descending branch. The secant-Picard reference of `arclength_oracle.py`
drives the solve in *displacement* control: it prescribes the load-patch
deflection delta and reads the reaction off the assembled secant stiffness.
This module instead keeps the load a free unknown (the load factor lambda)
and follows the equilibrium curve with a *cylindrical arc-length*
predictor-corrector and full-residual Newton iterations.

Two solvers agreeing on the descending branch under fundamentally
different continuation schemes -- prescribed-displacement Picard vs
load-controlled arc-length Newton -- is the path-following validation
previously deferred.

Why a CONSISTENT TANGENT (not the secant operator):
  Load-controlled arc-length must traverse the limit point. At the limit
  point the consistent tangent K_T = d F_int / d u becomes singular and
  then indefinite; that sign change is exactly what lets the predictor
  reverse lambda onto the descending branch. The secant operator that
  `arclength_oracle.py::membrane()` returns is clipped positive-definite
  (E_i in [Emin, Ec0]) and therefore NEVER sees the limit point -- a pure
  secant operator cannot follow the descending branch under load control.

  We obtain K_T by finite-differencing the *same* validated branched
  constitutive map `membrane()` (the clean numpy CSFM map shared with the
  displacement-controlled reference), element by element, with NO
  positive-definite clipping so the assembled K_T stays indefinite past
  the peak. The constitutive map is therefore identical to the secant
  reference; only the solver (arc-length Newton vs displacement Picard)
  and its linearisation (consistent FD tangent vs clipped secant) differ.

Numerical scheme -- Crisfield's cylindrical arc-length form:
  K_T(u) du = lambda * F_ref - F_int(u)        (Newton)
  ||u_{n+1} - u_start||^2 = (Delta l)^2         (cylindrical constraint)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Reuse the existing problem + mesh + the validated branched membrane.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arclength_oracle import Material, Mesh, Problem, build_mesh, membrane


def _sigma(ex: float, ey: float, gxy: float,
           rho_x: float, rho_y: float, mat: Material) -> np.ndarray:
    """Global-frame stress vector from the validated branched membrane."""
    sigma, _, _ = membrane(ex, ey, gxy, rho_x, rho_y, mat)
    return sigma


def _tangent_fd(ex: float, ey: float, gxy: float,
                rho_x: float, rho_y: float, mat: Material,
                h: float = 1e-7) -> tuple[np.ndarray, np.ndarray]:
    """Consistent 3x3 tangent K_T = d sigma / d eps by central finite
    differences of the branched membrane. No positive-definite clipping:
    the tangent is allowed to go indefinite (that is the point)."""
    eps = np.array([ex, ey, gxy], dtype=float)
    sigma = _sigma(ex, ey, gxy, rho_x, rho_y, mat)
    D = np.zeros((3, 3))
    for j in range(3):
        ep = eps.copy(); ep[j] += h
        em = eps.copy(); em[j] -= h
        sp = _sigma(ep[0], ep[1], ep[2], rho_x, rho_y, mat)
        sm = _sigma(em[0], em[1], em[2], rho_x, rho_y, mat)
        D[:, j] = (sp - sm) / (2.0 * h)
    return sigma, D


def assemble_tangent(u: np.ndarray, prob: Problem, mesh: Mesh
                     ) -> tuple[csr_matrix, np.ndarray]:
    """Assemble the consistent (possibly indefinite) global tangent K_T
    and internal force F_int at displacement state u."""
    ndof = mesh.ndof
    t = prob.thickness
    n_e = len(mesh.tris)
    rows = np.empty(n_e * 36, dtype=np.int32)
    cols = np.empty(n_e * 36, dtype=np.int32)
    vals = np.empty(n_e * 36, dtype=np.float64)
    F_int = np.zeros(ndof)
    k = 0
    for e, (nodes, rx, ry) in enumerate(mesh.tris):
        B = mesh.B[e]
        area = mesh.area[e]
        dofs = np.array([2 * nodes[0], 2 * nodes[0] + 1,
                         2 * nodes[1], 2 * nodes[1] + 1,
                         2 * nodes[2], 2 * nodes[2] + 1])
        eps = B @ u[dofs]
        sigma, D = _tangent_fd(eps[0], eps[1], eps[2], rx, ry, prob.mat)
        vol = area * t
        Ke = (B.T @ D @ B) * vol
        F_int[dofs] += (B.T @ sigma) * vol
        for i in range(6):
            for j in range(6):
                rows[k] = dofs[i]
                cols[k] = dofs[j]
                vals[k] = Ke[i, j]
                k += 1
    K = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    return K, F_int


def assemble_internal(u: np.ndarray, prob: Problem, mesh: Mesh) -> np.ndarray:
    """Internal force vector only (no tangent). One membrane evaluation per
    element -- ~6x cheaper than `assemble_tangent`, used inside the Newton
    line search where only the residual is needed."""
    F_int = np.zeros(mesh.ndof)
    t = prob.thickness
    for e, (nodes, rx, ry) in enumerate(mesh.tris):
        B = mesh.B[e]
        dofs = np.array([2 * nodes[0], 2 * nodes[0] + 1,
                         2 * nodes[1], 2 * nodes[1] + 1,
                         2 * nodes[2], 2 * nodes[2] + 1])
        eps = B @ u[dofs]
        sigma = _sigma(eps[0], eps[1], eps[2], rx, ry, prob.mat)
        F_int[dofs] += (B.T @ sigma) * (mesh.area[e] * t)
    return F_int


@dataclass
class DCPoint:
    """One displacement-controlled equilibrium point."""
    delta: float
    lam: float
    newton_iters: int
    converged: bool
    resid: float
    # the converged displacement field, kept so callers can interrogate the
    # state at a limit point without re-solving to reach it. Optional and
    # defaulted, so existing constructions are unaffected.
    u: object = None


def newton_displacement_control(
        prob: Problem, mesh: Mesh,
        delta_max: float = 10.0, n_steps: int = 40,
        newton_tol_rel: float = 5e-4, newton_max_iter: int = 120,
        lm_tries: int = 25, verbose: bool = True) -> list[DCPoint]:
    """Levenberg-Marquardt-damped consistent-tangent Newton under
    DISPLACEMENT control, mirroring the secant-Picard reference's control
    scheme so the two solvers can be compared apples-to-apples (same
    prescribed load-patch deflection, independent linearisation).

    The load-patch y-DOFs are prescribed to -delta (rigid bearing, as in
    `arclength_oracle.py`); the load factor is read from the patch reaction
    lambda = -sum(F_int_patch) / P_ref.

    Why LM damping (not plain / line-search Newton): once the cracked
    membrane softens, the consistent tangent goes near-singular and plain
    Newton produces a huge, wrong step that no line search can rescue (it
    stalls with the residual pinned at ~1e6). Adding a diagonal shift
    mu*scale*I makes the step a robust interpolation between the Newton
    direction (small mu) and scaled gradient descent (large mu); mu is
    raised until the step reduces the residual and lowered on success.
    This is the analogue of the secant reference's 0.30 relaxation, but
    built on the *consistent* tangent, so it remains an independent
    linearisation of the same equilibrium.
    """
    from scipy.sparse import diags

    Pref = abs(prob.P_ref)
    patch_y = [2 * n + 1 for n in mesh.load_dofs]
    prescribed = mesh.fixed.copy()
    for d in patch_y:
        prescribed[d] = True
    free = ~prescribed
    tol = newton_tol_rel * Pref

    u = np.zeros(mesh.ndof)
    history: list[DCPoint] = []
    schedule = np.linspace(0.0, delta_max, n_steps + 1)[1:]
    for k, delta in enumerate(schedule):
        u[mesh.fixed] = 0.0
        for d in patch_y:
            u[d] = -float(delta)
        mu = 1e-3                       # reset damping each step

        rn = float(np.linalg.norm(assemble_internal(u, prob, mesh)[free]))
        converged = False
        nit = 0
        for it in range(newton_max_iter):
            nit = it + 1
            if rn < tol:
                converged = True
                break
            K_T, F_int = assemble_tangent(u, prob, mesh)
            rn = float(np.linalg.norm(F_int[free]))
            if rn < tol:
                converged = True
                break
            Kff = K_T[free][:, free]
            dscale = float(np.abs(Kff.diagonal()).mean()) or 1.0
            accepted = False
            for _ in range(lm_tries):
                A = Kff + diags(mu * dscale * np.ones(Kff.shape[0]))
                du = spsolve(A, -F_int[free])
                u_try = u.copy()
                u_try[free] = u[free] + du
                rn_try = float(
                    np.linalg.norm(assemble_internal(u_try, prob, mesh)[free]))
                if np.isfinite(rn_try) and rn_try < rn:
                    accepted = True
                    mu = max(mu * 0.5, 1e-8)
                    break
                mu *= 3.0
            if not accepted:
                break
            u = u_try
            rn = rn_try

        lam = -float(np.sum(assemble_internal(u, prob, mesh)[patch_y])) / Pref
        history.append(DCPoint(delta=float(delta), lam=lam,
                               newton_iters=nit, converged=converged,
                               resid=rn, u=u.copy()))
        if verbose and (k % 4 == 0 or k == n_steps - 1):
            flag = "" if converged else "  [no conv]"
            print(f"  step {k + 1:>2}/{n_steps}  delta={delta:6.3f} mm  "
                  f"lam={lam:+.4f}  newton={nit}  r={rn:.1e}{flag}")
    return history


@dataclass
class CrisfieldPoint:
    delta: float
    lam: float
    arc_len_increment: float
    newton_iters: int
    converged: bool


def _solve_free(K: csr_matrix, rhs: np.ndarray,
                free: np.ndarray) -> np.ndarray:
    return spsolve(K[free][:, free], rhs[free])


def crisfield_solve(prob: Problem, mesh: Mesh,
                    arc: float = 5.0,
                    n_steps: int = 80,
                    newton_tol: float = 1e-3,
                    newton_max_iter: int = 25,
                    max_cuts: int = 6
                    ) -> list[CrisfieldPoint]:
    """Cylindrical arc-length predictor-corrector with full-residual
    Newton on the consistent FD tangent.

    `arc` is the arc-length increment in the free-DOF displacement norm
    (units: mm, Euclidean over all free DOFs). Adaptive step cutting
    halves `arc` for a step that fails to converge.
    """
    ndof = mesh.ndof
    free = ~mesh.fixed
    F_ref = mesh.F_ref.copy()
    f_ref_norm = float(np.linalg.norm(F_ref[free]))
    load_ydofs = [2 * n + 1 for n in mesh.load_dofs]

    u = np.zeros(ndof)
    lam = 0.0
    du_prev = None                      # previous converged free increment
    history: list[CrisfieldPoint] = []

    step = 0
    arc_step = arc
    while step < n_steps:
        u_start = u.copy()
        lam_start = lam

        # ---- Predictor: solve K_T du_t = F_ref ----
        K_T, _ = assemble_tangent(u, prob, mesh)
        du_t = np.zeros(ndof)
        du_t[free] = _solve_free(K_T, F_ref, free)
        norm_t = float(np.linalg.norm(du_t[free]))
        if norm_t < 1e-30:
            print("  [predictor] singular tangent, stopping")
            break

        # Sign control: align with the previous converged increment so
        # lambda reverses automatically once K_T turns indefinite.
        if du_prev is None:
            sign = 1.0
        else:
            sign = 1.0 if float(du_prev @ du_t[free]) >= 0.0 else -1.0

        dlam = sign * arc_step / norm_t
        u[free] = u_start[free] + dlam * du_t[free]
        lam = lam_start + dlam
        du_pred = (dlam * du_t[free]).copy()    # predictor increment

        # ---- Corrector: Newton enforcing equilibrium + arc constraint ----
        converged = False
        r_norm = np.inf
        inew = 0
        for inew in range(1, newton_max_iter + 1):
            K_T, F_int = assemble_tangent(u, prob, mesh)
            resid = lam * F_ref - F_int
            r_norm = float(np.linalg.norm(resid[free]))
            if r_norm < newton_tol * max(f_ref_norm, 1.0):
                converged = True
                break
            du_bar = np.zeros(ndof)
            du_bar[free] = _solve_free(K_T, resid, free)
            du_r = np.zeros(ndof)
            du_r[free] = _solve_free(K_T, F_ref, free)

            # cylindrical constraint ||(u - u_start) + du_bar + dl*du_r||^2 = arc^2
            p = (u[free] - u_start[free]) + du_bar[free]
            a = float(du_r[free] @ du_r[free])
            b = 2.0 * float(p @ du_r[free])
            cc = float(p @ p) - arc_step ** 2
            disc = b * b - 4.0 * a * cc
            if a < 1e-30:
                dl = 0.0
            elif disc < 0.0:
                # no real intersection: minimise constraint residual
                dl = -b / (2.0 * a)
            else:
                sq = np.sqrt(disc)
                r1 = (-b + sq) / (2.0 * a)
                r2 = (-b - sq) / (2.0 * a)
                # choose the root keeping the increment aligned with the
                # predictor (avoids doubling back on the path)
                d1 = p + r1 * du_r[free]
                d2 = p + r2 * du_r[free]
                dl = r1 if float(d1 @ du_pred) >= float(d2 @ du_pred) else r2

            u[free] = u[free] + du_bar[free] + dl * du_r[free]
            lam = lam + dl

        if not converged and arc_step > arc / (2 ** max_cuts):
            # step cut: restore and retry with a smaller arc
            u = u_start.copy()
            lam = lam_start
            arc_step *= 0.5
            continue

        du_inc = (u[free] - u_start[free]).copy()
        du_prev = du_inc
        delta_y = -float(np.mean(u[load_ydofs]))
        history.append(CrisfieldPoint(
            delta=delta_y, lam=lam,
            arc_len_increment=float(np.linalg.norm(du_inc)),
            newton_iters=inew, converged=converged))
        if step % 5 == 0 or step == n_steps - 1:
            flag = "" if converged else "  [no conv]"
            print(f"  step {step + 1:>2}/{n_steps}  lam={lam:+.4f}  "
                  f"delta_y={delta_y:7.3f} mm  newton={inew}  "
                  f"arc={arc_step:.3f}{flag}")

        # gently grow the arc back toward nominal after a successful step
        if converged and inew <= 4:
            arc_step = min(arc, arc_step * 1.3)
        step += 1

    return history


def main() -> None:
    import json
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from deepbeam import deepbeam
    prob = deepbeam()
    mesh = build_mesh(prob)
    here = Path(__file__).resolve().parent
    print(f"Crisfield-C1: deep beam, ndof={mesh.ndof}, "
          f"F_ref_sum={np.sum(mesh.F_ref[1::2]):.1f} N")

    # ---- Primary cross-check: displacement control (matches the secant
    # reference's control scheme; independent consistent-tangent LM solver).
    print("\n[displacement control] LM-damped consistent-tangent Newton")
    dc = newton_displacement_control(prob, mesh, delta_max=10.0, n_steps=40)
    dc_out = [{"delta": float(p.delta), "lam": float(p.lam),
               "newton_iters": int(p.newton_iters),
               "converged": bool(p.converged), "resid": float(p.resid)}
              for p in dc]
    with open(here / "deepbeam_crisfield.json", "w") as f:
        json.dump({"method": "displacement_control_LM_consistent_tangent",
                   "curve": dc_out}, f, indent=2)
    dlam = [p["lam"] for p in dc_out]
    print(f"-> deepbeam_crisfield.json ({len(dc_out)} pts, displacement "
          f"control), peak lam = {max(dlam):.3f}")

    # ---- Secondary: load-controlled cylindrical arc-length (records the
    # load-control limit point, which differs from displacement control
    # under unregularised softening -- see module docstring).
    print("\n[load control] cylindrical arc-length")
    arc = crisfield_solve(prob, mesh, arc=4.0, n_steps=80)
    arc_out = [{"delta": float(p.delta), "lam": float(p.lam),
                "newton_iters": int(p.newton_iters),
                "converged": bool(p.converged)} for p in arc]
    with open(here / "deepbeam_crisfield_arclength.json", "w") as f:
        json.dump({"method": "load_control_cylindrical_arclength",
                   "curve": arc_out}, f, indent=2)
    alam = [p["lam"] for p in arc_out]
    print(f"-> deepbeam_crisfield_arclength.json ({len(arc_out)} pts, load "
          f"control), peak lam = {max(alam):.3f}")


if __name__ == "__main__":
    main()
