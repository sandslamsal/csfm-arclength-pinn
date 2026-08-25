"""Background dead-load field for the VK1 PINN.

The VK1 wall pier in the Bimschas test is loaded by a constant axial
N = 1370 kN on the top edge and a parametrised horizontal V on the
left face. The PINN's hard initial condition u(s=0) = 0 is
incompatible with the non-zero static deformation that N alone
imposes. This module pre-computes the elastic FE solution of the
wall pier under N alone (no V) and provides a torch lookup of its
strain field at arbitrary (x, y). The arc-length-PINN ansatz is
then lifted to
  u_total(s, x, y) = u_N_background(x, y) + s * tilde_N(s, x, y)
so u_total(s=0) = u_N_background satisfies the dead-load BC sigma_yy
= -p_N at the top edge by construction. The PINN learns only the
V-driven incremental deformation on top of the background.

We pre-compute the strain field (epsilon_x, epsilon_y, gamma_xy) per
CST element, then provide a torch-compatible lookup of the background
strain at any (x, y) by element membership. This bypasses the
discontinuous gradients that a bilinear-displacement interpolation
would feed into autograd.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oracle"))
from arclength_oracle_vk1 import (                                         # noqa: E402
    VK1Problem, build_vk1_mesh, vk1_default,
    assemble_vk1_elastic,
)
from scipy.sparse.linalg import spsolve                                    # noqa: E402


@dataclass
class VK1Background:
    """Pre-computed elastic background under axial-N only."""
    nx: int
    ny: int
    L: float
    H: float
    dx: float
    dy: float
    # ex, ey, gxy stored per CST triangle, indexed (j, i, k) with
    # k in {0, 1} (two triangles per cell)
    ex: np.ndarray  # shape (ny, nx, 2)
    ey: np.ndarray
    gxy: np.ndarray
    # Bilinear-interpolated nodal displacement (for the ansatz)
    u_node: np.ndarray  # shape (nny*nnx, 2)
    xy_node: np.ndarray  # shape (n_node, 2)


def solve_background(prob: VK1Problem) -> VK1Background:
    """Solve the elastic FE under N alone (no V-patch prescribed
    displacement; the V-patch nodes are FREE so the wall responds to
    N only). Return per-element strain + nodal displacement."""
    mesh = build_vk1_mesh(prob)
    K_e = assemble_vk1_elastic(prob, mesh)
    ndof = mesh.ndof
    u = np.zeros(ndof)
    # Only the bottom clamp is prescribed; the V-patch is free
    prescribed = mesh.fixed.copy()
    free = ~prescribed
    rhs = mesh.F_dead[free] - K_e[:, prescribed][free, :] @ u[prescribed]
    u[free] = spsolve(K_e[:, free][free, :], rhs)
    u_node = u.reshape(-1, 2)

    nx, ny = prob.nx, prob.ny
    dx, dy = prob.L / nx, prob.H / ny
    nnx = nx + 1

    ex = np.zeros((ny, nx, 2))
    ey = np.zeros((ny, nx, 2))
    gxy = np.zeros((ny, nx, 2))

    # Same triangle ordering as build_vk1_mesh: per cell (i, j),
    # two triangles: (a, b, c) and (a, c, d) where
    # a = nid(i, j), b = nid(i+1, j), c = nid(i+1, j+1), d = nid(i, j+1)
    def nid(i, j):
        return j * nnx + i

    for j in range(ny):
        for i in range(nx):
            a, b, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            for k, nodes in enumerate([[a, b, c], [a, c, d]]):
                # Recompute B for this triangle
                p = mesh.xy[nodes]
                b1, b2, b3 = p[1, 1] - p[2, 1], p[2, 1] - p[0, 1], p[0, 1] - p[1, 1]
                c1, c2, c3 = p[2, 0] - p[1, 0], p[0, 0] - p[2, 0], p[1, 0] - p[0, 0]
                det = p[0, 0] * b1 + p[1, 0] * b2 + p[2, 0] * b3
                inv = 1.0 / det
                B = np.array([
                    [b1 * inv, 0, b2 * inv, 0, b3 * inv, 0],
                    [0, c1 * inv, 0, c2 * inv, 0, c3 * inv],
                    [c1 * inv, b1 * inv, c2 * inv, b2 * inv, c3 * inv, b3 * inv],
                ])
                dofs = np.array([2 * nodes[0], 2 * nodes[0] + 1,
                                 2 * nodes[1], 2 * nodes[1] + 1,
                                 2 * nodes[2], 2 * nodes[2] + 1])
                eps = B @ u[dofs]
                ex[j, i, k] = eps[0]
                ey[j, i, k] = eps[1]
                gxy[j, i, k] = eps[2]

    return VK1Background(
        nx=nx, ny=ny, L=prob.L, H=prob.H, dx=dx, dy=dy,
        ex=ex, ey=ey, gxy=gxy, u_node=u_node, xy_node=mesh.xy,
    )


# Cached singleton -- rebuilds only once per process even with many
# epoch calls.
_CACHED: dict[tuple, VK1Background] = {}


def get_background(prob: VK1Problem) -> VK1Background:
    key = (prob.L, prob.H, prob.thickness, prob.nx, prob.ny,
           prob.N, prob.bearing_N_half)
    if key not in _CACHED:
        print(f"  [vk1_background] solving elastic FE under N alone "
              f"(nx={prob.nx}, ny={prob.ny})")
        _CACHED[key] = solve_background(prob)
    return _CACHED[key]


def bg_strain_at(bg: VK1Background, x: Tensor, y: Tensor
                 ) -> tuple[Tensor, Tensor, Tensor]:
    """Look up the background strain (ex, ey, gxy) at each (x, y).
    Returns three (N, 1) torch tensors. The strain is constant per
    triangle; we pick the triangle within each cell by the diagonal
    test (a, b, c) for y - j*dy <= (i+1)*dx - x else (a, c, d).
    """
    xq = x.detach().cpu().numpy().squeeze(-1)
    yq = y.detach().cpu().numpy().squeeze(-1)
    nx, ny, dx, dy = bg.nx, bg.ny, bg.dx, bg.dy
    i = np.clip((xq / dx).astype(int), 0, nx - 1)
    j = np.clip((yq / dy).astype(int), 0, ny - 1)
    # Triangle index within cell: triangle 0 is (a, b, c) below the
    # cell's lower-right-to-upper-left diagonal; triangle 1 above.
    # Diagonal at x + y = (i+1)*dx + j*dy ... use a simple split.
    local_x = xq - i * dx
    local_y = yq - j * dy
    # Triangle 0: nodes (i,j), (i+1,j), (i+1,j+1) -- lower-right
    # Triangle 1: nodes (i,j), (i+1,j+1), (i,j+1) -- upper-left
    # Diagonal: y = (x/dx) * dy, i.e. local_y < local_x*(dy/dx) -> tri 0
    is_lower = local_y * dx < local_x * dy
    k = np.where(is_lower, 0, 1).astype(int)
    ex = torch.from_numpy(bg.ex[j, i, k].astype(np.float32))
    ey = torch.from_numpy(bg.ey[j, i, k].astype(np.float32))
    gxy = torch.from_numpy(bg.gxy[j, i, k].astype(np.float32))
    return ex.unsqueeze(-1), ey.unsqueeze(-1), gxy.unsqueeze(-1)


def bg_disp_at(bg: VK1Background, x: Tensor, y: Tensor
               ) -> tuple[Tensor, Tensor]:
    """Bilinear-interpolated background displacement at each (x, y).
    Returns (ux, uy) as (N, 1) torch tensors. Used for the ansatz
    additive term and for the directional arc-length probe."""
    xq = x.detach().cpu().numpy().squeeze(-1)
    yq = y.detach().cpu().numpy().squeeze(-1)
    nx, ny, dx, dy = bg.nx, bg.ny, bg.dx, bg.dy
    nnx = nx + 1
    i = np.clip((xq / dx).astype(int), 0, nx - 1)
    j = np.clip((yq / dy).astype(int), 0, ny - 1)
    fx = (xq - i * dx) / dx
    fy = (yq - j * dy) / dy
    n00 = j * nnx + i
    n10 = j * nnx + (i + 1)
    n01 = (j + 1) * nnx + i
    n11 = (j + 1) * nnx + (i + 1)
    u00 = bg.u_node[n00]
    u10 = bg.u_node[n10]
    u01 = bg.u_node[n01]
    u11 = bg.u_node[n11]
    w00 = (1 - fx) * (1 - fy); w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy;       w11 = fx * fy
    ux = w00 * u00[:, 0] + w10 * u10[:, 0] + w01 * u01[:, 0] + w11 * u11[:, 0]
    uy = w00 * u00[:, 1] + w10 * u10[:, 1] + w01 * u01[:, 1] + w11 * u11[:, 1]
    return (torch.from_numpy(ux.astype(np.float32)).unsqueeze(-1),
            torch.from_numpy(uy.astype(np.float32)).unsqueeze(-1))


if __name__ == "__main__":
    prob = vk1_default()
    bg = solve_background(prob)
    # Sanity: top-edge u_y should be roughly -N / (E * effective area) * H
    Ec = prob.mat.Ec0
    A = prob.L * prob.thickness
    expected_uy_top = -prob.N * prob.H / (Ec * A)
    print(f"  expected uniform-N top u_y ~ {expected_uy_top:.4f} mm")
    nnx = prob.nx + 1
    top_uys = bg.u_node[-nnx:, 1]
    print(f"  actual top u_y range: [{top_uys.min():.4f}, "
          f"{top_uys.max():.4f}] mm")
    # Background sigma_yy at top: epsilon_y * (E + rho*Es) ~ -p_N
    print(f"  top eps_y[0..3] cells: {bg.ey[-1, :3, 0]}")
