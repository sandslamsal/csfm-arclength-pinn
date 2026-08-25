"""(x, y, delta) -> CSFM-strain lookup for the VK1 strain-anchor PINN.

Reads the full-displacement VK1 CSFM trace
`oracle/vk1_reference_full.json` produced by `run_vk1_withN_full.py`,
which holds nodal displacements at every Picard step. For each step
we recompute the per-element strain via B @ u_elem and store a
(n_steps, ny, nx, 2, 3) tensor of strains
(ny*nx cells, 2 triangles per cell, 3 strain components).

The lookup `strain_at_state(x, y, delta_mm)`:
  1. linearly interpolates between the two CSFM steps that bracket
     `delta_mm`
  2. picks the (cell_i, cell_j, triangle_k) containing (x, y)
  3. returns the corresponding (ex, ey, gxy) torch tensor
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass
class VK1StrainTarget:
    nx: int
    ny: int
    L: float
    H: float
    dx: float
    dy: float
    deltas: np.ndarray            # (n_steps,)
    strain: np.ndarray            # (n_steps, ny, nx, 2, 3)


def build_target(json_path: Path) -> VK1StrainTarget:
    print(f"  [vk1_strain_target] loading {json_path.name}")
    data = json.load(open(json_path))
    nx, ny = data["nx"], data["ny"]
    L, H = data["L"], data["H"]
    dx, dy = L / nx, H / ny
    nnx = nx + 1
    deltas = np.array([p["delta_x"] for p in data["curve"]], dtype=float)
    u_all = np.array(data["u_all"], dtype=float)   # (n_steps, n_node, 2)

    def nid(i, j):
        return j * nnx + i

    n_steps = u_all.shape[0]
    strain = np.zeros((n_steps, ny, nx, 2, 3), dtype=np.float32)

    # Pre-compute B per (cell, triangle); both triangles in a cell
    # share the same vertex coordinates so B is the same for every
    # step
    B_lo = np.zeros((ny, nx, 3, 6))
    B_hi = np.zeros((ny, nx, 3, 6))
    dofs_lo = np.zeros((ny, nx, 6), dtype=int)
    dofs_hi = np.zeros((ny, nx, 6), dtype=int)
    xy_nodes = np.array([[i * dx, j * dy] for j in range(ny + 1)
                          for i in range(nnx)], dtype=float)
    for j in range(ny):
        for i in range(nx):
            a, b = nid(i, j), nid(i + 1, j)
            c, d = nid(i + 1, j + 1), nid(i, j + 1)
            for k, nodes in enumerate([[a, b, c], [a, c, d]]):
                p = xy_nodes[nodes]
                b1, b2, b3 = (p[1, 1] - p[2, 1], p[2, 1] - p[0, 1],
                              p[0, 1] - p[1, 1])
                c1, c2, c3 = (p[2, 0] - p[1, 0], p[0, 0] - p[2, 0],
                              p[1, 0] - p[0, 0])
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
                if k == 0:
                    B_lo[j, i] = B
                    dofs_lo[j, i] = dofs
                else:
                    B_hi[j, i] = B
                    dofs_hi[j, i] = dofs

    # Per-step strain
    for s_idx in range(n_steps):
        u_flat = u_all[s_idx].flatten()
        for j in range(ny):
            for i in range(nx):
                u_lo = u_flat[dofs_lo[j, i]]
                u_hi = u_flat[dofs_hi[j, i]]
                strain[s_idx, j, i, 0] = (B_lo[j, i] @ u_lo).astype(np.float32)
                strain[s_idx, j, i, 1] = (B_hi[j, i] @ u_hi).astype(np.float32)

    return VK1StrainTarget(nx=nx, ny=ny, L=L, H=H, dx=dx, dy=dy,
                           deltas=deltas, strain=strain)


_CACHED: dict[str, VK1StrainTarget] = {}


def get_strain_target(json_path: Path) -> VK1StrainTarget:
    key = str(json_path)
    if key not in _CACHED:
        _CACHED[key] = build_target(json_path)
    return _CACHED[key]


def strain_at_state(tgt: VK1StrainTarget,
                    x: Tensor, y: Tensor, delta: Tensor
                    ) -> tuple[Tensor, Tensor, Tensor]:
    """Per-sample lookup of target (ex, ey, gxy) at the given physical
    (x, y) and current delta. delta is per-sample (shape (N, 1)).
    """
    xq = x.detach().cpu().numpy().squeeze(-1)
    yq = y.detach().cpu().numpy().squeeze(-1)
    dq = delta.detach().cpu().numpy().squeeze(-1)
    nx, ny, dx, dy = tgt.nx, tgt.ny, tgt.dx, tgt.dy
    # cell + triangle
    i = np.clip((xq / dx).astype(int), 0, nx - 1)
    j = np.clip((yq / dy).astype(int), 0, ny - 1)
    local_x = xq - i * dx
    local_y = yq - j * dy
    is_lower = local_y * dx < local_x * dy
    k = np.where(is_lower, 0, 1).astype(int)
    # bracketing CSFM step indices by delta
    d_clamped = np.clip(dq, tgt.deltas[0], tgt.deltas[-1])
    idx = np.clip(np.searchsorted(tgt.deltas, d_clamped, side="right"),
                  1, len(tgt.deltas) - 1)
    d_lo = tgt.deltas[idx - 1]
    d_hi = tgt.deltas[idx]
    w = (d_clamped - d_lo) / (d_hi - d_lo + 1e-12)
    s_lo = tgt.strain[idx - 1, j, i, k]    # (N, 3)
    s_hi = tgt.strain[idx, j, i, k]
    s = s_lo + w[:, None] * (s_hi - s_lo)
    ex = torch.from_numpy(s[:, 0].astype(np.float32)).unsqueeze(-1)
    ey = torch.from_numpy(s[:, 1].astype(np.float32)).unsqueeze(-1)
    gxy = torch.from_numpy(s[:, 2].astype(np.float32)).unsqueeze(-1)
    return ex, ey, gxy


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    tgt = build_target(here.parent / "oracle" / "vk1_reference_full.json")
    print(f"  built target: nx={tgt.nx} ny={tgt.ny}  "
          f"n_steps={len(tgt.deltas)} delta_max={tgt.deltas[-1]:.1f}")
    # Smoke
    x = torch.tensor([[750.0], [200.0]])
    y = torch.tensor([[1850.0], [3300.0]])
    d = torch.tensor([[30.0], [10.0]])
    ex, ey, gxy = strain_at_state(tgt, x, y, d)
    print(f"  ex={ex.flatten().tolist()}  ey={ey.flatten().tolist()}  "
          f"gxy={gxy.flatten().tolist()}")
