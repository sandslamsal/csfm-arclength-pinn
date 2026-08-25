"""Pre-train the P3 PINN on the elastic FE solution to escape the trivial
zero-attractor.

The arc-length PINN has a strong local minimum at `u = lambda = 0`: the
equilibrium, support, traction, and free-edge losses are all *minimised*
by the trivial state, and only the arc-length constraint pushes away from
it. The arc gradient alone (weight ~1.3 from ReLoBraLo) is not strong
enough to escape that saddle from the standard small-weight init.

The fix used here is the standard one for this class of pathology: start
the optimiser from a non-trivial basin. Specifically:

  1. Solve the deepbeam under unit load with a self-contained linear-
     elastic FE (CST triangles, same mesh layout as the oracle).
  2. Scale the resulting displacement field so the loaded-patch deflection
     equals the first stage's S_max — that way the arc-length constraint
     is approximately satisfied at init.
  3. Supervised-fit both the field head and the lambda head to that scaled
     elastic state for a few hundred Adam iterations.
  4. Save the pre-trained weights; `pinn_arclength.py` loads them before
     the staged arc-length training begins.

The elastic FE here is a thin self-contained NumPy port — independent of
P2's `elastic_fe.py` to keep this script standalone, and small enough
(~80 lines) that it's not worth abstracting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

# Import local P3 model FIRST, then add P2's path for problem.py.
# Reversing this order makes P2/pinn/model.py shadow P3/pinn/model.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ArclengthPINN                                             # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "P2" / "pinn"))
from problem import DeepBeam                                                # noqa: E402

torch.set_default_dtype(torch.float32)
SEED = 20260522
U0 = 1.0e-3


# --------------------------------------------------------------------------- #
# Self-contained elastic CST FE solver
# --------------------------------------------------------------------------- #


def elastic_fe(prob: DeepBeam, nx: int = 40, ny: int = 20,
               nu: float = 0.2) -> tuple[np.ndarray, np.ndarray, dict]:
    """Linear-elastic plane-stress CST solve. Returns:
      nodes : (n_node, 2) nodal coordinates (mm)
      u     : (n_node, 2) nodal displacement (mm)
      info  : dict with mesh metadata
    """
    L, H, t = prob.L, prob.H, prob.t
    dx, dy = L / nx, H / ny
    nnx, nny = nx + 1, ny + 1
    n_node = nnx * nny
    ndof = 2 * n_node

    def nid(i: int, j: int) -> int:
        return j * nnx + i

    xy = np.array([[i * dx, j * dy] for j in range(nny) for i in range(nnx)],
                  dtype=float)

    Ec = prob.mat.Ec0
    Es = prob.mat.Es
    coef = Ec / (1.0 - nu * nu)
    Dc = coef * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, 0.5 * (1.0 - nu)],
    ])

    K = np.zeros((ndof, ndof))
    for j in range(ny):
        for i in range(nx):
            a, b = nid(i, j), nid(i + 1, j)
            c, d = nid(i + 1, j + 1), nid(i, j + 1)
            yc = (j + 0.5) * dy
            rx = prob.rho_tie if yc < prob.band else prob.rho_min
            ry = prob.rho_min + prob.rho_stirrup
            for nodes in ([a, b, c], [a, c, d]):
                p = xy[nodes]
                b1, b2, b3 = p[1, 1] - p[2, 1], p[2, 1] - p[0, 1], p[0, 1] - p[1, 1]
                c1, c2, c3 = p[2, 0] - p[1, 0], p[0, 0] - p[2, 0], p[1, 0] - p[0, 0]
                det = p[0, 0] * b1 + p[1, 0] * b2 + p[2, 0] * b3
                area = abs(det) / 2.0
                inv = 1.0 / det
                B = np.array([
                    [b1 * inv, 0, b2 * inv, 0, b3 * inv, 0],
                    [0, c1 * inv, 0, c2 * inv, 0, c3 * inv],
                    [c1 * inv, b1 * inv, c2 * inv, b2 * inv, c3 * inv, b3 * inv],
                ])
                D = Dc.copy()
                D[0, 0] += rx * Es
                D[1, 1] += ry * Es
                ke = (B.T @ D @ B) * area * t
                dofs = np.array([2 * nodes[0], 2 * nodes[0] + 1,
                                 2 * nodes[1], 2 * nodes[1] + 1,
                                 2 * nodes[2], 2 * nodes[2] + 1])
                K[np.ix_(dofs, dofs)] += ke

    # load: total P over the bearing patch on the top edge
    F = np.zeros(ndof)
    half = prob.bearing / 2.0
    top = [nid(i, ny) for i in range(nnx)
           if abs(i * dx - prob.x_load) <= half + 1e-6]
    for n in top:
        F[2 * n + 1] -= prob.P / len(top)

    # supports: fix u_y at both, u_x at the left support (pin + roller)
    fixed: list[int] = []
    for k, xc in enumerate(prob.x_supp):
        for n in range(n_node):
            if abs(xy[n, 0] - xc) <= half + 10 and xy[n, 1] < 1e-6:
                fixed.append(2 * n + 1)
                if k == 0:
                    fixed.append(2 * n)
    for g in set(fixed):
        K[g, :] = 0.0
        K[:, g] = 0.0
        K[g, g] = 1.0
        F[g] = 0.0

    u = np.linalg.solve(K, F)
    u_node = u.reshape(n_node, 2)
    info = {"nx": nx, "ny": ny, "dx": dx, "dy": dy, "load_nodes": top}
    return xy, u_node, info


# --------------------------------------------------------------------------- #
# Bilinear interpolation from FE nodes to arbitrary (x, y)
# --------------------------------------------------------------------------- #


def bilinear_sample(xy_grid: np.ndarray, u_grid: np.ndarray,
                    nx: int, ny: int, dx: float, dy: float,
                    x: Tensor, y: Tensor) -> Tensor:
    """Bilinearly sample the FE field u_grid (n_node, 2) at points (x, y).
    `xy_grid` is the FE node array; nx, ny, dx, dy describe the grid.
    Returns (N, 2) torch tensor."""
    xq = x.detach().cpu().numpy().squeeze(-1)
    yq = y.detach().cpu().numpy().squeeze(-1)

    nnx = nx + 1

    i = np.clip((xq / dx).astype(int), 0, nx - 1)
    j = np.clip((yq / dy).astype(int), 0, ny - 1)
    fx = (xq - i * dx) / dx
    fy = (yq - j * dy) / dy

    n00 = j * nnx + i
    n10 = j * nnx + (i + 1)
    n01 = (j + 1) * nnx + i
    n11 = (j + 1) * nnx + (i + 1)

    u00 = u_grid[n00]
    u10 = u_grid[n10]
    u01 = u_grid[n01]
    u11 = u_grid[n11]
    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy
    u = (w00[:, None] * u00 + w10[:, None] * u10
         + w01[:, None] * u01 + w11[:, None] * u11)
    return torch.from_numpy(u.astype(np.float32))


# --------------------------------------------------------------------------- #
# Supervised pre-training
# --------------------------------------------------------------------------- #


def pretrain(net: ArclengthPINN, prob: DeepBeam,
             S_max_mm: float = 0.5,
             n_iter: int = 1500, lr: float = 2e-3,
             n_int: int = 1024, verbose: bool = True
             ) -> dict:
    """Fit the PINN to a scaled elastic FE solution.

    Targets:
      u_target(x, y, s) = lambda_target(s) * u_FE(x, y) * elastic_scale
      lambda_target(s)  = s * lambda_max

    where `elastic_scale` and `lambda_max` are chosen so the loaded-patch
    deflection magnitude at s=1 equals S_max_mm, and so the load BC under
    lambda_max is consistent with the FE solve. Specifically:

      delta_FE_at_load  = |u_FE_y at the loaded patch|        (mm at P_ref)
      elastic_scale     = S_max_mm / delta_FE_at_load
      lambda_max        = elastic_scale                       (since u_FE
                           was solved at lambda=1 against P_ref)

    Returns a dict with the chosen scale, lambda_max, and the loss
    history.
    """
    gen = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)

    # ---- run the elastic FE -----------------------------------------------
    nx_fe, ny_fe = 40, 20
    xy_fe, u_fe, fe_info = elastic_fe(prob, nx=nx_fe, ny=ny_fe)
    # mean downward deflection at the load patch (positive magnitude)
    load_nodes = fe_info["load_nodes"]
    delta_fe = float(-u_fe[load_nodes, 1].mean())
    elastic_scale = S_max_mm / max(delta_fe, 1e-9)
    lambda_max = elastic_scale

    if verbose:
        print(f"  elastic FE: delta@load = {delta_fe:.4f} mm at lambda=1")
        print(f"  scaled to S_max={S_max_mm} mm "
              f"=> elastic_scale = lambda_max = {lambda_max:.4f}")

    # ---- pre-training loop ------------------------------------------------
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    history: list[dict] = []
    for it in range(n_iter):
        # sample (x, y, s)
        x = torch.rand(n_int, 1, generator=gen) * prob.L
        y = torch.rand(n_int, 1, generator=gen) * prob.H
        s = torch.rand(n_int, 1, generator=gen)
        # FE target at (x, y), scaled
        u_fe_xy = bilinear_sample(xy_fe, u_fe,
                                  nx_fe, ny_fe,
                                  fe_info["dx"], fe_info["dy"],
                                  x, y) * elastic_scale
        target_ux = s * u_fe_xy[:, 0:1]
        target_uy = s * u_fe_xy[:, 1:2]
        target_lam = s * lambda_max

        # forward through the PINN (mirroring pinn_arclength.displacements)
        xy_n = torch.cat([x / prob.L, y / prob.H], dim=-1)
        out, lam = net(xy_n, s)
        ux = out[:, 0:1] * U0 * prob.L
        uy = out[:, 1:2] * U0 * prob.H

        # MSE losses
        loss_ux = ((ux - target_ux) ** 2).mean()
        loss_uy = ((uy - target_uy) ** 2).mean()
        loss_lam = ((lam - target_lam) ** 2).mean()
        total = loss_ux + loss_uy + loss_lam

        opt.zero_grad()
        total.backward()
        opt.step()

        if verbose and (it % 200 == 0 or it == n_iter - 1):
            history.append({"iter": it, "total": float(total),
                            "ux": float(loss_ux), "uy": float(loss_uy),
                            "lam": float(loss_lam)})
            print(f"  it={it:5d}  total={float(total):.4e}  "
                  f"ux={float(loss_ux):.3e} uy={float(loss_uy):.3e} "
                  f"lam={float(loss_lam):.3e}")

    return {"elastic_scale": elastic_scale, "lambda_max": lambda_max,
            "delta_fe_at_load": delta_fe, "history": history}


def main() -> None:
    """Stand-alone smoke run: pre-train and save weights."""
    prob = DeepBeam()
    net = ArclengthPINN(width=96, depth=6)
    print("pretrain on elastic FE, S_max=0.5 mm")
    info = pretrain(net, prob, S_max_mm=0.5, n_iter=1500)
    out_dir = Path(__file__).resolve().parent / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out_dir / "pretrained_elastic.pt")
    print(f"-> {out_dir}/pretrained_elastic.pt")
    print(f"   elastic_scale = lambda_max = {info['lambda_max']:.4f}")

    # quick sanity: evaluate the network at the load patch
    net.eval()
    with torch.no_grad():
        s = torch.linspace(0.0, 1.0, 6).unsqueeze(-1)
        xy = torch.tensor([[prob.x_load / prob.L, prob.H / prob.H]]
                          ).repeat(6, 1)
        out, lam = net(xy, s)
        uy = out[:, 1:2] * U0 * prob.H
    print("\npost-pretrain probe at load patch:")
    for i in range(6):
        print(f"  s={float(s[i]):.2f}  lam={float(lam[i]):+.4f}  "
              f"uy={float(uy[i]):+.4f} mm")


if __name__ == "__main__":
    main()
