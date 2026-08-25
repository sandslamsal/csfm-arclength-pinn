"""Arc-length-parametrised continuum PINN for P3.

Network signature: `(x_n, y_n, s_n) -> (u_x, u_y, lambda)`.

Two heads on a shared SiLU trunk:
  - field head: 2-component nondimensional displacement (u_x, u_y) at the
                spacetime-like point (x, y, s)
  - scalar head: load factor lambda(s) (single scalar, depends on s only —
                 enforced by routing only the s coordinate through a
                 separate small MLP rather than the shared trunk)

Hard initial condition baked in via the ansatz
    u(x, y, s) = s * tilde_u(x, y, s)
    lambda(s)  = s * tilde_lambda(s)
so that at s = 0 the structure is exactly undeformed and unloaded — no
penalty term is needed to anchor the start of the equilibrium path.

SiLU activation per Balmer, Kaufmann & Kraus (2024) and P2: smooth, so the
second derivatives the equilibrium residual needs are well defined.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class ArclengthPINN(nn.Module):
    """(x_n, y_n, s_n) -> (u_x, u_y, lambda)."""

    def __init__(self, width: int = 96, depth: int = 6,
                 lam_width: int = 32, lam_depth: int = 3):
        super().__init__()
        # ---- field trunk: (x_n, y_n, s_n) -> (u_x, u_y) ----------------
        layers: list[nn.Module] = [nn.Linear(3, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, 2)]
        self.field = nn.Sequential(*layers)

        # ---- load-factor head: s_n -> lambda --------------------------
        lam_layers: list[nn.Module] = [nn.Linear(1, lam_width), nn.SiLU()]
        for _ in range(lam_depth - 1):
            lam_layers += [nn.Linear(lam_width, lam_width), nn.SiLU()]
        lam_layers += [nn.Linear(lam_width, 1)]
        self.lam = nn.Sequential(*lam_layers)

        with torch.no_grad():
            self.field[-1].weight.mul_(0.1)
            self.field[-1].bias.zero_()
            self.lam[-1].weight.mul_(0.1)
            self.lam[-1].bias.zero_()

    def forward(self, xy: Tensor, s: Tensor) -> tuple[Tensor, Tensor]:
        """Args:
          xy : (N, 2) normalised coordinates (in roughly [0, 1])
          s  : (N, 1) normalised arc-length (in [0, 1])
        Returns:
          u   : (N, 2) nondimensional displacement field, hard-zero at s=0
          lam : (N, 1) load factor, hard-zero at s=0
        """
        if getattr(self, "symmetric", False):
            # Enforce the physical symmetry of a centred-load deep beam about
            # midspan x_n = 0.5: horizontal displacement odd, vertical even.
            xy_m = torch.cat([1.0 - xy[:, 0:1], xy[:, 1:2]], dim=-1)
            u = self.field(torch.cat([xy, s], dim=-1))
            um = self.field(torch.cat([xy_m, s], dim=-1))
            ux = 0.5 * (u[:, 0:1] - um[:, 0:1])   # odd about midspan
            uy = 0.5 * (u[:, 1:2] + um[:, 1:2])   # even about midspan
            tilde_u = torch.cat([ux, uy], dim=-1)
        else:
            tilde_u = self.field(torch.cat([xy, s], dim=-1))
        tilde_lam = self.lam(s)
        return s * tilde_u, s * tilde_lam


class ParametricArclengthPINN(nn.Module):
    """(x_n, y_n, s_n, theta_n) -> (u_x, u_y, lambda).

    Parametric extension of ArclengthPINN: a normalised design parameter
    theta (here the concrete strength f_c) joins the trunk input, and the
    load-factor head reads (s, theta) so the whole equilibrium path
    lambda(s; theta) is theta-dependent. The hard initial condition
    u = lambda = 0 at s = 0 is kept via the same s-multiplied ansatz.
    """

    def __init__(self, width: int = 96, depth: int = 6,
                 lam_width: int = 32, lam_depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(4, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, 2)]
        self.field = nn.Sequential(*layers)

        lam_layers: list[nn.Module] = [nn.Linear(2, lam_width), nn.SiLU()]
        for _ in range(lam_depth - 1):
            lam_layers += [nn.Linear(lam_width, lam_width), nn.SiLU()]
        lam_layers += [nn.Linear(lam_width, 1)]
        self.lam = nn.Sequential(*lam_layers)

        with torch.no_grad():
            self.field[-1].weight.mul_(0.1)
            self.field[-1].bias.zero_()
            self.lam[-1].weight.mul_(0.1)
            self.lam[-1].bias.zero_()

    def forward(self, xy: Tensor, s: Tensor,
                theta: Tensor) -> tuple[Tensor, Tensor]:
        """Args:
          xy    : (N, 2) normalised coordinates
          s     : (N, 1) normalised arc-length in [0, 1]
          theta : (N, 1) normalised design parameter
        Returns:
          u   : (N, 2) nondimensional displacement, hard-zero at s = 0
          lam : (N, 1) load factor, hard-zero at s = 0
        """
        if getattr(self, "symmetric", False):
            xy_m = torch.cat([1.0 - xy[:, 0:1], xy[:, 1:2]], dim=-1)
            u = self.field(torch.cat([xy, s, theta], dim=-1))
            um = self.field(torch.cat([xy_m, s, theta], dim=-1))
            ux = 0.5 * (u[:, 0:1] - um[:, 0:1])
            uy = 0.5 * (u[:, 1:2] + um[:, 1:2])
            tilde_u = torch.cat([ux, uy], dim=-1)
        else:
            tilde_u = self.field(torch.cat([xy, s, theta], dim=-1))
        tilde_lam = self.lam(torch.cat([s, theta], dim=-1))
        return s * tilde_u, s * tilde_lam
