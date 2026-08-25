"""Deep-beam problem instance — same geometry/reinforcement as
Research/P2/pinn/problem.py::DeepBeam, expressed as a `Problem` for the
oracle solver."""
from __future__ import annotations

from arclength_oracle import Material, Problem


def deepbeam(
    P_ref: float = 800.0e3,
    nx: int = 40, ny: int = 20,
) -> Problem:
    L, H, t = 2000.0, 1000.0, 300.0
    a = 250.0
    bearing = 200.0
    rho_tie = 0.012
    rho_stirrup = 0.0015
    rho_min = 0.0010
    band = 150.0

    def rho_x(x: float, y: float) -> float:
        return rho_tie if y < band else rho_min

    def rho_y(x: float, y: float) -> float:
        return rho_min + rho_stirrup

    return Problem(
        L=L, H=H, thickness=t, nx=nx, ny=ny,
        rho_x=rho_x, rho_y=rho_y,
        x_load=L / 2.0, bearing=bearing,
        P_ref=P_ref,
        supports=((a, True, True), (L - a, False, True)),
        mat=Material(fc=30.0),
    )
