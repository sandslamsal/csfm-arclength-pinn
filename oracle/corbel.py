"""Corbel D-region problem instance.

A cantilever-bracket idealisation of a precast corbel: a rectangular
D-region fully clamped along its left face (the proxy for the
column-to-corbel interface) and loaded vertically over a short bearing
patch at the top-right free end. Mirrors `Research/P2/pinn/problem.py::Corbel`.

Compression flow: a single inclined strut from the loaded patch to the
bottom of the clamped face, balanced by a horizontal tie along the top
of the bracket. The textbook 2D abstraction of Kaufmann & Marti /
ACI 318 Section 16.5.
"""
from __future__ import annotations

from arclength_oracle import Material, Problem


def corbel(
    P_ref: float = 90.0e3,
    nx: int = 30, ny: int = 24,        # h ~ 17 mm, height-priority aspect
) -> Problem:
    L, H, t = 500.0, 400.0, 300.0
    bearing = 100.0
    x_load = L - bearing / 2.0          # bearing is at the right-hand top edge
    rho_tie = 0.012                     # top horizontal tie (rho_x)
    rho_stirrup = 0.0015                # vertical stirrups
    rho_min = 0.0010
    band = 80.0                         # top tie band thickness (mm)

    def rho_x(x: float, y: float) -> float:
        # top tie-band along the full corbel length
        in_band = (y > H - band)
        return rho_tie if in_band else rho_min

    def rho_y(x: float, y: float) -> float:
        return rho_min + rho_stirrup

    return Problem(
        L=L, H=H, thickness=t, nx=nx, ny=ny,
        rho_x=rho_x, rho_y=rho_y,
        x_load=x_load, bearing=bearing,
        P_ref=P_ref,
        supports=(),                    # no bottom-edge patches
        mat=Material(fc=30.0),
        clamped_left=True,              # left face is fully clamped
    )
