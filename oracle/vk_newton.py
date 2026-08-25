"""VK1 and VK3 wall piers, re-traced on equilibrium-converged states.

The wall-pier comparison is the study's second experimental check and the only
one in which the longitudinal reinforcement is the variable and the response is
measured on the same specimen geometry. Its earlier numbers came from
`picard_displacement_controlled_vk1`, which converges to the fixed point of a
clipped secant map rather than to equilibrium, so they are withdrawn rather
than adjusted.

Porting the equilibrium-converged scheme to this problem is not a change of
call. The deep-beam routine drives the internal force itself to zero on the
free degrees of freedom, which is the equilibrium condition only when nothing
external acts on them. Here a constant axial compression of 1370 kN is carried
as a dead-load vector, so the condition is R = F_int - F_dead = 0, and the
lateral load is read from that same residual at the prescribed patch.

VK1 and VK3 differ only in longitudinal reinforcement ratio, which is exactly
the parameter this study varies. What the comparison has to reproduce is not
the level of capacity, which a local softening model without crack-band scaling
cannot be asked for, but the RATIO between the two, since every result in the
paper is a ratio of a deteriorated state to an intact one.

Run one specimen per process:  python3.12 vk_newton.py vk1|vk3
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arclength_oracle_vk1 import (build_vk1_mesh,                        # noqa: E402
                                  newton_displacement_control_vk1,
                                  vk1_default)

# VK3 carries 1.5 times the longitudinal reinforcement of VK1; the transverse
# reinforcement and the axial load are the same, which is what makes the pair
# a test of sensitivity to the tie rather than of anything else.
RHO_L_VK1 = 0.0082
RHO_SCALE_VK3 = 1.5
DELTA_MAX, N_STEPS = 45.0, 45


def problem(which: str):
    base = vk1_default()
    if which == "vk1":
        return base
    rho_l = RHO_L_VK1 * RHO_SCALE_VK3
    return dataclasses.replace(base, rho_y=lambda x, y: max(rho_l, 0.0010))


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "vk1"
    prob = problem(which)
    mesh = build_vk1_mesh(prob)
    print(f"{which.upper()}: mesh {prob.nx}x{prob.ny}, {mesh.ndof} dof, "
          f"V_ref = {prob.V_ref / 1e3:.0f} kN, N = {prob.N / 1e3:.0f} kN")
    print(f"{DELTA_MAX} mm in {N_STEPS} steps "
          f"({DELTA_MAX / N_STEPS:.2f} mm each)\n", flush=True)

    t0 = time.time()
    # the iteration count climbs along the path as the field softens, from
    # about 50 at 1 mm to well past the default cap further along, so the cap
    # is raised rather than allowing states to be reported unconverged
    out = newton_displacement_control_vk1(prob, mesh, delta_max=DELTA_MAX,
                                          n_steps=N_STEPS,
                                          newton_max_iter=400, verbose=True)
    d = np.array([o["delta_x"] for o in out])
    l = np.array([o["lam"] for o in out])
    ok = np.array([o["converged"] for o in out])
    r = np.array([o["resid"] for o in out])
    i = int(np.argmax(np.where(ok, l, -np.inf)))
    print(f"\npeak over converged states: V = {l[i] * prob.V_ref / 1e3:.1f} kN "
          f"at {d[i]:.2f} mm  (lambda = {l[i]:.4f}, residual {r[i]:.2e})")
    print(f"{int(ok.sum())} of {len(ok)} states converged, "
          f"{time.time() - t0:.0f} s")

    json.dump({"which": which, "delta": d.tolist(), "lam": l.tolist(),
               "converged": ok.tolist(), "resid": r.tolist(),
               "V_ref": prob.V_ref, "V_peak_kN": l[i] * prob.V_ref / 1e3,
               "delta_peak": float(d[i])},
              open(f"{which}_newton.json", "w"))
    print(f"-> {which}_newton.json")


if __name__ == "__main__":
    main()
