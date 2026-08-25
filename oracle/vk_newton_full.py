"""VK1 on equilibrium-converged states, WITH the displacement history.

Identical trace to vk_newton.py vk1 (same problem, same 45 mm window, same
45 steps) but keeping u at every converged state, in the schema of
vk1_reference_full.json, so that the strain-level anchor of
pinn_arclength_vk1_strain.py can target equilibrium states instead of the
secant fixed points it was anchored to before. Non-converged states are
excluded: an anchor must never target a state that does not satisfy
equilibrium.
"""
import sys; sys.path.insert(0, '.')
import json, time
import numpy as np
from arclength_oracle_vk1 import (build_vk1_mesh,
                                  newton_displacement_control_vk1,
                                  vk1_default)

DELTA_MAX, N_STEPS = 45.0, 45          # the window of vk_newton.py

prob = vk1_default()
mesh = build_vk1_mesh(prob)
print(f"VK1 full: mesh {prob.nx}x{prob.ny}, {mesh.ndof} dof, "
      f"{DELTA_MAX} mm in {N_STEPS} steps", flush=True)
t0 = time.time()
out = newton_displacement_control_vk1(prob, mesh, delta_max=DELTA_MAX,
                                      n_steps=N_STEPS, verbose=True)
conv = [o for o in out if o["converged"]]
print(f"{len(conv)} of {len(out)} states converged, {time.time()-t0:.0f} s")

lam = np.array([o["lam"] for o in conv])
i = int(np.argmax(lam))
print(f"peak over converged states: lam = {lam[i]:.4f} "
      f"(V = {lam[i]*abs(prob.V_ref)/1e3:.1f} kN) at "
      f"{conv[i]['delta_x']:.2f} mm")

u_all = np.stack([o["u"].reshape(-1, 2) for o in conv])
json.dump({
    "L": prob.L, "H": prob.H, "t": prob.thickness,
    "nx": prob.nx, "ny": prob.ny, "V_ref": abs(prob.V_ref),
    "curve": [{"delta_x": o["delta_x"], "lam": o["lam"],
               "resid": o["resid"]} for o in conv],
    "peak": {"delta_x": conv[i]["delta_x"], "lam": float(lam[i])},
    "u_all": u_all.tolist(),
    "wall_s": time.time() - t0,
}, open("vk1_newton_full.json", "w"))
print("-> vk1_newton_full.json")
