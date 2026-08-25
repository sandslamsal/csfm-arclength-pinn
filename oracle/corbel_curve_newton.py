"""The corbel equilibrium curve, recomputed with the equilibrium-converged solver.

The secant-Picard corbel reference (corbel_reference.json) peaks at
lambda = 3.078 at 14.62 mm. On the deep beam the same solver overstated the
intact capacity by 66 per cent, placing the peak at roughly twice the
deflection (the reference peak is flat to 1 per cent over 3.0 to 4.75 mm,
so no sharp ratio is quotable), because
it converges to the fixed point of a clipped secant map rather than to
equilibrium. This driver traces the corbel on newton_displacement_control,
same problem, same mesh, same 15 mm window, so the two curves differ in the
solver and nothing else.
"""
import sys; sys.path.insert(0, '.')
import json, time
import numpy as np
from arclength_oracle import build_mesh
from arclength_oracle_crisfield import newton_displacement_control
from corbel import corbel

DELTA_MAX, N_STEPS = 15.0, 60      # 0.25 mm steps, the reference's window

prob = corbel()
mesh = build_mesh(prob)
t0 = time.time()
pts = newton_displacement_control(prob, mesh, delta_max=DELTA_MAX,
                                  n_steps=N_STEPS, verbose=False)
lam = np.array([p.lam for p in pts])
dd  = np.array([p.delta for p in pts])
res = np.array([p.resid for p in pts])
cvg = np.array([p.converged for p in pts])
i = int(np.argmax(np.where(cvg, lam, -np.inf)))
print(f"corbel newton: lam_max {lam[i]:.4f} at {dd[i]:.2f} mm "
      f"({int(cvg.sum())}/{len(pts)} converged, wall {time.time()-t0:.0f}s)")
print(f"secant reference said 3.078 at 14.62 mm")
json.dump({"delta": dd.tolist(), "lam": lam.tolist(),
           "converged": cvg.tolist(), "resid": res.tolist(),
           "lam_max": float(lam[i]), "delta_peak": float(dd[i]),
           "delta_max": DELTA_MAX, "n_steps": N_STEPS,
           "wall_s": time.time() - t0},
          open("corbel_newton_curve.json", "w"))
print("-> corbel_newton_curve.json")
