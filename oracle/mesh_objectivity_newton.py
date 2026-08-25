"""Are the deterioration sensitivities mesh-objective? Recomputed on equilibrium.

The earlier version of this test ran on the secant-Picard reference, which does
not satisfy equilibrium, so its answer is not evidence about anything. The
question it asked is still the right one, and it now matters more, because the
measured deep beams have just shown that this formulation's capacity moves with
element size: the same specimen gives 1540 kN at 100 mm elements, 997 kN at
50 mm and 1193 kN at 33 mm, a scatter of a quarter with no trend.

Absolute capacity from a local softening continuum with no crack-band scaling
is therefore not a quotable quantity, and the paper must not quote one. What
the paper actually needs is narrower: that the RATIO of capacity at a given
section loss to capacity intact does not depend on the mesh, and likewise for
the service deflection. A common bias divides out of a ratio; a mesh-dependent
slope does not.

The service deflection is included because it is read far below the peak,
where softening has not localised, and it was the most mesh-stable quantity in
the earlier run (1.042, 0.979, 1.044 mm across a 3:1 element range). If the
peak quantities move with the mesh and the service quantity does not, that is
the finding, and the study restricts itself to the service quantity.

Run one mesh per process:  python3.12 mesh_objectivity_newton.py <nx> <ny>
or one level per process on the finer meshes, where a level costs half an
hour:                      python3.12 mesh_objectivity_newton.py <nx> <ny> <i>
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arclength_oracle import build_mesh                                   # noqa: E402
from arclength_oracle_crisfield import newton_displacement_control        # noqa: E402
from oracle_rho_sweep import RHO_NOM, deepbeam_rho                        # noqa: E402

LEVELS = [0.0, 0.10, 0.30]
# The solver's convergence test is an absolute 2-norm of the force residual,
# 5e-4 * P_ref. That norm accumulates over free degrees of freedom, so the same
# per-element imbalance produces a larger value on a finer mesh and the finer
# mesh is held to a stricter criterion rather than an equal one. Comparing
# meshes requires scaling the tolerance by the square root of the DOF count,
# referred to the 40x20 mesh the study uses.
NDOF_REF = 41 * 21 * 2
DELTA_MAX, N_STEPS = 7.0, 28           # as the family, so the two compare
LAM_SERVICE = 0.65                     # fraction of intact capacity, set below
TIE_DEPTH_MM = 150.0


def tie_area_correction(prob, ny: int) -> float:
    """Restore the same total tie area on every mesh.

    An element carries tie reinforcement only if its centroid falls inside the
    150 mm band, so the depth actually reinforced is 100 mm at h = 100, 150 mm
    at h = 50 and 133 mm at h = 33. Without this the coarse mesh would simply
    have a third less tie steel, which would be read as a softening effect.
    """
    dy = prob.H / ny
    rows = sum(1 for j in range(ny) if (j + 0.5) * dy < TIE_DEPTH_MM)
    return TIE_DEPTH_MM / max(rows * dy, 1e-9)


def main() -> None:
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    ny = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    only = int(sys.argv[3]) if len(sys.argv) > 3 else None
    levels = LEVELS if only is None else [LEVELS[only]]
    key = f"{nx}x{ny}" if only is None else f"{nx}x{ny}_L{only}"
    tol = 5e-4 * np.sqrt((nx + 1) * (ny + 1) * 2 / NDOF_REF)
    h = 2000.0 / nx
    corr0 = tie_area_correction(deepbeam_rho(RHO_NOM), ny)
    print(f"mesh {key}  (h = {h:.0f} mm, tie rho x{corr0:.3f} to hold the "
          f"tie area constant)")
    print(f"residual tolerance {tol:.2e} * P_ref, scaled by sqrt(DOF) from "
          f"5.00e-04 on 40x20")
    print(f"{'loss':>6}{'lam_peak':>11}{'d_peak':>10}{'resid@pk':>11}"
          f"{'conv':>6}{'wall s':>9}", flush=True)

    runs = {}
    for loss in levels:
        corr = tie_area_correction(deepbeam_rho(RHO_NOM * (1.0 - loss)), ny)
        prob = dataclasses.replace(
            deepbeam_rho(RHO_NOM * (1.0 - loss) * corr), nx=nx, ny=ny)
        mesh = build_mesh(prob)
        t0 = time.time()
        pts = newton_displacement_control(prob, mesh, delta_max=DELTA_MAX,
                                          n_steps=N_STEPS,
                                          newton_tol_rel=tol,
                                          newton_max_iter=400,
                                          verbose=False)
        d = np.array([p.delta for p in pts])
        l = np.array([p.lam for p in pts])
        r = np.array([p.resid for p in pts])
        ok = np.array([p.converged for p in pts])
        i = int(np.argmax(np.where(ok, l, -np.inf)))
        runs[f"{loss:.2f}"] = {"lam_peak": float(l[i]), "d_peak": float(d[i]),
                               "resid_peak": float(r[i]),
                               "delta": d.tolist(), "lam": l.tolist(),
                               "resid": r.tolist(),
                               "n_converged": int(ok.sum()),
                               "converged": ok.tolist()}
        print(f"{loss * 100:>5.0f}%{l[i]:>11.4f}{d[i]:>9.2f}m{r[i]:>11.2e}"
              f"{'  ok' if ok[i] else '  NO':>6}{time.time() - t0:>9.0f}"
              f"   {int(ok.sum())}/{len(ok)} states",
              flush=True)

    if only is not None:
        json.dump({"mesh": [nx, ny], "h_mm": h, "levels": levels,
                   "runs": runs}, open(f"mesh_obj_{key}.json", "w"))
        print(f"\n-> mesh_obj_{key}.json")
        return

    lam0 = runs[f"{LEVELS[0]:.2f}"]["lam_peak"]
    lam_s = LAM_SERVICE * lam0
    print(f"\nservice load factor {lam_s:.3f} "
          f"({LAM_SERVICE:.2f} of this mesh's intact capacity)")
    print(f"{'loss':>6}{'cap ratio':>12}{'d_service':>12}{'growth':>10}")
    d_ref = None
    for loss in LEVELS:
        rr = runs[f"{loss:.2f}"]
        d = np.array(rr["delta"]); l = np.array(rr["lam"])
        i = int(np.argmax(l))
        ds = float(np.interp(lam_s, l[:i + 1], d[:i + 1]))
        rr["d_service"] = ds
        d_ref = ds if d_ref is None else d_ref
        print(f"{loss * 100:>5.0f}%{rr['lam_peak'] / lam0:>12.4f}"
              f"{ds:>11.3f}m{(ds / d_ref - 1) * 100:>9.1f}%")

    json.dump({"mesh": [nx, ny], "h_mm": h, "levels": LEVELS,
               "lam_service_frac": LAM_SERVICE, "runs": runs},
              open(f"mesh_obj_{key}.json", "w"))
    print(f"\n-> mesh_obj_{key}.json")


if __name__ == "__main__":
    main()
