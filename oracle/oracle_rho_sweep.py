"""Tie-reinforcement sweep of the displacement-controlled CSFM oracle for
the deep beam: equilibrium paths at a family of soffit-tie ratios rho_tie,
representing 0-40% corrosion section loss of the main tension tie.

Uses the SAME solver, mesh (40x20 CST) and stepping (delta_max = 20 mm in
80 steps) as run_deepbeam.py, whose nominal-rho curve is the manuscript's
displacement-controlled reference (peak lambda = 2.29). These curves are
the validation references for the parametric (x, y, s, theta) arc-length
PINN with theta = rho_tie.

Output: deepbeam_oracle_rhosweep.json (curve + peak + wall time per rho).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arclength_oracle import Material, Problem, build_mesh, trace_curve  # noqa: E402

RHO_NOM = 0.012
# 0-70 % section loss + two held-out levels (5, 25 %); the range
# deliberately crosses the strut/tie balance point so the capacity
# surface shows both the strut-governed plateau and the tie-governed decline
LOSS_TRAIN = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
LOSS_HELD = [0.05, 0.25]
NX, NY = 40, 20
DELTA_MAX, N_STEPS = 20.0, 80


def deepbeam_rho(rho_tie: float, P_ref: float = 800.0e3,
                 nx: int = NX, ny: int = NY) -> Problem:
    """oracle/deepbeam.py::deepbeam with rho_tie injectable."""
    L, H, t = 2000.0, 1000.0, 300.0
    a = 250.0
    bearing = 200.0
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


def main() -> None:
    here = Path(__file__).resolve().parent
    out = {"rho_nominal": RHO_NOM, "nx": NX, "ny": NY,
           "delta_max": DELTA_MAX, "n_steps": N_STEPS, "curves": []}
    for loss in LOSS_TRAIN + LOSS_HELD:
        rho = RHO_NOM * (1.0 - loss)
        prob = deepbeam_rho(rho)
        mesh = build_mesh(prob)
        print(f"\n[rho-sweep] section loss = {loss * 100:.0f}%  "
              f"rho_tie = {rho:.5f}", flush=True)
        t0 = time.time()
        curve, _u, _d = trace_curve(prob, mesh, delta_max=DELTA_MAX,
                                    n_steps=N_STEPS, soften=True,
                                    verbose=False)
        dt = time.time() - t0
        deltas = [p.delta for p in curve]
        lams = [p.lam for p in curve]
        conv = [p.converged for p in curve]
        lam_arr = np.where(np.array(conv), np.array(lams), -np.inf)
        ipk = int(np.argmax(lam_arr))
        print(f"  -> peak lam = {lams[ipk]:.3f} at delta = "
              f"{deltas[ipk]:.2f} mm   ({dt:.0f} s)", flush=True)
        out["curves"].append({
            "loss": loss, "rho_tie": rho, "held_out": loss in LOSS_HELD,
            "delta": deltas, "lam": lams, "converged": conv,
            "peak_lam": float(lams[ipk]), "peak_delta": float(deltas[ipk]),
            "wall_time_s": dt})
    with open(here / "deepbeam_oracle_rhosweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> deepbeam_oracle_rhosweep.json ({len(out['curves'])} curves)")


if __name__ == "__main__":
    main()
