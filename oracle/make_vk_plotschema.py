"""Rebuild the wall-pier plot schemas from the corrected references.

The schemas the figure reads were written from the t = 200 mm runs. The
specimen piers are 350 mm thick (Bimschas Tab. 5.1), so those schemas
hold capacities that are 33 per cent low. This regenerates them from
vk1_newton.json / vk3_newton.json, which are the corrected traces.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

for which in ("vk1", "vk3"):
    d = json.load(open(HERE / f"{which}_newton.json"))
    delta = np.array(d["delta"], float)
    lam = np.array(d["lam"], float)
    cv = np.array(d["converged"], bool)
    curve = [{"delta_x": float(a), "lam": float(b), "converged": bool(c)}
             for a, b, c in zip(delta, lam, cv)]
    k = int(np.argmax(np.where(cv, lam, -np.inf)))    # peak over converged
    out = {"V_ref": d["V_ref"],
           "curve": curve,
           "peak": {"delta_x": float(delta[k]), "lam": float(lam[k]),
                    "V_kN": float(lam[k] * d["V_ref"] / 1e3)},
           "n_converged": int(cv.sum()), "n_states": int(cv.size)}
    p = HERE / f"{which}_newton_plotschema.json"
    json.dump(out, open(p, "w"))
    print(f"{p.name}: peak {out['peak']['V_kN']:.1f} kN at "
          f"{out['peak']['delta_x']:.0f} mm, "
          f"{out['n_converged']}/{out['n_states']} converged")
