"""Sensitivity of the deep-beam result to ANCHOR_WEIGHT.

The anchor weight is the one fixed weight in the recipe that no study in
the paper justifies (w_free has the validity sweep, S_max has its own).
This trains the identical finetune at several weights from the same
warm start and reports what each buys: curve fidelity against the
equilibrium-converged reference, and what it costs in the physics
residuals the anchor competes with.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pinn_anchor_finetune as F                                     # noqa: E402
from anchor_loss import CSFMCurveTarget                              # noqa: E402
from model import ArclengthPINN                                      # noqa: E402

WEIGHTS = [25.0, 50.0, 200.0, 800.0]
U0 = 1.0e-3


def trace(net, prob, kind, S_max, n=241):
    """(delta, lambda) along the learned path at the load-patch centre."""
    s = torch.linspace(1e-3, 1.0, n).unsqueeze(-1)
    x = torch.full_like(s, prob.x_load)
    y = torch.full_like(s, prob.H)
    with torch.no_grad():
        out, lam = net(torch.cat([x / prob.L, y / prob.H], dim=-1), s)
        delta = -(out[:, 1:2] * U0 * prob.H)
    return delta.squeeze(-1).numpy(), lam.squeeze(-1).numpy()


def main():
    fam = json.load(open(HERE.parent / "oracle"
                         / "deepbeam_family_newton.json"))
    ref = fam["curves"]["0.00"]                     # intact deep beam
    rl = float(ref["lam_max"]); rd = float(ref["delta_peak"])
    print(f"reference (equilibrium-converged): lam_max {rl:.4f} at {rd:.2f} mm\n")

    rows = []
    for w in WEIGHTS:
        cfg, kind = F.deepbeam_cfg()
        cfg.out_dir = HERE / "runs" / f"anchor_w_{int(w)}"
        F.ANCHOR_WEIGHT = w                       # module global, read in-loop
        t0 = time.time()
        res = F.finetune(cfg, kind)
        net = ArclengthPINN(width=96, depth=6)
        net.load_state_dict(torch.load(cfg.out_dir / "pinn.pt"))
        net.eval()
        d, l = trace(net, cfg.prob, kind, cfg.S_max_mm)
        k = int(np.argmax(l))
        h = res["history"][-1]
        row = {"w": w, "lam_max": float(l[k]), "delta_peak": float(d[k]),
               "lam_err_pct": (float(l[k]) / rl - 1) * 100.0,
               "delta_err_mm": float(d[k]) - rd,
               "anchor": h["anchor"], "eq": h["eq"],
               "free": h.get("free"), "load": h.get("load"),
               "wall_s": time.time() - t0}
        rows.append(row)
        print(f"\n>>> w={w:6.0f}  lam {row['lam_max']:.4f} "
              f"({row['lam_err_pct']:+.2f}%)  peak {row['delta_peak']:.2f} mm "
              f"({row['delta_err_mm']:+.2f})  eq {row['eq']:.2e} "
              f"free {row['free']}\n")
        json.dump({"reference": {"lam_max": rl, "delta_peak": rd},
                   "rows": rows},
                  open(HERE / "runs" / "anchor_weight_sweep.json", "w"),
                  indent=1)

    print(f"\n{'w':>7} {'lam_max':>9} {'err %':>8} {'peak mm':>9} "
          f"{'d err':>7} {'eq':>10} {'free':>10}")
    for r in rows:
        fr = "n/a" if r["free"] is None else f"{r['free']:.2e}"
        print(f"{r['w']:7.0f} {r['lam_max']:9.4f} {r['lam_err_pct']:+8.2f} "
              f"{r['delta_peak']:9.2f} {r['delta_err_mm']:+7.2f} "
              f"{r['eq']:10.2e} {fr:>10}")


if __name__ == "__main__":
    main()
