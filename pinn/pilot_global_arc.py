"""Pilot 2 (reviewer item 3.2): the global L2 arc-length constraint.

The canonical arc loss constrains the speed of the LOADED PATCH alone;
the corbel showed that surrogate admits a trivial-lateral attractor and
Section 4.2 concedes it is strictly weaker as a path-length constraint.
This pilot replaces it with the global configuration-space form,
  X(s) = integral_Omega |du/ds|^2 dOmega / |Omega|,
sampled on the interior collocation set, with the same pointwise
enforcement against S_ref^2. Everything else, curriculum included, is
the canonical deep-beam run. Two lines change, as the review predicted.

S_ref for the global norm is calibrated so the elastic pre-training
state satisfies the constraint exactly at the first stage: the global
RMS displacement rate of the elastic solution differs from the patch
deflection rate by a geometry factor, measured once at s = 1.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pinn_arclength as base

_orig = base.compute_losses

def compute_losses_global(net, prob, alpha, gen, S_max_mm=1.0, **kw):
    losses = _orig(net, prob, alpha, gen, S_max_mm=S_max_mm, **kw)
    # replace the patch-restricted arc term with the global L2 form on a
    # fresh interior sample
    xi, yi = prob.interior(base.N_INT // 2, gen)
    si = torch.rand(base.N_INT // 2, 1, generator=gen).requires_grad_(True)
    ux, uy, _ = base.displacements(net, prob, xi, yi, si)
    dux = base.grad(ux, si); duy = base.grad(uy, si)
    speed2 = dux ** 2 + duy ** 2
    # geometry factor: global RMS rate vs patch rate, from the elastic
    # pre-training target (measured once, cached on the function)
    if not hasattr(compute_losses_global, "_gf"):
        compute_losses_global._gf = 0.35   # elastic FE: RMS(u)/u_patch
    target2 = (compute_losses_global._gf * S_max_mm) ** 2
    losses["arc"] = (((speed2 - target2) / max(target2, 1e-12)) ** 2).mean()
    return losses

base.compute_losses = compute_losses_global


# measure the true geometry factor from the elastic FE solution
from problem import DeepBeam
import numpy as np
from pretrain_elastic import elastic_fe
prob = DeepBeam()
xy, u, info = elastic_fe(prob, nx=40, ny=20)
u_patch = float(-u[info["load_nodes"], 1].mean())
rms = float(np.sqrt((u[:, 0] ** 2 + u[:, 1] ** 2).mean()))
compute_losses_global._gf = rms / u_patch
print(f"[pilot2] geometry factor RMS(u)/u_patch = {rms/u_patch:.4f}")

if __name__ == "__main__":
    t0 = time.time()
    out_dir = HERE / "runs" / "pilot_global_arc"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pilot2] global-L2 arc constraint -> {out_dir}")
    info = base.train(prob, out_dir, do_pretrain=True)
    json.dump({"geometry_factor": compute_losses_global._gf,
               "info": {k: v for k, v in (info or {}).items()
                        if isinstance(v, (int, float, str, list))}},
              open(out_dir / "pilot_summary.json", "w"), indent=1)
    print(f"[pilot2] wall {time.time()-t0:.0f}s")
