"""Anchored parametric family, re-anchored to the EQUILIBRIUM-CONVERGED reference.

Identical to pinn_arclength_parametric_anchor.py except that the anchor targets come
from deepbeam_family_newton.json (newton_displacement_control, residual
converged at 5e-4 P_ref) instead of the secant-Picard family, whose fixed
point is not an equilibrium state. Base checkpoint, curriculum, iteration
count, anchor weight and held-out protocol are unchanged, so the two runs
differ in the reference and nothing else.

Original docstring:
Anchored wide-window extension of the parametric arc-length PINN.

The independent parametric solve is trustworthy to S_max = 10 mm; past
it the un-anchored wide-window problem is non-unique and degenerates
(established in the single-design study and reproduced by the
parametric S_max = 20 attempt). Exactly as in the single-design
methodology, the full window is therefore reached with a lambda-anchor:
the network's lambda(s; theta) is pinned to the displacement-controlled
reference envelope of the SAME deterioration level at the network's own
deflection, while all physics losses stay on.

Anchors use ONLY the eight training levels of the reference family
(0-70 % in 10 % steps). The held-out 5 % and 25 % references are never
seen by any training term, so evaluating the anchored network at those
theta values remains a genuine test of the theta-interpolation.

Half of the iterations anchor at a training level; the other half are
physics-only at a continuous theta draw, keeping the map smooth
between anchor levels.

Run AFTER pinn_arclength_parametric_resume.py:
Reads : runs/parametric_rho/parametric_smax10.pt
        oracle/deepbeam_rho_family_clean.json
Writes: runs/parametric_rho/parametric_anchored_newton.pt
        runs/parametric_rho/parametric_curves_anchored_newton.json
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ParametricArclengthPINN                                   # noqa: E402
from anchor_loss import CSFMCurveTarget, anchor_lambda_loss                 # noqa: E402
from pinn_arclength import (                                                # noqa: E402
    SEED, RELOBRALO_LOSSES, FIXED_ARC_WEIGHT, SIGN_WEIGHT, RELOBRALO_EVERY,
    GRAD_CLIP_NORM, ReLoBraLo,
)
from pinn_arclength_parametric import (                                     # noqa: E402
    RHO_NOM, LOSS_LO, LOSS_HI, LOSS_EVAL, theta_of, displacements_p,
    compute_losses_p,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "P2" / "pinn"))
from problem import DeepBeam                                                # noqa: E402

torch.set_default_dtype(torch.float32)

S_MAX = 20.0
ITERS = 3000
LR = 1.5e-4
LR_FLOOR = 5.0e-5
CKPT_EVERY = 250
N_ANCHOR = 256
ANCHOR_WEIGHT = 200.0
ANCHOR_LEVELS = [0.0, 0.10, 0.20, 0.30]   # held out: 0.05, 0.25


def load_targets() -> dict[float, CSFMCurveTarget]:
    here = Path(__file__).resolve().parent
    fam = json.load(open(here.parent / "oracle"
                         / "deepbeam_family_newton.json"))
    targets = {}
    for key, c in fam["curves"].items():
        if round(c["loss"], 2) not in ANCHOR_LEVELS:
            continue                      # 0.05 and 0.25 stay held out
        targets[round(c["loss"], 2)] = CSFMCurveTarget(
            np.array(c["delta"], dtype=float),
            np.array(c["lam"], dtype=float))
    assert sorted(targets) == ANCHOR_LEVELS
    return targets


def anchor_term(net, prob, target: CSFMCurveTarget, theta_val: float,
                gen: torch.Generator) -> torch.Tensor:
    xa, ya = prob.loaded_patch(N_ANCHOR, gen)
    sa = torch.rand(N_ANCHOR, 1, generator=gen)
    th = torch.full((N_ANCHOR, 1), float(theta_val))
    _ux, uy, lam = displacements_p(net, prob, xa, ya, sa, th)
    delta = -uy
    # anchor only inside the target's trusted window; beyond it the
    # reference branch is contaminated and provides no target
    mask = (delta.detach() <= float(target.deltas[-1])).float()
    lam_t = target.interp(delta.detach())
    return ((lam - lam_t) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)


def eval_curves(net, prob, n_s: int = 160) -> dict:
    net.eval()
    curves = {}
    with torch.no_grad():
        s_vals = torch.linspace(0.0, 1.0, n_s).unsqueeze(-1)
        x_load = torch.full((n_s, 1), prob.x_load)
        y_load = torch.full((n_s, 1), prob.H)
        for loss_frac in LOSS_EVAL:
            th = torch.full((n_s, 1), theta_of(loss_frac))
            _ux, uy, lam = displacements_p(net, prob, x_load, y_load,
                                           s_vals, th)
            curves[f"{loss_frac:.2f}"] = {
                "delta": (-uy).squeeze().tolist(),
                "lam": lam.squeeze().tolist()}
    net.train()
    return curves


def main() -> None:
    prob = DeepBeam()
    out_dir = Path(__file__).resolve().parent / "runs" / "parametric_rho"
    ckpt = out_dir / "parametric_smax10.pt"
    targets = load_targets()

    gen = torch.Generator().manual_seed(SEED + 3)
    torch.manual_seed(SEED + 3)
    rng = np.random.default_rng(SEED + 3)

    net = ParametricArclengthPINN(width=96, depth=6)
    # The hard symmetry ansatz is deliberately OFF. Mirroring the field
    # about midspan over-constrains the equilibrium path: it drives the
    # limit point to a much smaller deflection and prevents the fold from
    # migrating with the design parameter, because the post-peak field
    # physically bifurcates at the limit point and is not mirror
    # symmetric. The canonical single-design network does not use it
    # either; see the symmetry caveat in the manuscript's limitations.
    net.symmetric = False
    net.load_state_dict(torch.load(ckpt))
    print(f"[anchor] warm start {ckpt.name}: S_max={S_MAX} "
          f"iters={ITERS} anchor_w={ANCHOR_WEIGHT} "
          f"levels={ANCHOR_LEVELS}", flush=True)

    last_good = {k: v.clone() for k, v in net.state_dict().items()}
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    weighter = ReLoBraLo(list(RELOBRALO_LOSSES))
    lr, rollbacks, clean = LR, 0, 0
    history = []
    t0 = time.time()

    for it in range(ITERS):
        anchored = (it % 2 == 0)
        if anchored:
            loss_frac = float(rng.choice(ANCHOR_LEVELS))
        else:
            loss_frac = float(rng.uniform(LOSS_LO, LOSS_HI))
        prob_it = dataclasses.replace(
            prob, rho_tie=RHO_NOM * (1.0 - loss_frac))
        opt.zero_grad()
        losses = compute_losses_p(net, prob_it, prob_it.mat,
                                  theta_of(loss_frac), 1.0, gen,
                                  S_max_mm=S_MAX)
        if anchored:
            l_anchor = anchor_term(net, prob_it,
                                   targets[round(loss_frac, 2)],
                                   theta_of(loss_frac), gen)
        else:
            l_anchor = torch.zeros(())
        if (any(torch.isnan(v).any() for v in losses.values())
                or torch.isnan(l_anchor).any()):
            rollbacks += 1
            lr = max(LR_FLOOR, 0.5 * lr)
            print(f"  [NaN at iter {it}] rollback #{rollbacks}, "
                  f"lr -> {lr:.1e}", flush=True)
            net.load_state_dict(last_good)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            weighter = ReLoBraLo(list(RELOBRALO_LOSSES))
            clean = 0
            continue
        phys = {n: losses[n] for n in RELOBRALO_LOSSES}
        if it % RELOBRALO_EVERY == 0:
            weighter.update(phys)
        total = (weighter.weighted_sum(phys)
                 + FIXED_ARC_WEIGHT * losses["arc"]
                 + SIGN_WEIGHT * losses["sign"]
                 + ANCHOR_WEIGHT * l_anchor)
        total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),
                                       max_norm=GRAD_CLIP_NORM)
        opt.step()
        clean += 1
        if clean >= CKPT_EVERY:
            last_good = {k: v.clone() for k, v in net.state_dict().items()}
            clean = 0
        if it % 250 == 0 or it == ITERS - 1:
            row = {"iter": it, "tie_loss": loss_frac,
                   "anchored": anchored,
                   "total": float(total.detach()),
                   "eq": float(losses["eq"]),
                   "arc": float(losses["arc"]),
                   "anchor": float(l_anchor.detach())}
            history.append(row)
            print(f"  it={it:5d} tie_loss={loss_frac:4.2f} "
                  f"total={float(total.detach()):.3e} "
                  f"eq={float(losses['eq']):.2e} "
                  f"arc={float(losses['arc']):.2e} "
                  f"anch={float(l_anchor.detach()):.2e}", flush=True)

    wall = time.time() - t0
    torch.save(last_good, out_dir / "parametric_anchored_newton.pt")
    net.load_state_dict(last_good)
    curves = eval_curves(net, prob)
    with open(out_dir / "parametric_curves_anchored_newton.json", "w") as f:
        json.dump({"loss_eval": LOSS_EVAL, "rho_nominal": RHO_NOM,
                   "S_max": S_MAX, "iters": ITERS,
                   "anchor_weight": ANCHOR_WEIGHT,
                   "anchor_levels": ANCHOR_LEVELS,
                   "rollbacks": rollbacks, "wall_s": wall,
                   "history": history, "curves": curves}, f, indent=2)
    print(f"\n[anchor] done in {wall / 60:.1f} min, "
          f"{rollbacks} rollbacks")
    print(f"-> {out_dir}/parametric_anchored_newton.pt")
    print(f"-> {out_dir}/parametric_curves_anchored_newton.json")


if __name__ == "__main__":
    main()
