"""Corbel PINN v3: continue from v2 stage-9 weights with the directional
arc-length loss AND a soft lambda-floor constraint, to escape the
stress-trivial attractor documented in the v2 writeup.

The v2 trained network reached u_y(load_patch, s=1) = -10 mm correctly
(directional arc-loss bound the loaded-direction speed) but the
cracked-membrane admitted a near-rigid-body field with sigma ~ 0 at
the load patch over the full sweep, so lambda peaked at only 0.17
before descending. The fix tried here adds a soft constraint that
lambda at each loaded-patch sample point cannot fall below
`lambda_floor_rate * s` (a linear-in-s floor). With
lambda_floor_rate = 1.5, the floor at s=1 is 1.5 -- comfortably below
the CSFM peak 3.08, so it does not constrain the peak position, but
firmly blocks the lambda-near-zero trivial state.

Continues only stage 9 (alpha=1.0, S_max=10 mm) since v2 already
passed through the full curriculum and stage 9 is what determines the
final equilibrium trace.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ArclengthPINN                                             # noqa: E402
from pinn_arclength import (                                                # noqa: E402
    ADAM_PER_STAGE, FIXED_ARC_WEIGHT, GRAD_CLIP_NORM,
    RELOBRALO_EVERY, RELOBRALO_LOSSES, SEED,
    ReLoBraLo, compute_losses, plot_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "P2" / "pinn"))
from problem import Corbel                                                  # noqa: E402


LAMBDA_FLOOR_RATE = 2.0         # rate per unit s; with S_max=15 the
                                 # floor at s=1 is 2.0 (well below the
                                 # CSFM peak 3.08 so it never caps the
                                 # peak position)
LAMBDA_FLOOR_WEIGHT = 50.0      # fixed-weight; held out of ReLoBraLo
N_ITER_CONT = 1500              # converges quickly when warm-started
S_MAX_MM = 15.0                  # extend arc-length window to include
                                 # the CSFM far peak at delta=14.6 mm


def main() -> None:
    prob = Corbel()
    here = Path(__file__).resolve().parent
    out_dir = here / "runs" / "corbel_v5"
    out_dir.mkdir(parents=True, exist_ok=True)
    # v5 warm-starts from v4 (already in the non-trivial basin) and
    # extends the arc-length window to S_max = 15 mm so the trace
    # reaches the CSFM far peak at delta = 14.6 mm.
    warmstart = here / "runs" / "corbel_v4" / "corbel_pinn.pt"
    if not warmstart.exists():
        raise FileNotFoundError(warmstart)

    gen = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)
    net = ArclengthPINN(width=96, depth=6)
    net.load_state_dict(torch.load(warmstart))
    print(f"warm-start from {warmstart}")

    alpha, S_max = 1.0, S_MAX_MM
    lr = 2.0e-4
    print(f"\n[continuation] alpha={alpha:.2f}  S_max={S_max:.2f} mm  "
          f"lr={lr:.1e}  lambda_floor_rate={LAMBDA_FLOOR_RATE}  "
          f"lambda_floor_weight={LAMBDA_FLOOR_WEIGHT}")

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    weighter = ReLoBraLo(list(RELOBRALO_LOSSES))
    last_good_state = {k: v.clone() for k, v in net.state_dict().items()}

    history: list[dict] = []
    t0 = time.time()
    for it in range(N_ITER_CONT):
        opt.zero_grad()
        losses = compute_losses(
            net, prob, alpha, gen, S_max_mm=S_max,
            arc_direction=(0.0, -1.0),
            lambda_floor_rate=LAMBDA_FLOOR_RATE,
        )
        if any(torch.isnan(v).any() for v in losses.values()):
            print(f"  [NaN at iter {it}] rolling back to last good")
            net.load_state_dict(last_good_state)
            break

        physics = {n: losses[n] for n in RELOBRALO_LOSSES}
        if it % RELOBRALO_EVERY == 0:
            weighter.update(physics)
        total = (weighter.weighted_sum(physics)
                 + FIXED_ARC_WEIGHT * losses["arc"]
                 + LAMBDA_FLOOR_WEIGHT * losses["lam_floor"])
        total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),
                                       max_norm=GRAD_CLIP_NORM)
        opt.step()

        if it % 250 == 0 or it == N_ITER_CONT - 1:
            row = {"iter": it, "total": float(total.detach())}
            row.update({n: float(losses[n].detach()) for n in losses})
            history.append(row)
            print(f"  it={it:5d}  total={float(total.detach()):.3e}  "
                  f"eq={float(losses['eq']):.2e}  "
                  f"supp={float(losses['supp']):.2e}  "
                  f"arc={float(losses['arc']):.2e}  "
                  f"load={float(losses['load']):.2e}  "
                  f"lam_floor={float(losses['lam_floor']):.2e}")
            # snapshot of network lambda at the patch
            with torch.no_grad():
                s_probe = torch.linspace(0, 1, 6).unsqueeze(-1)
                xp = torch.full_like(s_probe, prob.x_load)
                yp = torch.full_like(s_probe, prob.H)
                xy_n = torch.cat([xp / prob.L, yp / prob.H], dim=-1)
                _, lam_pr = net(xy_n, s_probe)
                vals = ", ".join(f"{float(lam_pr[k]):+.2f}" for k in range(6))
                print(f"         lambda(s in [0..1]) = [{vals}]")
            last_good_state = {k: v.clone() for k, v in net.state_dict().items()}

    wall = time.time() - t0
    torch.save(net.state_dict(), out_dir / "corbel_pinn.pt")
    curve = plot_curve(net, prob, out_dir)
    src = out_dir / "deepbeam_curve.png"
    if src.exists():
        src.replace(out_dir / "corbel_pinn_curve.png")
    with open(out_dir / "training_history.json", "w") as f:
        json.dump({"history": history, "curve": curve,
                   "wall_s": wall,
                   "lambda_floor_rate": LAMBDA_FLOOR_RATE,
                   "lambda_floor_weight": LAMBDA_FLOOR_WEIGHT,
                   "n_iter": N_ITER_CONT,
                   "warmstart": str(warmstart.name)}, f)
    print(f"\nwall: {wall:.1f}s")
    print(f"-> {out_dir}/corbel_pinn.pt")


if __name__ == "__main__":
    main()
