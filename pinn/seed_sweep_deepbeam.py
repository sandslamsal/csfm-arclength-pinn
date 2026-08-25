"""Multi-seed sweep on the deep-beam PINN. For each seed in
{1, 2, 3, 4, 5} we run the working {1, 2, 8, 9}-stage trajectory
(elastic warm-up, full-cracked at S_max=5 mm, full-cracked at
S_max=10 mm) followed by the anchor finetune and the directional
extension to S_max = 15 mm. Reports peak lambda and path-RMSE
against the displacement-controlled CSFM reference, with mean +/-
range across seeds.

Each seed reuses the same code path as the canonical v9 + anchor +
directional runs, just with a different RNG. The total per-seed
wall-time on a single CPU core is roughly 15 minutes; 5 seeds take
~75 minutes total.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_loss import CSFMCurveTarget                                     # noqa: E402
from model import ArclengthPINN                                             # noqa: E402
from pinn_arclength import (                                                # noqa: E402
    ADAM_PER_STAGE, FIXED_ARC_WEIGHT, GRAD_CLIP_NORM,
    LR_BY_ALPHA, RELOBRALO_EVERY, RELOBRALO_LOSSES,
    ReLoBraLo, compute_losses, plot_curve,
)
from pinn_anchor_finetune import ANCHOR_WEIGHT, anchor_term, make_probe     # noqa: E402
from pretrain_elastic import pretrain                                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "P2" / "pinn"))
from problem import DeepBeam                                                # noqa: E402

U0 = 1.0e-3
SEEDS = [1, 2, 3, 4, 5]

# Compressed working trajectory: stage 1 (elastic, S=0.5), stage 2
# (elastic, S=2.0), stage 8 (cracked S=5), stage 9 (cracked S=10).
# This is the {1,2,8,9} path documented in the v9 NaN-rollback summary.
WORKING_STAGES = [
    (0.00, 0.5),
    (0.00, 2.0),
    (1.00, 5.0),
    (1.00, 10.0),
]
ANCHOR_ITERS = 2500
EXTEND_S_MAX = 15.0
EXTEND_ITERS = 2500


def train_one(seed: int, prob: DeepBeam, target: CSFMCurveTarget,
              out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== seed {seed} ===")
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    net = ArclengthPINN(width=96, depth=6)

    # ---- pretrain ----
    pretrain(net, prob, S_max_mm=0.5, n_iter=1500, verbose=False)

    # ---- working stages ----
    last_good = {k: v.clone() for k, v in net.state_dict().items()}
    stage_outcomes: list[dict] = []
    for stage_idx, (alpha, S_max) in enumerate(WORKING_STAGES):
        lr_stage = LR_BY_ALPHA.get(alpha, 5e-4)
        opt = torch.optim.Adam(net.parameters(), lr=lr_stage)
        weighter = ReLoBraLo(list(RELOBRALO_LOSSES))
        failed = False
        for it in range(ADAM_PER_STAGE):
            opt.zero_grad()
            losses = compute_losses(net, prob, alpha, gen, S_max_mm=S_max)
            if any(torch.isnan(v).any() for v in losses.values()):
                failed = True
                net.load_state_dict(last_good)
                break
            physics = {n: losses[n] for n in RELOBRALO_LOSSES}
            if it % RELOBRALO_EVERY == 0:
                weighter.update(physics)
            total = (weighter.weighted_sum(physics)
                     + FIXED_ARC_WEIGHT * losses["arc"])
            total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),
                                           max_norm=GRAD_CLIP_NORM)
            opt.step()
        if failed:
            stage_outcomes.append({"stage": stage_idx, "alpha": alpha,
                                   "S_max": S_max, "completed": False})
        else:
            last_good = {k: v.clone() for k, v in net.state_dict().items()}
            stage_outcomes.append({"stage": stage_idx, "alpha": alpha,
                                   "S_max": S_max, "completed": True})
        print(f"  seed {seed} stage {stage_idx + 1} "
              f"alpha={alpha:.2f} S={S_max:.1f}  "
              f"{'NaN' if failed else 'OK'}")

    # ---- anchor finetune (S_max=10) ----
    # Guarded like the staged / extension phases: a NaN loss or a
    # non-finite parameter after the step restores the last good state
    # and abandons the phase, so the network can never be left NaN here
    # (the unguarded version of this loop was the cause of the
    # multi-seed NaN divergences).
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    weighter = ReLoBraLo(list(RELOBRALO_LOSSES))
    probe_fn = make_probe(prob, "deepbeam")
    last_good = {k: v.clone() for k, v in net.state_dict().items()}
    for it in range(ANCHOR_ITERS):
        opt.zero_grad()
        losses = compute_losses(net, prob, 1.0, gen, S_max_mm=10.0,
                                arc_direction=None)
        if any(torch.isnan(v).any() for v in losses.values()):
            net.load_state_dict(last_good)
            break
        l_anchor = anchor_term(net, prob, target, probe_fn, gen,
                               "deepbeam")
        physics = {n: losses[n] for n in RELOBRALO_LOSSES}
        if it % RELOBRALO_EVERY == 0:
            weighter.update(physics)
        total = (weighter.weighted_sum(physics)
                 + FIXED_ARC_WEIGHT * losses["arc"]
                 + ANCHOR_WEIGHT * l_anchor)
        total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),
                                       max_norm=GRAD_CLIP_NORM)
        opt.step()
        if any(not torch.isfinite(p).all() for p in net.parameters()):
            net.load_state_dict(last_good)
            break
        if it % 100 == 0:
            last_good = {k: v.clone() for k, v in net.state_dict().items()}

    # ---- directional extension (S_max=15) ----
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    weighter = ReLoBraLo(list(RELOBRALO_LOSSES))
    last_good = {k: v.clone() for k, v in net.state_dict().items()}
    for it in range(EXTEND_ITERS):
        opt.zero_grad()
        losses = compute_losses(net, prob, 1.0, gen,
                                S_max_mm=EXTEND_S_MAX,
                                arc_direction=(0.0, -1.0))
        if any(torch.isnan(v).any() for v in losses.values()):
            net.load_state_dict(last_good)
            break
        l_anchor = anchor_term(net, prob, target, probe_fn, gen,
                               "deepbeam")
        physics = {n: losses[n] for n in RELOBRALO_LOSSES}
        if it % RELOBRALO_EVERY == 0:
            weighter.update(physics)
        total = (weighter.weighted_sum(physics)
                 + FIXED_ARC_WEIGHT * losses["arc"]
                 + ANCHOR_WEIGHT * l_anchor)
        total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),
                                       max_norm=GRAD_CLIP_NORM)
        opt.step()
        if any(not torch.isfinite(p).all() for p in net.parameters()):
            net.load_state_dict(last_good)
            break

    # ---- read peak from final state ----
    with torch.no_grad():
        s = torch.linspace(0, 1, 200).unsqueeze(-1)
        x = torch.full_like(s, prob.x_load)
        y = torch.full_like(s, prob.H)
        xy_n = torch.cat([x / prob.L, y / prob.H], dim=-1)
        out, lam = net(xy_n, s)
        uy = out[:, 1:2] * U0 * prob.H
    delta = (-uy).squeeze().numpy()
    lam_np = lam.squeeze().numpy()
    torch.save(net.state_dict(), out_dir / f"pinn_seed{seed}.pt")

    failed = bool(np.isnan(lam_np).any() or not np.isfinite(lam_np).all())
    if failed:
        return {"seed": seed, "failed": True,
                "stage_outcomes": stage_outcomes}

    i_peak = int(np.argmax(lam_np))
    peak_lam = float(lam_np[i_peak])
    peak_delta = float(delta[i_peak])
    # Path-RMSE vs CSFM reference at sampled delta values
    csfm_at_pinn = target.interp(torch.tensor(delta).unsqueeze(-1)
                                  ).squeeze().numpy()
    rmse_lam = float(np.sqrt(np.mean((lam_np - csfm_at_pinn) ** 2)))
    return {"seed": seed, "failed": False, "peak_lam": peak_lam,
            "peak_delta": peak_delta, "path_rmse_lam": rmse_lam,
            "stage_outcomes": stage_outcomes}


def main() -> None:
    here = Path(__file__).resolve().parent
    out_dir = here / "runs" / "seed_sweep_newton"
    out_dir.mkdir(parents=True, exist_ok=True)
    prob = DeepBeam()
    # route 2: anchor to the equilibrium-converged intact curve
    target = CSFMCurveTarget.from_deepbeam_newton()

    results: list[dict] = []
    t0 = time.time()
    for seed in SEEDS:
        t_seed = time.time()
        r = train_one(seed, prob, target, out_dir)
        r["wall_s"] = time.time() - t_seed
        if r["failed"]:
            print(f"  seed {seed}: FAILED (NaN)  wall={r['wall_s']:.0f}s")
        else:
            print(f"  seed {seed}: peak_lam={r['peak_lam']:.3f} "
                  f"@ delta={r['peak_delta']:.2f} mm  "
                  f"path_RMSE_lam={r['path_rmse_lam']:.3f}  "
                  f"wall={r['wall_s']:.0f}s")
        results.append(r)

    ok = [r for r in results if not r["failed"]]
    peaks = np.array([r["peak_lam"] for r in ok]) if ok else np.array([np.nan])
    deltas = np.array([r["peak_delta"] for r in ok]) if ok else np.array([np.nan])
    rmses = np.array([r["path_rmse_lam"] for r in ok]) if ok else np.array([np.nan])
    summary = {
        "seeds": SEEDS,
        "n_success": len(ok),
        "n_total": len(results),
        "peak_lam_mean": float(peaks.mean()),
        "peak_lam_std": float(peaks.std()),
        "peak_lam_min": float(peaks.min()),
        "peak_lam_max": float(peaks.max()),
        "peak_delta_mean": float(deltas.mean()),
        "peak_delta_std": float(deltas.std()),
        "path_rmse_mean": float(rmses.mean()),
        "path_rmse_std": float(rmses.std()),
        "results": results,
        "total_wall_s": time.time() - t0,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== SUMMARY ===")
    print(f"success: {summary['n_success']}/{summary['n_total']} seeds")
    print(f"peak_lam (successes) = {summary['peak_lam_mean']:.3f} +/- "
          f"{summary['peak_lam_std']:.3f}  "
          f"(min {summary['peak_lam_min']:.3f}, "
          f"max {summary['peak_lam_max']:.3f})")
    print(f"path_RMSE_lam = {summary['path_rmse_mean']:.3f} +/- "
          f"{summary['path_rmse_std']:.3f}")
    print(f"total wall: {summary['total_wall_s']:.0f}s")
    print(f"-> {out_dir}/summary.json")


if __name__ == "__main__":
    main()
