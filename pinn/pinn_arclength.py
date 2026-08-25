"""Arc-length-parametrised continuum CSFM PINN — P3 capstone.

Implements the METHOD.md (Research/P3/METHOD.md) design. The network maps
(x, y, s) -> (u(x,y,s), lambda(s)) and is trained against:

  * equilibrium  : div sigma(eps(u)) + lambda(s) b = 0  on interior points
                   (body force b = 0; load enters through tractions)
  * traction BC  : sigma . n = -lambda(s) p on the loaded patch,
                   sigma . n = 0 elsewhere
  * support BC   : u = 0 at the supports
  * cylindrical
    arc-length  : ||du/ds||^2_loaded ~ S_max^2   (curriculum on S_max)

Hard initial condition u(s=0) = 0, lambda(s=0) = 0 is baked into the
network ansatz (see `model.ArclengthPINN`).

**Anti-trivial-zero tactic.** Before stage 0 of the staged training, the
field + lambda heads are pre-trained on the elastic FE solution (see
`pretrain_elastic.py`) so the optimiser starts in a non-trivial basin
where all losses pull in the same direction. Without this, the trivial
solution `u = lambda = 0` is a strong attractor that ReLoBraLo alone
cannot escape.

Two nested continuations from METHOD.md:
  - inner: constitutive alpha = 0 (elastic) -> 1 (full CSFM with softening)
  - outer: arc-length S_max ramps from S0 (pre-peak) to S_max (post-peak)
Each (alpha, S_max) pair is one warm-started training stage.

Run:  python pinn_arclength.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

# Local model FIRST, then P2's modules (otherwise model.py is shadowed)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ArclengthPINN                                             # noqa: E402
from pretrain_elastic import pretrain                                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "P2" / "pinn"))
from problem import DeepBeam                                                # noqa: E402

# Use the C1-regularised constitutive (see `csfm_smooth.py`) instead of
# P2's hard `torch.where`-branched membrane(). The hard version returns
# NaN gradients on the first backward pass at any constitutive-homotopy
# alpha > 0, blocking all cracked-regime training. The smooth version
# agrees with P2 to within 1% on representative strain states
# (test_smooth_membrane.py) and is C1 across the e1 = 0, e2 = 0, steel
# yield and parabola-plateau lines.
from csfm_smooth import membrane_homotopy_smooth as membrane_homotopy       # noqa: E402

torch.set_default_dtype(torch.float32)
SEED = 20260522

U0 = 1.0e-3

# Curriculum schedule. Preliminary experiments with a finer alpha
# curriculum (0 -> 0.1 -> 0.25 -> 0.5 -> 0.75 -> 1.0) found that
# every intermediate alpha stage produced a NaN gradient between
# iterations 355 and 1129 of its 2500-iteration budget, while the
# direct elastic-to-full-cracked jump from a converged elastic stage
# trained successfully. The final schedule used in this study is
# therefore four stages: two elastic and two full-CSFM, with no
# intermediate constitutive blend. See METHOD.md for the empirical
# basis of this choice.
STAGE_SCHEDULE = [
    (0.00, 0.5),       # pure elastic, matches pre-training
    (0.00, 2.0),       # extend elastic regime
    (0.10, 2.0),       # first nudge into cracked map
    (0.25, 2.0),       # quarter-cracked
    (0.50, 2.0),       # half-cracked
    (0.50, 5.0),       # half-cracked, longer path
    (0.75, 5.0),       # three-quarter cracked
    (1.00, 5.0),       # full CSFM, up to and including the peak
    (1.00, 10.0),      # full CSFM, post-peak softening branch
]
LR_BY_ALPHA = {0.00: 1.5e-3, 0.10: 1.0e-3, 0.25: 7e-4,
               0.50: 5e-4, 0.75: 3e-4, 1.00: 2e-4}
PRETRAIN_ITERS = 1500
PRETRAIN_S_MAX = 0.5     # matches STAGE_SCHEDULE[0]
ADAM_PER_STAGE = 2500
N_INT = 2000
N_BC = 400
# `arc` is held OUT of the ReLoBraLo pool and given its own fixed large
# weight. ReLoBraLo's relative-improvement rule down-weights arc once it
# stops improving, even when it has not actually converged. The previous
# run reached only delta=0.55mm against a target S_max=2.0mm because of
# that. A fixed large weight forces the optimiser to keep prioritising
# the arc-length constraint regardless of how the physics losses move.
RELOBRALO_LOSSES = ["eq", "supp", "load", "free"]
FIXED_ARC_WEIGHT = 20.0
# Fixed weight on the loaded-patch sign anchor that breaks the
# (u, lambda) -> (-u, -lambda) branch degeneracy (see compute_losses).
SIGN_WEIGHT = 10.0
RELOBRALO_EVERY = 10
RELOBRALO_ALPHA = 0.95
RELOBRALO_T = 0.1
RELOBRALO_RHO = 0.95
GRAD_CLIP_NORM = 1.0
NAN_CHECK_EVERY = 1


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def grad(out: Tensor, inp: Tensor) -> Tensor:
    return torch.autograd.grad(
        out, inp, grad_outputs=torch.ones_like(out),
        create_graph=True, retain_graph=True,
    )[0]


def normalise(prob: DeepBeam, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    return x / prob.L, y / prob.H


def displacements(net: ArclengthPINN, prob: DeepBeam,
                  x: Tensor, y: Tensor, s: Tensor
                  ) -> tuple[Tensor, Tensor, Tensor]:
    xn, yn = normalise(prob, x, y)
    xy_n = torch.cat([xn, yn], dim=-1)
    out, lam = net(xy_n, s)
    ux = out[:, 0:1] * U0 * prob.L
    uy = out[:, 1:2] * U0 * prob.H
    return ux, uy, lam


def strains(ux: Tensor, uy: Tensor, x: Tensor, y: Tensor
            ) -> tuple[Tensor, Tensor, Tensor]:
    ex = grad(ux, x)
    ey = grad(uy, y)
    gxy = grad(ux, y) + grad(uy, x)
    return ex, ey, gxy


def stresses_homotopy(ex: Tensor, ey: Tensor, gxy: Tensor,
                       rho_x: Tensor, rho_y: Tensor, prob: DeepBeam,
                       alpha: float) -> tuple[Tensor, Tensor, Tensor]:
    st = membrane_homotopy(ex, ey, gxy, rho_x, rho_y, prob.mat, alpha=alpha)
    return st["sigma_x"], st["sigma_y"], st["tau_xy"]


def equilibrium_residual(sx: Tensor, sy: Tensor, txy: Tensor,
                          x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    rx = grad(sx, x) + grad(txy, y)
    ry = grad(txy, x) + grad(sy, y)
    return rx, ry


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #


def compute_losses(net: ArclengthPINN, prob: DeepBeam, alpha: float,
                   gen: torch.Generator,
                   S_max_mm: float = 1.0,
                   arc_direction: tuple[float, float] | None = None,
                   lambda_floor_rate: float | None = None,
                   ) -> dict[str, Tensor]:
    fc = prob.mat.fc

    # ---- interior equilibrium ----------------------------------------
    xi, yi = prob.interior(N_INT, gen)
    si = torch.rand(N_INT, 1, generator=gen)
    xi = xi.clone().requires_grad_(True)
    yi = yi.clone().requires_grad_(True)
    ux_i, uy_i, _ = displacements(net, prob, xi, yi, si)
    ex, ey, gxy = strains(ux_i, uy_i, xi, yi)
    rho_x = prob.rho_x(xi, yi)
    rho_y = prob.rho_y(xi, yi)
    sx, sy, txy = stresses_homotopy(ex, ey, gxy, rho_x, rho_y, prob, alpha)
    rx, ry = equilibrium_residual(sx, sy, txy, xi, yi)
    l_eq = ((rx * prob.L / fc) ** 2 + (ry * prob.L / fc) ** 2).mean()

    # ---- support BC (u = 0) ------------------------------------------
    xs, ys = prob.supports(N_BC, gen)
    ss = torch.rand(N_BC, 1, generator=gen)
    ux_s, uy_s, _ = displacements(net, prob, xs, ys, ss)
    l_supp = prob.support_residual(ux_s / U0, uy_s / U0, xs)

    # ---- loaded-patch traction: sigma_yy = -lambda * pressure -------
    xl, yl = prob.loaded_patch(N_BC, gen)
    sl = torch.rand(N_BC, 1, generator=gen)
    xl = xl.clone().requires_grad_(True)
    yl = yl.clone().requires_grad_(True)
    ux_l, uy_l, lam_l = displacements(net, prob, xl, yl, sl)
    ex_l, ey_l, gxy_l = strains(ux_l, uy_l, xl, yl)
    rho_x_l = prob.rho_x(xl, yl)
    rho_y_l = prob.rho_y(xl, yl)
    sx_l, sy_l, txy_l = stresses_homotopy(ex_l, ey_l, gxy_l,
                                          rho_x_l, rho_y_l, prob, alpha)
    target_p = -lam_l * prob.pressure
    l_load = (((sy_l - target_p) / fc) ** 2 + (txy_l / fc) ** 2).mean()

    # ---- traction-free edges -----------------------------------------
    xf, yf, nf = prob.free_edges(N_BC, gen)
    sf = torch.rand(xf.shape[0], 1, generator=gen)
    xf = xf.clone().requires_grad_(True)
    yf = yf.clone().requires_grad_(True)
    ux_f, uy_f, _ = displacements(net, prob, xf, yf, sf)
    ex_f, ey_f, gxy_f = strains(ux_f, uy_f, xf, yf)
    rho_x_f = prob.rho_x(xf, yf)
    rho_y_f = prob.rho_y(xf, yf)
    sx_f, sy_f, txy_f = stresses_homotopy(ex_f, ey_f, gxy_f,
                                          rho_x_f, rho_y_f, prob, alpha)
    nx, ny = nf[:, 0:1], nf[:, 1:2]
    tx = sx_f * nx + txy_f * ny
    ty = txy_f * nx + sy_f * ny
    l_free = ((tx / fc) ** 2 + (ty / fc) ** 2).mean()

    # ---- cylindrical arc-length over the loaded patch ----------------
    # Pointwise (Cauchy-Schwarz-tight) form. Penalising the deviation of
    # the MEAN of speed^2 from target^2, as we did initially, lets the
    # optimiser game the constraint by concentrating all motion into a
    # narrow window of s: mean(speed^2) = target^2 is satisfied while the
    # integrated path length, integral of speed ds, is strictly smaller
    # by the Cauchy-Schwarz gap. The pointwise mean-of-squares variant
    # below penalises both the mean and the variance of speed^2(s), so
    # the only way to drive the loss to zero is to have speed^2(s) close
    # to target^2 at every s.
    xa, ya = prob.loaded_patch(N_BC, gen)
    sa = torch.rand(N_BC, 1, generator=gen).requires_grad_(True)
    ux_a, uy_a, _ = displacements(net, prob, xa, ya, sa)
    dux_ds = grad(ux_a, sa)
    duy_ds = grad(uy_a, sa)
    if arc_direction is None:
        # Isotropic-speed form (deep beam): penalise |du/ds|^2 deviation
        # from S_max^2. Symmetric BCs cap the lateral component naturally.
        speed2 = dux_ds ** 2 + duy_ds ** 2
        target2 = S_max_mm ** 2
        l_arc = (((speed2 - target2) / target2) ** 2).mean()
    else:
        # Directional form (corbel and similar single-clamp BCs):
        # penalise the LOADED-DIRECTION speed only. Without this, the
        # network can satisfy |du/ds|^2 = S_max^2 by moving mostly in
        # the unconstrained transverse direction, leaving the loaded-
        # direction deflection (and therefore lambda) near zero. The
        # load_direction is the unit vector pointing in the direction
        # of the applied traction (e.g. (0, -1) for a downward patch).
        dx, dy = float(arc_direction[0]), float(arc_direction[1])
        speed_load_dir = dux_ds * dx + duy_ds * dy
        # we want speed_in_load_direction = +S_max_mm (positive: the
        # patch follows the load at unit rate per arc-length)
        l_arc = (((speed_load_dir - S_max_mm) / S_max_mm) ** 2).mean()

    out = {"eq": l_eq, "supp": l_supp, "load": l_load,
           "free": l_free, "arc": l_arc}

    # Sign anchor: the loaded patch must move in the applied-load direction
    # (downward, u_y <= 0). The arc-length loss |du/ds|^2 = S_max^2 is
    # sign-agnostic and the map (u, lambda) -> (-u, -lambda) is an
    # approximate symmetry of the equilibrium and load-BC losses, so without
    # this term the optimiser converges to the mirrored branch on a
    # seed-dependent fraction of runs. The penalty is identically zero on
    # the physical branch (u_y <= 0) and only activates on the mirror.
    out["sign"] = (torch.relu(uy_l / U0) ** 2).mean()

    # Soft lambda-floor: penalise lambda being below a linear floor
    # `lambda_floor_rate * s` at the loaded patch. Blocks the
    # "stress-trivial attractor" observed on asymmetric BCs (corbel)
    # where the cracked-membrane admits a strain field with sigma ~ 0
    # and the load BC sigma_y = -lambda * p is satisfied trivially
    # with both sides near zero. Inactive when lambda_floor_rate is
    # None (deep-beam default behaviour preserved).
    if lambda_floor_rate is not None:
        target_floor = lambda_floor_rate * sl
        l_lam_floor = (torch.relu(target_floor - lam_l) ** 2).mean()
        out["lam_floor"] = l_lam_floor

    return out


# --------------------------------------------------------------------------- #
# ReLoBraLo with random lookback (Bischof & Kraus 2021)
# --------------------------------------------------------------------------- #


class ReLoBraLo:
    def __init__(self, names: list[str],
                 alpha: float = RELOBRALO_ALPHA, T: float = RELOBRALO_T,
                 rho: float = RELOBRALO_RHO):
        self.names = list(names)
        self.alpha = alpha
        self.T = T
        self.rho = rho
        self.weights = {n: 1.0 for n in self.names}
        self.prev: dict[str, float] | None = None
        self.init: dict[str, float] | None = None
        self._rng = np.random.default_rng(SEED)

    def update(self, losses: dict[str, Tensor]) -> None:
        cur = {n: float(losses[n].detach().item()) for n in self.names}
        if self.init is None:
            self.init = cur
            self.prev = cur
            return
        ref = self.prev if self._rng.random() < self.rho else self.init
        ratios = {n: cur[n] / (ref[n] + 1e-12) for n in self.names}
        x = np.array([ratios[n] / self.T for n in self.names])
        x -= x.max()
        ex = np.exp(x)
        bal = len(self.names) * ex / ex.sum()
        for i, n in enumerate(self.names):
            self.weights[n] = (
                self.alpha * self.weights[n]
                + (1.0 - self.alpha) * float(bal[i])
            )
        self.prev = cur

    def weighted_sum(self, losses: dict[str, Tensor]) -> Tensor:
        return sum(self.weights[n] * losses[n] for n in self.names)


# --------------------------------------------------------------------------- #
# Training pipeline
# --------------------------------------------------------------------------- #


def train(prob: DeepBeam, out_dir: Path, do_pretrain: bool = True) -> dict:
    gen = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)
    net = ArclengthPINN(width=96, depth=6)

    out_dir.mkdir(parents=True, exist_ok=True)
    pre_info: dict | None = None

    # ---- pre-train on the elastic FE solution ------------------------
    if do_pretrain:
        print(f"\n[pre-train] elastic FE warm-start, "
              f"{PRETRAIN_ITERS} iters, S_max={PRETRAIN_S_MAX} mm")
        pre_info = pretrain(net, prob, S_max_mm=PRETRAIN_S_MAX,
                            n_iter=PRETRAIN_ITERS, verbose=True)
        torch.save(net.state_dict(), out_dir / "pretrained_elastic.pt")
    else:
        print("[skip pre-train] starting from random init")

    # The pretrained state is our first checkpoint. `last_good_state` is
    # the network state we roll back to if a stage NaNs; `last_good_stage`
    # is the index of the last stage that completed cleanly (-1 = only
    # pretraining is trusted).
    last_good_state = {k: v.clone() for k, v in net.state_dict().items()}
    last_good_stage = -1
    last_good_alpha = 0.0
    last_good_S_max = PRETRAIN_S_MAX

    # ---- staged Adam with ReLoBraLo on physics losses + fixed arc weight ----
    loss_names = list(RELOBRALO_LOSSES)   # eq, supp, load, free
    all_loss_names = loss_names + ["arc"]
    history: list[dict] = []
    completed_stages: list[dict] = []

    for stage_idx, (alpha, S_max) in enumerate(STAGE_SCHEDULE):
        t0 = time.time()
        # LR decreases as the constitutive stiffens
        lr_stage = LR_BY_ALPHA.get(alpha, 5e-4)
        print(f"\n[stage {stage_idx + 1}/{len(STAGE_SCHEDULE)}] "
              f"alpha={alpha:.2f}  S_max={S_max:.2f} mm  lr={lr_stage:.1e}")

        # Fresh optimizer state each stage: warm-starting Adam's moments
        # across a constitutive jump tends to amplify the first bad
        # gradient. A fresh moment estimate adapts to the new geometry.
        opt = torch.optim.Adam(net.parameters(), lr=lr_stage)
        # Fresh ReLoBraLo state too: the absolute-magnitude reference
        # losses from the previous stage are stale at the new alpha.
        weighter = ReLoBraLo(loss_names)

        stage_failed = False
        nan_iter = -1
        for it in range(ADAM_PER_STAGE):
            opt.zero_grad()
            # Directional arc-length constraint on the load direction
            # (downward, (0, -1)). The isotropic form lets the network
            # satisfy |du/ds| = S_max with non-load-direction motion, so the
            # loaded-patch deflection stalls on a seed-dependent fraction of
            # runs; constraining the load-direction speed forces delta to
            # advance. Combined with the SIGN_WEIGHT anchor this removes the
            # two seed-dependent failure modes (mirror branch + stalled path).
            losses = compute_losses(net, prob, alpha, gen, S_max_mm=S_max,
                                    arc_direction=(0.0, -1.0))

            if any(torch.isnan(v).any() for v in losses.values()):
                stage_failed = True
                nan_iter = it
                print(f"  [NaN at iter {it}] aborting stage; "
                      f"rolling back to stage {last_good_stage + 1} weights")
                net.load_state_dict(last_good_state)
                break

            # ReLoBraLo on physics losses only; arc gets a fixed large
            # weight (see RELOBRALO_LOSSES / FIXED_ARC_WEIGHT above).
            physics_losses = {n: losses[n] for n in loss_names}
            if it % RELOBRALO_EVERY == 0:
                weighter.update(physics_losses)
            total = (weighter.weighted_sum(physics_losses)
                     + FIXED_ARC_WEIGHT * losses["arc"]
                     + SIGN_WEIGHT * losses["sign"])
            total.backward()

            # Gradient clipping: prevents a single anomalous batch from
            # poisoning Adam's running moment estimates and cascading the
            # parameters to NaN over subsequent iterations.
            torch.nn.utils.clip_grad_norm_(net.parameters(),
                                           max_norm=GRAD_CLIP_NORM)
            opt.step()

            if it % 250 == 0 or it == ADAM_PER_STAGE - 1:
                row = {"stage": stage_idx, "alpha": alpha, "S_max": S_max,
                       "iter": it, "total": float(total.detach())}
                row.update({k: float(v.detach()) for k, v in losses.items()})
                row.update({f"w_{k}": weighter.weights[k]
                            for k in loss_names})
                row["w_arc"] = FIXED_ARC_WEIGHT
                history.append(row)
                w_str = " ".join(f"w_{k}={weighter.weights[k]:.2f}"
                                 for k in loss_names)
                print(f"  it={it:5d}  total={float(total.detach()):.3e}  "
                      f"eq={float(losses['eq']):.2e}  "
                      f"supp={float(losses['supp']):.2e}  "
                      f"arc={float(losses['arc']):.2e}  "
                      f"load={float(losses['load']):.2e}  | {w_str} "
                      f"w_arc={FIXED_ARC_WEIGHT:.1f} [fixed]")

        wall = time.time() - t0
        if stage_failed:
            print(f"  stage failed at iter {nan_iter} after {wall:.1f}s "
                  f"(rolled back to last good)")
            completed_stages.append({"stage": stage_idx, "alpha": alpha,
                                     "S_max": S_max, "completed": False,
                                     "nan_iter": nan_iter, "wall_s": wall})
            # do not save a stage checkpoint for a failed stage; continue
            # to the next stage with the rolled-back weights (skip-ahead).
            continue

        # Stage completed without NaN. Save the checkpoint, update the
        # last_good_state, and continue.
        ckpt = out_dir / f"arclength_pinn_stage{stage_idx + 1}_a{int(alpha * 100):03d}_S{int(S_max * 10):03d}.pt"
        torch.save(net.state_dict(), ckpt)
        torch.save(net.state_dict(), out_dir / "arclength_pinn_latest.pt")
        last_good_state = {k: v.clone() for k, v in net.state_dict().items()}
        last_good_stage = stage_idx
        last_good_alpha = alpha
        last_good_S_max = S_max
        completed_stages.append({"stage": stage_idx, "alpha": alpha,
                                 "S_max": S_max, "completed": True,
                                 "wall_s": wall, "checkpoint": str(ckpt.name)})
        print(f"  stage done in {wall:.1f}s -> {ckpt.name}")

    # Final artifact is the last_good_state, which may not be the very
    # last stage if later ones NaN'd. validate.py reads this file.
    torch.save(last_good_state, out_dir / "arclength_pinn.pt")
    return {"history": history,
            "completed_stages": completed_stages,
            "last_good_stage": last_good_stage,
            "last_good_alpha": last_good_alpha,
            "last_good_S_max": last_good_S_max,
            "stage_schedule": [(a, s) for a, s in STAGE_SCHEDULE],
            "pre_info": pre_info}


def plot_curve(net: ArclengthPINN, prob: DeepBeam, out_dir: Path) -> dict:
    net.eval()
    with torch.no_grad():
        s_vals = torch.linspace(0.0, 1.0, 50).unsqueeze(-1)
        x_load = torch.full((50, 1), prob.x_load)
        y_load = torch.full((50, 1), prob.H)
        _ux, uy, lam = displacements(net, prob, x_load, y_load, s_vals)
        delta = -uy.squeeze().numpy()
        lam_v = lam.squeeze().numpy()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(delta, lam_v, "o-", label="P3 arc-length PINN")
    ax.set_xlabel("load-patch deflection delta (mm)")
    ax.set_ylabel("load factor lambda")
    ax.set_title("P3 deepbeam — equilibrium path from PINN")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "deepbeam_curve.png", dpi=120)
    plt.close(fig)
    return {"delta": delta.tolist(), "lam": lam_v.tolist()}


def main() -> None:
    prob = DeepBeam()
    out_dir = Path(__file__).resolve().parent / "runs"
    print(f"P3 arc-length PINN — deepbeam — out: {out_dir}")
    info = train(prob, out_dir, do_pretrain=True)
    net = ArclengthPINN(width=96, depth=6)
    net.load_state_dict(torch.load(out_dir / "arclength_pinn.pt"))
    curve = plot_curve(net, prob, out_dir)
    with open(out_dir / "training_history.json", "w") as f:
        json.dump({"history": info["history"],
                   "curve": curve,
                   "completed_stages": info["completed_stages"],
                   "last_good_stage": info["last_good_stage"],
                   "last_good_alpha": info["last_good_alpha"],
                   "last_good_S_max": info["last_good_S_max"],
                   "stage_schedule": info["stage_schedule"]}, f)
    print(f"-> {out_dir}/arclength_pinn.pt")
    print(f"-> {out_dir}/deepbeam_curve.png")
    print(f"-> {out_dir}/training_history.json")


if __name__ == "__main__":
    main()
