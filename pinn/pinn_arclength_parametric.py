"""Parametric arc-length CSFM PINN: one network, a family of equilibrium
paths.

Extends the deep-beam arc-length PINN (pinn_arclength.py) with the soffit
tie reinforcement ratio as a network input: (x, y, s, theta) -> (u, lambda),
theta parameterising a 0-70% corrosion section loss of the main tension tie
(rho_tie = 0.012 (1 - loss)). Each training iteration draws one loss level
uniformly, so the physics losses are satisfied across the whole family;
after training, the full equilibrium path for any deterioration state is a
single forward sweep in s.

Rationale: corrosion loss of the tension tie is the dominant deterioration
mode of aging concrete bridge members; a parametric network amortises one
training over every deterioration scenario queried afterwards, giving
capacity and ductility as continuous functions of section loss.

Pre-training: the elastic response is independent of rho_tie (reinforcement
enters only the cracked constitutive), so the elastic warm-start target is
the same for every theta.

Everything else (stage schedule, ReLoBraLo pool, fixed arc weight, sign
anchor, NaN rollback, seeds) is identical to the single-design trainer.

Run:  python pinn_arclength_parametric.py
"""
from __future__ import annotations

import dataclasses
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ParametricArclengthPINN                                   # noqa: E402
from pretrain_elastic import elastic_fe, bilinear_sample                    # noqa: E402
from pinn_arclength import (                                                # noqa: E402
    SEED, U0, STAGE_SCHEDULE, LR_BY_ALPHA, ADAM_PER_STAGE, N_INT, N_BC,
    RELOBRALO_LOSSES, FIXED_ARC_WEIGHT, SIGN_WEIGHT, RELOBRALO_EVERY,
    GRAD_CLIP_NORM, ReLoBraLo, grad, normalise,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "P2" / "pinn"))
from problem import DeepBeam                                                # noqa: E402
from csfm_constitutive import CsfmMaterial                                  # noqa: E402
from csfm_smooth import membrane_homotopy_smooth as membrane_homotopy       # noqa: E402

torch.set_default_dtype(torch.float32)

RHO_NOM = 0.012                 # nominal soffit-tie ratio
# Corrosion section loss of the tie. The range stops at 30 % because the
# displacement-controlled reference is only trustworthy that far: beyond
# it the fold migrates into the region where the secant solver snaps onto
# a spurious stiff branch, and the reported capacity RISES with section
# loss (2.35 to 2.47 over 40-70 %, against 2.29 for the intact beam),
# which removing tension steel cannot do. Validating against those curves
# would be validating against solver artefacts.
LOSS_LO, LOSS_HI = 0.0, 0.30
LOSS_EVAL = [0.0, 0.05, 0.10, 0.20, 0.25, 0.30]
PRETRAIN_ITERS = 2500          # a touch longer: the map now covers theta
PRETRAIN_S_MAX = 0.5


def theta_of(loss: float) -> float:
    return (loss - 0.5 * (LOSS_LO + LOSS_HI)) / (0.5 * (LOSS_HI - LOSS_LO))


def displacements_p(net: ParametricArclengthPINN, prob: DeepBeam,
                    x: Tensor, y: Tensor, s: Tensor, theta: Tensor
                    ) -> tuple[Tensor, Tensor, Tensor]:
    xn, yn = normalise(prob, x, y)
    xy_n = torch.cat([xn, yn], dim=-1)
    out, lam = net(xy_n, s, theta)
    ux = out[:, 0:1] * U0 * prob.L
    uy = out[:, 1:2] * U0 * prob.H
    return ux, uy, lam


def strains(ux, uy, x, y):
    ex = grad(ux, x)
    ey = grad(uy, y)
    gxy = grad(ux, y) + grad(uy, x)
    return ex, ey, gxy


def stresses_h(ex, ey, gxy, rho_x, rho_y, mat: CsfmMaterial, alpha: float):
    st = membrane_homotopy(ex, ey, gxy, rho_x, rho_y, mat, alpha=alpha)
    return st["sigma_x"], st["sigma_y"], st["tau_xy"]


def compute_losses_p(net: ParametricArclengthPINN, prob: DeepBeam,
                     mat: CsfmMaterial, theta_val: float, alpha: float,
                     gen: torch.Generator, S_max_mm: float,
                     ) -> dict[str, Tensor]:
    """Losses of pinn_arclength.compute_losses, with (mat, theta) explicit
    and the deep-beam directional arc form (load direction (0, -1))."""
    fc = mat.fc

    def th(n: int) -> Tensor:
        return torch.full((n, 1), float(theta_val))

    # ---- interior equilibrium ----------------------------------------
    xi, yi = prob.interior(N_INT, gen)
    si = torch.rand(N_INT, 1, generator=gen)
    xi = xi.clone().requires_grad_(True)
    yi = yi.clone().requires_grad_(True)
    ux_i, uy_i, _ = displacements_p(net, prob, xi, yi, si, th(N_INT))
    ex, ey, gxy = strains(ux_i, uy_i, xi, yi)
    sx, sy, txy = stresses_h(ex, ey, gxy, prob.rho_x(xi, yi),
                             prob.rho_y(xi, yi), mat, alpha)
    rx = grad(sx, xi) + grad(txy, yi)
    ry = grad(txy, xi) + grad(sy, yi)
    l_eq = ((rx * prob.L / fc) ** 2 + (ry * prob.L / fc) ** 2).mean()

    # ---- support BC ---------------------------------------------------
    xs, ys = prob.supports(N_BC, gen)
    ss = torch.rand(N_BC, 1, generator=gen)
    ux_s, uy_s, _ = displacements_p(net, prob, xs, ys, ss, th(N_BC))
    l_supp = prob.support_residual(ux_s / U0, uy_s / U0, xs)

    # ---- loaded-patch traction ---------------------------------------
    xl, yl = prob.loaded_patch(N_BC, gen)
    sl = torch.rand(N_BC, 1, generator=gen)
    xl = xl.clone().requires_grad_(True)
    yl = yl.clone().requires_grad_(True)
    ux_l, uy_l, lam_l = displacements_p(net, prob, xl, yl, sl, th(N_BC))
    ex_l, ey_l, gxy_l = strains(ux_l, uy_l, xl, yl)
    sx_l, sy_l, txy_l = stresses_h(ex_l, ey_l, gxy_l, prob.rho_x(xl, yl),
                                   prob.rho_y(xl, yl), mat, alpha)
    target_p = -lam_l * prob.pressure
    l_load = (((sy_l - target_p) / fc) ** 2 + (txy_l / fc) ** 2).mean()

    # ---- traction-free edges -----------------------------------------
    xf, yf, nf = prob.free_edges(N_BC, gen)
    sf = torch.rand(xf.shape[0], 1, generator=gen)
    xf = xf.clone().requires_grad_(True)
    yf = yf.clone().requires_grad_(True)
    ux_f, uy_f, _ = displacements_p(net, prob, xf, yf, sf, th(xf.shape[0]))
    ex_f, ey_f, gxy_f = strains(ux_f, uy_f, xf, yf)
    sx_f, sy_f, txy_f = stresses_h(ex_f, ey_f, gxy_f, prob.rho_x(xf, yf),
                                   prob.rho_y(xf, yf), mat, alpha)
    nx, ny = nf[:, 0:1], nf[:, 1:2]
    tx = sx_f * nx + txy_f * ny
    ty = txy_f * nx + sy_f * ny
    l_free = ((tx / fc) ** 2 + (ty / fc) ** 2).mean()

    # ---- directional arc-length on the loaded patch ------------------
    xa, ya = prob.loaded_patch(N_BC, gen)
    sa = torch.rand(N_BC, 1, generator=gen).requires_grad_(True)
    _ux_a, uy_a, _ = displacements_p(net, prob, xa, ya, sa, th(N_BC))
    duy_ds = grad(uy_a, sa)
    speed_load_dir = -duy_ds          # load direction (0, -1)
    l_arc = (((speed_load_dir - S_max_mm) / S_max_mm) ** 2).mean()

    out = {"eq": l_eq, "supp": l_supp, "load": l_load,
           "free": l_free, "arc": l_arc}
    out["sign"] = (torch.relu(uy_l / U0) ** 2).mean()
    return out


# --------------------------------------------------------------------------- #
# Parametric elastic pre-training
# --------------------------------------------------------------------------- #


def pretrain_parametric(net: ParametricArclengthPINN, prob: DeepBeam,
                        S_max_mm: float, n_iter: int,
                        lr: float = 2e-3, n_int: int = 1024) -> dict:
    nx_fe, ny_fe = 40, 20
    xy_fe, u_fe, fe_info = elastic_fe(prob, nx=nx_fe, ny=ny_fe)
    delta_fe = float(-u_fe[fe_info["load_nodes"], 1].mean())
    scale = S_max_mm / max(delta_fe, 1e-9)
    lam_max_ref = scale               # elastic load factor at f_c = FC_REF
    gen = torch.Generator().manual_seed(SEED)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(SEED)
    print(f"[pretrain-p] elastic warm-start over tie loss in "
          f"[{LOSS_LO}, {LOSS_HI}], {n_iter} iters "
          f"(delta_FE={delta_fe:.4f} mm, lam_max_ref={lam_max_ref:.4f})",
          flush=True)
    for it in range(n_iter):
        loss_frac = float(rng.uniform(LOSS_LO, LOSS_HI))
        lam_max = lam_max_ref        # elastic response is rho_tie-independent
        thv = theta_of(loss_frac)
        x = torch.rand(n_int, 1, generator=gen) * prob.L
        y = torch.rand(n_int, 1, generator=gen) * prob.H
        s = torch.rand(n_int, 1, generator=gen)
        u_t = bilinear_sample(xy_fe, u_fe, nx_fe, ny_fe,
                              fe_info["dx"], fe_info["dy"], x, y) * scale
        target_ux = s * u_t[:, 0:1]
        target_uy = s * u_t[:, 1:2]
        target_lam = s * lam_max
        th = torch.full((n_int, 1), thv)
        ux, uy, lam = displacements_p(net, prob, x, y, s, th)
        loss = (((ux - target_ux) / S_max_mm) ** 2
                + ((uy - target_uy) / S_max_mm) ** 2).mean() \
            + (((lam - target_lam) / max(lam_max_ref, 1e-6)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 500 == 0 or it == n_iter - 1:
            print(f"  it={it:5d}  loss={float(loss):.3e}  "
                  f"(tie loss={loss_frac:.2f})", flush=True)
    return {"scale": scale, "lam_max_ref": lam_max_ref,
            "delta_fe": delta_fe}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def train(prob: DeepBeam, out_dir: Path) -> dict:
    gen = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    net = ParametricArclengthPINN(width=96, depth=6)
    # The hard symmetry ansatz is deliberately OFF. Mirroring the field
    # about midspan over-constrains the equilibrium path: it drives the
    # limit point to a much smaller deflection and prevents the fold from
    # migrating with the design parameter, because the post-peak field
    # physically bifurcates at the limit point and is not mirror
    # symmetric. The canonical single-design network does not use it
    # either; see the symmetry caveat in the manuscript's limitations.
    net.symmetric = False
    out_dir.mkdir(parents=True, exist_ok=True)

    pre = pretrain_parametric(net, prob, PRETRAIN_S_MAX, PRETRAIN_ITERS)
    torch.save(net.state_dict(), out_dir / "parametric_pretrained.pt")

    last_good = {k: v.clone() for k, v in net.state_dict().items()}
    loss_names = list(RELOBRALO_LOSSES)
    history: list[dict] = []
    completed: list[dict] = []

    for stage_idx, (alpha, S_max) in enumerate(STAGE_SCHEDULE):
        t0 = time.time()
        lr_stage = LR_BY_ALPHA.get(alpha, 5e-4)
        print(f"\n[stage {stage_idx + 1}/{len(STAGE_SCHEDULE)}] "
              f"alpha={alpha:.2f}  S_max={S_max:.2f}  lr={lr_stage:.1e}",
              flush=True)
        opt = torch.optim.Adam(net.parameters(), lr=lr_stage)
        weighter = ReLoBraLo(loss_names)
        failed, nan_iter = False, -1

        for it in range(ADAM_PER_STAGE):
            loss_frac = float(rng.uniform(LOSS_LO, LOSS_HI))
            prob_it = dataclasses.replace(
                prob, rho_tie=RHO_NOM * (1.0 - loss_frac))
            opt.zero_grad()
            losses = compute_losses_p(net, prob_it, prob_it.mat,
                                      theta_of(loss_frac),
                                      alpha, gen, S_max_mm=S_max)
            if any(torch.isnan(v).any() for v in losses.values()):
                failed, nan_iter = True, it
                print(f"  [NaN at iter {it}] rollback", flush=True)
                net.load_state_dict(last_good)
                break
            phys = {n: losses[n] for n in loss_names}
            if it % RELOBRALO_EVERY == 0:
                weighter.update(phys)
            total = (weighter.weighted_sum(phys)
                     + FIXED_ARC_WEIGHT * losses["arc"]
                     + SIGN_WEIGHT * losses["sign"])
            total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),
                                           max_norm=GRAD_CLIP_NORM)
            opt.step()
            if it % 250 == 0 or it == ADAM_PER_STAGE - 1:
                row = {"stage": stage_idx, "alpha": alpha, "S_max": S_max,
                       "iter": it, "tie_loss": loss_frac,
                       "total": float(total.detach())}
                row.update({k: float(v.detach()) for k, v in losses.items()})
                history.append(row)
                print(f"  it={it:5d} tie_loss={loss_frac:4.2f} "
                      f"total={float(total.detach()):.3e} "
                      f"eq={float(losses['eq']):.2e} "
                      f"arc={float(losses['arc']):.2e}", flush=True)

        wall = time.time() - t0
        if failed:
            completed.append({"stage": stage_idx, "alpha": alpha,
                              "S_max": S_max, "completed": False,
                              "nan_iter": nan_iter, "wall_s": wall})
            continue
        torch.save(net.state_dict(), out_dir / "parametric_latest.pt")
        last_good = {k: v.clone() for k, v in net.state_dict().items()}
        completed.append({"stage": stage_idx, "alpha": alpha,
                          "S_max": S_max, "completed": True, "wall_s": wall})
        print(f"  stage done in {wall:.1f}s", flush=True)

    torch.save(last_good, out_dir / "parametric_arclength.pt")
    return {"history": history, "completed_stages": completed,
            "pre_info": pre}


def eval_curves(net: ParametricArclengthPINN, prob: DeepBeam) -> dict:
    net.eval()
    curves = {}
    with torch.no_grad():
        s_vals = torch.linspace(0.0, 1.0, 60).unsqueeze(-1)
        x_load = torch.full((60, 1), prob.x_load)
        y_load = torch.full((60, 1), prob.H)
        for loss_frac in LOSS_EVAL:
            th = torch.full((60, 1), theta_of(loss_frac))
            _ux, uy, lam = displacements_p(net, prob, x_load, y_load,
                                           s_vals, th)
            curves[f"{loss_frac:.2f}"] = {
                "delta": (-uy).squeeze().tolist(),
                "lam": lam.squeeze().tolist()}
    return curves


def main() -> None:
    prob = DeepBeam()
    out_dir = Path(__file__).resolve().parent / "runs" / "parametric_rho"
    t0 = time.time()
    info = train(prob, out_dir)
    net = ParametricArclengthPINN(width=96, depth=6)
    net.symmetric = False
    net.load_state_dict(torch.load(out_dir / "parametric_arclength.pt"))
    curves = eval_curves(net, prob)
    with open(out_dir / "parametric_curves.json", "w") as f:
        json.dump({"loss_eval": LOSS_EVAL, "rho_nominal": RHO_NOM,
                   "curves": curves,
                   "completed_stages": info["completed_stages"],
                   "wall_total_s": time.time() - t0}, f, indent=2)
    print(f"\nTotal wall: {(time.time() - t0) / 60:.1f} min")
    print(f"-> {out_dir}/parametric_arclength.pt")
    print(f"-> {out_dir}/parametric_curves.json")


if __name__ == "__main__":
    main()
