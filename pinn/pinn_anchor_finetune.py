"""Anchor-loss fine-tuning: warm-start from a trained PINN checkpoint
and continue training with an additional anchor-loss term that pins
the network's lambda(s) trajectory to the CSFM reference curve at
the network's own delta(s). Use for deep beam, corbel, and VK1
(no-N) to align the PINN shape with the displacement-controlled
CSFM reference (which the un-anchored arc-length PINN matches at
the peak but not along the curve).

Usage: import build_finetuner and call build_finetuner('deepbeam'
| 'corbel' | 'vk1').run(...).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_loss import CSFMCurveTarget                                     # noqa: E402
from model import ArclengthPINN                                             # noqa: E402
from pinn_arclength import (                                                # noqa: E402
    FIXED_ARC_WEIGHT, RELOBRALO_EVERY, RELOBRALO_LOSSES,
    SEED, ReLoBraLo, compute_losses,
)
from compute_losses_vk1 import compute_losses_vk1                            # noqa: E402
from pinn_arclength_vk1_v3 import sanitise_grads_                            # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "P2" / "pinn"))
from problem import Corbel, DeepBeam                                         # noqa: E402

U0 = 1.0e-3
N_ANCHOR = 256                # collocation count for the anchor loss
ANCHOR_WEIGHT = 200.0          # fixed, large; the anchor IS the shape


@dataclass
class FineTuneConfig:
    name: str
    prob: object
    warmstart_ckpt: Path
    out_dir: Path
    target: CSFMCurveTarget
    # Loss configuration
    alpha: float
    S_max_mm: float
    lr: float
    n_iter: int
    grad_clip: float
    # Geometry-specific load-patch probe to get (delta, lambda) at
    # the patch centre. Should return tensors of shape (N, 1) each.
    probe_xy: callable
    # arc_direction passed to compute_losses (corbel/VK1); None for
    # deepbeam (isotropic legacy)
    arc_direction: tuple[float, float] | None
    # lambda_floor_rate (None = inactive)
    lambda_floor_rate: float | None
    lambda_floor_weight: float
    # Whether to use the VK1-specific compute_losses_vk1
    use_vk1_loss: bool
    # Whether to use gradient sanitisation (VK1 only)
    use_grad_sanitise: bool
    # delta_PINN_from_disp converts (ux, uy) at probe -> delta the
    # CSFM curve indexes against (= -uy for deepbeam/corbel,
    # = +ux for VK1)
    delta_sign_uy: float  # +1 for VK1 (delta = ux is +x), 0 for n/a
    delta_sign_ux: float


def deepbeam_probe(prob: DeepBeam, gen: torch.Generator) -> tuple[Tensor, Tensor]:
    s = torch.rand(N_ANCHOR, 1, generator=gen).requires_grad_(False)
    x = torch.full_like(s, prob.x_load)
    y = torch.full_like(s, prob.H)
    return torch.cat([x, y, s], dim=-1), s   # bundle: x|y|s for caller


def make_probe(prob, kind: str):
    """Return a callable probe_xy(gen) -> (xs, ys, ss) that samples
    s in (0, 1] at the load-patch centre point."""
    def _probe(gen: torch.Generator):
        s = torch.rand(N_ANCHOR, 1, generator=gen)
        if kind == "deepbeam":
            x = torch.full_like(s, prob.x_load)
            y = torch.full_like(s, prob.H)
        elif kind == "corbel":
            x = torch.full_like(s, prob.x_load)
            y = torch.full_like(s, prob.H)
        elif kind == "vk1":
            x = torch.full_like(s, 0.0)
            y = torch.full_like(s, prob.h_eff)
        else:
            raise ValueError(kind)
        return x, y, s
    return _probe


def anchor_term(net: ArclengthPINN, prob, target: CSFMCurveTarget,
                probe_fn, gen: torch.Generator,
                kind: str) -> Tensor:
    x, y, s = probe_fn(gen)
    xy_n = torch.cat([x / prob.L, y / prob.H], dim=-1)
    out, lam = net(xy_n, s)
    ux = out[:, 0:1] * U0 * prob.L
    uy = out[:, 1:2] * U0 * prob.H
    if kind in ("deepbeam", "corbel"):
        delta = -uy
    elif kind == "vk1":
        delta = ux
    else:
        raise ValueError(kind)
    delta_clamped = torch.clamp(delta, min=0.0)
    # Anchor ONLY inside the target's window. interp clamps to the
    # boundary value beyond it, which for a truncated target would pull
    # every free state toward a constant and fake the fold location;
    # masking leaves the path beyond the window to the physics.
    mask = (delta_clamped.detach()
            <= float(target.deltas[-1])).float()
    lam_target = target.interp(delta_clamped.detach())
    return (((lam - lam_target) ** 2) * mask).sum()         / mask.sum().clamp(min=1.0)


def finetune(cfg: FineTuneConfig, kind: str) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(SEED + 7)
    torch.manual_seed(SEED + 7)
    net = ArclengthPINN(width=96, depth=6)
    net.load_state_dict(torch.load(cfg.warmstart_ckpt))
    print(f"[{cfg.name}] warm-start from {cfg.warmstart_ckpt.name}")
    print(f"  alpha={cfg.alpha}  S_max={cfg.S_max_mm} mm  lr={cfg.lr}  "
          f"n_iter={cfg.n_iter}  anchor_w={ANCHOR_WEIGHT}")

    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    weighter = ReLoBraLo(list(RELOBRALO_LOSSES))
    last_good = {k: v.clone() for k, v in net.state_dict().items()}
    n_rollbacks = 0
    history: list[dict] = []
    skipped = 0
    t0 = time.time()
    probe_fn = make_probe(cfg.prob, kind)

    for it in range(cfg.n_iter):
        opt.zero_grad()
        if cfg.use_vk1_loss:
            losses = compute_losses_vk1(
                net, cfg.prob, cfg.alpha, gen, S_max_mm=cfg.S_max_mm,
                lambda_floor_rate=cfg.lambda_floor_rate,
            )
        else:
            losses = compute_losses(
                net, cfg.prob, cfg.alpha, gen, S_max_mm=cfg.S_max_mm,
                arc_direction=cfg.arc_direction,
                lambda_floor_rate=cfg.lambda_floor_rate,
            )
        if any(torch.isnan(v).any() for v in losses.values()):
            skipped += 1
            continue
        l_anchor = anchor_term(net, cfg.prob, cfg.target, probe_fn,
                                gen, kind)
        physics = {n: losses[n] for n in RELOBRALO_LOSSES}
        if it % RELOBRALO_EVERY == 0:
            weighter.update(physics)
        total = (weighter.weighted_sum(physics)
                 + FIXED_ARC_WEIGHT * losses["arc"]
                 + ANCHOR_WEIGHT * l_anchor)
        if cfg.lambda_floor_rate is not None and "lam_floor" in losses:
            total = total + cfg.lambda_floor_weight * losses["lam_floor"]
        total.backward()
        if cfg.use_grad_sanitise:
            sanitise_grads_(net)
        torch.nn.utils.clip_grad_norm_(net.parameters(),
                                       max_norm=cfg.grad_clip)
        opt.step()
        if any(not torch.isfinite(p).all() for p in net.parameters()):
            # Roll back to the last good state and continue at half the
            # learning rate rather than terminating: a single non-finite
            # step early in the run must not end the whole finetune.
            n_rollbacks += 1
            if n_rollbacks > 6:
                print(f"  [non-finite params at iter {it}] rollback limit "
                      f"reached, stopping")
                net.load_state_dict(last_good)
                break
            for g in opt.param_groups:
                g["lr"] *= 0.5
            print(f"  [non-finite params at iter {it}] rollback "
                  f"#{n_rollbacks}, lr -> {opt.param_groups[0]['lr']:.1e}")
            net.load_state_dict(last_good)
            opt = torch.optim.Adam(net.parameters(),
                                   lr=opt.param_groups[0]["lr"])
            continue
        if it % 250 == 0 or it == cfg.n_iter - 1:
            with torch.no_grad():
                xq, yq, sq = probe_fn(gen)
                sq_lin = torch.linspace(0, 1, 6).unsqueeze(-1)
                xl = torch.full_like(sq_lin, float(xq[0]))
                yl = torch.full_like(sq_lin, float(yq[0]))
                xy_n = torch.cat([xl / cfg.prob.L, yl / cfg.prob.H],
                                 dim=-1)
                out, lam = net(xy_n, sq_lin)
                ux = out[:, 0:1] * U0 * cfg.prob.L
                uy = out[:, 1:2] * U0 * cfg.prob.H
                if kind in ("deepbeam", "corbel"):
                    delta = -uy
                else:
                    delta = ux
                lam_vals = ", ".join(f"{float(lam[k]):+.3f}"
                                     for k in range(6))
                d_vals = ", ".join(f"{float(delta[k]):+.2f}"
                                   for k in range(6))
            row = {"iter": it, "total": float(total.detach()),
                   "anchor": float(l_anchor.detach())}
            row.update({n: float(losses[n].detach()) for n in losses})
            history.append(row)
            print(f"  it={it:5d}  total={float(total.detach()):.3e}  "
                  f"anchor={float(l_anchor):.2e}  "
                  f"eq={float(losses['eq']):.2e}  "
                  f"load={float(losses['load']):.2e}  skipped={skipped}")
            print(f"         delta=[{d_vals}]  lam=[{lam_vals}]")
            last_good = {k: v.clone() for k, v in net.state_dict().items()}

    torch.save(net.state_dict(), cfg.out_dir / "pinn.pt")
    wall = time.time() - t0
    with open(cfg.out_dir / "training_history.json", "w") as f:
        json.dump({"history": history, "wall_s": wall,
                   "skipped": skipped,
                   "anchor_weight": ANCHOR_WEIGHT,
                   "warmstart": str(cfg.warmstart_ckpt.name)}, f)
    print(f"\nwall: {wall:.1f}s  skipped: {skipped}")
    return {"wall_s": wall, "skipped": skipped, "history": history}


def deepbeam_cfg() -> tuple[FineTuneConfig, str]:
    here = Path(__file__).resolve().parent
    cfg = FineTuneConfig(
        name="deepbeam",
        prob=DeepBeam(),
        warmstart_ckpt=here / "runs" / "arclength_pinn_latest.pt",
        out_dir=here / "runs" / "deepbeam_anchor",
        target=CSFMCurveTarget.from_deepbeam_oracle(),
        alpha=1.0, S_max_mm=10.0, lr=2e-4, n_iter=2500,
        grad_clip=1.0,
        probe_xy=None,
        arc_direction=None,
        lambda_floor_rate=None,
        lambda_floor_weight=0.0,
        use_vk1_loss=False,
        use_grad_sanitise=False,
        delta_sign_uy=-1.0, delta_sign_ux=0.0,
    )
    return cfg, "deepbeam"


def deepbeam_ascent_cfg() -> tuple[FineTuneConfig, str]:
    """The fold test: anchor ONLY the ascending branch (delta <= 3.5 mm)
    of the equilibrium-converged intact curve and leave the fold free.
    What is then tested is whether the arc-length parametrisation locates
    the limit point (reference: lambda = 1.3818 at 4.50 mm) on its own.
    """
    here = Path(__file__).resolve().parent
    cfg = FineTuneConfig(
        name="deepbeam_ascent",
        prob=DeepBeam(),
        warmstart_ckpt=here / "runs" / "arclength_pinn_latest.pt",
        out_dir=here / "runs" / "deepbeam_ascent_newton",
        target=CSFMCurveTarget.from_deepbeam_newton(delta_cut=3.5),
        alpha=1.0, S_max_mm=10.0, lr=2e-4, n_iter=2500,
        grad_clip=1.0,
        probe_xy=None,
        arc_direction=None,
        lambda_floor_rate=None,
        lambda_floor_weight=0.0,
        use_vk1_loss=False,
        use_grad_sanitise=False,
        delta_sign_uy=-1.0, delta_sign_ux=0.0,
    )
    return cfg, "deepbeam"


def corbel_cfg() -> tuple[FineTuneConfig, str]:
    here = Path(__file__).resolve().parent
    cfg = FineTuneConfig(
        name="corbel",
        prob=Corbel(),
        warmstart_ckpt=here / "runs" / "corbel_v4" / "corbel_pinn.pt",
        out_dir=here / "runs" / "corbel_anchor",
        target=CSFMCurveTarget.from_corbel_oracle(),
        alpha=1.0, S_max_mm=10.0, lr=2e-4, n_iter=2500,
        grad_clip=1.0,
        probe_xy=None,
        arc_direction=(0.0, -1.0),
        lambda_floor_rate=2.5,
        lambda_floor_weight=50.0,
        use_vk1_loss=False,
        use_grad_sanitise=False,
        delta_sign_uy=-1.0, delta_sign_ux=0.0,
    )
    return cfg, "corbel"


def corbel_newton_cfg() -> tuple[FineTuneConfig, str]:
    """corbel_cfg re-anchored to the equilibrium-converged reference."""
    here = Path(__file__).resolve().parent
    cfg = FineTuneConfig(
        name="corbel_newton",
        prob=Corbel(),
        warmstart_ckpt=here / "runs" / "corbel_v4" / "corbel_pinn.pt",
        out_dir=here / "runs" / "corbel_anchor_newton",
        target=CSFMCurveTarget.from_corbel_newton(),
        alpha=1.0, S_max_mm=10.0, lr=2e-4, n_iter=2500,
        grad_clip=1.0,
        probe_xy=None,
        arc_direction=(0.0, -1.0),
        # The floor is rate*s and must sit well BELOW the reference peak,
        # or it forces lambda above it. The secant-era 2.5 was chosen
        # against a peak of 3.08; the equilibrium-converged corbel peaks
        # at 1.7654, so 1.0 restores the original safety margin.
        lambda_floor_rate=1.0,
        lambda_floor_weight=50.0,
        use_vk1_loss=False,
        use_grad_sanitise=False,
        delta_sign_uy=-1.0, delta_sign_ux=0.0,
    )
    return cfg, "corbel"


def vk1_cfg() -> tuple[FineTuneConfig, str]:
    from wallpier_vk1 import WallPierVK1
    here = Path(__file__).resolve().parent
    cfg = FineTuneConfig(
        name="vk1",
        prob=WallPierVK1(include_N=False),
        warmstart_ckpt=here / "runs" / "vk1_v5" / "vk1_pinn.pt",
        out_dir=here / "runs" / "vk1_anchor",
        target=CSFMCurveTarget.from_vk1_noN_oracle(),
        alpha=1.0, S_max_mm=30.0, lr=5e-5, n_iter=3000,
        grad_clip=0.2,
        probe_xy=None,
        arc_direction=None,   # vk1-specific loss uses its own direction
        lambda_floor_rate=1.05,
        lambda_floor_weight=50.0,
        use_vk1_loss=True,
        use_grad_sanitise=True,
        delta_sign_uy=0.0, delta_sign_ux=1.0,
    )
    return cfg, "vk1"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("which",
                        choices=["deepbeam", "deepbeam_ascent", "corbel", "corbel_newton", "vk1", "all"])
    args = parser.parse_args()
    if args.which in ("deepbeam", "all"):
        cfg, kind = deepbeam_cfg()
        finetune(cfg, kind)
    if args.which == "deepbeam_ascent":
        cfg, kind = deepbeam_ascent_cfg()
        finetune(cfg, kind)
    if args.which in ("corbel", "all"):
        cfg, kind = corbel_cfg()
        finetune(cfg, kind)
    if args.which == "corbel_newton":
        cfg, kind = corbel_newton_cfg()
        finetune(cfg, kind)
    if args.which in ("vk1", "all"):
        cfg, kind = vk1_cfg()
        finetune(cfg, kind)


if __name__ == "__main__":
    main()
