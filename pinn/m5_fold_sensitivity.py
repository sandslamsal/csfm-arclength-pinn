"""M5: is the differentiability claim demonstrable, not just asserted?

The Conclusions call differentiability in the design parameter "the
methodological reason to prefer the learned formulation", and
Section 3.6 derives the fold sensitivity dlambda*/dtheta, but neither is
ever evaluated on the trained network. This script closes that gap by
computing the same derivative three independent ways and comparing them:

  autodiff   one backward pass through the trained parametric network
  reference  central finite differences of the equilibrium-converged
             family, which never sees the network
  network FD central finite differences of the network's own peak, which
             isolates autodiff error from model error

Agreement between the first two is the claim the paper needs. Agreement
between the first and third only shows the autodiff is correct.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import ParametricArclengthPINN                                  # noqa: E402
from problem import DeepBeam                                               # noqa: E402
from pinn_arclength_parametric import (                                    # noqa: E402
    LOSS_LO, LOSS_HI, theta_of, displacements_p,
)

CKPT = HERE / "runs" / "parametric_rho" / "parametric_anchored_newton.pt"
N_S = 400


def peak_lambda(net, prob, theta_val, requires_grad=False):
    """Smooth-max peak load factor at a given theta, differentiable in theta."""
    th = torch.tensor([[float(theta_val)]], dtype=torch.float32,
                      requires_grad=requires_grad)
    s = torch.linspace(0.0, 1.0, N_S).unsqueeze(-1)
    x = torch.full_like(s, prob.x_load)
    y = torch.full_like(s, prob.H)
    thr = th.expand(N_S, 1)
    _ux, uy, lam = displacements_p(net, prob, x, y, s, thr)
    delta = -uy
    # restrict to the reference window so the max is the physical peak
    mask = (delta.squeeze() <= 7.0).float()
    lam_m = lam.squeeze() * mask - 1e3 * (1.0 - mask)
    # log-sum-exp smooth max: differentiable, and tight for beta large
    beta = 400.0
    peak = torch.logsumexp(beta * lam_m, dim=0) / beta
    return peak, th


def main():
    prob = DeepBeam()
    net = ParametricArclengthPINN(width=96, depth=6)
    net.load_state_dict(torch.load(CKPT))
    net.eval()

    fam = json.load(open(HERE.parent / "oracle"
                         / "deepbeam_family_newton.json"))
    ref = {float(k): v["lam_max"] for k, v in fam["curves"].items()}
    ref_losses = np.array(sorted(ref))
    ref_lams = np.array([ref[k] for k in ref_losses])

    print("d(lambda_peak) / d(section loss), three independent routes\n")
    print(f"{'loss':>7}{'autodiff':>12}{'network FD':>13}{'reference FD':>14}"
          f"{'autodiff vs ref':>17}")

    rows = []
    for loss in (0.05, 0.10, 0.20, 0.25):
        th_val = theta_of(loss)
        # --- 1. autodiff through the network
        peak, th = peak_lambda(net, prob, th_val, requires_grad=True)
        peak.backward()
        dlam_dtheta = float(th.grad)
        # chain rule: theta = (loss - mid) / half  =>  dtheta/dloss = 1/half
        half = 0.5 * (LOSS_HI - LOSS_LO)
        auto = dlam_dtheta / half

        # --- 2. finite differences of the network itself
        h = 0.01
        with torch.no_grad():
            p_hi, _ = peak_lambda(net, prob, theta_of(loss + h))
            p_lo, _ = peak_lambda(net, prob, theta_of(loss - h))
        net_fd = float(p_hi - p_lo) / (2 * h)

        # --- 3. finite differences of the reference family
        ref_fd = float(np.gradient(ref_lams, ref_losses)[
            int(np.argmin(np.abs(ref_losses - loss)))])

        rows.append((loss, auto, net_fd, ref_fd))
        err = (auto - ref_fd) / abs(ref_fd) * 100.0
        print(f"{loss*100:>6.0f}%{auto:>12.3f}{net_fd:>13.3f}{ref_fd:>14.3f}"
              f"{err:>16.1f}%")

    a = np.array([r[1] for r in rows]); r_ = np.array([r[3] for r in rows])
    nfd = np.array([r[2] for r in rows])
    print(f"\nautodiff vs its own finite differences: "
          f"max |err| {np.max(np.abs(a - nfd) / np.abs(nfd)) * 100:.2f} %"
          "  (checks the gradient, not the model)")
    print(f"autodiff vs reference finite differences: "
          f"mean |err| {np.mean(np.abs(a - r_) / np.abs(r_)) * 100:.1f} %,"
          f" max {np.max(np.abs(a - r_) / np.abs(r_)) * 100:.1f} %")
    print(f"\nreference sensitivity spans {r_.min():.3f} to {r_.max():.3f} "
          f"per unit section loss")
    print("one backward pass returns this; the reference needs two extra "
          "solves per level")
    json.dump({"rows": [{"loss": r[0], "autodiff": r[1], "network_fd": r[2],
                         "reference_fd": r[3]} for r in rows]},
              open(HERE / "runs" / "parametric_rho" / "m5_sensitivity.json", "w"))


if __name__ == "__main__":
    main()
