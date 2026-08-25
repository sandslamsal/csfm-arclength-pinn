"""Pilot 1: the arc-length formulation on the mixed (Airy + displacement)
ansatz.

The displacement-only ansatz cannot hold interior equilibrium and the
traction-free boundary at once (F5): the free edge is the pressure
valve. Here equilibrium is exact by construction, sigma from an Airy
head, the constitutive is evaluated in its well-posed strain-to-stress
direction on a displacement head, and the two are tied by an algebraic
coupling residual sigma_phi = sigma_CSFM(eps(u)). The load factor is
not a head at all: lambda(s) is read from the patch traction of the
Airy field, so the fold is available with no special status exactly as
in the displacement formulation.

Success criteria, decided before running: (i) reach comparable to the
requested window; (ii) free-edge tractions at the sub-MPa level of the
fixed-load Airy runs rather than the -30 MPa of the displacement
ansatz; (iii) a fold, with peak lambda against the 1.35/1.48 bounds.
Any of the three failing is a documented negative.

FIRST ATTEMPT (v1, no F1 cure): collapsed to the trivial-zero
attractor, peak lambda 0.003. F1 transfers verbatim to the mixed
ansatz. This version adds the paper's own basin selector, a soft
lambda-floor at the elastic rate of the first window with the
standard de-rating of two, held fixed across stages so it can never
sit above the cracked peak (the floor-above-peak trap of the
taxonomy).
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "P2" / "pinn"))
from csfm_smooth import membrane_smooth                                   # noqa: E402
from problem import DeepBeam                                              # noqa: E402

torch.set_default_dtype(torch.float64)
LR_N = 1000.0
S_STAGES = [0.5, 2.0, 10.0]         # mm, the {1,2,9} windows
ITERS = 2500
N_INT, N_BC, N_ARC = 2000, 400, 256
W_ARC = 20.0
W_FLOOR = 50.0
FLOOR_RATE = (S_STAGES[0] / 0.287) / 2.0   # elastic rate of the first
                                           # window, de-rated by 2
LR = 3e-4
CLIP = 1.0
SEED = 0


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


class AiryArcNet(torch.nn.Module):
    """(X, Y, s) -> (Phi, ux, uy), all three vanishing at s = 0."""

    def __init__(self, n_freq=24, width=128, depth=5, sigma=3.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.B = torch.nn.Parameter(
            torch.randn(2, n_freq, generator=g) * sigma, requires_grad=False)
        din = 2 * n_freq + 1
        layers = [torch.nn.Linear(din, width), torch.nn.Tanh()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
        layers += [torch.nn.Linear(width, 3)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, X, Y, s):
        proj = 2 * math.pi * torch.cat([X, Y], 1) @ self.B
        feat = torch.cat([torch.sin(proj), torch.cos(proj), s], 1)
        out = self.net(feat)
        return (s * out[:, 0:1], s * out[:, 1:2], s * out[:, 2:3])


def fields(net, x, y, s):
    X = (x / LR_N).requires_grad_(True)
    Y = (y / LR_N).requires_grad_(True)
    Phi, ux, uy = net(X, Y, s)
    Phi_X = grad(Phi, X); Phi_Y = grad(Phi, Y)
    sx = grad(Phi_Y, Y)          # * fc below
    sy = grad(Phi_X, X)
    txy = -grad(Phi_X, Y)
    return sx, sy, txy, ux, uy, X, Y


def main():
    prob = DeepBeam()
    fc, mat, p = prob.mat.fc, prob.mat, prob.pressure
    g = torch.Generator().manual_seed(SEED)
    net = AiryArcNet(seed=SEED)
    out_dir = HERE / "runs" / "pilot_airy_arclength"
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = []
    t00 = time.time()

    for stage, S_max in enumerate(S_STAGES, 1):
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        print(f"[stage {stage}] S_max = {S_max} mm", flush=True)
        for it in range(ITERS):
            opt.zero_grad()
            # ---- coupling residual, interior ------------------------
            xi, yi = prob.interior(N_INT, g)
            si = torch.rand(N_INT, 1, generator=g)
            sx, sy, txy, ux, uy, X, Y = fields(net, xi, yi, si)
            ux_X = grad(ux, X); uy_Y = grad(uy, Y)
            ux_Y = grad(ux, Y); uy_X = grad(uy, X)
            ex, ey, gxy = ux_X / LR_N, uy_Y / LR_N, (ux_Y + uy_X) / LR_N
            m = membrane_smooth(ex, ey, gxy, prob.rho_x(xi, yi),
                                prob.rho_y(xi, yi), mat)
            l_con = (((fc * sx - m["sigma_x"]) / fc) ** 2
                     + ((fc * sy - m["sigma_y"]) / fc) ** 2
                     + ((fc * txy - m["tau_xy"]) / fc) ** 2).mean()
            # ---- free edges ----------------------------------------
            xf, yf, nf = prob.free_edges(N_BC, g)
            sf = torch.rand(xf.shape[0], 1, generator=g)
            sxf, syf, txyf, *_ = fields(net, xf, yf, sf)
            nx_f, ny_f = nf[:, 0:1], nf[:, 1:2]
            tx = sxf * nx_f + txyf * ny_f
            ty = txyf * nx_f + syf * ny_f
            l_free = (tx ** 2 + ty ** 2).mean()          # already /fc units
            # ---- supports ------------------------------------------
            xs_, ys_ = prob.supports(N_BC, g)
            ss = torch.rand(xs_.shape[0], 1, generator=g)
            Xs = (xs_ / LR_N).requires_grad_(False)
            Ys = (ys_ / LR_N).requires_grad_(False)
            _, uxs, uys = net(Xs, Ys, ss)
            left = (xs_ < prob.L / 2.0)
            l_supp = (uys ** 2).mean() + (uxs[left] ** 2).mean()
            # ---- patch shape: uniform sigma_y, zero shear ----------
            xl, yl = prob.loaded_patch(N_BC, g)
            sl = torch.rand(xl.shape[0], 1, generator=g)
            sxl, syl, txyl, uxl, uyl, Xl, Yl = fields(net, xl, yl, sl)
            # per-s uniformity: subtract the mean within each of 8 s-bins
            bins = (sl * 8).long().clamp(max=7).squeeze(-1)
            sy_mean = torch.zeros(8, dtype=syl.dtype)
            cnt = torch.zeros(8, dtype=syl.dtype)
            sy_mean.scatter_add_(0, bins, syl.squeeze(-1))
            cnt.scatter_add_(0, bins, torch.ones_like(syl.squeeze(-1)))
            sy_mean = sy_mean / cnt.clamp(min=1.0)
            l_shape = ((syl.squeeze(-1) - sy_mean[bins]) ** 2).mean() \
                + (txyl ** 2).mean()
            # sign: patch must descend, and carry compression
            l_sign = (torch.relu(uyl) ** 2).mean() \
                + (torch.relu(fc * syl) / fc ** 2).mean()
            # ---- arc-length, pointwise, patch ----------------------
            xa, ya = prob.loaded_patch(N_ARC, g)
            sa = torch.rand(xa.shape[0], 1, generator=g).requires_grad_(True)
            Xa = (xa / LR_N); Ya = (ya / LR_N)
            _, uxa, uya = net(Xa, Ya, sa)
            dux = grad(uxa, sa); duy = grad(uya, sa)
            speed2 = dux ** 2 + duy ** 2
            l_arc = (((speed2 - S_max ** 2) / S_max ** 2) ** 2).mean()

            # lambda-floor (F1 cure): the patch traction level must
            # grow at least at the de-rated elastic rate of the FIRST
            # window. Differentiable through sigma_y of the Airy head.
            lam_a = -fc * syl / p
            l_floor = (torch.relu(FLOOR_RATE * sl - lam_a) ** 2).mean()
            total = (l_con + 5.0 * l_free + l_supp + l_shape
                     + 10.0 * l_sign + W_ARC * l_arc
                     + W_FLOOR * l_floor)
            total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
            opt.step()

            if it % 250 == 0 or it == ITERS - 1:
                with torch.no_grad():
                    # lambda(s) readout at s = 1
                    pass
                lam1, d1 = readout(net, prob, fc, p)
                row = dict(stage=stage, it=it, con=float(l_con),
                           floor=float(l_floor),
                           free=float(l_free), supp=float(l_supp),
                           shape=float(l_shape), arc=float(l_arc),
                           lam_s1=lam1, delta_s1=d1)
                hist.append(row)
                print(f"  it={it:5d} con={row['con']:.2e} "
                      f"free={row['free']:.2e} arc={row['arc']:.2e} "
                      f"lam(1)={lam1:+.3f} d(1)={d1:+.2f} mm", flush=True)
        torch.save(net.state_dict(), out_dir / f"stage{stage}.pt")

    torch.save(net.state_dict(), out_dir / "pilot_airy.pt")
    # trace the full curve
    curve = trace(net, prob, fc, p)
    json.dump({"hist": hist, "curve": curve,
               "wall_s": time.time() - t00},
              open(out_dir / "pilot_airy.json", "w"), indent=1)
    lam = curve["lam"]
    k = max(range(len(lam)), key=lambda i: lam[i])
    print(f"\npeak lambda {lam[k]:.4f} at {curve['delta'][k]:.2f} mm; "
          f"reference 1.3817 at 4.00; bounds 1.35/1.48")
    print(f"free-edge traction rms at s near peak: {curve['free_rms_mpa']:.3f} MPa")


def readout(net, prob, fc, p, s_val=1.0):
    xq = torch.linspace(prob.x_load - prob.bearing / 2,
                        prob.x_load + prob.bearing / 2, 100).unsqueeze(-1)
    yq = torch.full_like(xq, prob.H)
    sq = torch.full_like(xq, s_val)
    sx, sy, txy, ux, uy, *_ = fields(net, xq, yq, sq)
    lam = float((-fc * sy.mean() / p).detach())
    d = float((-uy.mean()).detach())
    return lam, d


def trace(net, prob, fc, p, n=101):
    lams, deltas = [], []
    for s_val in [i / (n - 1) for i in range(1, n)]:
        lam, d = readout(net, prob, fc, p, s_val)
        lams.append(lam); deltas.append(d)
    # free-edge traction at the state nearest the peak
    k = max(range(len(lams)), key=lambda i: lams[i])
    s_pk = (k + 1) / (n - 1)
    g = torch.Generator().manual_seed(1)
    xf, yf, nf = prob.free_edges(800, g)
    sf = torch.full((xf.shape[0], 1), s_pk)
    sxf, syf, txyf, *_ = fields(net, xf, yf, sf)
    tx = fc * (sxf * nf[:, 0:1] + txyf * nf[:, 1:2])
    ty = fc * (txyf * nf[:, 0:1] + syf * nf[:, 1:2])
    frms = float(torch.sqrt((tx ** 2 + ty ** 2).mean()).detach())
    return {"lam": lams, "delta": deltas, "free_rms_mpa": frms}


if __name__ == "__main__":
    main()
