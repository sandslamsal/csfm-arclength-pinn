"""Anchor-point lambda loss: tie the PINN's lambda(s) trajectory to
the CSFM displacement-controlled reference curve at the SAME delta.

The arc-length-parametrised PINN finds A solution on the equilibrium
manifold but not necessarily the SAME (delta, lambda) trajectory the
displacement-controlled CSFM solver traces. The anchor loss pins the
PINN's lambda(s) to lambda_CSFM(delta(s)) at every collocation
point on the loaded patch, removing the parameterisation freedom.

This module is a thin helper used by `compute_losses` (deepbeam +
corbel) and `compute_losses_vk1` (VK1). The curve is loaded once at
module level from the JSON reference of the corresponding benchmark
and bilinearly interpolated against the network's |displacement| at
each iteration.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


class CSFMCurveTarget:
    """Holds a sorted (delta, lambda) array and provides a torch-
    differentiable interpolation: target_lam(delta) for any tensor
    delta in [0, delta_max]. Outside [0, delta_max] the value is
    clamped to the boundary.
    """

    def __init__(self, deltas: np.ndarray, lams: np.ndarray):
        order = np.argsort(deltas)
        self.deltas = torch.tensor(deltas[order], dtype=torch.float32)
        self.lams = torch.tensor(lams[order], dtype=torch.float32)

    @classmethod
    def from_json(cls, path: Path, delta_key: str = "delta",
                  lam_key: str = "lam") -> "CSFMCurveTarget":
        data = json.load(open(path))
        # Most JSONs have a `curve` list of dicts
        if "curve" in data:
            d = np.array([p[delta_key] for p in data["curve"]],
                         dtype=float)
            l = np.array([p[lam_key] for p in data["curve"]],
                         dtype=float)
        else:
            d = np.array(data[delta_key], dtype=float)
            l = np.array(data[lam_key], dtype=float)
        # Build monotone-increasing envelope past the peak so that
        # the interp gives a sensible target on the descending branch
        # (raw post-peak CSFM data is noisy).
        i_peak = int(np.argmax(l))
        env = l.copy()
        for i in range(i_peak + 1, len(env)):
            env[i] = min(env[i], env[i - 1])
        return cls(d, env)

    @classmethod
    def from_deepbeam_oracle(cls) -> "CSFMCurveTarget":
        here = Path(__file__).resolve().parent
        path = here.parent / "oracle" / "deepbeam_oracle.json"
        return cls.from_json(path, "delta", "lam")

    @classmethod
    def from_corbel_oracle(cls) -> "CSFMCurveTarget":
        here = Path(__file__).resolve().parent
        path = here.parent / "oracle" / "corbel_reference.json"
        return cls.from_json(path, "delta", "lam")

    @classmethod
    def from_vk1_noN_oracle(cls) -> "CSFMCurveTarget":
        here = Path(__file__).resolve().parent
        path = here.parent / "oracle" / "vk1_reference_noN.json"
        return cls.from_json(path, "delta_x", "lam")

    @classmethod
    def from_corbel_newton(cls) -> "CSFMCurveTarget":
        """Corbel target on equilibrium-converged states (route-2 reference).

        Only the converged prefix anchors: past it the trace stops meeting
        the residual criterion and an anchor must not target those states.
        """
        import json as _json
        import numpy as _np
        here = Path(__file__).resolve().parent
        d = _json.load(open(here.parent / "oracle" / "corbel_newton_curve.json"))
        dd = _np.array(d["delta"], dtype=float)
        ll = _np.array(d["lam"], dtype=float)
        cv = _np.array(d["converged"], dtype=bool)
        return cls(dd[cv], ll[cv])

    @classmethod
    def from_deepbeam_newton(cls, delta_cut: float | None = None
                             ) -> "CSFMCurveTarget":
        """Deep-beam intact-level target from the Newton family.

        delta_cut truncates the target: with a cut below the reference
        peak (4.50 mm) the anchor covers only the ascending branch and
        the anchor mask leaves everything beyond it free, so the fold
        location is the network's own.
        """
        import json as _json
        import numpy as _np
        here = Path(__file__).resolve().parent
        fam = _json.load(open(here.parent / "oracle"
                              / "deepbeam_family_newton.json"))
        c = fam["curves"]["0.00"]
        d = _np.array(c["delta"], dtype=float)
        l = _np.array(c["lam"], dtype=float)
        if delta_cut is not None:
            keep = d <= delta_cut
            d, l = d[keep], l[keep]
        return cls(d, l)

    @classmethod
    def from_vk1_newton(cls) -> "CSFMCurveTarget":
        """VK1 with-N lambda target on equilibrium-converged states."""
        here = Path(__file__).resolve().parent
        path = here.parent / "oracle" / "vk1_newton_full.json"
        return cls.from_json(path, "delta_x", "lam")

    def interp(self, delta: Tensor) -> Tensor:
        """Differentiable linear interpolation of lambda at the
        provided delta values. Inputs outside [0, delta_max] are
        clamped to the boundary.
        """
        d_min = self.deltas[0]
        d_max = self.deltas[-1]
        d = torch.clamp(delta, min=float(d_min), max=float(d_max))
        # Find the upper-bin index for each sample via right-bisect
        # on the sorted CSFM delta vector
        d_flat = d.reshape(-1)
        idx = torch.searchsorted(self.deltas, d_flat,
                                 right=True).clamp(1, len(self.deltas) - 1)
        d_lo = self.deltas[idx - 1]
        d_hi = self.deltas[idx]
        l_lo = self.lams[idx - 1]
        l_hi = self.lams[idx]
        w = (d_flat - d_lo) / (d_hi - d_lo + 1e-12)
        l = l_lo + w * (l_hi - l_lo)
        return l.reshape(delta.shape)


def anchor_lambda_loss(lam_pred: Tensor, delta_pred: Tensor,
                       target: CSFMCurveTarget) -> Tensor:
    """Squared deviation of lam_pred from the CSFM target at the
    network's own delta. Mean over the sample axis."""
    lam_target = target.interp(delta_pred.detach())
    return ((lam_pred - lam_target) ** 2).mean()
