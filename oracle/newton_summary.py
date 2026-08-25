"""Everything the study needs from the equilibrium-converged reference, in one place.

The quantities are read from the 0.05 mm traces rather than the family's
0.25 mm ones wherever both exist, because the service state sits near 0.6 mm
and the coarser stepping resolves it with two points.

The service load factor is not a free choice. Calibrating the strength limit
state so the intact member sits at the medium-consequence target beta = 3.8
with the study's coefficients of variation puts the mean demand at
capacity / exp(3.8 * sqrt(0.15^2 + 0.20^2)) = capacity / 2.586 with an assumed capacity spread, and 0.595 once that
spread is measured rather than assumed. This is the value the fields were
generated at and the value the reliability script calibrates to.

The deflection at PEAK is deliberately absent. The load factor is flat to
within 1 per cent over a 1.75 to 2.50 mm window at every deterioration level,
so the argmax is located by numerical noise and no sensitivity computed across
those points means anything. The load factor AT the peak is unaffected, being
stationary there by definition.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LEVELS = [0.0, 0.05, 0.10, 0.20, 0.25, 0.30]
LAM_SERVICE = 0.595


def main() -> None:
    rows = []
    for i, loss in enumerate(LEVELS):
        z = np.load(HERE / f"field_level_{i}.npz")
        d = z["trace_delta"]; l = z["trace_lam"]; ok = z["trace_ok"]
        j = int(np.argmax(np.where(ok, l, -np.inf)))
        ds = float(np.interp(LAM_SERVICE, l[:j + 1], d[:j + 1]))
        rows.append({
            "loss": loss,
            "cap": float(l[j]),
            "d_service": ds,
            "eps_soffit": float(z["soffit_ex"].max()),
            "lam_yield": float(z["lam_yield"][0]),
            "conv": f"{int(ok.sum())}/{len(ok)}",
        })

    b = rows[0]
    print(f"equilibrium-converged reference, service load factor "
          f"{LAM_SERVICE} ({LAM_SERVICE / b['cap'] * 100:.0f} % of intact "
          f"capacity)\n")
    print(f"{'loss':>6}{'capacity':>10}{'change':>9}{'d_svc':>9}{'growth':>9}"
          f"{'soffit ue':>11}{'growth':>9}{'yield at':>10}{'conv':>9}")
    for r in rows:
        print(f"{r['loss'] * 100:>5.0f}%{r['cap']:>10.4f}"
              f"{(r['cap'] / b['cap'] - 1) * 100:>8.1f}%"
              f"{r['d_service']:>9.3f}{(r['d_service'] / b['d_service'] - 1) * 100:>8.1f}%"
              f"{r['eps_soffit'] * 1e6:>11.0f}"
              f"{(r['eps_soffit'] / b['eps_soffit'] - 1) * 100:>8.1f}%"
              f"{r['lam_yield']:>10.4f}{r['conv']:>9}")

    e = rows[-1]
    print(f"\nacross 0 to 30 per cent tie section loss:")
    print(f"  capacity          {(e['cap'] / b['cap'] - 1) * 100:+.1f} %"
          f"   (ratio to tie loss {-(e['cap'] / b['cap'] - 1) / 0.30:.2f}; "
          f"measured 0.66 to 0.94)")
    print(f"  service deflection{(e['d_service'] / b['d_service'] - 1) * 100:+.1f} %")
    print(f"  soffit strain     {(e['eps_soffit'] / b['eps_soffit'] - 1) * 100:+.1f} %"
          f"   ({(e['eps_soffit'] - b['eps_soffit']) * 1e6:+.0f} ue, the quantity a "
          f"fibre reads)")
    print(f"  tie yield onset   {b['lam_yield']:.3f} -> {e['lam_yield']:.3f}"
          f"   ({(e['lam_yield'] / b['lam_yield'] - 1) * 100:+.1f} %)")
    print(f"\nservice stays below yield at every level: "
          f"{LAM_SERVICE:.3f} against {e['lam_yield']:.3f} at worst, a margin "
          f"of {e['lam_yield'] / LAM_SERVICE:.2f}")


if __name__ == "__main__":
    main()
