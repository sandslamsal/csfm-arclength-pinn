"""Reviewer item: one genuine extrapolation test.

Anchor at 0/10/20 per cent tie loss and hold out 30 per cent, which
lies OUTSIDE the anchored range; the family's existing held-out levels
(5, 25) only interpolate. The canonical checkpoint and curves file are
backed up and restored, since the trainer writes fixed paths.
"""
from __future__ import annotations
import json, shutil, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RHO = HERE / 'runs' / 'parametric_rho'
CANON = [RHO / 'parametric_anchored_newton.pt',
         RHO / 'parametric_curves_anchored_newton.json']

backups = []
for f in CANON:
    b = f.with_suffix(f.suffix + '.canonical_bak')
    shutil.copy2(f, b)
    backups.append((f, b))

try:
    import pinn_arclength_parametric_anchor_newton as par
    par.ANCHOR_LEVELS = [0.0, 0.10, 0.20]
    par.main()

    fam = json.load(open(HERE.parent / 'oracle' / 'deepbeam_family_newton.json'))['curves']
    d = json.load(open(RHO / 'parametric_curves_anchored_newton.json'))
    net = d['curves'] if 'curves' in d else d
    res = {}
    print(f"{'level':>6} {'net':>9} {'ref':>9} {'dev %':>8}")
    for k in ('0.00', '0.05', '0.10', '0.20', '0.25', '0.30'):
        c = net[k]
        nl, nd = np.array(c['lam']), np.array(c['delta'])
        m = nd <= 7.0
        dev = (float(np.max(nl[m])) / fam[k]['lam_max'] - 1) * 100
        res[k] = dev
        tag = ('  EXTRAPOLATED' if k == '0.30'
               else (' withheld' if k in ('0.05', '0.25') else ''))
        print(f"{float(k)*100:5.0f}% {float(np.max(nl[m])):9.4f} "
              f"{fam[k]['lam_max']:9.4f} {dev:+8.2f}{tag}", flush=True)
    shutil.copy2(RHO / 'parametric_curves_anchored_newton.json',
                 RHO / 'parametric_curves_extrap.json')
    shutil.copy2(RHO / 'parametric_anchored_newton.pt',
                 RHO / 'parametric_extrap.pt')
    json.dump(res, open(RHO / 'extrapolation_test.json', 'w'), indent=1)
finally:
    for f, b in backups:
        shutil.copy2(b, f)
        b.unlink()
    print("canonical checkpoint and curves restored")
