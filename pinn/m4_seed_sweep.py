"""Reviewer item 2.2: both M4 arms over three seeds.

The physics-vs-no-physics comparison rested on one seed per arm, and the
run-to-run spread between the headline family and the M4 arms is of the
same size as the effect claimed. This runs each arm at three seed
offsets and reports the withheld-level errors per seed, so the claim can
be stated with a spread or retired.
"""
from __future__ import annotations
import importlib, json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OFFSETS = [3, 11, 23]          # SEED+3 is the arm's original offset

def run_arm(mod_name, offset):
    import torch
    import pinn_arclength as base
    # Set the seed BEFORE (re)loading the arm module: both arms bind
    # SEED by from-import at module top level, so the reload is what
    # propagates the changed value. The first sweep set it after the
    # reload, and the no-physics arm (fully deterministic) returned
    # identical numbers at every "seed", which is how the flaw showed.
    old = base.SEED
    base.SEED = old + (offset - 3)
    try:
        mod = importlib.import_module(mod_name)
        importlib.reload(mod)
        mod.main()
    finally:
        base.SEED = old

def withheld_errors(curves_path):
    fam = json.load(open(HERE.parent / 'oracle' / 'deepbeam_family_newton.json'))['curves']
    d = json.load(open(curves_path))
    net = d['curves'] if 'curves' in d else d
    out = {}
    for k in ('0.05', '0.25'):
        c = net[k]
        nl, nd = np.array(c['lam']), np.array(c['delta'])
        m = nd <= 7.0
        out[k] = (float(np.max(nl[m])) / fam[k]['lam_max'] - 1) * 100
    return out

def main():
    results = {}
    for arm, curves in [('m4_physics_newton', 'm4_physics_newton_curves.json'),
                        ('m4_nophysics_newton', 'm4_nophysics_newton_curves.json')]:
        rows = []
        for off in OFFSETS:
            t0 = time.time()
            run_arm(arm, off)
            errs = withheld_errors(HERE / 'runs' / 'parametric_rho' / curves)
            rows.append({'seed_offset': off, **errs,
                         'mean_abs': float(np.mean([abs(v) for v in errs.values()])),
                         'wall_s': time.time() - t0})
            print(f"[{arm} +{off}] 5%: {errs['0.05']:+.2f}  25%: {errs['0.25']:+.2f}  "
                  f"({rows[-1]['wall_s']:.0f}s)", flush=True)
            json.dump(results | {arm: rows},
                      open(HERE / 'runs' / 'm4_seed_sweep_v2.json', 'w'), indent=1)
        results[arm] = rows
    json.dump(results, open(HERE / 'runs' / 'm4_seed_sweep_v2.json', 'w'), indent=1)
    for arm, rows in results.items():
        ms = [r['mean_abs'] for r in rows]
        print(f"{arm}: mean|err| per seed {['%.2f' % m for m in ms]}  "
              f"-> {np.mean(ms):.2f} +/- {np.std(ms, ddof=1):.2f}")

if __name__ == '__main__':
    main()
