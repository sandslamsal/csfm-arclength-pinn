"""Assemble the Figure-3 data from the equilibrium-converged mesh runs.

mesh_objectivity_newton.py writes one file per mesh; plot_mesh_sensitivity.py
reads a single `mesh_sensitivity.json` with a `resolutions` list. This
adapter joins them, keeping ONLY converged states and only meshes whose
trace meets the residual criterion over at least half its steps. The
80x40 mesh converges at no state and is therefore excluded rather than
plotted as if it carried information.
"""
import glob, json
import numpy as np

MIN_CONVERGED_FRACTION = 0.5

res, dropped = [], []
for f in sorted(glob.glob('mesh_obj_*x*_L0.json')):
    d = json.load(open(f))
    r = list(d['runs'].values())[0]
    ok = np.array(r['converged'], dtype=bool)
    frac = ok.sum() / len(ok)
    nx, ny = d['mesh']
    if frac < MIN_CONVERGED_FRACTION:
        dropped.append((nx, ny, d['h_mm'], frac))
        continue
    dd = np.array(r['delta'])[ok]
    ll = np.array(r['lam'])[ok]
    i = int(np.argmax(ll))
    res.append({"nx": nx, "ny": ny, "h_mm": d['h_mm'],
                "delta": dd.tolist(), "lam": ll.tolist(),
                "peak_lam": float(ll[i]), "peak_delta": float(dd[i]),
                "n_converged": int(ok.sum()), "n_steps": int(len(ok)),
                "wall_s": r.get('wall_s', 0.0)})

res.sort(key=lambda e: -e['h_mm'])
json.dump({"resolutions": res}, open('mesh_sensitivity.json', 'w'))
pk = [e['peak_lam'] for e in res]
print(f"{len(res)} meshes kept, {len(dropped)} dropped")
for e in res:
    print(f"   {e['nx']}x{e['ny']} h={e['h_mm']:.0f}mm  peak {e['peak_lam']:.4f} "
          f"at {e['peak_delta']:.2f}mm  ({e['n_converged']}/{e['n_steps']})")
for nx, ny, h, fr in dropped:
    print(f"   dropped {nx}x{ny} h={h:.0f}mm: only {fr*100:.0f}% of states converged")
print(f"spread {(max(pk)-min(pk))/np.mean(pk)*100:.1f}% of the mean")
print("-> mesh_sensitivity.json")
