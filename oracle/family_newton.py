"""The deterioration family, recomputed with the equilibrium-converged solver.

The benchmark's failure engages the tie: at its limit point the tie band runs
at 2.03 times yield strain on average and 6.82 at worst, while the concrete
reaches its softened compressive limit in only 1 per cent of elements. A
failure that yields the tie that heavily must depend on how much tie there is,
so capacity should fall roughly in proportion to section loss.

The secant-Picard family reported the opposite -- capacity flat to within
2.7 per cent across 0 to 30 per cent loss -- and that solver is now known to
converge to the fixed point of a clipped secant map rather than to
equilibrium, overstating the intact capacity by 66 per cent and the deflection
at peak by roughly a factor of two (no sharp ratio is quotable: the
reference is flat to 1 per cent over 3.0 to 4.75 mm).

Prediction, with its test: capacity at 30 per cent tie loss should fall by
roughly 25 to 35 per cent. Under 5 per cent would mean the insensitivity
survives a residual-converged solve and is a real property of the member,
which would make it a far stronger result than it was before.
"""
import sys; sys.path.insert(0,'.')
import json, time
import numpy as np
from arclength_oracle import build_mesh
from arclength_oracle_crisfield import newton_displacement_control
from oracle_rho_sweep import RHO_NOM, deepbeam_rho

LEVELS = [0.0, 0.05, 0.10, 0.20, 0.25, 0.30]
LAM_SERVICE_FRAC = 0.65      # service load as a fraction of intact capacity

out = {}
print('%8s%10s%12s%12s%10s'%('loss','lam_max','delta_pk','max resid','wall s'))
t00 = time.time()
for loss in LEVELS:
    prob = deepbeam_rho(RHO_NOM*(1.0-loss)); mesh = build_mesh(prob)
    t0 = time.time()
    pts = newton_displacement_control(prob, mesh, delta_max=7.0, n_steps=28,
                                      verbose=False)
    lam = np.array([p.lam for p in pts]); dd = np.array([p.delta for p in pts])
    res = np.array([p.resid for p in pts])
    i = int(np.argmax(lam))
    out['%.2f'%loss] = {'loss': loss, 'lam_max': float(lam[i]),
                        'delta_peak': float(dd[i]),
                        'delta': dd.tolist(), 'lam': lam.tolist(),
                        'resid': res.tolist()}
    print('%7.0f%%%10.4f%12.2f%12.2e%10.0f'%(loss*100, lam[i], dd[i],
                                             np.nanmax(res), time.time()-t0),
          flush=True)

cap = np.array([out['%.2f'%l]['lam_max'] for l in LEVELS])
dpk = np.array([out['%.2f'%l]['delta_peak'] for l in LEVELS])
print()
print('across 0 to %.0f %% tie section loss, equilibrium-converged:'%(LEVELS[-1]*100))
print('   capacity        %+.1f %%   (secant-Picard family said -1.6 %%)'%((cap[-1]/cap[0]-1)*100))
print('   deflection peak %+.1f %%   (diagnostic only: the peaks sit on '
      '1 %%-flat plateaus, so this ratio is not quotable; the secant '
      'family said +32.4 %%)'%((dpk[-1]/dpk[0]-1)*100))
print()
print('   capacity loss / tie loss = %.2f   (measured on deep beams: 0.66 to 0.94)'
      %(-(cap[-1]/cap[0]-1)/LEVELS[-1]))
json.dump({'levels': LEVELS, 'curves': out, 'wall_s': time.time()-t00},
          open('deepbeam_family_newton.json','w'))
print('\n-> deepbeam_family_newton.json')
