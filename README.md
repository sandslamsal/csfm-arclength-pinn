# csfm-arclength-pinn

Code and generated data for an arc-length physics-informed neural network
that traces the post-peak equilibrium path of concrete discontinuity
regions.

The load factor is an output of the network rather than an input, so the
path is free to fold and the limit point is an ordinary interior state.
Four training ingredients are needed for the cracked regime to converge:
supervised elastic pre-training, a pointwise arc-length loss, a fixed
weight on that loss held outside the adaptive balancer, and a
`C1`-regularised cracked-membrane constitutive.

## Layout

    oracle/    reference solver and the curves it generates
    pinn/      network, losses, training and figure scripts
    pinn/runs/ trained checkpoints for the reported results
    figures/   figures as published

## Reference solver

`arclength_oracle_crisfield.newton_displacement_control` advances a
displacement-controlled trace with a consistent finite-difference tangent
and Levenberg-Marquardt damping, converged on the true force residual
below `5e-4 * P_ref`. States that fail that criterion are reported as
unconverged and are excluded from every result. A secant-Picard iteration
stopped on the displacement increment converges instead to the fixed
point of a clipped secant map, which is not an equilibrium state; on the
deep-beam benchmark the two differ by 66 per cent in capacity.

Every stored curve carries the residual its solve converged to, so the
convergence state of each point can be checked independently.

    family_newton.py            deterioration family, 0 to 30 % tie loss
    corbel_curve_newton.py      corbel
    vk_newton.py vk1|vk3        wall piers
    vk_newton_full.py           wall pier with the displacement history
    mesh_objectivity_newton.py  mesh sweep

## Training

    pretrain_elastic.py                       elastic pre-training
    pinn_arclength.py                         single design
    pinn_arclength_parametric_anchor_newton.py deterioration family
    pinn_anchor_finetune.py                   anchor fine-tune
    seed_sweep_deepbeam.py                    seed reproducibility
    m4_physics_newton.py, m4_nophysics_newton.py   physics ablation
    m5_fold_sensitivity.py                    design derivative by autodiff

Run with Python 3.12 and PyTorch. Reference solves need only NumPy and
SciPy.

## Measured data

No measured data is distributed here. The wall-pier backbones used in the
experimental comparison are digitised from a figure in Bimschas (2010),
belong to their original author, and are neither included nor extractable
from this repository; `plot_vk_combined.py` expects them as
`vk1_backbone.csv` and `vk3_backbone.csv` and will not run without them.
The deep-beam capacities are from Li et al. (2022), *Materials* 15, 6017,
which is open access.

## Citation

Lamsal, S. An arc-length physics-informed neural network for post-peak
equilibrium paths, with application to concrete D-regions.
