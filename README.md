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

Figures are not stored; the scripts in `pinn/` regenerate them into a
`figures/` directory from the data and checkpoints here.

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

This repository contains the author's own code and generated data. The
experimental measurements used for comparison are available in the
published sources that report them:

- Wall piers: Bimschas, M. (2010). *Displacement-Based Seismic Assessment
  of Existing Bridges in Regions of Moderate Seismicity.* ETH Zurich.
- Deep beams: Li, S., Wu, Z., Zhang, J. and Xie, W. (2022). Experimental
  study and calculation methods of shear capacity for high-strength
  reinforced concrete full-scale deep beams. *Materials* 15, 6017.
  https://doi.org/10.3390/ma15176017

`plot_vk_combined.py` reads the wall-pier backbones as
`vk1_backbone.csv` and `vk3_backbone.csv`, which should be obtained from
the source above.

## Citation

Lamsal, S. An arc-length physics-informed neural network for post-peak
equilibrium paths, with application to concrete D-regions.

## Corrected references (August 2026)

Two defects in the original reference generation were found and fixed,
and every affected result was regenerated:

1. **Wall-pier section.** The Bimschas (2010) test units are
   1500 x 350 mm; an earlier 200 mm thickness gave 57% of the section.
   `arclength_oracle_vk1.py` and `pinn/wallpier_vk1.py` both carry
   350 mm, and the two definitions are asserted equal in training.
2. **Newton iteration cap.** The default cap of 120 truncated the
   cracked stages of several traces and displaced apparent peaks; the
   default is now 400 and every stored curve was regenerated under it.
   Stored curves carry per-state `converged` flags and residuals, and
   peaks are taken over converged states only.

`requirements.txt` pins the environment used for every stored result
(python 3.12, single CPU core, no GPU).
