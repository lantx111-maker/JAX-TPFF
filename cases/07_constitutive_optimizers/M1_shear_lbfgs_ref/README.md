# M1_shear_lbfgs_ref

L-BFGS shear on the 80x80 production mesh, u to 0.020 mm.

- Category: Constitutive and optimizer comparisons
- Origin: `registered-case-catalogue`
- Mode/model: `shear` / `AT2`
- Mesh: `80 x 80 x 1`
- Tags: cpu, M1, lbfgs, reference
- Result status: `result-available`
- Result directory: `outputs/M1_shear_lbfgs_ref`

## Run

```bash
python -m jax_tpff.run_case M1_shear_lbfgs_ref --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
