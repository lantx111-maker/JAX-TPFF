# M2_cyclic_lbfgsb

CPU L-BFGS-B cyclic shear with box-constraint irreversibility.

- Category: Constitutive and optimizer comparisons
- Origin: `registered-case-catalogue`
- Mode/model: `shear` / `AT2`
- Mesh: `40 x 40 x 1`
- Tags: cpu, M2, lbfgs
- Result status: `result-available`
- Result directory: `outputs/M2_cyclic_lbfgsb`

## Run

```bash
python -m jax_tpff.run_case M2_cyclic_lbfgsb --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
