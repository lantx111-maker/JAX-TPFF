# patch_lbfgs

Homogeneous AT2 plane-stress patch with an analytical damage and reaction solution; solver=lbfgs.

- Category: Verification and solver checks
- Origin: `registered-case-catalogue`
- Mode/model: `patch` / `AT2`
- Mesh: `32 x 32 x 1`
- Tags: cpu, verification, solver-benchmark
- Result status: `result-available`
- Result directory: `results/verification/patch_lbfgs`

## Run

```bash
python -m jax_tpff.run_case patch_lbfgs --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
