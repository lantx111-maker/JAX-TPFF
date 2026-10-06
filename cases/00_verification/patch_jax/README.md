# patch_jax

Homogeneous AT2 plane-stress patch with an analytical damage and reaction solution; solver=jax-krylov.

- Category: Verification and solver checks
- Origin: `registered-case-catalogue`
- Mode/model: `patch` / `AT2`
- Mesh: `32 x 32 x 1`
- Tags: cpu, verification, solver-benchmark
- Result status: `result-available`
- Result directory: `results/verification/patch_jax`

## Run

```bash
python -m jax_tpff.run_case patch_jax --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
