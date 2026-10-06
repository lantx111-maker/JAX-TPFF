# M1_shear_lbfgs_cmame

L-BFGS on the 256×256 CMAME shear residual.

- Category: Adaptive and high-density meshes
- Origin: `registered-case-catalogue`
- Mode/model: `shear` / `AT2`
- Mesh: `256 x 256 x 1`
- Tags: cpu, M1, lbfgs, cmame
- Result status: `result-available`
- Result directory: `outputs/M1_shear_lbfgs_cmame`

## Run

```bash
python -m jax_tpff.run_case M1_shear_lbfgs_cmame --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
