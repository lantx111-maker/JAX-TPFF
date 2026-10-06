# T3_shock_lbfgs

CPU L-BFGS HEX8 thermal shock on a reduced 3D brick (not the 1e7-DOF production mesh).

- Category: Three-dimensional thermal shock
- Origin: `registered-case-catalogue`
- Mode/model: `quench3d` / `AT2`
- Mesh: `8 x 4 x 3`
- Tags: cpu, T3, lbfgs
- Result status: `result-available`
- Result directory: `outputs/T3_shock_lbfgs`

## Run

```bash
python -m jax_tpff.run_case T3_shock_lbfgs --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
