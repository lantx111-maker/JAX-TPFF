# T3_shock_lbfgs_mid

Reduced 16×8×5 HEX8 thermal-shock calculation with damage-dependent conductivity.

- Category: Three-dimensional thermal shock
- Origin: `registered-case-catalogue`
- Mode/model: `quench3d` / `AT2`
- Mesh: `16 x 8 x 5`
- Tags: cpu, T3, lbfgs, reference
- Result status: `result-available`
- Result directory: `outputs/T3_shock_lbfgs_mid`

## Run

```bash
python -m jax_tpff.run_case T3_shock_lbfgs_mid --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
