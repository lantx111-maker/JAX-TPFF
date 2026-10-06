# T3_shock_cmame

Reduced HEX8 40×16×10 thermal-shock calculation for comparison with the large-scale reference.

- Category: Three-dimensional thermal shock
- Origin: `registered-case-catalogue`
- Mode/model: `quench3d` / `AT2`
- Mesh: `40 x 16 x 10`
- Tags: cpu, T3, lbfgs, cmame
- Result status: `result-available`
- Result directory: `outputs/T3_shock_cmame`

## Run

```bash
python -m jax_tpff.run_case T3_shock_cmame --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
