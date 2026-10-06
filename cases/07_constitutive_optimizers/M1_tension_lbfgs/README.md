# M1_tension_lbfgs

CPU L-BFGS notched tension on the 40x40 Miehe square.

- Category: Constitutive and optimizer comparisons
- Origin: `registered-case-catalogue`
- Mode/model: `tension` / `AT2`
- Mesh: `40 x 40 x 1`
- Tags: cpu, M1, lbfgs
- Result status: `result-available`
- Result directory: `outputs/M1_tension_lbfgs`

## Run

```bash
python -m jax_tpff.run_case M1_tension_lbfgs --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
