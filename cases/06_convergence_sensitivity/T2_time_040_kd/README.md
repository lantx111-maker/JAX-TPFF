# T2_time_040_kd

40-step matched quench with damage-dependent conductivity.

- Category: Convergence and sensitivity studies
- Origin: `registered-case-catalogue`
- Mode/model: `quench` / `AT2`
- Mesh: `100 x 40 x 1`
- Tags: cpu, T2, verification, time-convergence, matched-pair
- Result status: `result-available`
- Result directory: `results/verification/T2_time_040_kd`

## Run

```bash
python -m jax_tpff.run_case T2_time_040_kd --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
