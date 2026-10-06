# T2_time_040

Fixed-mesh backward-Euler temporal refinement with 40 steps to 0.1 s.

- Category: Convergence and sensitivity studies
- Origin: `registered-case-catalogue`
- Mode/model: `quench` / `AT2`
- Mesh: `100 x 40 x 1`
- Tags: cpu, T2, verification, time-convergence
- Result status: `result-available`
- Result directory: `results/verification/T2_time_040`

## Run

```bash
python -m jax_tpff.run_case T2_time_040 --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
