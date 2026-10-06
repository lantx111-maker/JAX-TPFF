# T2_quench_ref

Production half-plate quench matching Jin case 6 (GPU/MPI).

- Category: Two-dimensional thermal shock
- Origin: `registered-case-catalogue`
- Mode/model: `quench` / `AT2`
- Mesh: `125 x 50 x 1`
- Tags: gpu, T2, production
- Result status: `result-available`
- Result directory: `outputs/T2_quench_ref`

## Run

```bash
python -m jax_tpff.run_case T2_quench_ref --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
