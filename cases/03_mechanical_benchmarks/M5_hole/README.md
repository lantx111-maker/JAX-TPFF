# M5_hole

Ambati-type notched plate with an eccentric hole, AT2, ell/h=3.2.

- Category: Mechanical fracture benchmarks
- Origin: `registered-case-catalogue`
- Mode/model: `hole` / `AT2`
- Mesh: `160 x 160 x 1`
- Tags: cpu, M5, benchmark
- Result status: `result-available`
- Result directory: `outputs/M5_hole`

## Run

```bash
python -m jax_tpff.run_case M5_hole --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
