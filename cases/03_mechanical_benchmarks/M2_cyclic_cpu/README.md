# M2_cyclic_cpu

Cyclic shear: load–unload–reload for history-field irreversibility.

- Category: Mechanical fracture benchmarks
- Origin: `registered-case-catalogue`
- Mode/model: `shear` / `AT2`
- Mesh: `40 x 40 x 1`
- Tags: cpu, M2
- Result status: `result-available`
- Result directory: `outputs/M2_cyclic_cpu`

## Run

```bash
python -m jax_tpff.run_case M2_cyclic_cpu --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
