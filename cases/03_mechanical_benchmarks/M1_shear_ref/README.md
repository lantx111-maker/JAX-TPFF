# M1_shear_ref

Production-scale uniform shear, same ℓ as tension_ref, for AMR comparison.

- Category: Mechanical fracture benchmarks
- Origin: `registered-case-catalogue`
- Mode/model: `shear` / `AT2`
- Mesh: `80 x 80 x 1`
- Tags: cpu, M1
- Result status: `result-available`
- Result directory: `outputs/M1_shear_ref`

## Run

```bash
python -m jax_tpff.run_case M1_shear_ref --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
