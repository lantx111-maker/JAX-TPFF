# M1_shear_ref_ext

Staggered 80x80 shear continued to u=0.020 mm for a developed process zone.

- Category: Mechanical fracture benchmarks
- Origin: `registered-case-catalogue`
- Mode/model: `shear` / `AT2`
- Mesh: `80 x 80 x 1`
- Tags: cpu, M1, reference
- Result status: `result-available`
- Result directory: `outputs/M1_shear_ref_ext`

## Run

```bash
python -m jax_tpff.run_case M1_shear_ref_ext --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
