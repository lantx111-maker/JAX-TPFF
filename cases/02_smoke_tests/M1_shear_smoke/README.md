# M1_shear_smoke

CPU smoke: notched shear, AT2.

- Category: Smoke and low-cost development cases
- Origin: `registered-case-catalogue`
- Mode/model: `shear` / `AT2`
- Mesh: `20 x 20 x 1`
- Tags: cpu, smoke, M1
- Result status: `result-available`
- Result directory: `outputs/M1_shear_smoke`

## Run

```bash
python -m jax_tpff.run_case M1_shear_smoke --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
