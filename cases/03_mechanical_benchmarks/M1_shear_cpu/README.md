# M1_shear_cpu

CPU standard shear. Jin simple-shear deck, coarsened ℓ.

- Category: Mechanical fracture benchmarks
- Origin: `registered-case-catalogue`
- Mode/model: `shear` / `AT2`
- Mesh: `40 x 40 x 1`
- Tags: cpu, M1
- Result status: `result-available`
- Result directory: `outputs/M1_shear_cpu`

## Run

```bash
python -m jax_tpff.run_case M1_shear_cpu --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
