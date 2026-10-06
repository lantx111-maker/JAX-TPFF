# M1_shear_amr

AMR shear: damage-driven longest-edge bisection, curved path.

- Category: Adaptive and high-density meshes
- Origin: `registered-case-catalogue`
- Mode/model: `shear` / `AT2`
- Mesh: `32 x 32 x 1`
- Tags: cpu, M1, amr
- Result status: `result-available`
- Result directory: `outputs/M1_shear_amr`

## Run

```bash
python -m jax_tpff.run_case M1_shear_amr --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
