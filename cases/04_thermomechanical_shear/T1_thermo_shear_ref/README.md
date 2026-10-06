# T1_thermo_shear_ref

T1 production: 80x80 shear plus vertical temperature gradient.

- Category: Thermomechanical shear
- Origin: `registered-case-catalogue`
- Mode/model: `shear` / `AT2`
- Mesh: `80 x 80 x 1`
- Tags: cpu, T1, reference
- Result status: `result-available`
- Result directory: `outputs/T1_thermo_shear_ref`

## Run

```bash
python -m jax_tpff.run_case T1_thermo_shear_ref --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
