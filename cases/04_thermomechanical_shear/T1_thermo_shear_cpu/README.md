# T1_thermo_shear_cpu

T1: top-edge shear plus a vertical temperature gradient with Gc(T).

- Category: Thermomechanical shear
- Origin: `registered-case-catalogue`
- Mode/model: `shear` / `AT2`
- Mesh: `40 x 40 x 1`
- Tags: cpu, T1
- Result status: `result-available`
- Result directory: `outputs/T1_thermo_shear_cpu`

## Run

```bash
python -m jax_tpff.run_case T1_thermo_shear_cpu --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
