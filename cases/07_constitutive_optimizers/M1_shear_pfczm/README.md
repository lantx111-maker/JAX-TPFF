# M1_shear_pfczm

Isothermal PF-CZM shear on the Miehe square, monolithic L-BFGS, 80x80.

- Category: Constitutive and optimizer comparisons
- Origin: `registered-case-catalogue`
- Mode/model: `shear` / `PFCZM`
- Mesh: `80 x 80 x 1`
- Tags: cpu, M1, pfczm, benchmark
- Result status: `result-available`
- Result directory: `outputs/M1_shear_pfczm`

## Run

```bash
python -m jax_tpff.run_case M1_shear_pfczm --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
