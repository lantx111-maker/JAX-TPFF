# T2_convective_Bi1p0

Convective quench with Bi=1.0 based on Ly/2.

- Category: Convergence and sensitivity studies
- Origin: `registered-case-catalogue`
- Mode/model: `quench` / `AT2`
- Mesh: `100 x 40 x 1`
- Tags: T2, expansion, convection
- Result status: `result-available`
- Result directory: `outputs_expansion/T2_convective_Bi1p0`

## Run

```bash
python -m jax_tpff.run_case T2_convective_Bi1p0 --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
