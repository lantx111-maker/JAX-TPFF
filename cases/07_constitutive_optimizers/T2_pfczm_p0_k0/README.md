# T2_pfczm_p0_k0

PF-CZM strength-controlled quench, constant conductivity.

- Category: Constitutive and optimizer comparisons
- Origin: `registered-case-catalogue`
- Mode/model: `quench` / `PFCZM`
- Mesh: `100 x 40 x 1`
- Tags: T2, expansion, PFCZM
- Result status: `configuration-only`
- Result directory: `Not included; configuration only.`

## Run

```bash
python -m jax_tpff.run_case T2_pfczm_p0_k0 --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
