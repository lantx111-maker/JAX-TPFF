# T2_p0_at1_k0

P0 resolved matched quench: AT1 strength-threshold surrogate, constant conductivity; ell/h=4.

- Category: Constitutive and optimizer comparisons
- Origin: `registered-case-catalogue`
- Mode/model: `quench` / `AT1`
- Mesh: `100 x 40 x 1`
- Tags: cpu, T2, P0, resolved
- Result status: `result-available`
- Result directory: `outputs_p0/T2_p0_at1_k0`

## Run

```bash
python -m jax_tpff.run_case T2_p0_at1_k0 --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
