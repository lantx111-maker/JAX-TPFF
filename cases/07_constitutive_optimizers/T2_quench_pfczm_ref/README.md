# T2_quench_pfczm_ref

PF-CZM quench on 80x32 (ell=0.2 mm) with kappa(d) and Gc(T).

- Category: Constitutive and optimizer comparisons
- Origin: `registered-case-catalogue`
- Mode/model: `quench` / `PFCZM`
- Mesh: `80 x 32 x 1`
- Tags: cpu, T2, lbfgs, PFCZM, reference
- Result status: `result-available`
- Result directory: `outputs/T2_quench_pfczm_ref`

## Run

```bash
python -m jax_tpff.run_case T2_quench_pfczm_ref --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
