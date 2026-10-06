# T2_quench_pfczm_cmame

PF-CZM quench on the 312×125 CMAME mesh (ℓ/h=5).

- Category: Adaptive and high-density meshes
- Origin: `registered-case-catalogue`
- Mode/model: `quench` / `PFCZM`
- Mesh: `312 x 125 x 1`
- Tags: cpu, T2, lbfgs, PFCZM, cmame
- Result status: `result-available`
- Result directory: `outputs/T2_quench_pfczm_cmame`

## Run

```bash
python -m jax_tpff.run_case T2_quench_pfczm_cmame --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
