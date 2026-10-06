# T2_amr_p0

Conforming TRI3 AMR quench with local ell/h target 4.

- Category: Adaptive and high-density meshes
- Origin: `registered-case-catalogue`
- Mode/model: `quench` / `AT2`
- Mesh: `25 x 10 x 1`
- Tags: T2, expansion, amr
- Result status: `result-available`
- Result directory: `outputs_expansion/T2_amr_p0`

## Run

```bash
python -m jax_tpff.run_case T2_amr_p0 --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
