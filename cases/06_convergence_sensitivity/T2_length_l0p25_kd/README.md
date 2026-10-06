# T2_length_l0p25_kd

Length-scale study, ell=0.25 mm, ell/h=4, degraded conductivity.

- Category: Convergence and sensitivity studies
- Origin: `registered-case-catalogue`
- Mode/model: `quench` / `AT2`
- Mesh: `200 x 80 x 1`
- Tags: T2, expansion, length-scale
- Result status: `configuration-only`
- Result directory: `Not included; configuration only.`

## Run

```bash
python -m jax_tpff.run_case T2_length_l0p25_kd --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
