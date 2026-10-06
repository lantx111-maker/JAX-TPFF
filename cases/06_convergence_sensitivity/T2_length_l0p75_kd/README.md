# T2_length_l0p75_kd

Length-scale study, ell=0.75 mm, ell/h approximately 4, degraded conductivity.

- Category: Convergence and sensitivity studies
- Origin: `registered-case-catalogue`
- Mode/model: `quench` / `AT2`
- Mesh: `67 x 27 x 1`
- Tags: T2, expansion, length-scale
- Result status: `result-available`
- Result directory: `outputs_expansion/T2_length_l0p75_kd`

## Run

```bash
python -m jax_tpff.run_case T2_length_l0p75_kd --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
