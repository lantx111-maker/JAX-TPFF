# M4_lshape

Winkler L-panel, Jin 2024 PF-CZM: bottom fixed, uy at (470, 250).

- Category: Constitutive and optimizer comparisons
- Origin: `registered-case-catalogue`
- Mode/model: `lshape` / `PFCZM`
- Mesh: `200 x 200 x 1`
- Tags: cpu, M4, benchmark
- Result status: `configuration-only`
- Result directory: `Not included; configuration only.`

## Run

```bash
python -m jax_tpff.run_case M4_lshape --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
