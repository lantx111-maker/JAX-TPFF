# M1_tension_at1_ref

AT1 tension on the 80x80 production mesh.

- Category: Constitutive and optimizer comparisons
- Origin: `registered-case-catalogue`
- Mode/model: `tension` / `AT1`
- Mesh: `80 x 80 x 1`
- Tags: cpu, M1, AT1, reference
- Result status: `result-available`
- Result directory: `outputs/M1_tension_at1_ref`

## Run

```bash
python -m jax_tpff.run_case M1_tension_at1_ref --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
