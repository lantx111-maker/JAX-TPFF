# M1_tension_at1_cpu

CPU AT1 tension on the 40x40 Miehe square.

- Category: Constitutive and optimizer comparisons
- Origin: `registered-case-catalogue`
- Mode/model: `tension` / `AT1`
- Mesh: `40 x 40 x 1`
- Tags: cpu, M1, AT1
- Result status: `result-available`
- Result directory: `outputs/M1_tension_at1_cpu`

## Run

```bash
python -m jax_tpff.run_case M1_tension_at1_cpu --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
