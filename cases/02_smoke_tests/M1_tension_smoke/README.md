# M1_tension_smoke

CPU smoke: notched tension, AT2, h/ℓ≈0.63, 12 load steps.

- Category: Smoke and low-cost development cases
- Origin: `registered-case-catalogue`
- Mode/model: `tension` / `AT2`
- Mesh: `20 x 20 x 1`
- Tags: cpu, smoke, M1
- Result status: `result-available`
- Result directory: `outputs/M1_tension_smoke`

## Run

```bash
python -m jax_tpff.run_case M1_tension_smoke --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
