# M1_tension_ref

Production tension: ℓ=0.015, h=ℓ/1.2. GPU recommended.

- Category: Mechanical fracture benchmarks
- Origin: `registered-case-catalogue`
- Mode/model: `tension` / `AT2`
- Mesh: `80 x 80 x 1`
- Tags: gpu, M1, production
- Result status: `result-available`
- Result directory: `outputs/M1_tension_ref`

## Run

```bash
python -m jax_tpff.run_case M1_tension_ref --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
