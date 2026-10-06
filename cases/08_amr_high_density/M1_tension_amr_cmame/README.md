# M1_tension_amr_cmame

AMR tension to ~50k TRI3, target h/ℓ=0.20 (ℓ/h=5) in the band.

- Category: Adaptive and high-density meshes
- Origin: `registered-case-catalogue`
- Mode/model: `tension` / `AT2`
- Mesh: `80 x 80 x 1`
- Tags: cpu, M1, amr, cmame
- Result status: `result-available`
- Result directory: `outputs/M1_tension_amr_cmame`

## Run

```bash
python -m jax_tpff.run_case M1_tension_amr_cmame --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
