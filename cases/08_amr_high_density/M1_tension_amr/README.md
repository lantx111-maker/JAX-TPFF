# M1_tension_amr

AMR tension: coarse 32x32 TRI3, longest-edge bisection to ℓ/h≥4 in the crack band.

- Category: Adaptive and high-density meshes
- Origin: `registered-case-catalogue`
- Mode/model: `tension` / `AT2`
- Mesh: `32 x 32 x 1`
- Tags: cpu, M1, amr
- Result status: `result-available`
- Result directory: `outputs/M1_tension_amr`

## Run

```bash
python -m jax_tpff.run_case M1_tension_amr --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
