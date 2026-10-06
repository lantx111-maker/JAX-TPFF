# M1_tension_cmame

CMAME-density AT2 tension: 256×256 QUAD4, ℓ/h=3.84, 65536 cells.

- Category: Adaptive and high-density meshes
- Origin: `registered-case-catalogue`
- Mode/model: `tension` / `AT2`
- Mesh: `256 x 256 x 1`
- Tags: cpu, M1, cmame
- Result status: `result-available`
- Result directory: `outputs/M1_tension_cmame`

## Run

```bash
python -m jax_tpff.run_case M1_tension_cmame --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
