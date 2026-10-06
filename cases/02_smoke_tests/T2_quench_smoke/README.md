# T2_quench_smoke

CPU smoke quench: half-scale plate, ℓ=0.8 mm, 10 thermal steps.

- Category: Smoke and low-cost development cases
- Origin: `registered-case-catalogue`
- Mode/model: `quench` / `AT2`
- Mesh: `25 x 10 x 1`
- Tags: cpu, smoke, T2
- Result status: `result-available`
- Result directory: `outputs/T2_quench_smoke`

## Run

```bash
python -m jax_tpff.run_case T2_quench_smoke --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
