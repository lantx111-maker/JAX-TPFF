# T2_quench_amr_cmame

AMR quench to ~50k TRI3, thermal-skin target h/ℓ=0.20.

- Category: Adaptive and high-density meshes
- Origin: `registered-case-catalogue`
- Mode/model: `quench` / `AT2`
- Mesh: `100 x 40 x 1`
- Tags: cpu, T2, amr, cmame
- Result status: `result-available`
- Result directory: `outputs/T2_quench_amr_cmame`

## Run

```bash
python -m jax_tpff.run_case T2_quench_amr_cmame --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
