# T2_quench_cpu_kappa

CPU quench with Gc(T) and kappa(d) shielding.

- Category: Two-dimensional thermal shock
- Origin: `registered-case-catalogue`
- Mode/model: `quench` / `AT2`
- Mesh: `40 x 16 x 1`
- Tags: cpu, T2
- Result status: `result-available`
- Result directory: `outputs/T2_quench_cpu_kappa`

## Run

```bash
python -m jax_tpff.run_case T2_quench_cpu_kappa --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
