# T2_quench_lbfgs_kd10

L-BFGS quench, kappa(d) and Gc(T), matching staggered T2_quench_cpu_kappa.

- Category: Constitutive and optimizer comparisons
- Origin: `registered-case-catalogue`
- Mode/model: `quench` / `AT2`
- Mesh: `40 x 16 x 1`
- Tags: cpu, T2, lbfgs, reference
- Result status: `result-available`
- Result directory: `outputs/T2_quench_lbfgs_kd10`

## Run

```bash
python -m jax_tpff.run_case T2_quench_lbfgs_kd10 --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
