# I1_gcT_inverse_cpu

I1 on the 40x16 T2-CPU mesh: FD outer L-BFGS for (Gc0, b1, b2).

- Category: Inverse identification
- Origin: `registered-case-catalogue`
- Mode/model: `inverse` / `AT2`
- Mesh: `40 x 16 x 1`
- Tags: cpu, I1, reference
- Result status: `result-available`
- Result directory: `outputs/I1_gcT_inverse_cpu`

## Run

```bash
python -m jax_tpff.run_case I1_gcT_inverse_cpu --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
