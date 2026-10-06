# I1_gcT_inverse

I1: FD outer L-BFGS identification of (Gc0, b1, b2) from a near-edge damage snapshot.

- Category: Inverse identification
- Origin: `registered-case-catalogue`
- Mode/model: `inverse` / `AT2`
- Mesh: `25 x 10 x 1`
- Tags: cpu, I1
- Result status: `result-available`
- Result directory: `outputs/I1_gcT_inverse`

## Run

```bash
python -m jax_tpff.run_case I1_gcT_inverse --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
