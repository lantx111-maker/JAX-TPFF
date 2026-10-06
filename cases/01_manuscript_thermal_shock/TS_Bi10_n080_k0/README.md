# TS_Bi10_n080_k0

Matched thermal-shock calculation used in the manuscript.

- Category: Manuscript thermal-shock cases
- Origin: `manuscript-study`
- Mode/model: `quench` / `AT2`
- Mesh: `100 x 40 x 1`
- Tags: manuscript, T2
- Result status: `manuscript-result`
- Result directory: `results/manuscript/TS_Bi10_n080_k0`

## Run

```bash
python -m jax_tpff.run_manuscript TS_Bi10_n080_k0 --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
