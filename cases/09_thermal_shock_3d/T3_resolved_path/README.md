# T3_resolved_path

Reduced HEX8 three-dimensional consistency path with ell/h=4.

- Category: Three-dimensional thermal shock
- Origin: `registered-case-catalogue`
- Mode/model: `quench3d` / `AT2`
- Mesh: `20 x 8 x 4`
- Tags: T3, expansion, 3d
- Result status: `result-available`
- Result directory: `outputs_expansion/T3_resolved_path`

## Run

```bash
python -m jax_tpff.run_case T3_resolved_path --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
