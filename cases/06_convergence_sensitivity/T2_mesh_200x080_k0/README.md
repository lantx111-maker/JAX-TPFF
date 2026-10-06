# T2_mesh_200x080_k0

Fixed-length mesh-density study: 200x80 QUAD4 cells, ell/h=8.00.

- Category: Convergence and sensitivity studies
- Origin: `registered-case-catalogue`
- Mode/model: `quench` / `AT2`
- Mesh: `200 x 80 x 1`
- Tags: cpu, T2, mesh-density, convergence
- Result status: `result-available`
- Result directory: `outputs_expansion/T2_mesh_200x080_k0`

## Run

```bash
python -m jax_tpff.run_case T2_mesh_200x080_k0 --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
