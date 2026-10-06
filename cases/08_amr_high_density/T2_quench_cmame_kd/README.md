# T2_quench_cmame_kd

CMAME-density quench κ(d), Gc(T): same 312×125 residual as k0.

- Category: Adaptive and high-density meshes
- Origin: `registered-case-catalogue`
- Mode/model: `quench` / `AT2`
- Mesh: `312 x 125 x 1`
- Tags: cpu, T2, cmame
- Result status: `result-available`
- Result directory: `outputs/T2_quench_cmame_kd`

## Run

```bash
python -m jax_tpff.run_case T2_quench_cmame_kd --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
