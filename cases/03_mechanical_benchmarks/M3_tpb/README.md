# M3_tpb

Three-point bending, Jin 2024: 8x2 mm, triangular notch to y=0.4, AT2, plane strain, AMR band.

- Category: Mechanical fracture benchmarks
- Origin: `registered-case-catalogue`
- Mode/model: `tpb` / `AT2`
- Mesh: `64 x 16 x 1`
- Tags: cpu, M3, benchmark
- Result status: `result-available`
- Result directory: `outputs/M3_tpb`

## Run

```bash
python -m jax_tpff.run_case M3_tpb --force
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
