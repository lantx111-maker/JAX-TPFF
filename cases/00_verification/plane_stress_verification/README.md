# plane_stress_verification

Plane-stress constitutive and finite-element verification.

- Category: Verification and solver checks
- Origin: `independent-verification`
- Mode/model: `verification` / `analytical-reference`
- Mesh: `not applicable`
- Tags: verification, manuscript
- Result status: `manuscript-result`
- Result directory: `results/verification/plane_stress_verification`

## Run

```bash
python -m jax_tpff.verify_plane_stress
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
