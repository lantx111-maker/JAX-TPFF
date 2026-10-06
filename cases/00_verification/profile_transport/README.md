# profile_transport

Stationary AT2 profile and diffuse-layer heat-flux convergence.

- Category: Verification and solver checks
- Origin: `independent-verification`
- Mode/model: `verification` / `analytical-reference`
- Mesh: `not applicable`
- Tags: verification, manuscript
- Result status: `manuscript-result`
- Result directory: `results/verification/profile_transport`

## Run

```bash
python -m jax_tpff.verify_profile_transport
```

Generated checkpoints and VTK files are excluded from Git. Use the command above
to reproduce a case. A `configuration-only` entry provides a reproducible case
definition without a bundled numerical result.
