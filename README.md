# JAX-TPFF

JAX-TPFF provides the numerical implementation and reproducibility cases for
the manuscript *Thermal resistance of phase-field cracks in ceramic thermal
shock*. It couples thermomechanical phase-field fracture with
damage-dependent heat conduction and includes the resistance calibration,
transport-resolution studies, verification problems, and ceramic-quench
calculations reported in the paper.

The repository contains 108 indexed cases: 18 manuscript thermal-shock cases,
3 independent verification problems, and additional mechanical,
thermomechanical, sensitivity, and solver examples. Every indexed case has a
human-readable `README.md` and a machine-readable `case.json`; global indices
are provided in `cases/CASE_INDEX.json` and `cases/CASE_INDEX.csv`.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment/requirements.txt
export JAX_FEM_PATH="$PWD/third_party/jax-fem"
export PYTHONPATH="$PWD:$JAX_FEM_PATH"
```

## Quick start

```bash
./reproduce.sh list
./reproduce.sh case M1_tension_smoke
./reproduce.sh verify
```

Run the manuscript thermal-shock matrix with

```bash
python -m jax_tpff.run_manuscript
```

Production and high-density cases can require substantial memory and several
hours of computation. Inspect the case metadata before using `--force`.

## Repository structure

- `jax_tpff/`: constitutive models, finite-element problems, solvers, and drivers
- `cases/00_verification/`: analytical and numerical verification problems
- `cases/01_manuscript_thermal_shock/`: calculations reported in the manuscript
- `cases/02_smoke_tests/`: low-cost installation checks
- `cases/03_mechanical_benchmarks/`–`cases/10_inverse_identification/`: supplementary examples
- `paper_artifacts/`: scripts for figures and tables
- `third_party/jax-fem/`: the bundled JAX-FEM dependency and its licence

Generated checkpoints, VTK files, and other large numerical outputs are
excluded from Git. The case definitions and commands needed to regenerate
them are retained.

## Licence and citation

JAX-TPFF is distributed under the GNU General Public License v3.0; see
`LICENSE`. The bundled JAX-FEM snapshot remains governed by its included
licence. Citation metadata are provided in `CITATION.cff`.
