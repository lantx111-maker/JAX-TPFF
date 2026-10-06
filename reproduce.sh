#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
python_executable="${PYTHON_EXECUTABLE:-python3}"
export JAX_FEM_PATH="${JAX_FEM_PATH:-$root/third_party/jax-fem}"
export PYTHONPATH="$root:$JAX_FEM_PATH${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$root/.mplconfig}"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
cd "$root"

case "${1:-help}" in
  list)
    "$python_executable" -m jax_tpff.run_case --list
    ;;
  case)
    test "$#" -ge 2 || { echo "usage: $0 case CASE_NAME" >&2; exit 2; }
    "$python_executable" -m jax_tpff.run_case "$2" --force
    ;;
  verify)
    "$python_executable" -m jax_tpff.verify_heat
    "$python_executable" -m jax_tpff.verify_profile_transport
    "$python_executable" -m jax_tpff.verify_plane_stress
    ;;
  validate)
    "$python_executable" tools/validate_repository.py
    ;;
  *)
    echo "usage: $0 {list|case CASE_NAME|verify|validate}" >&2
    exit 2
    ;;
esac
