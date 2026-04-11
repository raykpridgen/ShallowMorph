#!/usr/bin/env bash
# Optional HPC module loads (Lmod / Environment Modules).
#
# Usage in job script:
#   export MORPH_HPC_MODULES="cuda/12.1 cudnn/9.0 gcc/11"
#   source /path/to/morph/scripts/hpc/load_modules.sh
#
# Tries common ``module``/Lmod init paths if ``module`` is not already a shell function.
# If no ``module`` command exists after init, warns and skips (so local runs still work).
# If a listed module fails to load, exits with status 1.

set -u

_morph_init_module_cmd() {
  if type module >/dev/null 2>&1; then
    return 0
  fi
  local f
  for f in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash \
           /usr/share/modules/init/bash; do
    if [[ -f "${f}" ]]; then
      # shellcheck disable=SC1090
      source "${f}"
      return 0
    fi
  done
  return 1
}

load_morph_hpc_modules_or_exit() {
  if [[ -z "${MORPH_HPC_MODULES:-}" ]]; then
    return 0
  fi

  if ! _morph_init_module_cmd; then
    echo "[morph HPC] WARN: could not init 'module'; skipping MORPH_HPC_MODULES" >&2
    return 0
  fi

  if ! type module >/dev/null 2>&1; then
    echo "[morph HPC] WARN: 'module' unavailable; skipping MORPH_HPC_MODULES" >&2
    return 0
  fi

  local mod
  for mod in ${MORPH_HPC_MODULES}; do
    # shellcheck disable=SC2086
    if ! module load "${mod}"; then
      echo "[morph HPC] ERROR: module load failed: ${mod}" >&2
      return 1
    fi
    echo "[morph HPC] OK: module load ${mod}"
  done
  return 0
}

load_morph_hpc_modules_or_exit
