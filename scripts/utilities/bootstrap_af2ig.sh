#!/bin/bash
# Build the AF2 initial-guess scoring environment for the binder benchmark.
#
#   bash scripts/utilities/bootstrap_af2ig.sh              # venv + params, GPU jax
#   AF2IG_DEVICE=cpu bash scripts/utilities/bootstrap_af2ig.sh
#   AF2IG_SKIP_PARAMS=1 bash scripts/utilities/bootstrap_af2ig.sh
#
# Why this is a *separate* environment and not an extension of the training one:
# AF2 is JAX, training is torch+Protenix, and PXDesign's own scoring stack pins
# Protenix v0.5.0+pxd while this repo trains against v2.0.0 (modules moved:
# protenix.data.ccd -> protenix.data.core.ccd, protenix.data.parser ->
# protenix.data.core.parser). Importing both in one interpreter fails on that
# immediately. Scoring only ever consumes design PDBs, so it needs nothing from
# the training environment and the split costs nothing.
#
# What gets installed, and why these pieces:
#
#   ColabDesign  -- provides both halves of the filter in one package: AF2 with
#                   `initial_guess=True` (the AF2-IG protocol) and ProteinMPNN
#                   with weights bundled in the wheel. That second point is why
#                   the PMPNN arm of Table 4 needs no separate download; the
#                   original v_48_020 weights ship inside colabdesign/mpnn.
#   AlphaFold parameters -- NOT in any package. 4.7 GB tar from DeepMind, of
#                   which this harness reads exactly two files: model_1_ptm
#                   (the complex pass, which needs a template stack) and
#                   model_3_ptm (the unbound pass, which does not). Only those
#                   two are extracted unless AF2IG_ALL_PARAMS=1.
#
# Everything except ColabDesign goes in ONE pip invocation, and ColabDesign
# itself installs with --no-deps. That is not tidiness. Installing jax first and
# the rest afterwards does not work: chex and optax both depend on jax with no
# upper bound, so a second pip run happily upgrades the pinned jax underneath
# you -- measured, jax 0.4.35 became 0.6.2 while the CUDA plugin stayed at
# 0.4.35 -- and the first symptom is `AttributeError: jax.core.JaxprEqn was
# removed in JAX v0.6.0` from inside haiku, several minutes and 5 GB later.
# One resolver pass with every pin visible is what prevents that, and the
# version assertion at the end is what catches it if it happens anyway.

set -euo pipefail

AF2IG_ENV="${AF2IG_ENV:-${HOME}/.venvs/af2ig}"
AF2_PARAMS_DIR="${AF2_PARAMS_DIR:-${HOME}/af2_params}"
AF2IG_DEVICE="${AF2IG_DEVICE:-gpu}"
AF2IG_SKIP_PARAMS="${AF2IG_SKIP_PARAMS:-0}"
AF2IG_ALL_PARAMS="${AF2IG_ALL_PARAMS:-0}"
PARAMS_URL="${PARAMS_URL:-https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Pinned set: dm-haiku 0.0.13 tracks the jax 0.4.x API ColabDesign's AlphaFold
# port was written against, and chex/optax are pinned to their contemporaries so
# the resolver has no reason to move jax. `haiku.experimental.jaxpr_info`
# imports `jax.core.JaxprEqn`, which JAX removed in 0.6, so this is a hard
# ceiling and not a preference.
JAX_VERSION="${JAX_VERSION:-0.4.35}"
HAIKU_VERSION="${HAIKU_VERSION:-0.0.13}"
CHEX_VERSION="${CHEX_VERSION:-0.1.87}"
OPTAX_VERSION="${OPTAX_VERSION:-0.2.4}"

echo "env      : ${AF2IG_ENV}"
echo "params   : ${AF2_PARAMS_DIR}"
echo "device   : ${AF2IG_DEVICE}"

# ------------------------------------------------------------------- the venv

if [[ ! -x "${AF2IG_ENV}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${AF2IG_ENV}"
fi
PY="${AF2IG_ENV}/bin/python"
"${PY}" -m pip install --upgrade pip wheel >/dev/null

if [[ "${AF2IG_DEVICE}" == "gpu" ]]; then
  # nvidia-cuda-nvcc-cu12 12.9 ships `nvidia/cuda_nvcc/` WITHOUT an
  # `__init__.py`, which makes it a namespace package whose `__file__` is None.
  # jax 0.4.35 does `pathlib.Path(cuda_nvcc.__file__).parent` unguarded, so
  # `import jax` dies with `TypeError: expected str, bytes or os.PathLike
  # object, not NoneType` -- before any of this harness runs, and with a
  # traceback that says nothing about CUDA. 12.8 and earlier still ship the
  # file. (Fixed upstream in later jax; pinning the wheel keeps jax where this
  # harness was verified.)
  JAX_SPEC=("jax[cuda12]==${JAX_VERSION}" "nvidia-cuda-nvcc-cu12<12.9")
else
  JAX_SPEC=("jax==${JAX_VERSION}")
fi

# jaxlib is deliberately NOT pinned alongside jax: the cuda12 extra of jax
# 0.4.35 requires jaxlib==0.4.34 exactly, so pinning jaxlib to the jax version
# makes the resolve impossible. jax's own bound is the right pin. biotite is
# here because pxdesign_train/benchmarks/af2ig.py reads the design PDBs with it
# to work out which chain is the binder.
"${PY}" -m pip install \
  "${JAX_SPEC[@]}" \
  "dm-haiku==${HAIKU_VERSION}" \
  "chex==${CHEX_VERSION}" \
  "optax==${OPTAX_VERSION}" \
  dm-tree ml-collections immutabledict \
  absl-py biopython numpy scipy pandas joblib py3Dmol matplotlib biotite

# Pinned to the commit this harness was read against, not to a tag: the
# `plddt` log value is flipped back from a loss in af/design.py and `i_pae` is
# stored divided by 31, and scripts/evaluation/fold_af2ig.py undoes exactly
# those two conventions. A silent upstream change to either would move every
# number in Table 4 without erroring.
COLABDESIGN_REF="${COLABDESIGN_REF:-e31a56fe1d9b4de25c8697f3a28b75892941cc72}"
"${PY}" -m pip install --no-deps \
  "git+https://github.com/sokrypton/ColabDesign.git@${COLABDESIGN_REF}"

# -------------------------------------------------------- AlphaFold parameters

if [[ "${AF2IG_SKIP_PARAMS}" != "1" ]]; then
  mkdir -p "${AF2_PARAMS_DIR}/params"
  WANTED=(params_model_1_ptm.npz params_model_3_ptm.npz)

  have_all=1
  for f in "${WANTED[@]}"; do
    [[ -f "${AF2_PARAMS_DIR}/params/${f}" ]] || have_all=0
  done

  if [[ "${have_all}" == "1" && "${AF2IG_ALL_PARAMS}" != "1" ]]; then
    echo "AlphaFold parameters already present, skipping download"
  else
    TAR="${AF2_PARAMS_DIR}/alphafold_params_2022-12-06.tar"
    if [[ ! -f "${TAR}" ]]; then
      echo "downloading AlphaFold parameters (~4.7 GB) ..."
      # -C picks up an interrupted download rather than restarting it.
      curl -fL -C - -o "${TAR}" "${PARAMS_URL}"
    fi
    if [[ "${AF2IG_ALL_PARAMS}" == "1" ]]; then
      tar -xf "${TAR}" -C "${AF2_PARAMS_DIR}/params"
    else
      tar -xf "${TAR}" -C "${AF2_PARAMS_DIR}/params" "${WANTED[@]}"
    fi
    # The tar is kept only if asked for: it is 4.7 GB of nothing once extracted.
    if [[ "${AF2IG_KEEP_TAR:-0}" != "1" ]]; then rm -f "${TAR}"; fi
  fi

  for f in "${WANTED[@]}"; do
    if [[ ! -f "${AF2_PARAMS_DIR}/params/${f}" ]]; then
      echo "ERROR: ${AF2_PARAMS_DIR}/params/${f} is missing after extraction" >&2
      exit 2
    fi
  done
fi

# ------------------------------------------------------------------- verify

JAX_VERSION="${JAX_VERSION}" "${PY}" - <<'EOF'
import os, pathlib
import jax

want = os.environ["JAX_VERSION"]
if jax.__version__ != want:
    raise SystemExit(
        f"jax is {jax.__version__}, expected {want}. Something in the dependency "
        "set pulled it forward; haiku will fail on jax >= 0.6. Re-pin and rebuild."
    )

from colabdesign.af import mk_af_model          # noqa: F401
from colabdesign.mpnn import mk_mpnn_model      # noqa: F401
import colabdesign

weights = pathlib.Path(colabdesign.__file__).parent / "mpnn" / "weights"
print("jax          :", jax.__version__)
print("jax devices  :", jax.devices())
print("colabdesign  :", pathlib.Path(colabdesign.__file__).parent)
print("mpnn weights :", sorted(p.name for p in weights.glob("*.pkl")))
EOF

cat <<EOF

ready. To score a generation run:

  export AF2_PARAMS_DIR=${AF2_PARAMS_DIR}
  ${PY} scripts/evaluation/fold_af2ig.py \\
      --run-dir <run> --variants co_design pmpnn
  ${PY} scripts/evaluation/score_af2ig_designability.py \\
      --metrics-csv <run>/af2ig_metrics.csv
EOF
