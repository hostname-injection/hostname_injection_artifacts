#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-ccd}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
INSTALL_GRADCACHE="${INSTALL_GRADCACHE:-0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found. Install Miniconda/Anaconda first."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
conda activate "${ENV_NAME}"

conda install -y -c conda-forge -c pytorch \
  numpy \
  scipy \
  pytorch \
  sentence-transformers \
  idna \
  pytest \
  sentencepiece \
  scikit-learn

# Install the local package (editable) for dev/test workflows.
python -m pip install -e .

if [ "${INSTALL_GRADCACHE}" = "1" ]; then
  TMP_DIR="$(mktemp -d)"
  git clone https://github.com/luyug/GradCache "${TMP_DIR}/GradCache"
  python -m pip install "${TMP_DIR}/GradCache"
  rm -rf "${TMP_DIR}"
fi

echo "Environment '${ENV_NAME}' is ready."
