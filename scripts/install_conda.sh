#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-ccd}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

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
  sentencepiece

# GradCache is required for replay-scale CAHO training.
python -m pip install "GradCache @ git+https://github.com/luyug/GradCache.git"

# Install the local package (editable) for dev/test workflows.
python -m pip install -e .

echo "Environment '${ENV_NAME}' is ready."
