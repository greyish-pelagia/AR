#!/bin/bash -l
#SBATCH --job-name=langevin-cuda
#SBATCH --partition=plgrid-gpu-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --output=langevin-cuda-%j.out
#SBATCH --error=langevin-cuda-%j.err

# Uruchomienie z lab5: sbatch --account=NAZWA_KONTA_GPU langevin.slurm
# Wymaga uv w PATH oraz dostępu do pobrania Pythona i pakietów.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Uruchom przez sbatch z katalogu lab5}"

module purge
module load "${CUDA_MODULE:-cuda/12.9.1}"
module list
command -v uv

export UV_PROJECT_ENVIRONMENT="${SCRATCH:?Wymagana zmienna SCRATCH}/uv/lab5"
export UV_CACHE_DIR="$SCRATCH/uv/.cache"
export UV_PYTHON_INSTALL_DIR="$SCRATCH/uv/python"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
export PYTHONUNBUFFERED=1

# Python i pakiety w SCRATCH; biblioteki CUDA pochodzą wyłącznie z modułu.
# Bez dodatku numba-cuda[cu12], który instalowałby także CUDA z PyPI.
uv venv --allow-existing --managed-python --python 3.12 "$UV_PROJECT_ENVIRONMENT"
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
    'numpy==2.2.6' 'numba==0.62.0' 'numba-cuda==0.30.4' \
    'cuda-bindings>=12.9.1,<13'

srun "$UV_PROJECT_ENVIRONMENT/bin/python" langevin_cuda.py --output "wyniki/${SLURM_JOB_ID}"
