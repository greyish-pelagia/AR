#!/bin/bash -l
#SBATCH --job-name=sito-mpi
#SBATCH --partition=plgrid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --output=sito-%j.out
#SBATCH --error=sito-%j.err
# Grant podaj przy wysyłaniu: sbatch -A TWOJ_GRANT-cpu sito.slurm
# Wzorowane na załączonym Quickstarcie; nazwa modułu może wymagać aktualizacji.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Uruchom ten plik przez sbatch}"
module load "${MPI_PYTHON_MODULE:-scipy-bundle/2021.10-intel-2021b}"
export SLURM_OVERLAP=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

N="${1:-1000000}"
MODE="${2:-shared}"

mpiexec python quickstart.py
mpiexec python sito_mpi.py "$N" --mode "$MODE"
