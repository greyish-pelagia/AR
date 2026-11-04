#!/bin/bash -l
#SBATCH --job-name=membrana-mpi
#SBATCH --partition=plgrid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=2G
#SBATCH --output=membrana-%j.out
#SBATCH --error=membrana-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Uruchom ten plik przez sbatch z katalogu lab2}"
module load "${MPI_PYTHON_MODULE:-scipy-bundle/2021.10-intel-2021b}"
export SLURM_OVERLAP=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

OUTPUT="wyniki/${SLURM_JOB_ID}"

for P in 1 4; do
    mpiexec -n "$P" python membrana_mpi.py 32 --save-grid --output "$OUTPUT"
done

for N in 128 256; do
    for P in 1 2 4 8; do
        for R in 1 2 3; do
            mpiexec -n "$P" python membrana_mpi.py "$N" --trial "$R" --output "$OUTPUT"
        done
    done
done
