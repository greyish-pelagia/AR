#!/bin/bash -l
#SBATCH --job-name=ising-mpi
#SBATCH --partition=plgrid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=2G
#SBATCH --output=ising-%j.out
#SBATCH --error=ising-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Uruchom ten plik przez sbatch z katalogu lab6}"
module load "${MPI_PYTHON_MODULE:-scipy-bundle/2021.10-intel-2021b}"
export SLURM_OVERLAP=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

OUTPUT="wyniki/${SLURM_JOB_ID}"

for BETA in 1 4; do
    mpiexec -n 4 python ising_mpi.py 32 --beta "$BETA" --sweeps 10001 \
        --save-grid --output "$OUTPUT"
done

for P in 1 2 4 8; do
    for R in 1 2 3; do
        mpiexec -n "$P" python ising_mpi.py 128 --beta 1 --sweeps 200 \
            --trial "$R" --output "$OUTPUT"
    done
done
