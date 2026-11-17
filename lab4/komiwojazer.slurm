#!/bin/bash -l
#SBATCH --job-name=komiwojazer-mpi
#SBATCH --partition=plgrid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=2G
#SBATCH --output=komiwojazer-%j.out
#SBATCH --error=komiwojazer-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Uruchom ten plik przez sbatch z katalogu lab4}"
module load "${MPI_PYTHON_MODULE:-scipy-bundle/2021.10-intel-2021b}"
export SLURM_OVERLAP=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

OUTPUT="wyniki/${SLURM_JOB_ID}"

for D in 1 2 3 4 5; do
    for P in 1 2 4 8; do
        if [ "$D" -ne 2 ] && [ "$P" -ne 1 ] && [ "$P" -ne 8 ]; then
            continue
        fi
        for R in 1 2 3; do
            mpiexec -n "$P" python -m mpi4py.futures komiwojazer_mpi.py 12 \
                --depth "$D" --seed 42 --trial "$R" --output "$OUTPUT" --save-search
        done
    done
done
