#!/bin/bash -l
#SBATCH --job-name=julia-mpi
#SBATCH --partition=plgrid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --output=julia-%j.out
#SBATCH --error=julia-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Uruchom ten plik przez sbatch z katalogu lab3}"
module load "${MPI_PYTHON_MODULE:-scipy-bundle/2021.10-intel-2021b}"
export SLURM_OVERLAP=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

OUTPUT="wyniki/${SLURM_JOB_ID}"

mpiexec -n "$SLURM_NTASKS" python -m mpi4py.futures julia.py --output "$OUTPUT/bazowy"

DELAY=$(python - "$OUTPUT/bazowy/linie.csv" <<'PY'
import csv
import sys

times = {}
with open(sys.argv[1], newline="") as file:
    for row in csv.DictReader(file):
        rank = int(row["rank"])
        times[rank] = times.get(rank, 0.0) + float(row["seconds"])
print(max(times.values()))
PY
)

mpiexec -n "$SLURM_NTASKS" python -m mpi4py.futures julia.py --sleep "$DELAY" --output "$OUTPUT/sleep"
