#!/bin/bash -l
#SBATCH --job-name=test-numba-cuda
#SBATCH --partition=plgrid-gpu-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mem=2G
#SBATCH --output=test-numba-cuda-%j.out
#SBATCH --error=test-numba-cuda-%j.err

# Uruchomienie: sbatch --account=NAZWA_KONTA_GPU test_numba_cuda.slurm
# Konkretna wersja Numby: NUMBA_MODULE=numba/WERSJA sbatch --account=... test_numba_cuda.slurm
set -euo pipefail

module load "${NUMBA_MODULE:-numba}"
module load "${CUDA_MODULE:-cuda/12.9.1}"
module list
export PYTHONUNBUFFERED=1

srun python -X faulthandler - <<'PY'
import os
import subprocess
import sys

import numpy as np
import numba
from numba import cuda
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_normal_float64,
)


@cuda.jit
def dodawanie(a, b, wynik):
    i = cuda.grid(1)
    if i < wynik.size:
        wynik[i] = a[i] + b[i]


@cuda.jit
def losowanie(states, wynik):
    i = cuda.grid(1)
    if i < wynik.size:
        wynik[i] = xoroshiro128p_normal_float64(states, i)


print("Python:", sys.version)
print("NumPy:", np.__version__)
print("Numba:", numba.__version__)
print("Backend CUDA:", cuda.__file__)
for name in (
    "SLURM_JOB_ID", "SLURMD_NODENAME", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS",
    "CUDA_VISIBLE_DEVICES", "NUMBA_CUDA_DRIVER", "NUMBA_CUDA_USE_NVIDIA_BINDING",
    "LD_LIBRARY_PATH", "LD_PRELOAD",
):
    print(f"{name}={os.environ.get(name, '(nie ustawiono)')}")
subprocess.run(["nvidia-smi", "-L"], check=False)
if not cuda.is_available():
    raise RuntimeError("CUDA niedostępna w przydzielonym zadaniu.")
print("Liczba GPU widocznych dla Numby:", len(cuda.gpus))
from numba.cuda.cudadrv.driver import driver
with driver.get_active_context() as context:
    print("Indeks GPU aktywnego kontekstu:", context.devnum)
print("GPU:", cuda.get_current_device().name)

n = 1000
threads = 256
blocks = (n + threads - 1) // threads
a = np.arange(n, dtype=np.float64)
b = np.ones(n, dtype=np.float64)
print("ETAP: transfer tablicy a na GPU", flush=True)
d_a = cuda.to_device(a)
print("ETAP: transfer tablicy b na GPU", flush=True)
d_b = cuda.to_device(b)
print("ETAP: alokacja tablicy wynikowej na GPU", flush=True)
d_wynik = cuda.device_array(n, dtype=np.float64)

print("ETAP: kompilacja kernela dodawania", flush=True)
dodawanie.specialize(d_a, d_b, d_wynik)
print("ETAP: uruchomienie kernela dodawania", flush=True)
dodawanie[blocks, threads](d_a, d_b, d_wynik)
print("ETAP: synchronizacja po dodawaniu", flush=True)
cuda.synchronize()
print("ETAP: transfer wyniku na CPU", flush=True)
np.testing.assert_array_equal(d_wynik.copy_to_host(), a + b)
print("OK: kompilacja kernela, dodawanie i transfery CPU/GPU.")

print("ETAP: inicjalizacja generatora GPU", flush=True)
states = create_xoroshiro128p_states(n, seed=1)
print("ETAP: uruchomienie kernela losowania", flush=True)
losowanie[blocks, threads](states, d_wynik)
cuda.synchronize()
wynik = d_wynik.copy_to_host()
if not np.isfinite(wynik).all() or np.var(wynik) == 0:
    raise RuntimeError("Generator GPU zwrócił nieprawidłowe wyniki.")
print("OK: generator normalny float64 na GPU.")
print("Średnia:", wynik.mean(), "wariancja:", wynik.var())
print("Wszystkie testy zakończone powodzeniem.")
PY
