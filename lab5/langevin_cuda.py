import argparse
import csv
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import cuda
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_normal_float64,
)


@cuda.jit
def rozklad(states, wynik, dt, kroki):
    i = cuda.grid(1)
    if i < wynik.size:
        x = 0.0
        for _ in range(kroki):
            x += math.sqrt(dt) * xoroshiro128p_normal_float64(states, i)
        wynik[i] = x


@cuda.jit
def wyjscie(states, wynik, dt):
    i = cuda.grid(1)
    if i < wynik.size:
        x = 0.0
        krok = 0
        while abs(x) <= 1.0:
            x += math.sqrt(dt) * xoroshiro128p_normal_float64(states, i)
            krok += 1
        wynik[i] = krok * dt


def parse_args():
    parser = argparse.ArgumentParser(description="Równanie Langevina na GPU.")
    parser.add_argument(
        "--n", type=int, nargs="+", default=[1000, 10000, 100000, 1000000, 10000000]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("wyniki"))
    args = parser.parse_args()
    if min(args.n) < 1 or args.repeats < 1:
        parser.error("Liczby trajektorii i powtórzeń muszą być dodatnie.")
    return args


def main():
    args = parse_args()
    dt = 0.001
    tau = 1.0
    kroki = round(tau / dt)
    threads = 256
    gpu = cuda.get_current_device().name
    print(f"GPU: {gpu}, dt={dt}, tau={tau}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)

    states = create_xoroshiro128p_states(1, seed=1)
    wynik = cuda.device_array(1, dtype=np.float64)
    rozklad[1, threads](states, wynik, dt, kroki)
    wyjscie[1, threads](states, wynik, dt)
    cuda.synchronize()
    del states, wynik

    log = args.output / "pomiary.csv"
    write_header = not log.exists() or log.stat().st_size == 0
    with log.open("a", newline="") as file:
        for problem in ("rozklad", "wyjscie"):
            for n in args.n:
                for trial in range(1, args.repeats + 1):
                    seed = trial
                    start = perf_counter()
                    states = create_xoroshiro128p_states(n, seed=seed)
                    wynik = cuda.device_array(n, dtype=np.float64)
                    cuda.synchronize()
                    kernel_start = perf_counter()
                    blocks = (n + threads - 1) // threads
                    if problem == "rozklad":
                        rozklad[blocks, threads](states, wynik, dt, kroki)
                    else:
                        wyjscie[blocks, threads](states, wynik, dt)
                    cuda.synchronize()
                    kernel_seconds = perf_counter() - kernel_start
                    samples = wynik.copy_to_host()
                    seconds = perf_counter() - start
                    samples_file = ""
                    if n == max(args.n) and trial == args.repeats:
                        samples_file = f"{problem}_n{n}_r{trial}.npz"
                        np.savez(args.output / samples_file, samples=samples)
                    row = {
                        "problem": problem,
                        "n": n,
                        "trial": trial,
                        "seed": seed,
                        "dt": dt,
                        "tau": tau if problem == "rozklad" else "",
                        "threads": threads,
                        "gpu": gpu,
                        "seconds": seconds,
                        "kernel_seconds": kernel_seconds,
                        "samples_file": samples_file,
                    }
                    writer = csv.DictWriter(file, fieldnames=row.keys())
                    if write_header:
                        writer.writeheader()
                        write_header = False
                    writer.writerow(row)
                    file.flush()
                    print(
                        f"{problem} n={n} trial={trial} seconds={seconds:.6f} "
                        f"kernel_seconds={kernel_seconds:.6f}",
                        flush=True,
                    )
                    del states, wynik, samples


if __name__ == "__main__":
    main()
