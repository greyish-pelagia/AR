import argparse
import csv
from math import exp, isfinite
from pathlib import Path
from random import Random

import numpy as np
from mpi4py import MPI


def metropolis(n, beta, sweeps, seed, comm):
    rank, size = comm.Get_rank(), comm.Get_size()
    rows = n // size
    above, below = (rank - 1) % size, (rank + 1) % size
    sites = Random(seed)
    moves = Random(f"{seed}:{rank}")
    spins = [moves.choices((-1, 1), k=n) for _ in range(rows)]

    if size == 1:
        top, bottom = spins[-1], spins[0]
    else:
        top = comm.sendrecv(spins[-1], dest=below, source=above)
        bottom = comm.sendrecv(spins[0], dest=above, source=below)
    spins = [top, *spins, bottom]
    acceptance = {4: exp(-4 * beta), 8: exp(-8 * beta)}

    for _ in range(sweeps * rows * n):
        row, col = divmod(sites.randrange(rows * n), n)
        row += 1
        line = spins[row]
        spin = line[col]
        neighbours = (
            spins[row - 1][col] + spins[row + 1][col]
            + line[col - 1] + line[(col + 1) % n]
        )
        delta = 2 * spin * neighbours
        if delta <= 0 or moves.random() < acceptance[delta]:
            line[col] = -spin

        if size > 1:
            if row == 1:
                bottom[col] = comm.sendrecv(line[col], dest=above, source=below)
            elif row == rows:
                top[col] = comm.sendrecv(line[col], dest=below, source=above)

    return spins[1:-1]


def parse_args(size):
    parser = argparse.ArgumentParser(description="Model Isinga: algorytm Metropolisa z MPI.")
    parser.add_argument("n", type=int, help="Liczba spinów na boku siatki.")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--sweeps", type=int, default=10001)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("wyniki"))
    parser.add_argument("--save-grid", action="store_true")
    args = parser.parse_args()
    if args.n < 2 * size or args.n % size != 0:
        parser.error("n musi dzielić się przez liczbę procesów i dawać co najmniej dwa wiersze na proces.")
    if not isfinite(args.beta) or args.beta <= 0 or args.sweeps < 1 or args.trial < 1:
        parser.error("beta, sweeps i trial muszą być dodatnie, a beta skończona.")
    return args


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    args = parse_args(size)

    comm.Barrier()
    start = MPI.Wtime()
    local_spins = metropolis(args.n, args.beta, args.sweeps, args.trial, comm)
    seconds = comm.reduce(MPI.Wtime() - start, op=MPI.MAX, root=0)
    parts = comm.gather(local_spins, root=0) if args.save_grid else None
    if rank != 0:
        return

    args.output.mkdir(parents=True, exist_ok=True)
    grid_file = ""
    if args.save_grid:
        grid_file = f"siatka_n{args.n}_beta{args.beta}_P{size}_t{args.sweeps}_r{args.trial}.npz"
        np.savez(args.output / grid_file, spins=np.concatenate(parts).astype(np.int8))

    row = {
        "n": args.n,
        "beta": args.beta,
        "sweeps": args.sweeps,
        "processes": size,
        "trial": args.trial,
        "seconds": seconds,
        "grid_file": grid_file,
    }
    log = args.output / "pomiary.csv"
    write_header = not log.exists() or log.stat().st_size == 0
    with log.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(
        f"n={args.n} beta={args.beta} sweeps={args.sweeps} "
        f"processes={size} trial={args.trial} seconds={seconds:.6f}"
    )


if __name__ == "__main__":
    main()
