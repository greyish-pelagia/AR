import argparse
import csv
from pathlib import Path

import numpy as np
from mpi4py import MPI


def jacobi(n, a, p, tension, eps, max_iter, comm):
    rank, size = comm.Get_rank(), comm.Get_size()
    rows = (n - 1) * (rank + 1) // size - (n - 1) * rank // size
    old = np.zeros((rows + 2, n + 1))
    new = np.zeros_like(old)
    above = rank - 1 if rank > 0 else MPI.PROC_NULL
    below = rank + 1 if rank < size - 1 else MPI.PROC_NULL
    q = p * (a / n) ** 2 / tension

    for iteration in range(1, max_iter + 1):
        comm.Sendrecv(old[1], dest=above, sendtag=0,
                      recvbuf=old[-1], source=below, recvtag=0)
        comm.Sendrecv(old[-2], dest=below, sendtag=1,
                      recvbuf=old[0], source=above, recvtag=1)
        new[1:-1, 1:-1] = (
            old[:-2, 1:-1] + old[2:, 1:-1]
            + old[1:-1, :-2] + old[1:-1, 2:] + q
        ) / 4
        local_delta = float(np.max(np.abs(new[1:-1] - old[1:-1])))
        delta = comm.allreduce(local_delta, op=MPI.MAX)
        old, new = new, old
        if delta < eps:
            break
    return old[1:-1].copy(), iteration, delta


def parse_args():
    parser = argparse.ArgumentParser(description="Membrana: metoda Jacobiego z MPI.")
    parser.add_argument("n", type=int, help="Liczba przedziałów na boku siatki.")
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--tension", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=1000000)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("wyniki"))
    parser.add_argument("--save-grid", action="store_true")
    args = parser.parse_args()
    if args.n < 2 or MPI.COMM_WORLD.Get_size() > args.n - 1:
        parser.error("Wymagane n >= 2 oraz liczba procesów <= n - 1.")
    if not all(np.isfinite(x) and x > 0 for x in
               (args.a, args.p, args.tension, args.eps)):
        parser.error("a, p, tension i eps muszą być dodatnie i skończone.")
    if args.max_iter < 1 or args.trial < 1:
        parser.error("max-iter i trial muszą być dodatnie.")
    return args


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    try:
        comm.Barrier()
        start = MPI.Wtime()
        local_h, iterations, delta = jacobi(
            args.n, args.a, args.p, args.tension, args.eps, args.max_iter, comm
        )
        seconds = comm.reduce(MPI.Wtime() - start, op=MPI.MAX, root=0)
        converged = delta < args.eps
        parts = comm.gather(local_h, root=0) if args.save_grid else None

        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=True)
            grid_file = ""
            if args.save_grid:
                h = np.zeros((args.n + 1, args.n + 1))
                h[1:-1] = np.concatenate(parts)
                grid_file = f"siatka_n{args.n}_P{size}_r{args.trial}.npz"
                np.savez(args.output / grid_file, h=h, a=args.a,
                         p=args.p, tension=args.tension, eps=args.eps)
            row = dict(n=args.n, processes=size, trial=args.trial,
                       a=args.a, p=args.p, tension=args.tension, eps=args.eps,
                       max_iter=args.max_iter, iterations=iterations,
                       delta=delta, converged=converged, seconds=seconds,
                       grid_file=grid_file)
            log = args.output / "pomiary.csv"
            write_header = not log.exists() or log.stat().st_size == 0
            with log.open("a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=row.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            print(f"n={args.n} processes={size} iterations={iterations} "
                  f"delta={delta:.3e} seconds={seconds:.6f} "
                  f"converged={converged}", flush=True)
        return 0 if converged else 1
    except Exception as exc:
        print(f"Błąd w procesie {rank}: {exc}", flush=True)
        comm.Abort(1)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
