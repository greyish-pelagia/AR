import argparse
import csv
import math
from itertools import repeat
from pathlib import Path
from time import sleep

from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

x0, x1, w = -2.0, +2.0, 640 * 2
y0, y1, h = -1.5, +1.5, 480 * 2
dx = (x1 - x0) / w
dy = (y1 - y0) / h

c = complex(0, 0.65)


def julia(x, y):
    z = complex(x, y)
    n = 255
    while abs(z) < 3 and n > 1:
        z = z**2 + c
        n -= 1
    return n


def julia_line(k, delay=0.0):
    rank = MPI.COMM_WORLD.Get_rank()
    start = MPI.Wtime()
    line = bytearray(w)
    y = y1 - k * dy
    for j in range(w):
        x = x0 + j * dx
        line[j] = julia(x, y)
    compute_seconds = MPI.Wtime() - start
    if k == 1 and delay > 0:
        sleep(delay)
    seconds = MPI.Wtime() - start
    print(
        f"line={k} rank={rank} compute_seconds={compute_seconds:.6f} "
        f"seconds={seconds:.6f}",
        flush=True,
    )
    return line, k, rank, compute_seconds, seconds


def parse_args():
    parser = argparse.ArgumentParser(description="Zbiór Julii z MPIPoolExecutor.")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("wyniki/bazowy"))
    args = parser.parse_args()
    if not math.isfinite(args.sleep) or args.sleep < 0:
        parser.error("sleep musi być nieujemny i skończony.")
    if MPI.COMM_WORLD.Get_size() < 2:
        parser.error("Uruchom: mpiexec -n 8 python -m mpi4py.futures julia.py")
    return args


def main():
    start = MPI.Wtime()
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with MPIPoolExecutor() as executor:
        image = executor.map(julia_line, range(h), repeat(args.sleep), chunksize=1)
        with (args.output / "julia.pgm").open("wb") as file:
            file.write(b"P5 %d %d %d\n" % (w, h, 255))
            for line, k, rank, compute_seconds, seconds in image:
                file.write(line)
                rows.append((k, rank, compute_seconds, seconds))
    seconds = MPI.Wtime() - start

    with (args.output / "linie.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["line", "rank", "compute_seconds", "seconds"])
        writer.writerows(rows)

    with (args.output / "pomiary.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["processes", "workers", "width", "height", "sleep", "seconds"])
        size = MPI.COMM_WORLD.Get_size()
        writer.writerow([size, size - 1, w, h, args.sleep, seconds])
    print(
        f"workers={size - 1} sleep={args.sleep:.6f} seconds={seconds:.6f}", flush=True
    )


if __name__ == "__main__":
    main()
