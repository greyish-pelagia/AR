"""Równoległe sito Eratostenesa: komunikaty lub wspólna tablica MPI."""

import argparse
from math import isqrt

import numpy as np
from mpi4py import MPI


def base_primes(limit):
    """Sekwencyjne sito w domkniętym przedziale [2, limit]."""
    flags = np.ones(limit + 1, dtype=np.bool_)
    flags[:2] = False
    for d in range(2, isqrt(limit) + 1):
        if flags[d]:
            flags[d * d :: d] = False
    return np.flatnonzero(flags)


def split_interval(start, stop, rank, size):
    """Podział [start, stop) na size rozłącznych, prawie równych części."""
    length = stop - start
    return (start + length * rank // size, start + length * (rank + 1) // size)


def mark_segment(flags, lo, divisors):
    """flags[i] opisuje liczbę lo+i; modyfikujemy przekazany bufor."""
    hi = lo + flags.size
    for prime in divisors:
        d = int(prime)
        first = max(d * d, ((lo + d - 1) // d) * d)
        if first < hi:
            flags[first - lo :: d] = False


def sieve_message(n, comm: MPI.Intracomm):
    """Każdy proces replikuje B i zbiera pierwsze ze swojego fragmentu C."""
    rank, size = comm.Get_rank(), comm.Get_size()
    limit = isqrt(n)
    divisors = base_primes(limit)

    lo, hi = split_interval(limit + 1, n + 1, rank, size)

    flags = np.ones(hi - lo, dtype=np.bool_)
    mark_segment(flags, lo, divisors)
    local_primes = np.flatnonzero(flags) + lo

    parts = comm.gather(local_primes, root=0)
    if rank == 0:
        return np.concatenate([divisors] + parts)
    return None


def sieve_shared(n, comm: MPI.Intracomm, MPI: MPI):
    """Jedna współdzielona tablica flag, wyłącznie jedna domena pamięci."""
    rank, size = comm.Get_rank(), comm.Get_size()
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, key=rank)
    if node_comm.Get_size() != size:
        node_comm.Free()
        raise ValueError("Tryb shared wymaga jednej domeny pamięci; użyj --nodes=1.")

    limit = isqrt(n)
    divisors = base_primes(limit)
    lo, hi = split_interval(limit + 1, n + 1, rank, size)
    dtype = np.dtype(np.bool_)
    nbytes = (n + 1) * dtype.itemsize if rank == 0 else 0

    win = MPI.Win.Allocate_shared(nbytes, dtype.itemsize, comm=node_comm)
    buf_ptr, _ = win.Shared_query(0)
    shared_array = np.ndarray(buffer=buf_ptr, dtype=dtype, shape=(n + 1,))

    win.Fence()
    if rank == 0:
        shared_array[: limit + 1] = False
        shared_array[divisors] = True
    shared_array[lo:hi] = True
    win.Fence()

    local_view = shared_array[lo:hi]
    mark_segment(local_view, lo, divisors)
    win.Fence()

    primes = np.flatnonzero(shared_array) if rank == 0 else None
    win.Fence()

    del local_view, shared_array, buf_ptr
    win.Free()
    node_comm.Free()
    return primes


def reference_primes(n):
    """Niezależna, wolna weryfikacja przez dzielenie próbne."""
    return [
        x for x in range(2, n + 1) if all(x % d != 0 for d in range(2, isqrt(x) + 1))
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n", type=int, help="Górna granica domkniętego [2, n]; n >= 0.")
    parser.add_argument("--mode", choices=("message", "shared"), default="message")
    parser.add_argument(
        "--check", action="store_true", help="Sprawdź wynik, n <= 100000."
    )
    parser.add_argument(
        "--print-primes", action="store_true", help="Wypisz pełną listę."
    )
    args = parser.parse_args()
    if args.n < 0:
        parser.error("n musi być nieujemne")
    if args.check and args.n > 100000:
        parser.error("--check ograniczono do n <= 100000 (dzielenie próbne)")
    return args


def main():
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    try:
        comm.Barrier()
        start = MPI.Wtime()
        if args.mode == "message":
            primes = sieve_message(args.n, comm)
        else:
            primes = sieve_shared(args.n, comm, MPI)

        comm.Barrier()
        elapsed = comm.reduce(MPI.Wtime() - start, op=MPI.MAX, root=0)
        ok = True
        if rank == 0:
            print(
                f"mode={args.mode} n={args.n} processes={size} "
                f"count={primes.size} seconds={elapsed:.6f}",
                flush=True,
            )
            if args.print_primes:
                print(primes.tolist(), flush=True)
            if args.check:
                ok = primes.tolist() == reference_primes(args.n)
                print("check=OK" if ok else "check=FAIL", flush=True)

        ok = comm.bcast(ok, root=0)
        return 0 if ok else 1
    except Exception as exc:  # noqa: BLE001
        print(f"Błąd w procesie {rank}: {exc}", flush=True)

        comm.Abort(1)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
