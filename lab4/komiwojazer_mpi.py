import argparse
import csv
from itertools import permutations, repeat
from pathlib import Path

import numpy as np
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor


def search(prefix, distances, minimum, two_minimum):
    start = MPI.Wtime()
    n = len(distances)
    remaining = [city for city in range(1, n) if city not in prefix]
    cost = sum(distances[a, b] for a, b in zip(prefix, prefix[1:]))
    route = list(prefix)
    left = remaining.copy()
    best = cost
    while left:
        city = min(left, key=lambda city: distances[route[-1], city])
        best += distances[route[-1], city]
        route.append(city)
        left.remove(city)
    best += distances[route[-1], 0]
    best_route = route + [0]
    visited = 0
    pruned = 0

    def visit(path, left, length, bound_sum):
        nonlocal best, best_route, visited, pruned
        visited += 1
        if not left:
            total = length + distances[path[-1], 0]
            if total < best:
                best = total
                best_route = path + [0]
            return
        bound = length + (bound_sum + minimum[path[-1]] + minimum[0]) / 2
        if bound >= best:
            pruned += 1
            return
        for city in sorted(left, key=lambda city: distances[path[-1], city]):
            visit(
                path + [city],
                [other for other in left if other != city],
                length + distances[path[-1], city],
                bound_sum - two_minimum[city],
            )

    visit(list(prefix), remaining, cost, sum(two_minimum[remaining]))
    return {
        "prefix": "-".join(str(city + 1) for city in prefix),
        "rank": MPI.COMM_WORLD.Get_rank(),
        "seconds": MPI.Wtime() - start,
        "visited": visited,
        "pruned": pruned,
        "length": float(best),
        "route": best_route,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Komiwojażer: branch and bound z MPI.")
    parser.add_argument("n", type=int)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("wyniki"))
    parser.add_argument("--save-search", action="store_true")
    args = parser.parse_args()
    if args.n < 3 or not 1 <= args.depth <= args.n:
        parser.error("Wymagane n >= 3 i 1 <= depth <= n.")
    if args.seed < 0 or args.trial < 1:
        parser.error("seed musi być nieujemne, a trial dodatnie.")
    return args


def main():
    start = MPI.Wtime()
    args = parse_args()
    size = MPI.COMM_WORLD.Get_size()
    try:
        angles = np.random.default_rng(args.seed).uniform(0, 2 * np.pi, args.n)
        points = np.column_stack((np.cos(angles), np.sin(angles)))
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        edges = distances.copy()
        np.fill_diagonal(edges, np.inf)
        sorted_edges = np.sort(edges, axis=1)
        minimum = sorted_edges[:, 0]
        two_minimum = sorted_edges[:, :2].sum(axis=1)
        prefixes = [
            (0,) + tail for tail in permutations(range(1, args.n), args.depth - 1)
        ]
        if size == 1:
            results = [
                search(prefix, distances, minimum, two_minimum) for prefix in prefixes
            ]
        else:
            with MPIPoolExecutor(max_workers=size - 1) as pool:
                results = list(
                    pool.map(
                        search,
                        prefixes,
                        repeat(distances),
                        repeat(minimum),
                        repeat(two_minimum),
                        chunksize=1,
                    )
                )
        best = min(results, key=lambda result: result["length"])
        seconds = MPI.Wtime() - start
        args.output.mkdir(parents=True, exist_ok=True)
        name = f"n{args.n}_d{args.depth}_P{size}_s{args.seed}_r{args.trial}"
        solution_file = f"trasa_{name}.npz"
        np.savez(args.output / solution_file, points=points, route=best["route"])
        search_file = ""
        if args.save_search:
            search_file = f"zadania_{name}.csv"
            with (args.output / search_file).open("w", newline="") as file:
                fields = [key for key in results[0] if key != "route"]
                writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(results)
        row = {
            "n": args.n,
            "depth": args.depth,
            "processes": size,
            "workers": size - 1 if size > 1 else 1,
            "seed": args.seed,
            "trial": args.trial,
            "tasks": len(prefixes),
            "visited": sum(result["visited"] for result in results),
            "pruned": sum(result["pruned"] for result in results),
            "length": best["length"],
            "seconds": seconds,
            "solution_file": solution_file,
            "search_file": search_file,
        }
        log = args.output / "pomiary.csv"
        write_header = not log.exists() or log.stat().st_size == 0
        with log.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(
            f"n={args.n} depth={args.depth} processes={size} tasks={len(prefixes)} "
            f"length={best['length']:.8f} seconds={seconds:.6f}",
            flush=True,
        )
    except Exception as exc:
        print(f"Błąd: {exc}", flush=True)
        MPI.COMM_WORLD.Abort(1)


if __name__ == "__main__":
    main()
