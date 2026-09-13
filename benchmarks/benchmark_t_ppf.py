"""Benchmark quantiles, optionally checking exact equality against older macros.

Run from the repo root:
    .venv/bin/python benchmarks/benchmark_t_ppf.py --baseline /tmp/before.sql
"""

import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import duckdb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')

    current = Path(__file__).resolve().parents[1] / 'regression_macros.sql'
    paths = [('baseline', args.baseline)] if args.baseline else []
    paths.append(('current', current))
    connections = []
    try:
        for label, path in paths:
            con = duckdb.connect(config={'threads': 1})
            connections.append((label, con))
            con.execute(path.read_text())

        print(f'DuckDB {duckdb.__version__}; one thread; median of {args.repeats} runs')
        queries = {
            '100 quantiles': 'SELECT t_ppf((i+1)/101.0, 2.0+i%40) FROM range(100) t(i)',
            'tail/central grid': '''
                SELECT t_ppf(p, df) FROM
                (VALUES (1e-200), (1e-100), (1e-20), (1e-6), (.025),
                        (.49999999999999994), (.5), (.5000000000000001),
                        (.975), (.999999), (0.), (1.)) probabilities(p),
                (VALUES (.1), (.5), (1.), (2.), (30.), (1000.), (1e8)) degrees(df)
                ORDER BY p, df
            ''',
        }
        for name, query in queries.items():
            # Warm both variants and compare every result, including infinities.
            results = [con.execute(query).fetchall() for _, con in connections]
            if len(results) == 2 and results[0] != results[1]:
                raise AssertionError(f'{name}: baseline/current results differ')
            timings = [[] for _ in connections]
            for repeat in range(args.repeats):
                # Alternate order to reduce systematic timing bias.
                order = range(len(connections))
                if repeat % 2:
                    order = reversed(order)
                for index in order:
                    start = perf_counter()
                    connections[index][1].execute(query).fetchall()
                    timings[index].append(perf_counter() - start)
            print(name + (': exactly equal results' if len(results) == 2 else ''))
            for (label, _), samples in zip(connections, timings):
                print(f'  {label}: {median(samples):.4f}s')
    finally:
        for _, con in connections:
            con.close()


if __name__ == '__main__':
    main()
