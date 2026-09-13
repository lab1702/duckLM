"""Compare evaluation results, EXPLAIN ANALYZE timings, and memory with older SQL.

Example (run without other CPU-heavy jobs):
    .venv/bin/python benchmarks/benchmark_evaluate.py --baseline /tmp/before.sql

Timings are informational. Results must agree within floating-point reduction
roundoff; NULLs, NaNs, infinities, column schemas, and integer counts match exactly.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import median

import duckdb


FAMILIES = ['linreg', 'logit', 'poisson', 'gamma', 'tweedie', 'nbinom']
SCENARIOS = [(1000, 3), (10000, 3), (100000, 3), (10000, 24)]


def compare(left, right):
    assert left[0] == right[0], 'Output schemas differ'
    assert len(left[1]) == len(right[1]), 'Output row counts differ'
    for old_row, new_row in zip(left[1], right[1]):
        for (name, _), old, new in zip(left[0], old_row, new_row):
            if old is None or new is None or isinstance(old, int):
                assert old == new, (name, old, new)
            elif math.isnan(old):
                assert math.isnan(new), (name, old, new)
            elif math.isinf(old):
                assert old == new, (name, old, new)
            else:
                assert math.isclose(old, new, rel_tol=1e-10, abs_tol=1e-12), (name, old, new)


def setup(con, rows, features, family):
    terms = [f'sin(i*{j+1}*.618)/sqrt({features}) AS x{j}' for j in range(features)]
    score = '.3' + ''.join(f'+({(-1)**j * .2})*x{j}' for j in range(features))
    outcome = '(sin(i*1.7)>0)::DOUBLE' if family == 'logit' else f'exp({score})*(1.+.2*cos(i*.7))'
    con.execute(f"CREATE TABLE observations AS SELECT {', '.join(terms)}, {outcome} AS y "
                f'FROM range({rows}) t(i)')
    values = ["('(Intercept)', .3::DOUBLE)"]
    values += [f"('x{j}', {(-1)**j * .2}::DOUBLE)" for j in range(features)]
    con.execute('CREATE TABLE model AS SELECT * FROM (VALUES ' + ','.join(values)
                + ') t(feature, coefficient)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=FAMILIES)
    parser.add_argument('--output', type=Path, help='Optional JSON measurements file')
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error('--repeats and --threads must be positive')
    current = Path(__file__).resolve().parents[1] / 'regression_macros.sql'
    sources = [args.baseline.read_text(), current.read_text()]
    measurements = []
    print(f'DuckDB {duckdb.__version__}; {args.threads} thread(s); '
          f'median of {args.repeats} EXPLAIN ANALYZE runs', flush=True)
    for rows, features in SCENARIOS:
        for family in args.families:
            connections = []
            try:
                outputs = []
                query = f"SELECT * FROM {family}_evaluate('model','observations','y')"
                for sql in sources:
                    con = duckdb.connect(config={'threads': args.threads})
                    connections.append(con)
                    con.execute(sql)
                    setup(con, rows, features, family)
                    result = con.execute(query)
                    outputs.append(([(v[0], str(v[1])) for v in result.description], result.fetchall()))
                compare(*outputs)
                profiles = [[], []]
                for repeat in range(args.repeats):
                    for index in ([0, 1] if repeat % 2 == 0 else [1, 0]):
                        raw = connections[index].execute('EXPLAIN (ANALYZE, FORMAT JSON) ' + query).fetchone()[1]
                        profile = json.loads(raw)
                        profiles[index].append({key: profile[key] for key in
                                                ['latency', 'system_peak_buffer_memory',
                                                 'system_peak_temp_dir_size']})
                times = [median(p['latency'] for p in runs) for runs in profiles]
                record = {'family': family, 'rows': rows, 'features': features,
                          'baseline_s': times[0], 'current_s': times[1],
                          'reduction_pct': 100 * (1 - times[1] / times[0]),
                          'profiles': dict(zip(['baseline', 'current'], profiles))}
                measurements.append(record)
                print(f'{family:8} n={rows:6} p={features:2}: {times[0]:.4f}s -> '
                      f'{times[1]:.4f}s ({record["reduction_pct"]:.1f}% less); results agree', flush=True)
                if args.output:
                    args.output.write_text(json.dumps({'duckdb': duckdb.__version__,
                        'threads': args.threads, 'repeats': args.repeats,
                        'note': 'Fresh connections per variant/workload. Memory includes resident inputs '
                                'and may retain connection high-water marks; not incremental allocations.',
                        'measurements': measurements}, indent=2) + '\n')
            finally:
                for con in connections:
                    con.close()


if __name__ == '__main__':
    main()
