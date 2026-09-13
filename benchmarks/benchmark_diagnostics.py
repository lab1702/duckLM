"""Compare prediction intervals and influence profiles against older macros.

Run from the repository root without competing CPU-heavy jobs:
    .venv/bin/python benchmarks/benchmark_diagnostics.py --baseline /tmp/before.sql

Cases cover six families, linear row/feature scaling, weights and offsets with
incomplete rows, and separate scoring data. Each variant/workload uses a fresh
connection. Timings are informational; output schemas and numeric results must
agree with the baseline (allowing floating-point reduction roundoff).
"""

import argparse
import json
from pathlib import Path
from statistics import median

import duckdb

from benchmark_evaluate import FAMILIES, setup
from benchmark_summary import METRICS, compare


KINDS = ['predict_ci', 'influence']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=FAMILIES)
    parser.add_argument('--kinds', nargs='+', choices=KINDS, default=KINDS)
    parser.add_argument('--output', type=Path, help='Optional JSON measurements file')
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error('--repeats and --threads must be positive')
    current = Path(__file__).resolve().parents[1] / 'regression_macros.sql'
    sources = [args.baseline.read_text(), current.read_text()]
    cases = [(family, kind, 10000, 3, 'plain') for family in args.families for kind in args.kinds]
    if 'linreg' in args.families:
        cases += [('linreg', kind, n, d, 'plain') for kind in args.kinds
                  for n, d in [(1000, 3), (100000, 3), (10000, 24)]]
    cases += [(family, kind, 1000, 3, 'weighted') for family in args.families for kind in args.kinds]
    if 'predict_ci' in args.kinds:
        cases += [(family, 'predict_ci', 1000, 3, 'newdata') for family in args.families]
    measurements = []
    print(f'DuckDB {duckdb.__version__}; {args.threads} thread(s); '
          f'median of {args.repeats} detailed EXPLAIN ANALYZE runs', flush=True)
    for family, kind, rows, features, mode in cases:
        connections = []
        try:
            outputs = []
            options = ",offset_col:='expo',weights_col:='weight'" if mode == 'weighted' else ''
            if mode == 'newdata':
                options = ",newdata:='scoring'"
            query = f"SELECT * FROM {family}_{kind}('model','observations','y'{options})"
            for sql in sources:
                con = duckdb.connect(config={'threads': args.threads})
                connections.append(con)
                con.execute(sql)
                setup(con, rows, features, family)
                if mode == 'weighted':
                    con.execute('CREATE OR REPLACE TABLE observations AS '
                        'SELECT * EXCLUDE (rn) REPLACE '
                        '(CASE WHEN rn%37=0 THEN NULL ELSE x0 END AS x0, '
                        ' CASE WHEN rn%31=0 THEN NULL ELSE y END AS y), '
                        'CASE WHEN rn%29=0 THEN NULL ELSE sin(rn*.13)*.1 END AS expo, '
                        'CASE WHEN rn%19=0 THEN NULL WHEN rn%23=0 THEN 0. '
                        '     ELSE .5+abs(cos(rn)) END AS weight FROM '
                        '(SELECT *,row_number() OVER () AS rn FROM observations)')
                elif mode == 'newdata':
                    con.execute('CREATE TABLE scoring AS SELECT * EXCLUDE (rn,y) REPLACE '
                        '(CASE WHEN rn%17=0 THEN NULL ELSE x0+2. END AS x0) FROM '
                        '(SELECT *,row_number() OVER () AS rn FROM observations) WHERE rn<=200')
                result = con.execute(query)
                outputs.append(([(v[0], str(v[1])) for v in result.description], result.fetchall()))
                con.execute("SET profiling_mode='detailed'")
            compare(*outputs)
            profiles = [[], []]
            for repeat in range(args.repeats):
                for index in ([0, 1] if repeat % 2 == 0 else [1, 0]):
                    raw = connections[index].execute('EXPLAIN (ANALYZE, FORMAT JSON) ' + query).fetchone()[1]
                    profile = json.loads(raw)
                    profiles[index].append({key: profile[key] for key in METRICS})
            times = [median(p['latency'] for p in runs) for runs in profiles]
            binding = [median(p['planner_binding'] for p in runs) for runs in profiles]
            record = {'family': family, 'kind': kind, 'rows': rows, 'features': features,
                      'mode': mode, 'query': query, 'output_rows': len(outputs[1][1]),
                      'baseline_s': times[0], 'current_s': times[1],
                      'reduction_pct': 100 * (1 - times[1] / times[0]),
                      'exact_outputs': outputs[0] == outputs[1],
                      'profiles': dict(zip(['baseline', 'current'], profiles))}
            measurements.append(record)
            print(f'{family:8} {kind:10} n={rows:6} p={features:2} {mode:8}: '
                  f'{times[0]:.3f}s -> {times[1]:.3f}s ({record["reduction_pct"]:.1f}% less); '
                  f'binding {binding[0]:.3f}s -> {binding[1]:.3f}s; results agree', flush=True)
            if args.output:
                args.output.write_text(json.dumps({'duckdb': duckdb.__version__,
                    'threads': args.threads, 'repeats': args.repeats,
                    'note': 'Memory includes resident inputs and may retain connection high-water '
                            'marks. Binding is part of planning; nested operator timings overlap.',
                    'measurements': measurements}, indent=2) + '\n')
        finally:
            for con in connections:
                con.close()


if __name__ == '__main__':
    main()
