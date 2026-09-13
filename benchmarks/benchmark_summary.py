"""Compare summary outputs and detailed planning/execution profiles with older SQL.

Run from the repository root, without competing CPU-heavy jobs:
    .venv/bin/python benchmarks/benchmark_summary.py --baseline /tmp/before.sql

Each variant uses a fresh connection per workload. Default cases cover all six
families, linear row/feature scaling, and weighted HC3/cluster summaries with
offsets. Timing is informational; numeric results must agree within roundoff.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import median

import duckdb

from benchmark_evaluate import FAMILIES, setup


METRICS = ['latency', 'planner_binding', 'all_optimizers', 'physical_planner',
           'cpu_time', 'system_peak_buffer_memory', 'system_peak_temp_dir_size']


def compare(old, new):
    assert old[0] == new[0], 'Output schemas differ'
    assert len(old[1]) == len(new[1]), 'Output row counts differ'
    for old_row, new_row in zip(old[1], new[1]):
        for (name, _), a, b in zip(old[0], old_row, new_row):
            if isinstance(a, float) and isinstance(b, float):
                assert ((math.isnan(a) and math.isnan(b)) or a == b
                        or math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)), (name, a, b)
            else:
                assert a == b, (name, a, b)


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
    cases = [(family, 10000, 3, 'none') for family in args.families]
    if 'linreg' in args.families:
        cases += [('linreg', rows, features, 'none') for rows, features in
                  [(1000, 3), (100000, 3), (10000, 24)]]
    cases += [(family, 1000, 3, mode) for family in args.families for mode in ['hc3', 'cluster']]
    measurements = []
    print(f'DuckDB {duckdb.__version__}; {args.threads} thread(s); '
          f'median of {args.repeats} detailed EXPLAIN ANALYZE runs', flush=True)
    for family, rows, features, mode in cases:
        connections = []
        try:
            outputs = []
            options = ''
            if mode != 'none':
                options = ",offset_col:='expo',weights_col:='weight'"
                options += ",cluster_col:='cl'" if mode == 'cluster' else ",robust:='hc3'"
            query = f"SELECT * FROM {family}_summary('model','observations','y'{options})"
            for sql in sources:
                con = duckdb.connect(config={'threads': args.threads})
                connections.append(con)
                con.execute(sql)
                setup(con, rows, features, family)
                if mode != 'none':
                    con.execute('CREATE OR REPLACE TABLE observations AS SELECT * EXCLUDE (rn), '
                                'sin(rn*.13)*.1 AS expo, '
                                'CASE WHEN rn%23=0 THEN 0. ELSE .5+abs(cos(rn)) END AS weight, '
                                '(rn%37)::VARCHAR AS cl FROM '
                                '(SELECT *,row_number() OVER () AS rn FROM observations)')
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
            record = {'family': family, 'rows': rows, 'features': features, 'mode': mode,
                      'query': query, 'baseline_s': times[0], 'current_s': times[1],
                      'reduction_pct': 100 * (1 - times[1] / times[0]),
                      'exact_outputs': outputs[0] == outputs[1],
                      'profiles': dict(zip(['baseline', 'current'], profiles))}
            measurements.append(record)
            print(f'{family:8} n={rows:6} p={features:2} {mode:7}: '
                  f'{times[0]:.3f}s -> {times[1]:.3f}s ({record["reduction_pct"]:.1f}% less); '
                  f'binding {binding[0]:.3f}s -> {binding[1]:.3f}s; results agree', flush=True)
            if args.output:
                args.output.write_text(json.dumps({'duckdb': duckdb.__version__,
                    'threads': args.threads, 'repeats': args.repeats,
                    'note': 'Memory includes resident inputs and may retain connection high-water '
                            'marks. Binding is part of planning; nested CPU/operator timings overlap.',
                    'measurements': measurements}, indent=2) + '\n')
        finally:
            for con in connections:
                con.close()


if __name__ == '__main__':
    main()
