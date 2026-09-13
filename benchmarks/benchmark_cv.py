"""Compare CV scores, fitted solver state, and detailed execution profiles.

Run without competing CPU-heavy jobs:
    .venv/bin/python benchmarks/benchmark_cv.py --baseline /tmp/before.sql

Covers six families, L1/L2, power/dispersion, refinement, wider grids/designs,
missing rows, and singular fallbacks. Timing is informational. State inspection
uses a benchmark-only copy of the private CV CTEs with a diagnostic final SELECT.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import median

import duckdb

from benchmark_evaluate import FAMILIES, setup
from benchmark_summary import METRICS, compare


NAMES = dict(zip(FAMILIES, ['linear', 'logistic', 'poisson', 'gamma', 'tweedie', 'nbinom']))
GRIDS = {'l1': '[0.,.01,.1]', 'l2': '[0.,.1,1.]',
         'power': '[1.,1.5,2.]', 'alpha': '[.1,1.,10.]'}
WIDE_GRID = '[0.,.001,.003,.01,.03,.1,.3,1.,3.]'


def cases(families):
    for family in families:
        for sweep in ['l1', 'l2']:
            yield family, 1000, 3, sweep, GRIDS[sweep], 'ordinary'
        yield family, 1000, 24, 'l2', WIDE_GRID, 'ordinary'
        if family in ['linreg', 'logit']:
            yield family, 1000, 24, 'l1', GRIDS['l1'], 'ordinary'
            yield family, 10000, 24, 'l2', GRIDS['l2'], 'ordinary'
            for scenario in ['missing', 'singular']:
                yield family, 300, 3, 'l2', GRIDS['l2'], scenario
        if family in ['tweedie', 'nbinom']:
            sweep = 'power' if family == 'tweedie' else 'alpha'
            yield family, 1000, 3, sweep, GRIDS[sweep], 'ordinary'
        if family in ['linreg', 'logit', 'tweedie', 'nbinom']:
            sweep = {'linreg': 'l2', 'logit': 'l1', 'tweedie': 'power', 'nbinom': 'alpha'}[family]
            yield family, 300, 3, sweep, GRIDS[sweep], 'refine'


def install_probe(con, source):
    start = source.index('CREATE OR REPLACE MACRO __reg_cv(')
    end = source.index('__reg_cv_score AS (', start)
    prefix = source[start:end].rstrip().removesuffix(',')
    prefix = prefix.replace('MACRO __reg_cv(', 'MACRO __reg_cv_benchmark_state(', 1)
    con.execute(prefix + '''
SELECT B, (SELECT ok FROM __reg_cv_irls_ok) AS used_irls,
       (SELECT max(it) FROM __reg_cv_irls) AS irls_iterations,
       (SELECT max(it) FROM __reg_cv_gd) AS gd_iterations
FROM __reg_cv_sol;
''')


def compare_state(a, b):
    if isinstance(a, (list, tuple)):
        assert isinstance(b, type(a)) and len(a) == len(b), 'Solver state shape changed'
        for x, y in zip(a, b):
            compare_state(x, y)
    elif isinstance(a, float) and isinstance(b, float):
        assert (a == b or (math.isnan(a) and math.isnan(b))
                or math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)), (a, b)
    else:
        assert a == b, (a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=FAMILIES)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error('--repeats and --threads must be positive')
    current = Path(__file__).resolve().parents[1] / 'regression_macros.sql'
    sources = [args.baseline.read_text(), current.read_text()]
    measurements = []
    print(f'DuckDB {duckdb.__version__}; {args.threads} thread(s); '
          f'median of {args.repeats} detailed EXPLAIN ANALYZE runs', flush=True)
    for family, rows, features, sweep, grid, scenario in cases(args.families):
        connections, outputs, states = [], [], []
        try:
            family_arg = f",'{NAMES[family]}'" if sweep in ['l1', 'l2'] else ''
            suffix = '_refine' if scenario == 'refine' else ''
            extra = ',n_refine:=4' if suffix else ''
            query = (f"SELECT * FROM cv_{sweep}{suffix}('observations','y'{family_arg},"
                     f'{grid},k:=3,max_iter:=2000{extra})')
            probe = (f"SELECT * FROM __reg_cv_benchmark_state('observations','y',"
                     f"'{NAMES[family]}',{grid},'{sweep}',3,2000,NULL,1e-8)")
            for source in sources:
                con = duckdb.connect(config={'threads': args.threads})
                connections.append(con)
                con.execute(source)
                setup(con, rows, features, family)
                if scenario == 'singular':
                    con.execute('ALTER TABLE observations ADD COLUMN duplicate DOUBLE')
                    con.execute('UPDATE observations SET duplicate=x0')
                elif scenario == 'missing':
                    con.execute('CREATE OR REPLACE TABLE observations AS '
                                'SELECT * EXCLUDE (rn) REPLACE '
                                '(CASE WHEN rn%17=0 THEN NULL ELSE x0 END AS x0, '
                                'CASE WHEN rn%23=0 THEN NULL ELSE y END AS y) '
                                'FROM (SELECT *,row_number() OVER () rn FROM observations)')
                result = con.execute(query)
                outputs.append(([(v[0], str(v[1])) for v in result.description], result.fetchall()))
                if not suffix:
                    install_probe(con, source)
                    states.append(con.execute(probe).fetchone())
                con.execute("SET profiling_mode='detailed'")
            compare(*outputs)
            # Compare the chosen grid value, including original ordering for ties.
            winners = [min(out[1], key=lambda row: row[1])[0] for out in outputs]
            assert winners[0] == winners[1], ('Selected parameter changed', winners)
            if states:
                compare_state(*states)
            profiles = [[], []]
            for repeat in range(args.repeats):
                for index in ([0, 1] if repeat % 2 == 0 else [1, 0]):
                    raw = connections[index].execute('EXPLAIN (ANALYZE, FORMAT JSON) ' + query).fetchone()[1]
                    profile = json.loads(raw)
                    profiles[index].append({key: profile[key] for key in METRICS})
            times = [median(p['latency'] for p in runs) for runs in profiles]
            record = dict(family=family, rows=rows, features=features, sweep=sweep,
                          grid=grid, scenario=scenario, query=query,
                          baseline_s=times[0], current_s=times[1],
                          reduction_pct=100 * (1 - times[1] / times[0]),
                          exact_outputs=outputs[0] == outputs[1], selected_parameter=str(winners[0]),
                          exact_state=states[0] == states[1] if states else None,
                          profiles=dict(zip(['baseline', 'current'], profiles)))
            measurements.append(record)
            print(f'{family:8} n={rows:5} p={features:2} {sweep:5} {scenario:8}: '
                  f'{times[0]:.3f}s -> {times[1]:.3f}s ({record["reduction_pct"]:.1f}% less); '
                  'scores, selection, and checked state agree', flush=True)
            if args.output:
                args.output.write_text(json.dumps(dict(duckdb=duckdb.__version__,
                    threads=args.threads, repeats=args.repeats,
                    note='Memory includes resident inputs and diagnostic-query high-water marks. '
                         'State checks cover original grids; refinement checks public scores and selection. '
                         'Nested CPU/operator timings overlap.', measurements=measurements), indent=2) + '\n')
        finally:
            for con in connections:
                con.close()


if __name__ == '__main__':
    main()
