"""Compare fitted coefficients, solver state, and detailed execution profiles.

Run without competing CPU-heavy jobs:
    .venv/bin/python benchmarks/benchmark_fit.py --baseline /tmp/before.sql

Covers six families, narrow/wide designs, weights/offsets/missing rows, elastic
net, explicit IRLS/GD, and singular fallback. Timings are informational. A
benchmark-only copy of the fit CTEs exposes standardized coefficients and solver
state without changing the library's public result.
"""

import argparse
import json
from pathlib import Path
from statistics import median

import duckdb

from benchmark_cv import NAMES, compare_state
from benchmark_evaluate import FAMILIES, setup
from benchmark_summary import METRICS, compare


def cases(families):
    for family in families:
        for rows, features in [(1000, 3), (10000, 3), (10000, 24)]:
            yield family, rows, features, 'ordinary'
        yield family, 1000, 3, 'weighted'
        yield family, 1000, 24, 'elastic_net'
        yield family, 1000, 3, 'irls'
        if family in ['linreg', 'logit']:
            yield family, 300, 3, 'gd'
            yield family, 300, 3, 'singular'


def install_probe(con, source):
    start = source.index('CREATE OR REPLACE MACRO __reg_fit(')
    end = source.index('-- Map standardized-scale coefficients', start)
    prefix = source[start:end].replace('MACRO __reg_fit(', 'MACRO __reg_fit_benchmark_state(', 1)
    con.execute(prefix + '''
SELECT betas, (SELECT ok FROM __reg_irls_ok) AS used_irls,
       (SELECT max(it) FROM __reg_irls) AS irls_iterations,
       (SELECT max(it) FROM __reg_gd) AS gd_iterations,
       (SELECT move FROM __reg_irls_beta) AS irls_move,
       (SELECT proposed_move FROM __reg_irls_beta) AS proposed_move
FROM __reg_sol;
''')


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
    for family, rows, features, scenario in cases(args.families):
        connections, outputs, states = [], [], []
        try:
            solver = scenario if scenario in ['irls', 'gd'] else 'auto'
            l1 = '.01' if scenario == 'elastic_net' else '0.'
            l2 = '.1' if scenario in ['elastic_net', 'irls'] else '0.'
            offset = "'expo'" if scenario == 'weighted' else 'NULL'
            weight = "'weight'" if scenario == 'weighted' else 'NULL'
            power = ('1.3' if scenario == 'weighted' else '1.5') if family == 'tweedie' else 'NULL'
            alpha = ('.5' if scenario == 'weighted' else '1.') if family == 'nbinom' else 'NULL'
            extra = f',power:={power}' if family == 'tweedie' else ''
            extra += f',alpha:={alpha}' if family == 'nbinom' else ''
            query = (f"SELECT * FROM {family}_fit('observations','y',max_iter:=5000,"
                     f"l1:={l1},l2:={l2},offset_col:={offset},weights_col:={weight},solver:='{solver}'{extra})")
            probe = (f"SELECT * FROM __reg_fit_benchmark_state('observations','y','{NAMES[family]}',"
                     f"'{family}_fit',5000,NULL,1e-10,{l2},{offset},{weight},{power},{l1},{alpha},'{solver}')")
            for source in sources:
                con = duckdb.connect(config={'threads': args.threads})
                connections.append(con)
                con.execute(source)
                setup(con, rows, features, family)
                if scenario == 'singular':
                    con.execute('ALTER TABLE observations ADD COLUMN duplicate DOUBLE')
                    con.execute('UPDATE observations SET duplicate=x0')
                elif scenario == 'weighted':
                    con.execute('CREATE OR REPLACE TABLE observations AS '
                                'SELECT * EXCLUDE (rn) REPLACE '
                                '(CASE WHEN rn%17=0 THEN NULL ELSE x0 END AS x0, '
                                'CASE WHEN rn%29=0 THEN NULL ELSE y END AS y), '
                                'CASE WHEN rn%31=0 THEN NULL ELSE sin(rn*.13)*.1 END AS expo, '
                                'CASE WHEN rn%23=0 THEN 0. ELSE .5+abs(cos(rn)) END AS weight '
                                'FROM (SELECT *,row_number() OVER () rn FROM observations)')
                result = con.execute(query)
                outputs.append(([(v[0], str(v[1])) for v in result.description], result.fetchall()))
                install_probe(con, source)
                states.append(con.execute(probe).fetchone())
                con.execute("SET profiling_mode='detailed'")
            compare(*outputs)
            compare_state(*states)
            profiles = [[], []]
            for repeat in range(args.repeats):
                for index in ([0, 1] if repeat % 2 == 0 else [1, 0]):
                    raw = connections[index].execute('EXPLAIN (ANALYZE, FORMAT JSON) ' + query).fetchone()[1]
                    profile = json.loads(raw)
                    profiles[index].append({key: profile[key] for key in METRICS})
            times = [median(p['latency'] for p in runs) for runs in profiles]
            record = dict(family=family, rows=rows, features=features, scenario=scenario,
                          query=query, baseline_s=times[0], current_s=times[1],
                          reduction_pct=100 * (1 - times[1] / times[0]),
                          exact_outputs=outputs[0] == outputs[1], exact_state=states[0] == states[1],
                          profiles=dict(zip(['baseline', 'current'], profiles)))
            measurements.append(record)
            print(f'{family:8} n={rows:5} p={features:2} {scenario:11}: '
                  f'{times[0]:.3f}s -> {times[1]:.3f}s ({record["reduction_pct"]:.1f}% less); '
                  'coefficients and solver state agree', flush=True)
            if args.output:
                args.output.write_text(json.dumps(dict(duckdb=duckdb.__version__,
                    threads=args.threads, repeats=args.repeats,
                    note='Memory includes resident inputs and diagnostic-query high-water marks. '
                         'Nested CPU/operator timings overlap.', measurements=measurements), indent=2) + '\n')
        finally:
            for con in connections:
                con.close()


if __name__ == '__main__':
    main()
