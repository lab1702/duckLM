# Tests

Two independent test paths.

## Python suite — `test_regression_macros.py`

Checks every fit, predict, and evaluate against an equivalent **scikit-learn**
model on the same fixed-seed data, so a failure means the macros disagree with a
trusted reference (not just that a recorded number drifted). Covers all four
families (including Tweedie across powers, negative binomial, multinomial softmax,
offset/exposure, sample weights, k-fold cross-validation via cv_l2/cv_l1/
cv_power/cv_alpha, NB dispersion estimation via nbinom_dispersion, and
two-stage grid refinement via reg_grid/cv_*_refine/nbinom_dispersion_refine,
and Wald inference via `*_summary` — standard errors, z/t statistics, p-values
and confidence intervals for all six single-outcome families plus multinomial
(baseline-category Fisher information), robust HC0-HC3 and cluster-robust
(sandwich) standard errors, prediction intervals (`*_predict_ci`, CI on the
predicted mean), influence diagnostics (`*_influence`: leverage, Pearson/deviance
residuals, studentized residuals, Cook's distance), and the IRLS solver
(`solver := 'irls'`), all checked against an independent numpy/scipy reference, plus the pure-SQL
`norm_cdf`/`norm_ppf`/`t_cdf`/`t_ppf` helpers vs SciPy),
ridge/lasso/elastic-net with the documented sklearn equivalences
(and KKT-optimality checks for L1 where sklearn has no reference), predict
semantics, goodness-of-fit metrics (`*_evaluate` vs sklearn R²/AUC/log-loss/
deviance/`d2_tweedie_score`), NULL / constant-feature / type edge cases, and the
error + reserved-name contract.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r tests/requirements.txt   # Windows
# .venv/bin/python -m pip install -r tests/requirements.txt      # macOS/Linux
.venv/Scripts/python -m pytest tests/ -q
```

The root `pytest.ini` adds `-n auto` to every pytest run, so `pytest-xdist`
automatically chooses the number of parallel workers. Install or update the
dependencies above before running tests in an existing environment. For debugging,
pass `-n 0` to run serially.

To measure Student-t quantile performance, run
`.venv/bin/python benchmarks/benchmark_t_ppf.py` from the repo root. Optionally
pass `--baseline /path/to/previous_regression_macros.sql` to compare timings and
check exact output equality on ordinary, central, and extreme-tail inputs.
The benchmark uses one DuckDB thread and reports medians; run it without other
CPU-heavy jobs for comparable timings. Timing is informational, not a test gate.

For evaluation, run `.venv/bin/python benchmarks/benchmark_evaluate.py --baseline
/path/to/previous_regression_macros.sql` (as one command). This compares all six
single-outcome families at 1,000–100,000 rows and includes a 24-feature workload.
It checks metric agreement within floating-point reduction roundoff and reports
median `EXPLAIN ANALYZE` latency. Pass `--output /tmp/evaluate.json` to retain
timings and DuckDB's buffer-memory/spill measurements. Run it without concurrent
CPU-heavy jobs; memory measurements include resident inputs and may retain
connection high-water marks.

For coefficient summaries, run `.venv/bin/python benchmarks/benchmark_summary.py
--baseline /path/to/previous_regression_macros.sql` (as one command). This checks
all six single-outcome families, linear row/feature scaling, and weighted HC3 and
cluster summaries with offsets. Detailed `EXPLAIN ANALYZE` profiles separate
binding, optimization, and total latency. Use `--output /tmp/summary.json` to
retain measurements, including buffer memory and spill; the same timing and
memory caveats above apply. The script reuses setup from `benchmark_evaluate.py`.

For prediction intervals and influence diagnostics, run
`.venv/bin/python benchmarks/benchmark_diagnostics.py --baseline /path/to/previous_regression_macros.sql`.
It compares all six families, linear row/feature scaling, weighted data with
offsets and incomplete rows, and intervals on separate scoring data. Use
`--output /tmp/diagnostics.json` to retain detailed planning, latency, memory,
and spill measurements. It reuses helpers from the evaluation and summary
benchmarks; the same timing and memory caveats apply.

## SQL smoke test — `smoke.sql`

No Python required — just the DuckDB CLI. Fits each family on deterministic
inline data and aborts (non-zero exit) on the first failed check. Run from the
repo root:

```bash
duckdb < tests/smoke.sql
```
