# duckLM — GLM regression (logistic, linear, Poisson, Gamma, Tweedie, negative binomial, multinomial) in pure DuckDB SQL

Table macros for DuckDB **1.5+**, no extensions and no driver required. For each
of binary logistic, ordinary-least-squares linear, Poisson, Gamma, Tweedie,
negative-binomial (all but linear use a log link) and **multinomial (softmax)**
regression, duckLM provides **fit**, **predict**, **evaluate**, **summary**
(standard errors, p-values, confidence intervals), **prediction intervals** and
**influence diagnostics** — plus cross-validation, robust/cluster-robust SEs,
and a fast IRLS solver. The single-outcome families take optional
ridge/lasso/elastic-net regularization, an offset/exposure term, and sample
weights.

Everything runs inside DuckDB: training is Fisher-scoring IRLS by default, with
Nesterov-accelerated gradient descent for L1 and as an automatic fallback on a
rank-deficient design — both implemented with a recursive CTE and list lambdas,
sharing a single optimizer core across all families; even the coefficient
covariance (matrix inversion) and the normal/Student-t distributions are
computed in pure SQL. All outputs are verified against statsmodels / scikit-learn
to machine precision.

## Setup

```sql
.read regression_macros.sql
```

That's it — the whole library is one file of `CREATE OR REPLACE MACRO`
statements. Load it once per session (or `.read` it from your own script), then
call the macros. It works from the DuckDB CLI and from any driver (Python, R,
Node, …) — pass table and column names as **strings**.

## Documentation

- **[CHEATSHEET.md](CHEATSHEET.md)** — one-page signature reference for every
  macro (fit / predict / evaluate / summary / predict_ci / influence,
  cross-validation, utilities). Start here to look something up fast.
- **[GUIDE.md](GUIDE.md)** — the user's guide: task-oriented explanations,
  examples, statistical conventions, and the full contract / edge-case behavior.

Quick taste:

```sql
CREATE TABLE m AS SELECT * FROM poisson_fit('policies', 'n_claims');
SELECT * FROM poisson_summary('m', 'policies', 'n_claims');   -- coefficients + SE / p / CI
SELECT * FROM poisson_predict_ci('m', 'policies', 'n_claims', newdata := 'renewals');
```

## Testing

Two independent paths (details in [tests/README.md](tests/README.md)):

```bash
# Python suite: every fit/predict/summary checked against scikit-learn /
# statsmodels / a numpy-scipy reference on fixed-seed data
python -m venv .venv && .venv/Scripts/python -m pip install -r tests/requirements.txt
.venv/Scripts/python -m pytest tests/ -q

# Pure-SQL smoke test: no Python, just the DuckDB CLI
duckdb < tests/smoke.sql
```

## Files

- [regression_macros.sql](regression_macros.sql) — the entire library: every
  model macro (fit / predict / evaluate / summary / predict_ci / influence),
  cross-validation, `dummy_encode_sql`, the `norm_*` / `t_*` distribution
  helpers, and the shared core
- [CHEATSHEET.md](CHEATSHEET.md) — one-page signature reference
- [GUIDE.md](GUIDE.md) — the user's guide
- [tests/](tests) — pytest suite (vs scikit-learn / statsmodels) and a pure-SQL
  smoke test
- [LICENSE](LICENSE) — MIT

## License

MIT — see [LICENSE](LICENSE).
