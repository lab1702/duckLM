# duckLM — logistic, linear, Poisson & Gamma regression in pure DuckDB SQL

Eight table macros for DuckDB **1.5+**, no extensions required: fit and
predict for binary logistic regression, ordinary least-squares linear
regression, Poisson regression, and Gamma regression (both log link), each
with optional ridge (L2) regularization. Everything runs inside DuckDB —
training is Nesterov-accelerated gradient descent implemented with a
recursive CTE and list lambdas, sharing a single optimizer core across all
four model families.

## Setup

```sql
.read regression_macros.sql
```

## Fitting: `logit_fit` / `linreg_fit` / `poisson_fit` / `gamma_fit`

Fits a regression of `outcome` on **every other column** of the input table.

```sql
CREATE TABLE churn_model AS
SELECT * FROM logit_fit('training_data', 'churned');

CREATE TABLE rev_model AS
SELECT * FROM linreg_fit('sales', 'revenue');

CREATE TABLE claims_model AS
SELECT * FROM poisson_fit('policies', 'n_claims');

CREATE TABLE severity_model AS
SELECT * FROM gamma_fit('claims', 'claim_amount');

-- optional ridge regularization on any family
CREATE TABLE churn_model_reg AS
SELECT * FROM logit_fit('training_data', 'churned', l2 := 0.1);

SELECT * FROM rev_model;
-- ┌─────────────┬─────────────┐
-- │   feature   │ coefficient │
-- │ (Intercept) │      3.4097 │
-- │ ad_spend    │     -2.0046 │
-- │ headcount   │      0.7201 │
-- └─────────────┴─────────────┘
```

| argument | default | meaning |
|---|---|---|
| `tbl` | — | table/view name as a **string** (resolved via `query_table`; schema-qualified names work) |
| `outcome` | — | column to predict, as a string; for `logit_fit` it must be 0/1 or boolean, for `poisson_fit` non-negative, for `gamma_fit` strictly positive |
| `max_iter` | `50000` | hard cap on gradient iterations |
| `learning_rate` | `NULL` | step size on the standardized scale; `NULL` auto-picks a convergent default (`4/(d+1+4·l2)` logistic, `1/(d+1+l2)` otherwise; Poisson/Gamma steps are additionally damped each iteration by the largest curvature weight, since theirs is unbounded) |
| `tol` | `1e-10` | stop early when the gradient step is smaller than this |
| `l2` | `0.0` | ridge penalty `(l2/2)·Σβ²` added to the mean loss of the internally standardized problem, intercept unpenalized |

Coefficients are on the **original feature scale** (features — and for
linear/Poisson/Gamma regression the outcome — are rescaled internally only
for optimizer conditioning), so unpenalized results match R's `glm()`/`lm()`,
statsmodels, or unpenalized scikit-learn up to convergence tolerance.

**Ridge semantics.** Like glmnet, the penalty applies to *standardized*
coefficients, so a given `l2` has comparable strength regardless of feature
or outcome scale, and the intercept is never penalized. Exact scikit-learn
equivalents (fit on z-scored features, outcome transformed as the macro does
internally): `Ridge(alpha = n*l2)` for linear, `LogisticRegression(C =
1/(n*l2))` for logistic, `PoissonRegressor(alpha = l2)` / `GammaRegressor(
alpha = l2)` on the mean-scaled outcome. A small `l2` also gives perfectly
separable logistic data a finite, fast solution.

## Predicting: `logit_predict` / `linreg_predict` / `poisson_predict` / `gamma_predict`

Scores a table with a fitted model, matching model features to columns
**by name**. Extra columns (ids, the outcome itself, …) are passed through
untouched and ignored by the scoring.

```sql
-- logistic: adds prob DOUBLE and pred BOOLEAN (pred = prob >= threshold)
SELECT * FROM logit_predict('churn_model', 'new_customers');
SELECT * FROM logit_predict('churn_model', 'new_customers', threshold := 0.7);

-- linear: adds prediction DOUBLE
SELECT * FROM linreg_predict('rev_model', 'pipeline');

-- poisson: adds prediction DOUBLE = exp(score), the expected count
SELECT * FROM poisson_predict('claims_model', 'new_policies');

-- gamma: adds prediction DOUBLE = exp(score), the expected value
SELECT * FROM gamma_predict('severity_model', 'open_claims');
```

## Contract / fine print

- Feature columns must be castable to `DOUBLE` (numeric or boolean). Booleans
  become 1/0.
- Training drops rows with a `NULL` outcome or any `NULL` feature. A feature
  column that is *entirely* NULL is rejected with a clear error (as R's
  `glm()` and scikit-learn do) rather than silently excluded.
- Prediction returns `NULL` outputs for rows where a model feature is `NULL`
  or the column is missing entirely.
- Zero-variance (constant) feature columns get coefficient `0`. A constant
  *outcome* in `linreg_fit` is fine (intercept = mean, slopes 0).
- Exactly collinear features (singular design) don't error: the fit converges
  to one valid least-squares solution. Its *predictions* match any other
  solver's; the individual coefficient split across the collinear columns is
  arbitrary (as it is for every solver).
- Data with no finite maximum-likelihood solution — perfectly separable
  logistic data, or an all-zero-count Poisson/all-constant Gamma outcome —
  still terminates with large coefficients rather than erroring, but only
  after running all `max_iter` iterations. For **separable logistic** data a
  positive `l2` gives a finite, fast solution (it penalizes the diverging
  slopes). For an **all-zero Poisson** outcome it's the *intercept* that
  diverges, and `l2` does not penalize the intercept (by design, matching
  glmnet/sklearn), so lower `max_iter` instead.
- Guarded name collisions (clear errors, never silent misbehavior): column
  *and table* names beginning with `__reg_` are reserved everywhere, checked
  case-insensitively; a *feature* named `(Intercept)` is rejected at fit time
  (as the outcome it's fine); `prob`/`pred` columns (any case) are rejected by
  `logit_predict` and `prediction` by
  `linreg_predict`/`poisson_predict`/`gamma_predict` — drop them first with
  `SELECT * EXCLUDE (...)`.
- Prediction matches model features to columns by exact, case-sensitive name
  string — score with the same column spellings you trained with.
- The training set is materialized as an in-memory list during optimization —
  comfortable up to a few hundred thousand rows × dozens of features.
- `__reg_fit` and `__reg_score` are internal helpers; call the eight public
  macros instead.

## Testing

Two independent paths (details in [tests/README.md](tests/README.md)):

```bash
# Python suite: every fit/predict checked against scikit-learn on fixed-seed data
python -m venv .venv && .venv/Scripts/python -m pip install -r tests/requirements.txt
.venv/Scripts/python -m pytest tests/ -q

# Pure-SQL smoke test: no Python, just the DuckDB CLI
duckdb < tests/smoke.sql
```

## Files

- [regression_macros.sql](regression_macros.sql) — all eight macros + shared core
- [tests/](tests) — pytest suite (vs scikit-learn) and a pure-SQL smoke test
- [LICENSE](LICENSE) — MIT

## License

MIT — see [LICENSE](LICENSE).
