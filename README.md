# duckLM — logistic & linear regression in pure DuckDB SQL

Four table macros for DuckDB **1.5+**, no extensions required: fit and predict
for binary logistic regression and for ordinary least-squares linear
regression. Everything runs inside DuckDB — training is Nesterov-accelerated
gradient descent implemented with a recursive CTE and list lambdas, sharing a
single optimizer core between both model families.

## Setup

```sql
.read regression_macros.sql
```

## Fitting: `logit_fit` / `linreg_fit`

Fits a regression of `outcome` on **every other column** of the input table.

```sql
CREATE TABLE churn_model AS
SELECT * FROM logit_fit('training_data', 'churned');

CREATE TABLE rev_model AS
SELECT * FROM linreg_fit('sales', 'revenue');

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
| `outcome` | — | column to predict, as a string; for `logit_fit` it must be 0/1 or boolean |
| `max_iter` | `50000` | hard cap on gradient iterations |
| `learning_rate` | `NULL` | step size on the standardized scale; `NULL` auto-picks a provably convergent default (`4/(d+1)` logistic, `1/(d+1)` linear) |
| `tol` | `1e-10` | stop early when no coefficient moves more than this between iterations |

Coefficients are on the **original feature scale** (features — and for linear
regression the outcome — are z-scored internally only for optimizer
conditioning), so results match R's `glm()`/`lm()`, statsmodels, or
unpenalized scikit-learn up to convergence tolerance.

## Predicting: `logit_predict` / `linreg_predict`

Scores a table with a fitted model, matching model features to columns
**by name**. Extra columns (ids, the outcome itself, …) are passed through
untouched and ignored by the scoring.

```sql
-- logistic: adds prob DOUBLE and pred BOOLEAN (pred = prob >= threshold)
SELECT * FROM logit_predict('churn_model', 'new_customers');
SELECT * FROM logit_predict('churn_model', 'new_customers', threshold := 0.7);

-- linear: adds prediction DOUBLE
SELECT * FROM linreg_predict('rev_model', 'pipeline');
```

## Contract / fine print

- Feature columns must be castable to `DOUBLE` (numeric or boolean). Booleans
  become 1/0.
- Training drops rows with a `NULL` outcome or any `NULL` feature.
- Prediction returns `NULL` outputs for rows where a model feature is `NULL`
  or the column is missing entirely.
- Zero-variance (constant) feature columns get coefficient `0`. A constant
  *outcome* in `linreg_fit` is fine (intercept = mean, slopes 0).
- Exactly collinear features (singular design) don't error: the fit converges
  to one valid least-squares solution. Its *predictions* match any other
  solver's; the individual coefficient split across the collinear columns is
  arbitrary (as it is for every solver).
- Perfectly separable logistic data (including an all-0 or all-1 outcome) has
  no finite maximum-likelihood solution; the fit still terminates with large
  coefficients rather than erroring, but only after running all `max_iter`
  iterations — lower `max_iter` if that's too slow.
- Guarded name collisions (clear errors, never silent misbehavior): column
  *and table* names beginning with `__reg_` are reserved everywhere, checked
  case-insensitively; a *feature* named `(Intercept)` is rejected at fit time
  (as the outcome it's fine); `prob`/`pred` columns (any case) are rejected by
  `logit_predict` and `prediction` by `linreg_predict` — drop them first with
  `SELECT * EXCLUDE (...)`.
- Prediction matches model features to columns by exact, case-sensitive name
  string — score with the same column spellings you trained with.
- The training set is materialized as an in-memory list during optimization —
  comfortable up to a few hundred thousand rows × dozens of features.
- `__reg_fit` and `__reg_score` are internal helpers; call the four public
  macros instead.

## Files

- [regression_macros.sql](regression_macros.sql) — all four macros + shared core
- [LICENSE](LICENSE) — MIT
- `.venv/` — project-local Python venv (numpy/scikit-learn/duckdb) used only
  for validating the macros against reference implementations

## License

MIT — see [LICENSE](LICENSE).
