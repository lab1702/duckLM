# duckLM — logistic, linear, Poisson, Gamma & Tweedie regression in pure DuckDB SQL

Table macros for DuckDB **1.5+**, no extensions required: **fit**, **predict**,
and **evaluate** for binary logistic regression, ordinary least-squares linear
regression, Poisson regression, Gamma regression, and Tweedie regression (all
but linear use a log link), each with optional ridge/lasso/elastic-net
regularization, an offset/exposure term, and sample weights. Everything runs
inside DuckDB —
training is Nesterov-accelerated gradient descent implemented with a recursive
CTE and list lambdas, sharing a single optimizer core across all model
families.

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

-- Tweedie for zero-inflated positive data (insurance pure premium):
-- power=1.5 is the compound Poisson-Gamma; p=1 is Poisson, p=2 is Gamma
CREATE TABLE pure_premium AS
SELECT * FROM tweedie_fit('policies', 'loss_cost', power := 1.5);

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
| `l1` | `0.0` | lasso penalty `l1·Σ\|β\|` (feature selection); combine with `l2` for elastic net. Intercept unpenalized |
| `offset_col` | `NULL` | name of a column holding a per-row **offset** added to the linear predictor `η = offset + xβ` with a fixed coefficient of 1 (not fit, not penalized) |
| `weights_col` | `NULL` | name of a column of non-negative per-row **sample weights** — the loss (and internal standardization) are weighted by them |

Coefficients are on the **original feature scale** (features — and for
linear/Poisson/Gamma regression the outcome — are rescaled internally only
for optimizer conditioning), so unpenalized results match R's `glm()`/`lm()`,
statsmodels, or unpenalized scikit-learn up to convergence tolerance.

**Offset / exposure.** An offset is a known per-row term in the linear
predictor — most often `log(exposure)` for a Poisson/Gamma rate model (claims
per policy-year, events per person-time). Pass the column name via
`offset_col`, and pass the same `offset_col` to the matching `*_predict` /
`*_evaluate` so scoring includes it. Matches R's `offset=` /
statsmodels' GLM `offset=`.

```sql
-- claims modelled per unit of exposure: E[claims] = exposure · exp(xβ)
CREATE TABLE m AS
SELECT * FROM poisson_fit('policies', 'n_claims', offset_col := 'log_exposure');
SELECT * FROM poisson_predict('m', 'new_policies', offset_col := 'log_exposure');
```

**Sample weights.** `weights_col` names a column of non-negative per-row
weights; the loss and the internal standardization are weighted by them.
Matches scikit-learn's `sample_weight` and R's `weights=`. Integer weights
behave exactly like replicating each row that many times. Weights apply to
fitting only — `*_predict` and `*_evaluate` don't take them.

```sql
CREATE TABLE m AS
SELECT * FROM linreg_fit('survey', 'income', weights_col := 'sampling_weight');
```

**Tweedie.** `tweedie_fit(tbl, outcome, power := 1.5, ...)` takes a variance
power *p* that unifies the log-link families: *p*=1 is Poisson, *p*=2 is Gamma,
and **1<*p*<2** is the compound Poisson-Gamma that models data with **exact
zeros and positive continuous values together** — the classic insurance
pure-premium / loss-cost use case. `tweedie_predict` returns `exp(score)`;
`tweedie_evaluate(model, tbl, outcome, power := 1.5)` returns Tweedie deviance,
deviance-based pseudo-R², and Pearson dispersion. Outcomes must be ≥ 0 for
1≤*p*<2 (strictly positive for *p*≥2). Matches
`TweedieRegressor(power=p, alpha=0, link='log')`.

**Ridge semantics.** Like glmnet, the penalty applies to *standardized*
coefficients, so a given `l2` has comparable strength regardless of feature
or outcome scale, and the intercept is never penalized. Exact scikit-learn
equivalents (fit on z-scored features, outcome transformed as the macro does
internally): `Ridge(alpha = n*l2)` for linear, `LogisticRegression(C =
1/(n*l2))` for logistic, `PoissonRegressor(alpha = l2)` / `GammaRegressor(
alpha = l2)` on the mean-scaled outcome. A small `l2` also gives perfectly
separable logistic data a finite, fast solution.

**Lasso & elastic net.** `l1` adds an L1 penalty that drives coefficients to
**exactly zero** for feature selection (FISTA / proximal-gradient); combine
`l1` and `l2` for elastic net. Available on every family (the intercept is
never penalized). Linear matches `Lasso(alpha = l1)` /
`ElasticNet(alpha = l1+l2, l1_ratio = l1/(l1+l2))` on the standardized problem.

```sql
-- keep only the features that matter
CREATE TABLE sparse_model AS
SELECT * FROM linreg_fit('wide_table', 'target', l1 := 0.1);
SELECT feature FROM sparse_model WHERE coefficient != 0;   -- selected features
```

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

-- tweedie: adds prediction DOUBLE = exp(score), the expected value
SELECT * FROM tweedie_predict('pure_premium', 'renewals');
```

## Evaluating: `logit_evaluate` / `linreg_evaluate` / `poisson_evaluate` / `gamma_evaluate`

Scores `tbl` with a fitted model and its outcome column and returns a one-row
table of goodness-of-fit metrics. Pass the training table for in-sample fit or
a holdout for out-of-sample. Metrics follow the standard statsmodels /
scikit-learn definitions (verified against both).

```sql
SELECT * FROM linreg_evaluate('rev_model', 'sales', 'revenue');
-- ┌───────┬────────┬────────┬────────┬─────────┬──────────┬─────────┬─────────┐
-- │   n   │  rmse  │  mae   │   r2   │ adj_r2  │  loglik  │   aic   │   bic   │
-- └───────┴────────┴────────┴────────┴─────────┴──────────┴─────────┴─────────┘

SELECT * FROM logit_evaluate('churn_model', 'training_data', 'churned');
-- n, accuracy, auc, log_loss, loglik, deviance, null_deviance, pseudo_r2, aic, bic
```

| macro | metrics returned |
|---|---|
| `linreg_evaluate` | `n, rmse, mae, r2, adj_r2, loglik, aic, bic` |
| `logit_evaluate` | `n, accuracy, auc, log_loss, loglik, deviance, null_deviance, pseudo_r2, aic, bic` |
| `poisson_evaluate` | `n, rmse, mae, loglik, deviance, null_deviance, pseudo_r2, aic, bic` |
| `gamma_evaluate` | `n, rmse, mae, deviance, null_deviance, pseudo_r2, dispersion` |
| `tweedie_evaluate` | `n, rmse, mae, deviance, null_deviance, pseudo_r2, dispersion` |

`pseudo_r2` is McFadden's (logistic) or deviance-based (Poisson/Gamma); `aic`
and `bic` use *k* = number of model coefficients (intercept included). Gamma's
log-likelihood/AIC depend on the dispersion parameter, so it reports deviance,
deviance-based pseudo-R², and the Pearson `dispersion` instead.

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
- `__reg_fit`, `__reg_score`, and `__reg_eval` are internal helpers; call the
  fifteen public macros instead.

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

- [regression_macros.sql](regression_macros.sql) — all fifteen macros + shared core
- [tests/](tests) — pytest suite (vs scikit-learn) and a pure-SQL smoke test
- [LICENSE](LICENSE) — MIT

## License

MIT — see [LICENSE](LICENSE).
