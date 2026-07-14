# duckLM cheat sheet

One-page reference. `tbl`/`model`/`outcome`/`*_col` are **string** names; grids
are `DOUBLE[]`. Full docs: [GUIDE.md](GUIDE.md). `{fam}` ∈ `logit`, `linreg`,
`poisson`, `gamma`, `tweedie`, `nbinom`.

```sql
.read regression_macros.sql          -- load everything
```

## Typical workflow

```sql
CREATE TABLE m  AS SELECT * FROM poisson_fit('train', 'y');          -- fit
SELECT * FROM poisson_predict('m', 'newdata');                       -- score  (+ prediction)
SELECT * FROM poisson_evaluate('m', 'test', 'y');                    -- metrics (1 row)
SELECT * FROM poisson_summary('m', 'train', 'y');                    -- SE / p / CI per coef
SELECT * FROM poisson_predict_ci('m', 'train', 'y');                 -- mean + confidence band
SELECT * FROM poisson_influence('m', 'train', 'y');                  -- leverage / residuals / Cook's D
```

## Fit

```
{fam}_fit(tbl, outcome
          [, power := 1.5]      -- tweedie only
          [, alpha := 1.0]      -- nbinom only (fixed dispersion)
          , max_iter := 50000, learning_rate := NULL, tol := 1e-10
          , l2 := 0.0           -- ridge (standardized coefs, intercept free)
          , l1 := 0.0           -- lasso / elastic-net with l2
          , offset_col := NULL  -- +offset in eta, coef fixed at 1
          , weights_col := NULL -- per-row sample weights
          , solver := 'auto')    -- 'auto' | 'gd' | 'irls'
   -> (feature VARCHAR, coefficient DOUBLE)
```

Coefficients on the original scale; unpenalized ≙ R `glm`/`lm`, statsmodels.

`solver := 'auto'` (default) runs IRLS — typically 3–10× faster than gradient
descent for the same MLE — and falls back to `'gd'` automatically when `l1 > 0`
or when `XᵀWX` is singular (constant or perfectly collinear feature, complete
separation). `'gd'` and `'irls'` force one solver; forcing `'irls'` on a singular
design raises an error instead of falling back.

## Predict

```
logit_predict(model, tbl, threshold := 0.5, offset_col := NULL)   -- + prob, pred
{other}_predict(model, tbl, offset_col := NULL)                   -- + prediction
```

## Prediction intervals (CI on the mean)

```
{fam}_predict_ci(model, tbl, outcome, newdata := NULL, conf_level := 0.95,
                 offset_col := NULL, weights_col := NULL [, power|alpha])
   -> input cols + prediction, conf_low, conf_high
```
Covariance from `tbl`; scores `newdata` (default `tbl`). Singular design → NULL band.

## Evaluate  (returns one row)

```
{fam}_evaluate(model, tbl, outcome, offset_col := NULL [, power := 1.5 | alpha := 1.0])
```
| macro | metrics |
|---|---|
| `linreg_evaluate` | n, rmse, mae, r2, adj_r2, loglik, aic, bic |
| `logit_evaluate` | n, accuracy, auc, log_loss, loglik, deviance, null_deviance, pseudo_r2, aic, bic |
| `poisson_evaluate` | n, rmse, mae, loglik, deviance, null_deviance, pseudo_r2, aic, bic |
| `gamma`/`tweedie_evaluate` | n, rmse, mae, deviance, null_deviance, pseudo_r2, dispersion |
| `nbinom_evaluate` | n, rmse, mae, loglik, deviance, null_deviance, pseudo_r2, dispersion, aic, bic |

## Inference — coefficient table

```
{fam}_summary(model, tbl, outcome, conf_level := 0.95,
              offset_col := NULL, weights_col := NULL,
              robust := 'none',      -- 'hc0'|'hc1'|'hc2'|'hc3' sandwich SEs
              cluster_col := NULL     -- one-way cluster-robust SEs
              [, power|alpha])
   -> feature, coefficient, std_error, statistic, p_value, conf_low, conf_high
```

z-based for `logit`/`poisson`/`nbinom`, Student-t(n−d) for `linreg`/`gamma`/`tweedie`.
`robust`/`cluster` are always z. Singular/penalized/df≤0 → NULL SEs.

```
multinom_summary(model, tbl, outcome, conf_level := 0.95)
   -> class, feature, coefficient, std_error, statistic, p_value, conf_low, conf_high
```

## Influence diagnostics  (per training row)

```
{fam}_influence(model, tbl, outcome, offset_col := NULL, weights_col := NULL [, power|alpha])
   -> input cols + hat, pearson_resid, deviance_resid, std_resid, cooks_distance
```
`hat` ∈ [0,1] summing to `d`; flag rows with `cooks_distance > 4/n`.

## Multinomial (softmax)

```
multinom_fit(tbl, outcome, max_iter := 50000, learning_rate := NULL,
             tol := 1e-10, l2 := 0.0, l1 := 0.0)
   -> (class, feature, coefficient)     -- reference class = alphabetical-min label, all 0
multinom_predict(model, tbl)            -- + pred VARCHAR, probs MAP(VARCHAR, DOUBLE)
multinom_evaluate(model, tbl, outcome)  -- n, accuracy, log_loss
```

## Cross-validation  (min cv_deviance)

```
cv_l2(tbl, outcome, family, l2_grid, k := 5)    -- family: linear|logistic|poisson|gamma
cv_l1(tbl, outcome, family, l1_grid, k := 5)
cv_power(tbl, outcome, power_grid, k := 5)       -- tweedie
cv_alpha(tbl, outcome, alpha_grid, k := 5)       -- nbinom
   -> (l2|l1|power|alpha, cv_deviance)
```
Two-stage refine (coarse → fine around the best): `cv_l2_refine` / `cv_l1_refine`
/ `cv_power_refine` / `cv_alpha_refine(..., n_refine := 10)`.

## Dispersion (negative binomial)

```
nbinom_dispersion(tbl, outcome, alpha_grid)              -> (alpha, loglik)   -- argmax = MLE
nbinom_dispersion_refine(tbl, outcome, alpha_grid, n_refine := 10)
```

## Utilities

```
reg_grid(lo, hi, n, log_spaced := false)   -> DOUBLE[]   -- even / geometric grid
dummy_encode_sql(tbl, outcome)             -> VARCHAR     -- R-style k-1 dummy SQL (run as text)
norm_cdf(z)  norm_ppf(p)                    -- standard normal CDF / quantile
t_cdf(t, df) t_ppf(p, df)                   -- Student-t CDF / quantile
```
