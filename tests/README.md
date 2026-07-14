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

## SQL smoke test — `smoke.sql`

No Python required — just the DuckDB CLI. Fits each family on deterministic
inline data and aborts (non-zero exit) on the first failed check. Run from the
repo root:

```bash
duckdb < tests/smoke.sql
```
