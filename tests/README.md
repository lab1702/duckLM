# Tests

Two independent test paths.

## Python suite — `test_regression_macros.py`

Checks every fit, predict, and evaluate against an equivalent **scikit-learn**
model on the same fixed-seed data, so a failure means the macros disagree with a
trusted reference (not just that a recorded number drifted). Covers all four
families, ridge (L2) with the documented sklearn equivalences, predict
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
