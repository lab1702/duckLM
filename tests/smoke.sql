-- Pure-DuckDB smoke test for the duckLM regression macros -- no Python needed.
-- Run from the repo root:
--     duckdb < tests/smoke.sql
-- Every check aborts the script with a non-zero exit code if it fails.

.bail on
.read regression_macros.sql

-- Deterministic, noiseless linear data: y = 2 + 3*x1 - x2.
CREATE TABLE lin AS
  SELECT i AS x1, (i % 7) - 3 AS x2, 2 + 3.0 * i - ((i % 7) - 3) AS y
  FROM range(200) t(i);

CREATE TABLE lin_m AS SELECT * FROM linreg_fit('lin', 'y');
SELECT CASE
    WHEN abs((SELECT coefficient FROM lin_m WHERE feature = 'x1') - 3.0) < 1e-6
     AND abs((SELECT coefficient FROM lin_m WHERE feature = 'x2') + 1.0) < 1e-6
     AND abs((SELECT coefficient FROM lin_m WHERE feature = '(Intercept)') - 2.0) < 1e-6
    THEN 'PASS  linreg recovers y = 2 + 3*x1 - x2'
    ELSE error('SMOKE FAIL: linreg did not recover the exact linear relationship')
  END;

-- linreg_predict reproduces y on noiseless data and preserves row count.
SELECT CASE
    WHEN (SELECT count(*) FROM linreg_predict('lin_m', 'lin')) = 200
     AND (SELECT max(abs(prediction - y)) FROM linreg_predict('lin_m', 'lin')) < 1e-6
    THEN 'PASS  linreg_predict reproduces y'
    ELSE error('SMOKE FAIL: linreg_predict did not reproduce y')
  END;

-- Constant feature must get coefficient exactly 0 (even for 4.2).
CREATE TABLE cst AS SELECT x1, x2, 4.2 AS c, y FROM lin;
SELECT CASE
    WHEN (SELECT coefficient FROM linreg_fit('cst', 'y') WHERE feature = 'c') = 0.0
    THEN 'PASS  constant feature coefficient is exactly 0'
    ELSE error('SMOKE FAIL: constant feature coefficient is not exactly 0')
  END;

-- Ridge runs and shrinks the slopes relative to the unpenalized fit.
SELECT CASE
    WHEN abs((SELECT coefficient FROM linreg_fit('lin', 'y', l2 := 5.0) WHERE feature = 'x1'))
       < abs((SELECT coefficient FROM lin_m WHERE feature = 'x1'))
    THEN 'PASS  ridge (l2) shrinks the slope'
    ELSE error('SMOKE FAIL: ridge did not shrink the slope')
  END;

-- Logistic: fits with finite coefficients; probabilities land in [0, 1].
CREATE TABLE clf AS
  SELECT i AS x1, (i * 7 + 3) % 11 AS x2, ((i * 13 + 5) % 10 < 6)::INT AS y
  FROM range(300) t(i);
CREATE TABLE clf_m AS SELECT * FROM logit_fit('clf', 'y');
SELECT CASE
    WHEN (SELECT bool_and(NOT isnan(coefficient) AND NOT isinf(coefficient)) FROM clf_m)
     AND (SELECT count(*) FROM clf_m) = 3
     AND (SELECT bool_and(prob BETWEEN 0 AND 1) FROM logit_predict('clf_m', 'clf'))
    THEN 'PASS  logit fits finite coefficients and probabilities in [0,1]'
    ELSE error('SMOKE FAIL: logit produced non-finite coefficients or out-of-range probabilities')
  END;

-- Poisson: non-negative counts, strictly positive predicted means.
CREATE TABLE pois AS SELECT i AS x1, (i % 5) AS y FROM range(200) t(i);
CREATE TABLE pois_m AS SELECT * FROM poisson_fit('pois', 'y');
SELECT CASE
    WHEN (SELECT bool_and(prediction > 0 AND NOT isinf(prediction))
          FROM poisson_predict('pois_m', 'pois'))
    THEN 'PASS  poisson predicts positive means'
    ELSE error('SMOKE FAIL: poisson prediction not positive/finite')
  END;

-- Gamma: strictly positive outcome, strictly positive predicted means.
CREATE TABLE gam AS SELECT i AS x1, (i % 5) + 1.0 AS y FROM range(200) t(i);
CREATE TABLE gam_m AS SELECT * FROM gamma_fit('gam', 'y');
SELECT CASE
    WHEN (SELECT bool_and(prediction > 0 AND NOT isinf(prediction))
          FROM gamma_predict('gam_m', 'gam'))
    THEN 'PASS  gamma predicts positive means'
    ELSE error('SMOKE FAIL: gamma prediction not positive/finite')
  END;

-- Evaluate: on the noiseless linear data R^2 is ~1 and RMSE ~0.
SELECT CASE
    WHEN (SELECT n FROM linreg_evaluate('lin_m', 'lin', 'y')) = 200
     AND (SELECT r2 FROM linreg_evaluate('lin_m', 'lin', 'y')) > 0.9999
     AND (SELECT rmse FROM linreg_evaluate('lin_m', 'lin', 'y')) < 1e-6
    THEN 'PASS  linreg_evaluate reports R^2 ~ 1 on noiseless data'
    ELSE error('SMOKE FAIL: linreg_evaluate metrics wrong on noiseless data')
  END;

-- Evaluate: logistic metrics are in valid ranges.
SELECT CASE
    WHEN (SELECT auc FROM logit_evaluate('clf_m', 'clf', 'y')) BETWEEN 0 AND 1
     AND (SELECT accuracy FROM logit_evaluate('clf_m', 'clf', 'y')) BETWEEN 0 AND 1
     AND (SELECT deviance FROM logit_evaluate('clf_m', 'clf', 'y')) > 0
    THEN 'PASS  logit_evaluate reports valid AUC/accuracy/deviance'
    ELSE error('SMOKE FAIL: logit_evaluate metrics out of range')
  END;

-- Offset/exposure: fit and predict with a log-exposure offset; predictions
-- stay positive and finite through the offset path.
CREATE TABLE expo AS SELECT i AS x1, ln(1.0 + (i % 5)) AS logexp, (i % 4) AS y FROM range(300) t(i);
CREATE TABLE expo_m AS SELECT * FROM poisson_fit('expo', 'y', offset_col := 'logexp');
SELECT CASE
    WHEN (SELECT bool_and(prediction > 0 AND NOT isinf(prediction))
          FROM poisson_predict('expo_m', 'expo', offset_col := 'logexp'))
    THEN 'PASS  poisson with offset predicts positive means'
    ELSE error('SMOKE FAIL: poisson offset prediction not positive/finite')
  END;

-- Sample weights: a constant weight column reproduces the unweighted fit on
-- the noiseless linear data (still recovers y = 2 + 3*x1 - x2).
CREATE TABLE linw AS SELECT x1, x2, 2.0 AS w, y FROM lin;
SELECT CASE
    WHEN abs((SELECT coefficient FROM linreg_fit('linw', 'y', weights_col := 'w') WHERE feature = 'x1') - 3.0) < 1e-6
     AND abs((SELECT coefficient FROM linreg_fit('linw', 'y', weights_col := 'w') WHERE feature = 'x2') + 1.0) < 1e-6
    THEN 'PASS  weighted fit (constant weights) matches unweighted'
    ELSE error('SMOKE FAIL: weighted linreg wrong with constant weights')
  END;

-- Tweedie (power 1.5): fits zero-inflated positive data (compound
-- Poisson-Gamma) and predicts strictly positive, finite means.
CREATE TABLE tw AS SELECT i AS x1, CASE WHEN i % 3 = 0 THEN 0.0 ELSE 1.0 + (i % 5) END AS y FROM range(300) t(i);
CREATE TABLE tw_m AS SELECT * FROM tweedie_fit('tw', 'y', power := 1.5);
SELECT CASE
    WHEN (SELECT bool_and(prediction > 0 AND NOT isinf(prediction)) FROM tweedie_predict('tw_m', 'tw'))
     AND (SELECT pseudo_r2 FROM tweedie_evaluate('tw_m', 'tw', 'y', power := 1.5)) IS NOT NULL
    THEN 'PASS  tweedie fits zero-inflated data, predicts positive means'
    ELSE error('SMOKE FAIL: tweedie prediction not positive/finite')
  END;

-- L1 (lasso): a strong penalty drives an irrelevant feature's coefficient to
-- exactly 0 (feature selection) while a relevant one survives.
CREATE TABLE l1t AS SELECT i AS x1, (i * 7 + 3) % 13 AS noise, 2 + 3.0 * i AS y FROM range(300) t(i);
SELECT CASE
    WHEN (SELECT coefficient FROM linreg_fit('l1t', 'y', l1 := 0.3) WHERE feature = 'noise') = 0.0
     AND (SELECT coefficient FROM linreg_fit('l1t', 'y', l1 := 0.3) WHERE feature = 'x1') != 0.0
    THEN 'PASS  L1 zeros an irrelevant feature, keeps a relevant one'
    ELSE error('SMOKE FAIL: L1 did not select features as expected')
  END;

-- dummy_encode_sql: generates SQL that drops the categorical column and adds
-- k-1 indicator columns (reference = first level). Two levels -> one dummy.
CREATE TABLE cats AS SELECT i AS x, (CASE WHEN i % 2 = 0 THEN 'a' ELSE 'b' END) AS g, 1.0 * i AS y FROM range(50) t(i);
SELECT CASE
    WHEN dummy_encode_sql('cats', 'y') = 'SELECT * EXCLUDE (g), (g = ''b'')::INT AS "g_b" FROM cats'
    THEN 'PASS  dummy_encode_sql generates R-style k-1 dummy encoding'
    ELSE error('SMOKE FAIL: dummy_encode_sql output = ' || dummy_encode_sql('cats', 'y'))
  END;

-- Multinomial (softmax): 3-class fit; reference class has zero coefficients,
-- predicted probabilities are in [0,1] and sum to 1 per row.
CREATE TABLE mc AS SELECT i AS x1, (i % 3) AS y FROM range(150) t(i);
CREATE TABLE mc_m AS SELECT * FROM multinom_fit('mc', 'y');
SELECT CASE
    WHEN (SELECT count(DISTINCT class) FROM mc_m) = 3
     AND (SELECT bool_and(coefficient = 0) FROM mc_m WHERE class = '0')  -- reference class
     AND (SELECT bool_and(abs(list_sum(map_values(probs)) - 1.0) < 1e-9)
          FROM multinom_predict('mc_m', 'mc'))
    THEN 'PASS  multinomial fits 3 classes, probs normalize per row'
    ELSE error('SMOKE FAIL: multinomial output invalid')
  END;

SELECT 'ALL SMOKE CHECKS PASSED' AS result;
