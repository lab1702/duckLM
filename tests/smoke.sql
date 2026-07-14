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

-- Negative binomial: fits overdispersed counts, predicts positive means, and
-- alpha -> 0 approaches the Poisson fit.
CREATE TABLE nb AS SELECT i AS x1, (i % 7) * (1 + (i % 3)) AS y FROM range(300) t(i);
CREATE TABLE nb_m AS SELECT * FROM nbinom_fit('nb', 'y', alpha := 0.5);
SELECT CASE
    WHEN (SELECT bool_and(prediction > 0 AND NOT isinf(prediction)) FROM nbinom_predict('nb_m', 'nb'))
     AND abs((SELECT coefficient FROM nbinom_fit('nb','y', alpha := 1e-8) WHERE feature='x1')
             - (SELECT coefficient FROM poisson_fit('nb','y') WHERE feature='x1')) < 1e-3
    THEN 'PASS  negative binomial fits, predicts positive, -> Poisson as alpha->0'
    ELSE error('SMOKE FAIL: negative binomial output invalid')
  END;

-- Cross-validation: cv_l2 returns one row per grid value; over-regularizing a
-- clean linear signal (large l2) yields a worse held-out deviance than l2=0.
CREATE TABLE cvt AS SELECT i AS x1, (i % 11) AS x2, 2.0 + 3 * i - 0.5 * (i % 11) + (i % 5) AS y FROM range(400) t(i);
SELECT CASE
    WHEN (SELECT count(*) FROM cv_l2('cvt', 'y', 'linear', [0.0, 0.1, 1.0, 100.0], k := 5)) = 4
     AND (SELECT cv_deviance FROM cv_l2('cvt', 'y', 'linear', [100.0], k := 5))
       > (SELECT cv_deviance FROM cv_l2('cvt', 'y', 'linear', [0.0], k := 5))
    THEN 'PASS  cv_l2 scores the ridge grid; over-shrinkage worsens held-out fit'
    ELSE error('SMOKE FAIL: cv_l2 output invalid')
  END;

-- CV over other hyperparameters: cv_power (Tweedie) and cv_alpha (neg-binom)
-- each return one finite deviance per grid value.
CREATE TABLE cvtw AS SELECT i AS x1, CASE WHEN i % 3 = 0 THEN 0.0 ELSE 1.0 + (i % 5) END AS y FROM range(300) t(i);
CREATE TABLE cvnb AS SELECT i AS x1, (i % 7) * (1 + (i % 3)) AS y FROM range(300) t(i);
SELECT CASE
    WHEN (SELECT count(*) FROM cv_power('cvtw', 'y', [1.3, 1.5, 1.7])) = 3
     AND (SELECT bool_and(cv_deviance > 0 AND NOT isinf(cv_deviance)) FROM cv_power('cvtw', 'y', [1.3, 1.5, 1.7]))
     AND (SELECT count(*) FROM cv_alpha('cvnb', 'y', [0.2, 0.5, 1.0])) = 3
     AND (SELECT count(*) FROM cv_l1('cvtw', 'y', 'poisson', [0.0, 0.1])) = 2
    THEN 'PASS  cv_power / cv_alpha / cv_l1 score their grids'
    ELSE error('SMOKE FAIL: cv_power/cv_alpha/cv_l1 output invalid')
  END;

-- NB dispersion estimation: nbinom_dispersion returns a finite profile
-- log-likelihood per alpha, and the grid argmax is a valid dispersion value.
SELECT CASE
    WHEN (SELECT count(*) FROM nbinom_dispersion('cvnb', 'y', [0.2, 0.5, 1.0, 2.0])) = 4
     AND (SELECT alpha FROM nbinom_dispersion('cvnb', 'y', [0.2, 0.5, 1.0, 2.0])
          ORDER BY loglik DESC LIMIT 1) > 0
    THEN 'PASS  nbinom_dispersion returns a profile-likelihood curve'
    ELSE error('SMOKE FAIL: nbinom_dispersion output invalid')
  END;

-- Two-stage refinement: reg_grid builds grids; the *_refine wrappers re-sweep a
-- finer grid (n_refine rows) within the coarse span, and NB refinement raises
-- the profile log-likelihood by zooming in on its peak.
SELECT CASE
    WHEN len(reg_grid(0.0, 1.0, 5)) = 5
     AND len(reg_grid(0.01, 100.0, 5, log_spaced := true)) = 5
     -- cv_l2_refine: 10 finite deviances on a grid zoomed below the coarse max
     AND (SELECT count(*) = 10 AND bool_and(cv_deviance > 0 AND NOT isinf(cv_deviance))
                              AND min(l2) >= 0.0 AND max(l2) < 100.0
          FROM cv_l2_refine('cvt', 'y', 'linear', [0.0, 0.1, 1.0, 100.0], k := 5))
     -- nbinom_dispersion_refine: 10 points, and refining sharpens the peak loglik
     AND (SELECT count(*) FROM nbinom_dispersion_refine('cvnb', 'y', [0.2, 0.5, 1.0, 2.0])) = 10
     AND (SELECT max(loglik) FROM nbinom_dispersion_refine('cvnb', 'y', [0.2, 0.5, 1.0, 2.0]))
       >= (SELECT max(loglik) FROM nbinom_dispersion('cvnb', 'y', [0.2, 0.5, 1.0, 2.0])) - 0.01
    THEN 'PASS  reg_grid + two-stage refinement search a finer grid and sharpen the optimum'
    ELSE error('SMOKE FAIL: grid refinement output invalid')
  END;

-- Inference: the norm/t utilities hit known quantiles; *_summary returns one
-- finite (SE, statistic, p, CI) row per coefficient with the CI bracketing the
-- estimate; fixed-dispersion families use the z critical value ~1.95996 while
-- estimated-dispersion families use a wider Student-t critical value.
CREATE TABLE infm_p AS SELECT * FROM poisson_fit('cvnb', 'y');
CREATE TABLE infm_l AS SELECT * FROM linreg_fit('cvt', 'y');
SELECT CASE
    WHEN abs(norm_ppf(0.975) - 1.959963984540054) < 1e-9
     AND abs(norm_cdf(1.959963984540054) - 0.975) < 1e-9
     AND abs(t_ppf(0.975, 1.0) - 12.706204736174699) < 1e-6
     AND abs(t_cdf(0.0, 5.0) - 0.5) < 1e-12
     -- poisson_summary (fixed dispersion): finite, CI brackets estimate, z crit ~1.96
     AND (SELECT count(*) FROM poisson_summary('infm_p', 'cvnb', 'y')) = 2
     AND (SELECT bool_and(std_error > 0 AND p_value BETWEEN 0.0 AND 1.0
                          AND conf_low < coefficient AND coefficient < conf_high
                          AND abs((conf_high - coefficient)/std_error - 1.959963984540054) < 1e-6)
          FROM poisson_summary('infm_p', 'cvnb', 'y'))
     -- linreg_summary (estimated dispersion): Student-t critical value wider than z
     AND (SELECT bool_and((conf_high - coefficient)/std_error > 1.9599640)
          FROM linreg_summary('infm_l', 'cvt', 'y'))
    THEN 'PASS  norm/t quantiles match; *_summary gives valid SE/stat/p/CI; t wider than z'
    ELSE error('SMOKE FAIL: inference output invalid')
  END;

-- Multinomial inference: multinom_summary returns one z-based row per estimated
-- (class, feature) for the K-1 non-reference classes, with finite SE and a CI
-- that brackets the coefficient.
CREATE TABLE mns AS SELECT (i%21-10)::DOUBLE AS x1, ((i*7)%13)::DOUBLE AS x2,
  CASE WHEN (i*3+i%5) % 3 = 0 THEN 'a' WHEN (i*3+i%5) % 3 = 1 THEN 'b' ELSE 'c' END AS y
  FROM range(800) g(i);
CREATE TABLE mns_m AS SELECT * FROM multinom_fit('mns', 'y');
SELECT CASE
    WHEN (SELECT count(*) FROM multinom_summary('mns_m', 'mns', 'y')) = 6           -- 2 classes x 3 coefs
     AND (SELECT count(DISTINCT class) FROM multinom_summary('mns_m', 'mns', 'y')) = 2  -- reference 'a' excluded
     AND (SELECT bool_and(std_error > 0 AND p_value BETWEEN 0.0 AND 1.0
                          AND conf_low < coefficient AND coefficient < conf_high
                          AND abs((conf_high - coefficient)/std_error - 1.959963984540054) < 1e-6)
          FROM multinom_summary('mns_m', 'mns', 'y'))
    THEN 'PASS  multinom_summary gives valid per-class SE/stat/p/CI (z, reference excluded)'
    ELSE error('SMOKE FAIL: multinom_summary output invalid')
  END;

-- IRLS solver (solver := 'irls'): Fisher scoring reaches the same coefficients
-- as the default gradient-descent solver, in far fewer iterations.
CREATE TABLE irdat AS SELECT (i%10-5)::DOUBLE AS x1, ((i*3)%7-3)::DOUBLE AS x2,
  (i%5 + i%3)::DOUBLE AS y FROM range(400) g(i);
SELECT CASE
    WHEN (SELECT max(abs(g.coefficient - r.coefficient))
          FROM poisson_fit('irdat','y') g JOIN poisson_fit('irdat','y', solver := 'irls') r USING (feature)) < 1e-5
     AND (SELECT max(abs(g.coefficient - r.coefficient))
          FROM linreg_fit('irdat','y') g JOIN linreg_fit('irdat','y', solver := 'irls') r USING (feature)) < 1e-6
    THEN 'PASS  irls solver reaches the same coefficients as gradient descent'
    ELSE error('SMOKE FAIL: irls solver disagrees with gradient descent')
  END;

SELECT 'ALL SMOKE CHECKS PASSED' AS result;
