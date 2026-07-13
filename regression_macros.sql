-- ============================================================================
-- Logistic, linear & Poisson regression as pure DuckDB (>= 1.5) SQL macros.
--
-- Public macros:
--   logit_fit(tbl, outcome, ...)    -> table (feature VARCHAR, coefficient DOUBLE)
--   logit_predict(model, tbl, ...)  -> input rows + prob DOUBLE + pred BOOLEAN
--   linreg_fit(tbl, outcome, ...)   -> table (feature VARCHAR, coefficient DOUBLE)
--   linreg_predict(model, tbl)      -> input rows + prediction DOUBLE
--   poisson_fit(tbl, outcome, ...)  -> table (feature VARCHAR, coefficient DOUBLE)
--   poisson_predict(model, tbl)     -> input rows + prediction DOUBLE (mean count)
--
-- Internal helpers (do not call directly, subject to change):
--   __reg_fit(tbl, outcome, family, caller, max_iter, learning_rate, tol)
--   __reg_score(model, tbl, caller)
--
-- All macros take *table names as strings* and resolve them via query_table(),
-- so they work on any table or view visible in the current connection
-- (schema- and catalog-qualified names like 's.tbl' or 'db.s.tbl' included).
--
-- The fit macros train on every column of the input table except the outcome
-- column, using Nesterov-accelerated gradient descent inside a recursive CTE.
-- Features (and, for linear regression, the outcome) are z-score standardized
-- internally for conditioning; returned coefficients are transformed back to
-- the original scale, so results match R's glm()/lm(), statsmodels, or
-- unpenalized scikit-learn up to convergence tolerance.
--
-- Requirements / behavior:
--   * All columns must be castable to DOUBLE (numeric or boolean).
--   * logit_fit outcomes must be binary: 0/1 or boolean.
--   * poisson_fit outcomes must be non-negative (counts; log link, so
--     poisson_predict returns the predicted mean count exp(score)).
--   * Rows containing NULL in the outcome or any feature are dropped.
--   * Constant (zero-variance) columns get coefficient 0.
--   * Column AND table names beginning with "__reg_" (any case) are reserved.
--   * Prediction matches model features to columns by exact, case-sensitive
--     name string.
--   * Perfectly separable logit data has no finite MLE; the fit terminates at
--     max_iter with large coefficients rather than erroring.
--
-- Fit parameters:
--   max_iter      := 50000   hard cap on gradient iterations
--   learning_rate := NULL    step size on the standardized scale; NULL picks
--                            a convergent default (4/(d+1) logistic, 1/(d+1)
--                            linear and Poisson; Poisson steps are additionally
--                            damped each iteration by the largest fitted mean,
--                            since its curvature is unbounded)
--   tol           := 1e-10   stop when no coefficient moves more than this
--                            (standardized scale) between iterations
--
-- Example:
--   CREATE TABLE model AS SELECT * FROM linreg_fit('sales', 'revenue');
--   SELECT * FROM linreg_predict('model', 'pipeline');
-- ============================================================================

-- NOTE on naming: every internal CTE is prefixed __reg_ because query_table()
-- resolves bare table names against CTEs already defined in the enclosing
-- WITH; an internal CTE named e.g. "long" would shadow a user table of the
-- same name. The __reg_ prefix is reserved (and enforced below) so that
-- shadowing cannot happen.

CREATE OR REPLACE MACRO __reg_fit(tbl, outcome, family, caller, max_iter, learning_rate, tol) AS TABLE
WITH RECURSIVE
-- Every column cast to DOUBLE, with a synthetic row id.
__reg_wide AS MATERIALIZED (
    SELECT row_number() OVER () AS __reg_rid__, CAST(COLUMNS(*) AS DOUBLE)
    FROM query_table(tbl)
),
-- Long form: one row per (row, column). UNPIVOT drops NULL cells, which is
-- how incomplete rows get detected and excluded below.
__reg_long AS MATERIALIZED (
    SELECT __reg_rid__ AS rid, name AS col, value AS v
    FROM (UNPIVOT __reg_wide ON COLUMNS(* EXCLUDE (__reg_rid__)) INTO NAME name VALUE value)
),
__reg_ycheck AS (
    SELECT CASE
             WHEN count(*) = 0
               THEN error(caller || ': outcome column "' || outcome || '" not found, entirely NULL, or table is empty')
             WHEN family = 'logistic' AND NOT bool_and(v IN (0.0, 1.0))
               THEN error(caller || ': outcome column "' || outcome || '" must be binary (0/1 or boolean)')
             WHEN family = 'poisson' AND min(v) < 0
               THEN error(caller || ': outcome column "' || outcome || '" must be non-negative for Poisson regression')
             ELSE true
           END AS ok
    FROM __reg_long
    WHERE col = outcome
),
-- Every column name of the input table, even when the table has zero rows:
-- the sampled row is LEFT JOINed onto a constant row so the UNPIVOT always
-- has one row to enumerate (aggregates over an empty input would otherwise
-- return NULL and silently skip the checks below).
__reg_cols AS (
    SELECT colname
    FROM (SELECT *
          FROM (SELECT 1 AS __reg_one)
          LEFT JOIN (SELECT CAST(COLUMNS(*) AS VARCHAR) FROM query_table(tbl) LIMIT 1) ON true)
         UNPIVOT INCLUDE NULLS (v FOR colname IN (COLUMNS(* EXCLUDE (__reg_one))))
),
-- Reject names that would silently collide with internals. Identifier
-- comparisons are case-folded because DuckDB identifiers are case-insensitive.
__reg_namecheck AS (
    SELECT CASE
             WHEN starts_with(lower(tbl), '__reg_')
               THEN error(caller || ': table names beginning with "__reg_" are reserved for internal use; please rename')
             WHEN coalesce(bool_or(starts_with(lower(colname), '__reg_')), false)
               THEN error(caller || ': column names beginning with "__reg_" are reserved for internal use; please rename')
             WHEN coalesce(bool_or(colname = '(Intercept)' AND colname != outcome), false)
               THEN error(caller || ': a feature column named "(Intercept)" would collide with the intercept row in the model output; please rename it')
             ELSE true
           END AS ok
    FROM __reg_cols
),
-- Standardization stats per feature. Zero-variance columns get sigma = 1 so
-- they pass through harmlessly (their z-score is 0 => coefficient stays 0).
__reg_stats AS MATERIALIZED (
    SELECT col,
           avg(v) AS mu,
           CASE WHEN coalesce(stddev_pop(v), 0) < 1e-300 THEN 1.0 ELSE stddev_pop(v) END AS sigma
    FROM __reg_long
    WHERE col != outcome
    GROUP BY col
),
__reg_feats AS MATERIALIZED (
    SELECT list(col   ORDER BY col) AS names,
           list(mu    ORDER BY col) AS mus,
           list(sigma ORDER BY col) AS sigmas,
           count(*)::INT            AS d
    FROM __reg_stats
),
-- Outcome standardization: identity for logistic (y is 0/1), z-score for
-- linear so the optimization and tol are scale-free in y, and mean-scaling
-- for Poisson. Dividing a Poisson outcome by its mean shifts only the
-- intercept (by ln(mean), undone in the back-transform below) and makes the
-- optimizer start at fitted means ~= 1 from the zero initialization.
__reg_ystats AS (
    SELECT CASE WHEN family = 'linear' THEN avg(v) ELSE 0.0 END AS mu_y,
           CASE WHEN family = 'logistic' THEN 1.0
                WHEN family = 'poisson'
                  THEN (CASE WHEN avg(v) < 1e-300 THEN 1.0 ELSE avg(v) END)
                WHEN coalesce(stddev_pop(v), 0) < 1e-300 THEN 1.0
                ELSE stddev_pop(v)
           END AS sd_y
    FROM __reg_long
    WHERE col = outcome
),
-- The whole training set packed into one row: a list of {y, xs} structs where
-- xs = [1.0 (intercept), standardized features in name order].
__reg_packed AS MATERIALIZED (
    SELECT list(struct_pack(y := y, xs := xs)) AS rows, count(*)::DOUBLE AS n
    FROM (
        SELECT x.rid,
               (any_value(yv.v) - any_value(ys.mu_y)) / any_value(ys.sd_y) AS y,
               [1.0::DOUBLE] || list((x.v - s.mu) / s.sigma ORDER BY x.col) AS xs
        FROM __reg_long x
        JOIN __reg_stats s  ON s.col = x.col
        JOIN __reg_long  yv ON yv.rid = x.rid AND yv.col = outcome
        CROSS JOIN __reg_ystats ys
        WHERE x.col != outcome
        GROUP BY x.rid
        HAVING count(*) = (SELECT d FROM __reg_feats)   -- complete rows only
    )
),
__reg_cfg AS (
    SELECT CASE
             WHEN f.d = 0 THEN error(caller || ': no feature columns besides the outcome')
             -- A feature column with no non-NULL values never reaches stats;
             -- silently fitting without it would surprise, so reject it (as
             -- R's glm() and scikit-learn do).
             WHEN (SELECT count(*) FROM __reg_cols WHERE colname != outcome) != f.d
               THEN error(caller || ': feature column(s) entirely NULL: '
                          || (SELECT string_agg('"' || colname || '"', ', ')
                              FROM __reg_cols
                              WHERE colname != outcome
                                AND colname NOT IN (SELECT col FROM __reg_stats))
                          || '; drop them (e.g. SELECT * EXCLUDE (...)) or fill them')
             WHEN p.n = 0 THEN error(caller || ': no complete (non-NULL) rows to train on')
             -- Guaranteed-convergent steps: on standardized data the mean-loss
             -- gradient is L-Lipschitz with L <= (d+1)/4 (logistic, from the
             -- sigmoid derivative bound) or L <= d+1 (linear). Poisson has no
             -- global Lipschitz bound; its base step 1/(d+1) is locally safe at
             -- the mean-scaled start (fitted means ~= 1) and is damped each
             -- iteration by the largest fitted mean in the loop below.
             ELSE coalesce(learning_rate,
                           CASE WHEN family = 'logistic' THEN 4.0 ELSE 1.0 END / (f.d + 1))
           END AS step
    FROM __reg_feats f, __reg_packed p, __reg_ycheck y, __reg_namecheck nc
    WHERE y.ok AND nc.ok
),
-- Nesterov-accelerated gradient descent on the mean loss (negative
-- log-likelihood for logistic, squared error / 2 for linear).
-- The early-stop criterion is the size of the pure gradient step,
-- move = step * max_j |grad_j|, NOT the iterate displacement |betas - prev|:
-- momentum can make two consecutive iterates coincide exactly at a
-- non-stationary point (deterministically so for exactly collinear features),
-- while the gradient step is zero only at a true stationary point.
__reg_gd AS (
    SELECT 0 AS it,
           list_transform(range(f.d + 1), lambda i: 0.0::DOUBLE) AS betas,
           list_transform(range(f.d + 1), lambda i: 0.0::DOUBLE) AS prev,
           1e308::DOUBLE AS move
    FROM __reg_feats f
    UNION ALL
    SELECT it + 1,
           newbetas,
           betas,
           list_aggregate(list_transform(newbetas, lambda nb, j: abs(nb - look[j])), 'max')
    FROM (
        SELECT it, betas, look,
               -- beta_j <- look_j + (step/damp) * (1/n) * sum_i xs_ij * r_i
               list_transform(look, lambda b, j:
                   b + (step / damp) * list_sum(list_transform(res, lambda ob: ob.xs[j] * ob.r)) / n) AS newbetas
        FROM (
            SELECT it, betas, n, step, look, res,
                   -- Poisson curvature grows with the fitted mean exp(z), so
                   -- damp the step by the largest fitted mean (recovered as
                   -- y - r without recomputing exp). 1 for other families.
                   CASE WHEN family = 'poisson'
                        THEN greatest(1.0, list_aggregate(
                               list_transform(res, lambda ob, i: rows[i].y - ob.r), 'max'))
                        ELSE 1.0
                   END AS damp
            FROM (
                SELECT it, betas, rows, n, step, look,
                       -- residual per training row: y - yhat(xs . look)
                       list_transform(rows, lambda rw: struct_pack(
                           xs := rw.xs,
                           r  := rw.y - CASE WHEN family = 'logistic'
                                             THEN 1.0 / (1.0 + exp(-list_dot_product(rw.xs, look)))
                                             WHEN family = 'poisson'
                                             THEN exp(least(list_dot_product(rw.xs, look), 700.0))
                                             ELSE list_dot_product(rw.xs, look)
                                        END)) AS res
                FROM (
                    SELECT g.it, g.betas, p.rows, p.n, c.step,
                           -- Nesterov lookahead point
                           list_transform(g.betas, lambda b, j:
                               b + (g.it::DOUBLE / (g.it + 3)) * (b - g.prev[j])) AS look
                    FROM __reg_gd g, __reg_packed p, __reg_cfg c
                    WHERE g.it < max_iter AND g.move >= tol
                )
            )
        )
    )
),
__reg_sol AS (
    SELECT betas FROM __reg_gd ORDER BY it DESC LIMIT 1
)
-- Map standardized-scale coefficients back to the original scales of x and y.
-- Linear/logistic: the outcome scaling multiplies the whole linear predictor,
-- so slopes scale by sd_y and the intercept by sd_y plus the mu_y shift.
-- Poisson: y was divided by its mean, i.e. E[y] = sd_y * exp(z_std), which is
-- exp(z_std + ln(sd_y)) — only the intercept shifts; slopes are unchanged.
SELECT feature, coefficient
FROM (
    SELECT '(Intercept)' AS feature,
           CASE WHEN family = 'poisson'
                THEN ln(ys.sd_y) + (s.betas[1] - coalesce(list_sum(list_transform(f.names,
                     lambda nm, j: s.betas[j + 1] * f.mus[j] / f.sigmas[j])), 0.0))
                ELSE ys.mu_y + ys.sd_y * (s.betas[1] - coalesce(list_sum(list_transform(f.names,
                     lambda nm, j: s.betas[j + 1] * f.mus[j] / f.sigmas[j])), 0.0))
           END AS coefficient
    FROM __reg_sol s, __reg_feats f, __reg_ystats ys
    UNION ALL
    SELECT unnest(f.names),
           unnest(list_transform(f.names, lambda nm, j:
               (CASE WHEN family = 'poisson' THEN 1.0 ELSE ys.sd_y END) * s.betas[j + 1] / f.sigmas[j]))
    FROM __reg_sol s, __reg_feats f, __reg_ystats ys
)
ORDER BY (feature = '(Intercept)') DESC, feature;


-- Shared scorer: returns the input rows plus the linear score
-- __reg_score__ = intercept + sum(coefficient * feature value), NULL for rows
-- where a model feature is missing or NULL.
CREATE OR REPLACE MACRO __reg_score(model, tbl, caller) AS TABLE
WITH
__reg_numbered AS MATERIALIZED (
    SELECT row_number() OVER () AS __reg_rid__, *
    FROM query_table(tbl)
),
-- Long form of the scoring data. TRY_CAST: non-numeric columns become NULL
-- and are dropped by UNPIVOT; they only matter if the model references them.
__reg_long AS MATERIALIZED (
    SELECT __reg_rid__ AS rid, name AS col, value AS v
    FROM (UNPIVOT (SELECT __reg_rid__, TRY_CAST(COLUMNS(* EXCLUDE (__reg_rid__)) AS DOUBLE) FROM __reg_numbered)
          ON COLUMNS(* EXCLUDE (__reg_rid__)) INTO NAME name VALUE value)
),
__reg_coefs AS MATERIALIZED (
    SELECT feature, coefficient FROM query_table(model)
),
-- Reject names that would silently collide with internals or outputs.
-- Identifier comparisons are case-folded (DuckDB identifiers are
-- case-insensitive, so an input column "PREDICTION" would still capture a
-- downstream "SELECT prediction"). The sampled row is LEFT JOINed onto a
-- constant row so the check still sees every column name when the scoring
-- table has zero rows (bool_or over an empty input is NULL and would
-- silently skip the checks, leaking a malformed output schema).
__reg_namecheck AS (
    SELECT CASE
             WHEN starts_with(lower(tbl), '__reg_') OR starts_with(lower(model), '__reg_')
               THEN error(caller || ': table names beginning with "__reg_" are reserved for internal use; please rename')
             WHEN coalesce(bool_or(starts_with(lower(colname), '__reg_')), false)
               THEN error(caller || ': column names beginning with "__reg_" are reserved for internal use; please rename')
             WHEN caller = 'logit_predict' AND coalesce(bool_or(lower(colname) IN ('prob', 'pred')), false)
               THEN error('logit_predict: the input table already has a "prob" or "pred" column, which collides with the output columns; rename or drop it first (e.g. SELECT * EXCLUDE (prob, pred))')
             WHEN caller IN ('linreg_predict', 'poisson_predict') AND coalesce(bool_or(lower(colname) = 'prediction'), false)
               THEN error(caller || ': the input table already has a "prediction" column, which collides with the output column; rename or drop it first (e.g. SELECT * EXCLUDE (prediction))')
             ELSE true
           END AS ok
    FROM (SELECT *
          FROM (SELECT 1 AS __reg_one)
          LEFT JOIN (SELECT CAST(COLUMNS(*) AS VARCHAR) FROM query_table(tbl) LIMIT 1) ON true)
         UNPIVOT INCLUDE NULLS (v FOR colname IN (COLUMNS(* EXCLUDE (__reg_one))))
),
__reg_meta AS (
    SELECT coalesce((SELECT coefficient FROM __reg_coefs WHERE feature = '(Intercept)'), 0.0) AS b0,
           (SELECT count(*) FROM __reg_coefs WHERE feature != '(Intercept)')                  AS k
    FROM __reg_namecheck
    WHERE ok
),
-- Score is NULL for any row where a model feature is missing or NULL.
__reg_scores AS (
    SELECT l.rid,
           CASE WHEN count(*) = m.k
                THEN m.b0 + sum(c.coefficient * l.v)
           END AS z
    FROM __reg_coefs c
    JOIN __reg_long l ON l.col = c.feature
    CROSS JOIN __reg_meta m
    WHERE c.feature != '(Intercept)'
    GROUP BY l.rid, m.b0, m.k
)
SELECT n.*,
       coalesce(s.z, CASE WHEN m.k = 0 THEN m.b0 END) AS __reg_score__
FROM __reg_numbered n
CROSS JOIN __reg_meta m
LEFT JOIN __reg_scores s ON s.rid = n.__reg_rid__;


-- ---------------------------------------------------------------------------
-- Public wrappers
-- ---------------------------------------------------------------------------

CREATE OR REPLACE MACRO logit_fit(tbl, outcome, max_iter := 50000, learning_rate := NULL, tol := 1e-10) AS TABLE
SELECT * FROM __reg_fit(tbl, outcome, 'logistic', 'logit_fit', max_iter, learning_rate, tol);

CREATE OR REPLACE MACRO linreg_fit(tbl, outcome, max_iter := 50000, learning_rate := NULL, tol := 1e-10) AS TABLE
SELECT * FROM __reg_fit(tbl, outcome, 'linear', 'linreg_fit', max_iter, learning_rate, tol);

CREATE OR REPLACE MACRO logit_predict(model, tbl, threshold := 0.5) AS TABLE
SELECT * EXCLUDE (__reg_rid__, __reg_score__),
       1.0 / (1.0 + exp(-__reg_score__)) AS prob,
       1.0 / (1.0 + exp(-__reg_score__)) >= threshold AS pred
FROM __reg_score(model, tbl, 'logit_predict')
ORDER BY __reg_rid__;

CREATE OR REPLACE MACRO linreg_predict(model, tbl) AS TABLE
SELECT * EXCLUDE (__reg_rid__, __reg_score__),
       __reg_score__ AS prediction
FROM __reg_score(model, tbl, 'linreg_predict')
ORDER BY __reg_rid__;

CREATE OR REPLACE MACRO poisson_fit(tbl, outcome, max_iter := 50000, learning_rate := NULL, tol := 1e-10) AS TABLE
SELECT * FROM __reg_fit(tbl, outcome, 'poisson', 'poisson_fit', max_iter, learning_rate, tol);

CREATE OR REPLACE MACRO poisson_predict(model, tbl) AS TABLE
SELECT * EXCLUDE (__reg_rid__, __reg_score__),
       exp(__reg_score__) AS prediction
FROM __reg_score(model, tbl, 'poisson_predict')
ORDER BY __reg_rid__;
