-- ============================================================================
-- Logistic, linear, Poisson, Gamma & Tweedie regression as pure DuckDB
-- (>= 1.5) SQL macros, with optional ridge/lasso/elastic-net regularization,
-- an offset/exposure term, and sample weights.
--
-- Public macros (15): fit / predict / evaluate for each family.
--   {logit,linreg,poisson,gamma}_fit(tbl, outcome, ...)
--   tweedie_fit(tbl, outcome, power := 1.5, ...)
--       -> table (feature VARCHAR, coefficient DOUBLE), with an '(Intercept)' row
--   logit_predict(model, tbl, threshold := 0.5, offset_col := NULL)
--       -> input rows + prob DOUBLE + pred BOOLEAN
--   {linreg,poisson,gamma,tweedie}_predict(model, tbl, offset_col := NULL)
--       -> input rows + prediction DOUBLE (linear: score; log-link: exp(score))
--   {logit,linreg,poisson,gamma}_evaluate(model, tbl, outcome, offset_col := NULL)
--   tweedie_evaluate(model, tbl, outcome, power := 1.5, offset_col := NULL)
--       -> one-row table of goodness-of-fit metrics
--
-- Fit macros accept: max_iter, learning_rate, tol, l2, offset_col, weights_col
-- (tweedie_fit also power); see "Fit parameters" below.
--
-- Internal helpers (do not call directly, subject to change):
--   __reg_fit(tbl, outcome, family, caller, max_iter, learning_rate, tol, l2,
--             offset_col, weights_col, power, l1)
--   __reg_score(model, tbl, caller, offset_col)
--   __reg_eval(model, tbl, outcome, family, caller, offset_col, power)
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
--   * gamma_fit outcomes must be strictly positive (log link, like R's
--     glm(family = Gamma(link = "log")) or sklearn's GammaRegressor).
--   * tweedie_fit takes a variance power p (>= 1): p=1 Poisson, p=2 Gamma,
--     1<p<2 the compound Poisson-Gamma admitting exact zeros. Outcomes must be
--     non-negative (strictly positive for p >= 2). Matches sklearn
--     TweedieRegressor(power=p, alpha=0, link='log').
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
--                            a convergent default (4/(d+1+4*l2) logistic,
--                            1/(d+1+l2) otherwise; Poisson/Gamma steps are
--                            additionally damped each iteration by the largest
--                            curvature weight, since theirs is unbounded)
--   tol           := 1e-10   stop when the gradient step is smaller than this
--   offset_col    := NULL    name of a column added to the linear predictor
--                            eta = offset + x.beta with a fixed coefficient of
--                            1 (not fit, not penalized); pass the same
--                            offset_col to predict/evaluate. Standard for rate
--                            models, e.g. a log(exposure) offset. Matches R /
--                            statsmodels GLM offset=.
--   weights_col   := NULL    name of a column of non-negative per-row sample
--                            weights; the loss and internal standardization are
--                            weighted by them. Integer weights == replicating
--                            each row that many times. Matches sklearn
--                            sample_weight. Applies to fitting only.
--   l2            := 0.0     ridge penalty (l2/2)*sum(beta_j^2) added to the
--                            MEAN loss of the internally standardized problem,
--                            intercept unpenalized (glmnet-style: penalizing
--                            standardized coefficients makes the strength
--                            scale-invariant). Exact sklearn equivalents, all
--                            fit on z-scored features (population sd) with the
--                            outcome transformed as this macro does internally:
--                              linear:   Ridge(alpha = n*l2) on (X_std, y_zscored)
--                              logistic: LogisticRegression(C = 1/(n*l2)) on X_std
--                              poisson:  PoissonRegressor(alpha = l2) on (X_std, y/mean(y))
--                              gamma:    GammaRegressor(alpha = l2) on (X_std, y/mean(y))
--                            l2 > 0 also gives perfectly separable logistic
--                            data a finite optimum (fast convergence instead
--                            of running to max_iter).
--   l1            := 0.0     lasso penalty l1*sum(|beta_j|) on the standardized
--                            coefficients (intercept unpenalized), applied via
--                            a FISTA soft-threshold prox so coefficients are
--                            driven to EXACTLY zero (feature selection).
--                            Combine with l2 for elastic net. Linear matches
--                            Lasso(alpha=l1) / ElasticNet(alpha=l1+l2,
--                            l1_ratio=l1/(l1+l2)) on the standardized problem.
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

CREATE OR REPLACE MACRO __reg_fit(tbl, outcome, family, caller, max_iter, learning_rate, tol, l2, offset_col, weights_col, power, l1) AS TABLE
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
             WHEN family = 'gamma' AND min(v) <= 0
               THEN error(caller || ': outcome column "' || outcome || '" must be strictly positive for Gamma regression')
             WHEN family = 'tweedie' AND power < 1
               THEN error(caller || ': power must be >= 1 (1<power<2 for zero-inflated positive data; use linreg_fit for power=0)')
             WHEN family = 'tweedie' AND power >= 2 AND min(v) <= 0
               THEN error(caller || ': outcome column "' || outcome || '" must be strictly positive for Tweedie power >= 2')
             WHEN family = 'tweedie' AND min(v) < 0
               THEN error(caller || ': outcome column "' || outcome || '" must be non-negative for Tweedie regression')
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
-- Feature columns that have at least one non-NULL value anywhere. Used for
-- the "entirely NULL column" check and to define a complete row. The outcome
-- and the optional offset / weights columns are not features.
-- coalesce(col, '') turns "not supplied" into a name no real column can have.
__reg_featcols AS (
    SELECT DISTINCT col AS colname FROM __reg_long
    WHERE col != outcome
      AND col != coalesce(offset_col, '')
      AND col != coalesce(weights_col, '')
),
-- Number of columns every complete row must have non-NULL: features + outcome
-- + offset + weights (each when supplied).
__reg_reqn AS (
    SELECT (SELECT count(*) FROM __reg_featcols) + 1
           + CASE WHEN offset_col IS NOT NULL THEN 1 ELSE 0 END
           + CASE WHEN weights_col IS NOT NULL THEN 1 ELSE 0 END AS req
),
-- Rows used for training: every feature column, the outcome, the offset and
-- the weight (each if any) non-NULL. Standardization (below) is computed over
-- exactly these rows -- and weighted by the sample weights -- so that ridge,
-- whose penalty is on the standardized coefficients, matches "drop the NULL
-- rows, then (weighted-)standardize". For unpenalized fits with no NULLs this
-- changes nothing (the fit is invariant to the standardization scale, and with
-- no NULLs every row is complete).
__reg_complete AS (
    SELECT rid
    FROM __reg_long
    WHERE col = outcome
       OR col = coalesce(offset_col, '')
       OR col = coalesce(weights_col, '')
       OR col IN (SELECT colname FROM __reg_featcols)
    GROUP BY rid
    HAVING count(*) = (SELECT req FROM __reg_reqn)
),
__reg_clong AS MATERIALIZED (
    SELECT l.rid, l.col, l.v
    FROM __reg_long l SEMI JOIN __reg_complete c ON c.rid = l.rid
),
-- Sample weight per complete row (1.0 when no weights column is given).
__reg_w AS (
    SELECT c.rid, coalesce(wv.v, 1.0) AS w
    FROM __reg_complete c
    LEFT JOIN __reg_clong wv ON wv.rid = c.rid AND wv.col = coalesce(weights_col, '')
),
-- Standardization stats per feature (over complete rows). Constant columns
-- (detected exactly by min = max) get mu = min(v) and sigma = 1: centering on
-- an actual stored value makes every z-score exactly 0 (avg() of a
-- non-representable constant like 4.2 is not bit-exact, which would otherwise
-- leave a ~1e-16 z-score and drift the coefficient off 0), so the coefficient
-- stays exactly 0.
-- Weighted mean and weighted population sd. With equal weights these reduce to
-- the ordinary avg / stddev_pop, so no-weights fits are unchanged.
__reg_stats AS MATERIALIZED (
    SELECT s.col,
           CASE WHEN min(s.v) = max(s.v) THEN min(s.v)
                ELSE sum(w.w * s.v) / sum(w.w) END AS mu,
           CASE WHEN min(s.v) = max(s.v) THEN 1.0
                ELSE sqrt(greatest(sum(w.w * s.v * s.v) / sum(w.w)
                                   - (sum(w.w * s.v) / sum(w.w)) ^ 2, 0.0)) END AS sigma
    FROM __reg_clong s JOIN __reg_w w ON w.rid = s.rid
    WHERE s.col != outcome
      AND s.col != coalesce(offset_col, '')
      AND s.col != coalesce(weights_col, '')
    GROUP BY s.col
),
__reg_feats AS MATERIALIZED (
    SELECT list(col   ORDER BY col) AS names,
           list(mu    ORDER BY col) AS mus,
           list(sigma ORDER BY col) AS sigmas,
           count(*)::INT            AS d
    FROM __reg_stats
),
-- Outcome standardization (over complete rows): identity for logistic (y is
-- 0/1), z-score for linear so the optimization and tol are scale-free in y,
-- and mean-scaling for the log-link families (Poisson/Gamma). Dividing a
-- log-link outcome by its mean shifts only the intercept (by ln(mean), undone
-- in the back-transform below) and makes the optimizer start at fitted means
-- ~= 1 from the zero initialization.
__reg_ystats AS (
    SELECT CASE WHEN family = 'linear' THEN sum(w.w * s.v) / sum(w.w) ELSE 0.0 END AS mu_y,
           CASE WHEN family = 'logistic' THEN 1.0
                WHEN family IN ('poisson', 'gamma', 'tweedie')
                  THEN (CASE WHEN sum(w.w * s.v) / sum(w.w) < 1e-300 THEN 1.0
                             ELSE sum(w.w * s.v) / sum(w.w) END)
                ELSE (CASE WHEN sqrt(greatest(sum(w.w * s.v * s.v) / sum(w.w)
                                              - (sum(w.w * s.v) / sum(w.w)) ^ 2, 0.0)) < 1e-300
                           THEN 1.0
                           ELSE sqrt(greatest(sum(w.w * s.v * s.v) / sum(w.w)
                                              - (sum(w.w * s.v) / sum(w.w)) ^ 2, 0.0)) END)
           END AS sd_y
    FROM __reg_clong s JOIN __reg_w w ON w.rid = s.rid
    WHERE s.col = outcome
),
-- The whole training set packed into one row: a list of {y, xs} structs where
-- xs = [1.0 (intercept), standardized features in name order]. __reg_clong is
-- already restricted to complete rows.
-- Each row carries y (transformed), xs (standardized features), and o, the
-- internal offset. An offset is a known per-row term in the linear predictor
-- eta = o + xs.beta; it is not fit and not penalized. For the log-link
-- families it passes through unchanged (dividing y by its mean only shifts the
-- intercept); for linear the outcome is z-scored, so the offset is divided by
-- sd_y to live on the same scale. o = 0 when no offset column is given.
-- Each row carries y (transformed), xs (standardized features), o (internal
-- offset), and w (sample weight). sumw is the total weight; the gradient is
-- (1/sumw) * sum_i w_i xs_ij r_i.
__reg_packed AS MATERIALIZED (
    SELECT list(struct_pack(y := y, xs := xs, o := o, w := w)) AS rows,
           count(*)::DOUBLE AS n,
           sum(w) AS sumw
    FROM (
        SELECT x.rid,
               (any_value(yv.v) - any_value(ys.mu_y)) / any_value(ys.sd_y) AS y,
               [1.0::DOUBLE] || list((x.v - s.mu) / s.sigma ORDER BY x.col) AS xs,
               coalesce(any_value(ov.v), 0.0)
                 / (CASE WHEN family = 'linear' THEN any_value(ys.sd_y) ELSE 1.0 END) AS o,
               any_value(wt.w) AS w
        FROM __reg_clong x
        JOIN __reg_stats s  ON s.col = x.col
        JOIN __reg_clong yv ON yv.rid = x.rid AND yv.col = outcome
        JOIN __reg_w wt     ON wt.rid = x.rid
        LEFT JOIN __reg_clong ov ON ov.rid = x.rid AND ov.col = coalesce(offset_col, '')
        CROSS JOIN __reg_ystats ys
        WHERE x.col != outcome
          AND x.col != coalesce(offset_col, '')
          AND x.col != coalesce(weights_col, '')
        GROUP BY x.rid
    )
),
__reg_cfg AS (
    SELECT CASE
             WHEN offset_col IS NOT NULL
                    AND offset_col NOT IN (SELECT colname FROM __reg_cols)
               THEN error(caller || ': offset column "' || offset_col || '" not found')
             WHEN weights_col IS NOT NULL
                    AND weights_col NOT IN (SELECT colname FROM __reg_cols)
               THEN error(caller || ': weights column "' || weights_col || '" not found')
             WHEN (SELECT count(*) FROM __reg_cols
                   WHERE colname != outcome
                     AND colname != coalesce(offset_col, '')
                     AND colname != coalesce(weights_col, '')) = 0
               THEN error(caller || ': no feature columns besides the outcome')
             -- A feature column with no non-NULL values anywhere never reaches
             -- __reg_featcols; silently fitting without it would surprise, so
             -- reject it (as R's glm() and scikit-learn do). Checked on an
             -- all-rows basis so it fires independently of complete-row count.
             WHEN (SELECT count(*) FROM __reg_featcols)
                    != (SELECT count(*) FROM __reg_cols
                        WHERE colname != outcome
                          AND colname != coalesce(offset_col, '')
                          AND colname != coalesce(weights_col, ''))
               THEN error(caller || ': feature column(s) entirely NULL: '
                          || (SELECT string_agg('"' || colname || '"', ', ')
                              FROM __reg_cols
                              WHERE colname != outcome
                                AND colname != coalesce(offset_col, '')
                                AND colname != coalesce(weights_col, '')
                                AND colname NOT IN (SELECT colname FROM __reg_featcols))
                          || '; drop them (e.g. SELECT * EXCLUDE (...)) or fill them')
             WHEN p.n = 0 THEN error(caller || ': no complete (non-NULL) rows to train on')
             WHEN weights_col IS NOT NULL AND (SELECT min(w) FROM __reg_w) < 0
               THEN error(caller || ': weights must be non-negative')
             WHEN weights_col IS NOT NULL AND (SELECT coalesce(sum(w), 0) FROM __reg_w) <= 0
               THEN error(caller || ': sample weights sum to zero')
             WHEN l2 < 0 THEN error(caller || ': l2 must be >= 0, got ' || l2)
             WHEN l1 < 0 THEN error(caller || ': l1 must be >= 0, got ' || l1)
             -- Guaranteed-convergent steps: on standardized data the mean-loss
             -- gradient is L-Lipschitz with L <= (d+1)/4 (logistic, from the
             -- sigmoid derivative bound) or L <= d+1 (linear), plus l2 from the
             -- ridge penalty. Poisson/Gamma have no global Lipschitz bound;
             -- their base step is locally safe at the mean-scaled start
             -- (fitted means ~= 1, curvature weights ~= 1) and is damped each
             -- iteration by the largest curvature weight in the loop below.
             ELSE coalesce(learning_rate,
                           CASE WHEN family = 'logistic'
                                THEN 4.0 / (f.d + 1 + 4.0 * l2)
                                ELSE 1.0 / (f.d + 1 + l2)
                           END)
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
               -- L1 proximal step (soft-threshold): beta_j <- prox_{t*l1}(z_j),
               -- prox(z) = sign(z) * max(|z| - t*l1, 0), zeroing small coefficients
               -- exactly (feature selection). The intercept is never penalized; a
               -- no-op when l1 = 0, so unpenalized / pure-ridge fits are unchanged.
               list_transform(zstep, lambda zj, j:
                   CASE WHEN j = 1 THEN zj
                        ELSE sign(zj) * greatest(abs(zj) - threshl1, 0.0) END) AS newbetas
        FROM (
            SELECT it, betas, look, (step / damp) * l1 AS threshl1,
                   -- smooth-part gradient step z = look + (step/damp)*(grad - l2*look);
                   -- (1/sumw) sum_i w_i xs_ij r_i is the mean-loss gradient, and the
                   -- smooth L2 gradient l2*look_j stays here (not in the prox).
                   list_transform(look, lambda b, j:
                       b + (step / damp) * (list_sum(list_transform(res, lambda ob: ob.w * ob.xs[j] * ob.r)) / sumw
                                            - CASE WHEN j = 1 THEN 0.0 ELSE l2 * b END)) AS zstep
            FROM (
                SELECT it, betas, n, sumw, step, look, res,
                   -- The log-link families (Poisson/Gamma/Tweedie) have
                   -- unbounded curvature, so damp the step by the largest
                   -- per-row Hessian weight hw. 1 for the bounded families.
                   CASE WHEN family IN ('poisson', 'gamma', 'tweedie')
                        THEN greatest(1.0, list_aggregate(
                               list_transform(res, lambda ob: ob.hw), 'max'))
                        ELSE 1.0
                   END AS damp
            FROM (
                SELECT it, betas, rows, n, sumw, step, look,
                       -- Per training row: the residual r (the per-row gradient
                       -- in z, so the beta-gradient is w*xs*r) and hw (the per-row
                       -- Hessian weight used to damp the step for the unbounded-
                       -- curvature log-link families). The linear predictor is
                       -- the offset plus xs . look; the offset is fixed, so it
                       -- enters the fitted value but not the beta-gradient.
                       --   logistic:      r = y - sigmoid(eta)
                       --   linear:        r = y - eta
                       --   poisson (p=1): r = y - mu,          hw = mu
                       --   gamma   (p=2): r = y/mu - 1,        hw = y/mu
                       --   tweedie:       r = (y-mu)*mu^(1-p), hw = (2-p)mu^(2-p)
                       --                                            + (p-1)y*mu^(1-p)
                       -- with mu = exp(eta); Tweedie unifies p=1 (Poisson) and
                       -- p=2 (Gamma), and 1<p<2 admits exact zeros.
                       list_transform(rows, lambda rw: struct_pack(
                           xs := rw.xs,
                           w  := rw.w,
                           r  := CASE WHEN family = 'logistic'
                                      THEN rw.y - 1.0 / (1.0 + exp(-(list_dot_product(rw.xs, look) + rw.o)))
                                      WHEN family = 'poisson'
                                      THEN rw.y - exp(least(list_dot_product(rw.xs, look) + rw.o, 700.0))
                                      WHEN family = 'gamma'
                                      THEN rw.y / exp(greatest(least(list_dot_product(rw.xs, look) + rw.o, 700.0), -700.0)) - 1.0
                                      WHEN family = 'tweedie'
                                      THEN (rw.y - exp(greatest(least(list_dot_product(rw.xs, look) + rw.o, 700.0), -700.0)))
                                           * pow(exp(greatest(least(list_dot_product(rw.xs, look) + rw.o, 700.0), -700.0)), 1.0 - power)
                                      ELSE rw.y - (list_dot_product(rw.xs, look) + rw.o)
                                 END,
                           hw := CASE WHEN family = 'poisson'
                                      THEN exp(least(list_dot_product(rw.xs, look) + rw.o, 700.0))
                                      WHEN family = 'gamma'
                                      THEN rw.y / exp(greatest(least(list_dot_product(rw.xs, look) + rw.o, 700.0), -700.0))
                                      WHEN family = 'tweedie'
                                      THEN (2.0 - power) * pow(exp(greatest(least(list_dot_product(rw.xs, look) + rw.o, 700.0), -700.0)), 2.0 - power)
                                           + (power - 1.0) * rw.y * pow(exp(greatest(least(list_dot_product(rw.xs, look) + rw.o, 700.0), -700.0)), 1.0 - power)
                                      ELSE 0.0
                                 END)) AS res
                FROM (
                    SELECT g.it, g.betas, p.rows, p.n, p.sumw, c.step,
                           -- Nesterov lookahead point
                           list_transform(g.betas, lambda b, j:
                               b + (g.it::DOUBLE / (g.it + 3)) * (b - g.prev[j])) AS look
                    FROM __reg_gd g, __reg_packed p, __reg_cfg c
                    WHERE g.it < max_iter AND g.move >= tol
                )
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
-- Poisson/Gamma (log link): y was divided by its mean, i.e.
-- E[y] = sd_y * exp(z_std) = exp(z_std + ln(sd_y)) — only the intercept
-- shifts; slopes are unchanged.
SELECT feature, coefficient
FROM (
    SELECT '(Intercept)' AS feature,
           CASE WHEN family IN ('poisson', 'gamma', 'tweedie')
                THEN ln(ys.sd_y) + (s.betas[1] - coalesce(list_sum(list_transform(f.names,
                     lambda nm, j: s.betas[j + 1] * f.mus[j] / f.sigmas[j])), 0.0))
                ELSE ys.mu_y + ys.sd_y * (s.betas[1] - coalesce(list_sum(list_transform(f.names,
                     lambda nm, j: s.betas[j + 1] * f.mus[j] / f.sigmas[j])), 0.0))
           END AS coefficient
    FROM __reg_sol s, __reg_feats f, __reg_ystats ys
    UNION ALL
    SELECT unnest(f.names),
           unnest(list_transform(f.names, lambda nm, j:
               (CASE WHEN family IN ('poisson', 'gamma', 'tweedie') THEN 1.0 ELSE ys.sd_y END) * s.betas[j + 1] / f.sigmas[j]))
    FROM __reg_sol s, __reg_feats f, __reg_ystats ys
)
ORDER BY (feature = '(Intercept)') DESC, feature;


-- Shared scorer: returns the input rows plus the linear score
-- __reg_score__ = offset + intercept + sum(coefficient * feature value), NULL
-- for rows where a model feature (or the offset, when requested) is missing or
-- NULL. offset_col is a column name or NULL.
CREATE OR REPLACE MACRO __reg_score(model, tbl, caller, offset_col) AS TABLE
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
             WHEN caller IN ('linreg_predict', 'poisson_predict', 'gamma_predict') AND coalesce(bool_or(lower(colname) = 'prediction'), false)
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
),
-- Per-row offset (only when offset_col is given). NULL offsets were dropped by
-- the UNPIVOT, so a requested-but-missing offset nulls the score below.
__reg_offset AS (
    SELECT rid, v AS o FROM __reg_long WHERE col = offset_col
)
SELECT n.*,
       CASE WHEN offset_col IS NOT NULL AND o.o IS NULL THEN NULL
            ELSE coalesce(s.z, CASE WHEN m.k = 0 THEN m.b0 END) + coalesce(o.o, 0.0)
       END AS __reg_score__
FROM __reg_numbered n
CROSS JOIN __reg_meta m
LEFT JOIN __reg_scores s ON s.rid = n.__reg_rid__
LEFT JOIN __reg_offset o ON o.rid = n.__reg_rid__;


-- ---------------------------------------------------------------------------
-- Public wrappers
-- ---------------------------------------------------------------------------

CREATE OR REPLACE MACRO logit_fit(tbl, outcome, max_iter := 50000, learning_rate := NULL, tol := 1e-10, l2 := 0.0, offset_col := NULL, weights_col := NULL, l1 := 0.0) AS TABLE
SELECT * FROM __reg_fit(tbl, outcome, 'logistic', 'logit_fit', max_iter, learning_rate, tol, l2, offset_col, weights_col, NULL, l1);

CREATE OR REPLACE MACRO linreg_fit(tbl, outcome, max_iter := 50000, learning_rate := NULL, tol := 1e-10, l2 := 0.0, offset_col := NULL, weights_col := NULL, l1 := 0.0) AS TABLE
SELECT * FROM __reg_fit(tbl, outcome, 'linear', 'linreg_fit', max_iter, learning_rate, tol, l2, offset_col, weights_col, NULL, l1);

CREATE OR REPLACE MACRO logit_predict(model, tbl, threshold := 0.5, offset_col := NULL) AS TABLE
SELECT * EXCLUDE (__reg_rid__, __reg_score__),
       1.0 / (1.0 + exp(-__reg_score__)) AS prob,
       1.0 / (1.0 + exp(-__reg_score__)) >= threshold AS pred
FROM __reg_score(model, tbl, 'logit_predict', offset_col)
ORDER BY __reg_rid__;

CREATE OR REPLACE MACRO linreg_predict(model, tbl, offset_col := NULL) AS TABLE
SELECT * EXCLUDE (__reg_rid__, __reg_score__),
       __reg_score__ AS prediction
FROM __reg_score(model, tbl, 'linreg_predict', offset_col)
ORDER BY __reg_rid__;

CREATE OR REPLACE MACRO poisson_fit(tbl, outcome, max_iter := 50000, learning_rate := NULL, tol := 1e-10, l2 := 0.0, offset_col := NULL, weights_col := NULL, l1 := 0.0) AS TABLE
SELECT * FROM __reg_fit(tbl, outcome, 'poisson', 'poisson_fit', max_iter, learning_rate, tol, l2, offset_col, weights_col, NULL, l1);

CREATE OR REPLACE MACRO poisson_predict(model, tbl, offset_col := NULL) AS TABLE
SELECT * EXCLUDE (__reg_rid__, __reg_score__),
       exp(__reg_score__) AS prediction
FROM __reg_score(model, tbl, 'poisson_predict', offset_col)
ORDER BY __reg_rid__;

CREATE OR REPLACE MACRO gamma_fit(tbl, outcome, max_iter := 50000, learning_rate := NULL, tol := 1e-10, l2 := 0.0, offset_col := NULL, weights_col := NULL, l1 := 0.0) AS TABLE
SELECT * FROM __reg_fit(tbl, outcome, 'gamma', 'gamma_fit', max_iter, learning_rate, tol, l2, offset_col, weights_col, NULL, l1);

CREATE OR REPLACE MACRO gamma_predict(model, tbl, offset_col := NULL) AS TABLE
SELECT * EXCLUDE (__reg_rid__, __reg_score__),
       exp(__reg_score__) AS prediction
FROM __reg_score(model, tbl, 'gamma_predict', offset_col)
ORDER BY __reg_rid__;

-- Tweedie regression (log link): a single `power` (variance power p) unifies
-- Poisson (p=1) and Gamma (p=2); 1<p<2 is the compound Poisson-Gamma that
-- admits exact zeros alongside positive values (e.g. insurance pure premium).
-- Matches sklearn TweedieRegressor(power=p, alpha=0, link='log').
CREATE OR REPLACE MACRO tweedie_fit(tbl, outcome, power := 1.5, max_iter := 50000, learning_rate := NULL, tol := 1e-10, l2 := 0.0, offset_col := NULL, weights_col := NULL, l1 := 0.0) AS TABLE
SELECT * FROM __reg_fit(tbl, outcome, 'tweedie', 'tweedie_fit', max_iter, learning_rate, tol, l2, offset_col, weights_col, power, l1);

CREATE OR REPLACE MACRO tweedie_predict(model, tbl, offset_col := NULL) AS TABLE
SELECT * EXCLUDE (__reg_rid__, __reg_score__),
       exp(__reg_score__) AS prediction
FROM __reg_score(model, tbl, 'tweedie_predict', offset_col)
ORDER BY __reg_rid__;


-- ---------------------------------------------------------------------------
-- Goodness-of-fit evaluation
--
-- __reg_eval scores `tbl` with `model` and the outcome column, then computes
-- every metric as a single-row aggregation. The public *_evaluate wrappers
-- select the subset that is meaningful for each family. Metrics are computed
-- from the model's own predictions, so they measure the fit of that model on
-- that data (pass the training table for in-sample fit, a holdout for
-- out-of-sample). Rows where a model feature or the outcome is NULL are
-- dropped, as in training. AIC/BIC use k = number of model coefficients
-- (intercept included); log-likelihoods, deviances, R^2 and AUC follow the
-- standard statsmodels/scikit-learn definitions.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE MACRO __reg_eval(model, tbl, outcome, family, caller, offset_col, power) AS TABLE
WITH
__reg_numbered AS MATERIALIZED (
    SELECT row_number() OVER () AS __reg_rid__, * FROM query_table(tbl)
),
__reg_long AS MATERIALIZED (
    SELECT __reg_rid__ AS rid, name AS col, value AS v
    FROM (UNPIVOT (SELECT __reg_rid__, TRY_CAST(COLUMNS(* EXCLUDE (__reg_rid__)) AS DOUBLE) FROM __reg_numbered)
          ON COLUMNS(* EXCLUDE (__reg_rid__)) INTO NAME name VALUE value)
),
__reg_coefs AS MATERIALIZED (SELECT feature, coefficient FROM query_table(model)),
__reg_meta AS (
    SELECT coalesce((SELECT coefficient FROM __reg_coefs WHERE feature = '(Intercept)'), 0.0) AS b0,
           (SELECT count(*) FROM __reg_coefs WHERE feature != '(Intercept)')                  AS kfeat,
           (SELECT count(*) FROM __reg_coefs)                                                  AS kparams
),
__reg_offset AS (SELECT rid, v AS o FROM __reg_long WHERE col = offset_col),
-- Linear predictor z = offset + b0 + sum(coef * feature); NULL if a model
-- feature (or a requested offset) is missing or NULL for the row.
__reg_z AS (
    SELECT l.rid,
           CASE WHEN count(*) = m.kfeat
                     AND (offset_col IS NULL OR any_value(o.o) IS NOT NULL)
                THEN m.b0 + sum(c.coefficient * l.v) + coalesce(any_value(o.o), 0.0) END AS z
    FROM __reg_coefs c
    JOIN __reg_long l ON l.col = c.feature
    CROSS JOIN __reg_meta m
    LEFT JOIN __reg_offset o ON o.rid = l.rid
    WHERE c.feature != '(Intercept)'
    GROUP BY l.rid, m.b0, m.kfeat
),
__reg_y AS (SELECT rid, v AS y FROM __reg_long WHERE col = outcome),
-- One row per evaluated observation: actual y, linear predictor z, and the
-- mean response yhat under the family's inverse link.
__reg_rows AS (
    SELECT y.y, z.z,
           CASE family WHEN 'logistic' THEN 1.0 / (1.0 + exp(-z.z))
                       WHEN 'poisson'  THEN exp(z.z)
                       WHEN 'gamma'    THEN exp(z.z)
                       WHEN 'tweedie'  THEN exp(z.z)
                       ELSE z.z END AS yhat
    FROM __reg_z z JOIN __reg_y y ON y.rid = z.rid
    WHERE z.z IS NOT NULL AND y.y IS NOT NULL
),
__reg_evalcheck AS (
    SELECT CASE WHEN (SELECT count(*) FROM __reg_rows) = 0
                THEN error(caller || ': no rows with a non-NULL prediction and outcome to evaluate')
                ELSE true END AS ok
),
-- AUC via the Mann-Whitney statistic on average ranks of yhat (logistic only).
__reg_ranked AS (
    SELECT y, avg(rn) OVER (PARTITION BY yhat) AS rk
    FROM (SELECT y, yhat, row_number() OVER (ORDER BY yhat) AS rn FROM __reg_rows)
),
__reg_auc AS (
    SELECT CASE WHEN sum(y) = 0 OR sum(y) = count(*) THEN NULL
                ELSE (sum(CASE WHEN y = 1 THEN rk END) - sum(y) * (sum(y) + 1) / 2.0)
                     / (sum(y) * (count(*) - sum(y))) END AS auc
    FROM __reg_ranked
),
__reg_agg AS (
    SELECT count(*)::DOUBLE AS n,
           avg(y) AS ybar,
           sum((y - yhat) * (y - yhat)) AS sse,
           sum(abs(y - yhat)) AS sae,
           sum(y * ln(greatest(yhat, 1e-15)) + (1 - y) * ln(greatest(1 - yhat, 1e-15))) AS ll_bin,
           avg(CASE WHEN (yhat >= 0.5) = (y >= 0.5) THEN 1.0 ELSE 0.0 END) AS accuracy,
           sum(y * ln(yhat) - yhat - lgamma(y + 1)) AS ll_pois,
           sum((CASE WHEN y > 0 THEN y * ln(y / yhat) ELSE 0.0 END) - (y - yhat)) AS dev_pois_half,
           sum(-ln(y / yhat) + (y - yhat) / yhat) AS dev_gam_half,
           sum(((y - yhat) / yhat) * ((y - yhat) / yhat)) AS pearson_gam,
           -- Tweedie unit half-deviance and Pearson chi-square (power = p).
           sum(pow(greatest(y, 0.0), 2.0 - power) / ((1.0 - power) * (2.0 - power))
               - y * pow(yhat, 1.0 - power) / (1.0 - power)
               + pow(yhat, 2.0 - power) / (2.0 - power)) AS dev_tw_half,
           sum((y - yhat) * (y - yhat) / pow(yhat, power)) AS pearson_tw
    FROM __reg_rows
),
-- Null-model quantities need ybar, so aggregate a second time against it.
__reg_null AS (
    SELECT sum((y - a.ybar) * (y - a.ybar)) AS sst,
           sum(y * ln(a.ybar) + (1 - y) * ln(1 - a.ybar)) AS ll0_bin,
           sum((CASE WHEN y > 0 THEN y * ln(y / a.ybar) ELSE 0.0 END) - (y - a.ybar)) AS null_dev_pois_half,
           sum(-ln(y / a.ybar) + (y - a.ybar) / a.ybar) AS null_dev_gam_half,
           sum(pow(greatest(y, 0.0), 2.0 - power) / ((1.0 - power) * (2.0 - power))
               - y * pow(a.ybar, 1.0 - power) / (1.0 - power)
               + pow(a.ybar, 2.0 - power) / (2.0 - power)) AS null_dev_tw_half
    FROM __reg_rows r, __reg_agg a
)
SELECT
    a.n::BIGINT AS n,
    sqrt(a.sse / a.n) AS rmse,
    a.sae / a.n AS mae,
    CASE WHEN family = 'linear' THEN 1.0 - a.sse / nu.sst END AS r2,
    CASE WHEN family = 'linear' THEN 1.0 - (a.sse / (a.n - m.kparams)) / (nu.sst / (a.n - 1)) END AS adj_r2,
    a.accuracy AS accuracy,
    au.auc AS auc,
    CASE WHEN family = 'logistic' THEN -a.ll_bin / a.n END AS log_loss,
    CASE family WHEN 'linear'   THEN -a.n / 2.0 * (ln(2 * pi()) + ln(a.sse / a.n) + 1.0)
                WHEN 'logistic' THEN a.ll_bin
                WHEN 'poisson'  THEN a.ll_pois END AS loglik,
    CASE family WHEN 'logistic' THEN -2.0 * a.ll_bin
                WHEN 'poisson'  THEN 2.0 * a.dev_pois_half
                WHEN 'gamma'    THEN 2.0 * a.dev_gam_half
                WHEN 'tweedie'  THEN 2.0 * a.dev_tw_half END AS deviance,
    CASE family WHEN 'logistic' THEN -2.0 * nu.ll0_bin
                WHEN 'poisson'  THEN 2.0 * nu.null_dev_pois_half
                WHEN 'gamma'    THEN 2.0 * nu.null_dev_gam_half
                WHEN 'tweedie'  THEN 2.0 * nu.null_dev_tw_half END AS null_deviance,
    CASE family WHEN 'logistic' THEN 1.0 - a.ll_bin / nu.ll0_bin
                WHEN 'poisson'  THEN 1.0 - a.dev_pois_half / nu.null_dev_pois_half
                WHEN 'gamma'    THEN 1.0 - a.dev_gam_half / nu.null_dev_gam_half
                WHEN 'tweedie'  THEN 1.0 - a.dev_tw_half / nu.null_dev_tw_half END AS pseudo_r2,
    CASE WHEN family = 'gamma'   THEN a.pearson_gam / (a.n - m.kparams)
         WHEN family = 'tweedie' THEN a.pearson_tw / (a.n - m.kparams) END AS dispersion,
    CASE family WHEN 'linear'   THEN -2.0 * (-a.n / 2.0 * (ln(2 * pi()) + ln(a.sse / a.n) + 1.0)) + 2.0 * m.kparams
                WHEN 'logistic' THEN -2.0 * a.ll_bin  + 2.0 * m.kparams
                WHEN 'poisson'  THEN -2.0 * a.ll_pois + 2.0 * m.kparams END AS aic,
    CASE family WHEN 'linear'   THEN -2.0 * (-a.n / 2.0 * (ln(2 * pi()) + ln(a.sse / a.n) + 1.0)) + ln(a.n) * m.kparams
                WHEN 'logistic' THEN -2.0 * a.ll_bin  + ln(a.n) * m.kparams
                WHEN 'poisson'  THEN -2.0 * a.ll_pois + ln(a.n) * m.kparams END AS bic
FROM __reg_agg a, __reg_null nu, __reg_auc au, __reg_meta m, __reg_evalcheck ck
WHERE ck.ok;


CREATE OR REPLACE MACRO linreg_evaluate(model, tbl, outcome, offset_col := NULL) AS TABLE
SELECT n, rmse, mae, r2, adj_r2, loglik, aic, bic
FROM __reg_eval(model, tbl, outcome, 'linear', 'linreg_evaluate', offset_col, NULL);

CREATE OR REPLACE MACRO logit_evaluate(model, tbl, outcome, offset_col := NULL) AS TABLE
SELECT n, accuracy, auc, log_loss, loglik, deviance, null_deviance, pseudo_r2, aic, bic
FROM __reg_eval(model, tbl, outcome, 'logistic', 'logit_evaluate', offset_col, NULL);

CREATE OR REPLACE MACRO poisson_evaluate(model, tbl, outcome, offset_col := NULL) AS TABLE
SELECT n, rmse, mae, loglik, deviance, null_deviance, pseudo_r2, aic, bic
FROM __reg_eval(model, tbl, outcome, 'poisson', 'poisson_evaluate', offset_col, NULL);

CREATE OR REPLACE MACRO gamma_evaluate(model, tbl, outcome, offset_col := NULL) AS TABLE
SELECT n, rmse, mae, deviance, null_deviance, pseudo_r2, dispersion
FROM __reg_eval(model, tbl, outcome, 'gamma', 'gamma_evaluate', offset_col, NULL);

CREATE OR REPLACE MACRO tweedie_evaluate(model, tbl, outcome, power := 1.5, offset_col := NULL) AS TABLE
SELECT n, rmse, mae, deviance, null_deviance, pseudo_r2, dispersion
FROM __reg_eval(model, tbl, outcome, 'tweedie', 'tweedie_evaluate', offset_col, power);
