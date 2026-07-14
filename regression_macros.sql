-- ============================================================================
-- Logistic, linear, Poisson, Gamma, Tweedie, negative binomial & multinomial
-- (softmax) regression as pure DuckDB (>= 1.5) SQL macros. The single-outcome
-- families take optional ridge/lasso/elastic-net regularization, an
-- offset/exposure term, and weights.
--
-- Public macros (15): fit / predict / evaluate for each family.
--   {logit,linreg,poisson,gamma}_fit(tbl, outcome, ...)
--   tweedie_fit(tbl, outcome, power := 1.5, ...)
--   nbinom_fit(tbl, outcome, alpha := 1.0, ...)   (fixed dispersion alpha)
--       -> table (feature VARCHAR, coefficient DOUBLE), with an '(Intercept)' row
--   logit_predict(model, tbl, threshold := 0.5, offset_col := NULL)
--       -> input rows + prob DOUBLE + pred BOOLEAN
--   {linreg,poisson,gamma,tweedie,nbinom}_predict(model, tbl, offset_col := NULL)
--       -> input rows + prediction DOUBLE (linear: score; log-link: exp(score))
--   {logit,linreg,poisson,gamma}_evaluate(model, tbl, outcome, offset_col := NULL)
--   tweedie_evaluate(model, tbl, outcome, power := 1.5, offset_col := NULL)
--   nbinom_evaluate(model, tbl, outcome, alpha := 1.0, offset_col := NULL)
--       -> one-row table of goodness-of-fit metrics
--
-- Fit macros accept: max_iter, learning_rate, tol, l2, l1, offset_col,
-- weights_col (tweedie_fit also power); see "Fit parameters" below.
--
-- Multiclass (softmax; baseline-category parameterization):
--   multinom_fit(tbl, outcome, ...) -> (class, feature, coefficient)
--   multinom_predict(model, tbl)    -> input rows + pred + probs MAP
--   multinom_evaluate(model, tbl, outcome) -> (n, accuracy, log_loss)
--
-- Cross-validation (all k folds fit simultaneously in one recursive CTE),
-- returns (param, cv_deviance); pick the smallest cv_deviance:
--   cv_l2(tbl, outcome, family, l2_grid, k := 5)   ridge, over linear/logistic/poisson/gamma
--   cv_l1(tbl, outcome, family, l1_grid, k := 5)   lasso, same families
--   cv_power(tbl, outcome, power_grid, k := 5)     Tweedie variance power
--   cv_alpha(tbl, outcome, alpha_grid, k := 5)     negative binomial dispersion
--
-- Dispersion estimation:
--   nbinom_dispersion(tbl, outcome, alpha_grid) -> (alpha, loglik)
--       NB2 profile log-likelihood per alpha; argmax is the MLE dispersion.
--
-- Helper (returns SQL text, run it as a second step):
--   dummy_encode_sql(tbl, outcome) -> VARCHAR: a SELECT that R-style dummy
--       encodes every VARCHAR column except the outcome (see its own comment).
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

-- Dense matrix inverse of a d x d DOUBLE[][] as a single scalar expression:
-- Gauss-Jordan elimination folded over columns 1..d with list_reduce (the
-- accumulator carries the augmented [A|I]). No pivoting -- intended for the
-- symmetric positive-definite X'WX of the IRLS solver, whose pivots stay
-- positive. Used inside the IRLS recursive CTE, where a recursive-CTE inverse
-- cannot be nested.
CREATE OR REPLACE MACRO __reg_matinv(A) AS (
  list_transform(
    list_reduce(
      [ struct_pack(k := 0, M := list_transform(A, lambda row, i:
            row || list_transform(row, lambda v, j: CASE WHEN i = j THEN 1.0 ELSE 0.0 END))) ]
      || list_transform(range(1, len(A)+1), lambda i: struct_pack(k := i, M := [[0.0]]::DOUBLE[][])),
      (acc, e) -> struct_pack(k := e.k, M :=
        list_transform(acc.M, lambda row, i:
          CASE WHEN i = e.k THEN list_transform(acc.M[e.k], lambda v, j: v / acc.M[e.k][e.k])
               ELSE list_transform(row, lambda v, j: v - row[e.k]*(acc.M[e.k][j]/acc.M[e.k][e.k])) END))
    ).M,
    lambda row: list_slice(row, len(A)+1, 2*len(A))
  )
);

CREATE OR REPLACE MACRO __reg_fit(tbl, outcome, family, caller, max_iter, learning_rate, tol, l2, offset_col, weights_col, power, l1, alpha, solver) AS TABLE
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
             WHEN family = 'nbinom' AND min(v) < 0
               THEN error(caller || ': outcome column "' || outcome || '" must be non-negative for negative binomial regression')
             WHEN family = 'nbinom' AND alpha <= 0
               THEN error(caller || ': alpha (dispersion) must be > 0')
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
-- j is the feature's position in name order. Downstream, the per-row feature
-- vector is assembled with list(... ORDER BY j) rather than ORDER BY col: an
-- ordered list aggregate carries its sort key alongside every value, and a
-- VARCHAR key means n*d column-name strings in the sort buffers. Ordering by
-- the integer instead is the same order (j is assigned by ORDER BY col) for a
-- fraction of the time and memory.
__reg_stats AS MATERIALIZED (
    SELECT s.col,
           row_number() OVER (ORDER BY s.col) AS j,
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
                WHEN family IN ('poisson', 'gamma', 'tweedie', 'nbinom')
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
               [1.0::DOUBLE] || list((x.v - s.mu) / s.sigma ORDER BY s.j) AS xs,
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
             WHEN solver NOT IN ('gd', 'irls', 'auto')
               THEN error(caller || ': solver must be ''auto'', ''gd'' or ''irls'', got ''' || solver || '''')
             WHEN solver = 'irls' AND l1 > 0
               THEN error(caller || ': the irls solver does not support L1 (lasso/elastic-net); use solver := ''gd'' for l1 > 0')
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
           END AS step,
           -- Negative-binomial dispersion on the mean-scaled internal problem:
           -- dividing y by its mean multiplies the effective alpha by that mean.
           coalesce(alpha, 0.0) * (SELECT sd_y FROM __reg_ystats) AS alpha_int
    FROM __reg_feats f, __reg_packed p, __reg_ycheck y, __reg_namecheck nc
    WHERE y.ok AND nc.ok
),
-- IRLS / Fisher scoring (solver := 'irls'): each iteration solves the weighted
-- least squares beta <- (X'WX + sumw*l2*D)^-1 X'W z on the standardized data,
-- where W is the expected-information working weight and z the working response.
-- Converges to the exact (penalized) MLE in ~5-10 iterations; the matrix solve
-- uses the __reg_matinv fold (a recursive-CTE inverse cannot nest here). D omits
-- the intercept. Gated by solver = 'irls' so it is inert (base row only) under
-- the default gradient-descent solver, and vice versa.
__reg_irls(it, betas, move) AS (
    -- cross-join __reg_cfg and touch c.step so ALL input validation (which lives
    -- in the step CASE: l2/l1 sign, bad solver, l1-with-irls, missing offset/
    -- weights) fires for the irls path too, not just gradient descent
    SELECT 0, list_transform(range(f.d + 1), lambda i: 0.0::DOUBLE), 1e308::DOUBLE
    FROM __reg_feats f, __reg_cfg c
    WHERE c.step IS NOT NULL
      -- irls is the solver under solver := 'irls', and under the 'auto' default
      -- whenever there is no L1 penalty (irls cannot do L1). Otherwise no seed
      -- row is emitted and this recursion is inert.
      AND (solver = 'irls' OR (solver = 'auto' AND l1 = 0.0))
    UNION ALL
    SELECT it + 1, betas_new,
           list_aggregate(list_transform(betas_new, lambda v, j: abs(v - betas[j])), 'max')
    FROM (
        SELECT it, betas, list_transform(inv, lambda invrow: list_dot_product(invrow, XWr)) AS betas_new
        FROM (
            SELECT it, betas, XWr, __reg_matinv(XWXpen) AS inv
            FROM (
                -- ridge penalty on the diagonal (intercept at position 1 unpenalized)
                SELECT it, betas, XWr,
                       list_transform(XWX, lambda row, a:
                           list_transform(row, lambda v, b:
                               CASE WHEN a = b AND a > 1 THEN v + sumw * l2 ELSE v END)) AS XWXpen
                FROM (
                    SELECT it, betas, sumw,
                           list_transform(range(1, len(betas) + 1), lambda a:
                               list_transform(range(1, len(betas) + 1), lambda b:
                                   list_sum(list_transform(res, lambda ob: ob.wirls * ob.xs[a] * ob.xs[b])))) AS XWX,
                           list_transform(range(1, len(betas) + 1), lambda a:
                               list_sum(list_transform(res, lambda ob: ob.wr * ob.xs[a]))) AS XWr
                    FROM (
                        -- per row: expected-info weight wirls, and wr = wirls*(xs.beta) + w*residual
                        SELECT it, betas, sumw,
                               list_transform(mus, lambda e: struct_pack(
                                   xs := e.xs,
                                   wirls := e.w * (CASE family
                                              WHEN 'logistic' THEN e.mu * (1.0 - e.mu)
                                              WHEN 'linear'   THEN 1.0
                                              WHEN 'poisson'  THEN e.mu
                                              WHEN 'gamma'    THEN 1.0
                                              WHEN 'tweedie'  THEN pow(e.mu, 2.0 - power)
                                              WHEN 'nbinom'   THEN e.mu / (1.0 + alpha_int * e.mu) END),
                                   wr := e.w * ((CASE family
                                              WHEN 'logistic' THEN e.mu * (1.0 - e.mu)
                                              WHEN 'linear'   THEN 1.0
                                              WHEN 'poisson'  THEN e.mu
                                              WHEN 'gamma'    THEN 1.0
                                              WHEN 'tweedie'  THEN pow(e.mu, 2.0 - power)
                                              WHEN 'nbinom'   THEN e.mu / (1.0 + alpha_int * e.mu) END) * e.linpred
                                          + (CASE family
                                              WHEN 'gamma'    THEN e.y / e.mu - 1.0
                                              WHEN 'tweedie'  THEN (e.y - e.mu) * pow(e.mu, 1.0 - power)
                                              WHEN 'nbinom'   THEN (e.y - e.mu) / (1.0 + alpha_int * e.mu)
                                              ELSE e.y - e.mu END))
                                   )) AS res
                        FROM (
                            SELECT g.it, g.betas, p.sumw, c.alpha_int AS alpha_int,
                                   list_transform(p.rows, lambda rw: struct_pack(
                                       xs := rw.xs, w := rw.w, y := rw.y,
                                       linpred := list_dot_product(rw.xs, g.betas),
                                       mu := CASE family
                                               WHEN 'logistic' THEN 1.0 / (1.0 + exp(-greatest(least(list_dot_product(rw.xs, g.betas) + rw.o, 700.0), -700.0)))
                                               WHEN 'linear'   THEN list_dot_product(rw.xs, g.betas) + rw.o
                                               ELSE exp(greatest(least(list_dot_product(rw.xs, g.betas) + rw.o, 700.0), -700.0)) END
                                       )) AS mus
                            FROM __reg_irls g, __reg_packed p, __reg_cfg c
                            -- isfinite() stops promptly on a singular step (NaN move),
                            -- since NaN >= tol is TRUE in DuckDB and would otherwise loop
                            WHERE g.it < max_iter AND g.move >= tol AND isfinite(g.move)
                        )
                    )
                )
            )
        )
    )
),
-- Last irls iterate, and whether it can be trusted. A singular X'WX (a constant
-- or perfectly collinear feature) drives the coefficients to NaN; complete
-- separation drives them to ~1e305. Both are rejected here, which is what makes
-- solver := 'auto' fall back to gradient descent instead of returning garbage.
-- Empty (=> ok false) when irls did not run at all, which is exactly the gate gd
-- wants: solver := 'gd' and the l1 > 0 path both need gd to run.
__reg_irls_beta AS (SELECT betas FROM __reg_irls ORDER BY it DESC LIMIT 1),
__reg_irls_ok AS (
    SELECT coalesce(
             (SELECT list_aggregate(
                        list_transform(betas, lambda v: isfinite(v) AND abs(v) < 1e100),
                        'bool_and')
              FROM __reg_irls_beta),
             false) AS ok
),
-- Nesterov-accelerated gradient descent on the mean loss (negative
-- log-likelihood for logistic, squared error / 2 for linear).
-- The early-stop criterion is the size of the pure gradient step,
-- move = step * max_j |grad_j|, NOT the iterate displacement |betas - prev|:
-- momentum can make two consecutive iterates coincide exactly at a
-- non-stationary point (deterministically so for exactly collinear features),
-- while the gradient step is zero only at a true stationary point.
-- Gated so that it produces NO rows (not even the seed) when gradient descent is
-- not the solver in play: solver := 'gd' always runs it, solver := 'auto' runs it
-- only as the fallback when irls came back singular/divergent, and solver := 'irls'
-- never does. With no seed row the recursion terminates immediately, so the
-- unused solver costs nothing.
__reg_gd AS (
    SELECT 0 AS it,
           list_transform(range(f.d + 1), lambda i: 0.0::DOUBLE) AS betas,
           list_transform(range(f.d + 1), lambda i: 0.0::DOUBLE) AS prev,
           1e308::DOUBLE AS move
    FROM __reg_feats f
    WHERE solver = 'gd' OR (solver = 'auto' AND NOT (SELECT ok FROM __reg_irls_ok))
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
                   CASE WHEN family IN ('poisson', 'gamma', 'tweedie', 'nbinom')
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
                                      WHEN family = 'nbinom'
                                      -- NB2: r = (y - mu) / (1 + alpha*mu), mu = exp(eta);
                                      -- reduces to Poisson (y - mu) as alpha -> 0
                                      THEN (rw.y - exp(least(list_dot_product(rw.xs, look) + rw.o, 700.0)))
                                           / (1.0 + alpha_int * exp(least(list_dot_product(rw.xs, look) + rw.o, 700.0)))
                                      ELSE rw.y - (list_dot_product(rw.xs, look) + rw.o)
                                 END,
                           hw := CASE WHEN family = 'poisson'
                                      THEN exp(least(list_dot_product(rw.xs, look) + rw.o, 700.0))
                                      WHEN family = 'gamma'
                                      THEN rw.y / exp(greatest(least(list_dot_product(rw.xs, look) + rw.o, 700.0), -700.0))
                                      WHEN family = 'tweedie'
                                      THEN (2.0 - power) * pow(exp(greatest(least(list_dot_product(rw.xs, look) + rw.o, 700.0), -700.0)), 2.0 - power)
                                           + (power - 1.0) * rw.y * pow(exp(greatest(least(list_dot_product(rw.xs, look) + rw.o, 700.0), -700.0)), 1.0 - power)
                                      WHEN family = 'nbinom'
                                      -- NB Hessian weight mu(1+alpha*y)/(1+alpha*mu)^2
                                      THEN exp(least(list_dot_product(rw.xs, look) + rw.o, 700.0)) * (1.0 + alpha_int * rw.y)
                                           / pow(1.0 + alpha_int * exp(least(list_dot_product(rw.xs, look) + rw.o, 700.0)), 2.0)
                                      ELSE 0.0
                                 END)) AS res
                FROM (
                    SELECT g.it, g.betas, p.rows, p.n, p.sumw, c.step, c.alpha_int,
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
__reg_gd_beta AS (SELECT betas FROM __reg_gd ORDER BY it DESC LIMIT 1),
__reg_sol AS (
    -- Exactly one solver produced a seed row (see the gates above), except under
    -- solver := 'auto' with a singular X'WX, where irls ran, was rejected, and gd
    -- ran as the fallback -- so gd's answer wins whenever gd produced one.
    SELECT CASE
             WHEN (SELECT count(*) FROM __reg_gd_beta) > 0
               THEN (SELECT betas FROM __reg_gd_beta)
             WHEN (SELECT ok FROM __reg_irls_ok)
               THEN (SELECT betas FROM __reg_irls_beta)
             -- Only reachable for an explicit solver := 'irls'; 'auto' falls back.
             ELSE error(caller || ': the irls solver did not converge -- X''WX is singular '
                        || '(perfectly collinear features, or complete separation for logistic). '
                        || 'Use solver := ''auto'' (the default) or solver := ''gd'', or add l2 ridge.')
           END AS betas
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
           CASE WHEN family IN ('poisson', 'gamma', 'tweedie', 'nbinom')
                THEN ln(ys.sd_y) + (s.betas[1] - coalesce(list_sum(list_transform(f.names,
                     lambda nm, j: s.betas[j + 1] * f.mus[j] / f.sigmas[j])), 0.0))
                ELSE ys.mu_y + ys.sd_y * (s.betas[1] - coalesce(list_sum(list_transform(f.names,
                     lambda nm, j: s.betas[j + 1] * f.mus[j] / f.sigmas[j])), 0.0))
           END AS coefficient
    FROM __reg_sol s, __reg_feats f, __reg_ystats ys
    UNION ALL
    SELECT unnest(f.names),
           unnest(list_transform(f.names, lambda nm, j:
               (CASE WHEN family IN ('poisson', 'gamma', 'tweedie', 'nbinom') THEN 1.0 ELSE ys.sd_y END) * s.betas[j + 1] / f.sigmas[j]))
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

CREATE OR REPLACE MACRO logit_fit(tbl, outcome, max_iter := 50000, learning_rate := NULL, tol := 1e-10, l2 := 0.0, offset_col := NULL, weights_col := NULL, l1 := 0.0, solver := 'auto') AS TABLE
SELECT * FROM __reg_fit(tbl, outcome, 'logistic', 'logit_fit', max_iter, learning_rate, tol, l2, offset_col, weights_col, NULL, l1, NULL, solver);

CREATE OR REPLACE MACRO linreg_fit(tbl, outcome, max_iter := 50000, learning_rate := NULL, tol := 1e-10, l2 := 0.0, offset_col := NULL, weights_col := NULL, l1 := 0.0, solver := 'auto') AS TABLE
SELECT * FROM __reg_fit(tbl, outcome, 'linear', 'linreg_fit', max_iter, learning_rate, tol, l2, offset_col, weights_col, NULL, l1, NULL, solver);

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

CREATE OR REPLACE MACRO poisson_fit(tbl, outcome, max_iter := 50000, learning_rate := NULL, tol := 1e-10, l2 := 0.0, offset_col := NULL, weights_col := NULL, l1 := 0.0, solver := 'auto') AS TABLE
SELECT * FROM __reg_fit(tbl, outcome, 'poisson', 'poisson_fit', max_iter, learning_rate, tol, l2, offset_col, weights_col, NULL, l1, NULL, solver);

CREATE OR REPLACE MACRO poisson_predict(model, tbl, offset_col := NULL) AS TABLE
SELECT * EXCLUDE (__reg_rid__, __reg_score__),
       exp(__reg_score__) AS prediction
FROM __reg_score(model, tbl, 'poisson_predict', offset_col)
ORDER BY __reg_rid__;

CREATE OR REPLACE MACRO gamma_fit(tbl, outcome, max_iter := 50000, learning_rate := NULL, tol := 1e-10, l2 := 0.0, offset_col := NULL, weights_col := NULL, l1 := 0.0, solver := 'auto') AS TABLE
SELECT * FROM __reg_fit(tbl, outcome, 'gamma', 'gamma_fit', max_iter, learning_rate, tol, l2, offset_col, weights_col, NULL, l1, NULL, solver);

CREATE OR REPLACE MACRO gamma_predict(model, tbl, offset_col := NULL) AS TABLE
SELECT * EXCLUDE (__reg_rid__, __reg_score__),
       exp(__reg_score__) AS prediction
FROM __reg_score(model, tbl, 'gamma_predict', offset_col)
ORDER BY __reg_rid__;

-- Tweedie regression (log link): a single `power` (variance power p) unifies
-- Poisson (p=1) and Gamma (p=2); 1<p<2 is the compound Poisson-Gamma that
-- admits exact zeros alongside positive values (e.g. insurance pure premium).
-- Matches sklearn TweedieRegressor(power=p, alpha=0, link='log').
CREATE OR REPLACE MACRO tweedie_fit(tbl, outcome, power := 1.5, max_iter := 50000, learning_rate := NULL, tol := 1e-10, l2 := 0.0, offset_col := NULL, weights_col := NULL, l1 := 0.0, solver := 'auto') AS TABLE
SELECT * FROM __reg_fit(tbl, outcome, 'tweedie', 'tweedie_fit', max_iter, learning_rate, tol, l2, offset_col, weights_col, power, l1, NULL, solver);

CREATE OR REPLACE MACRO tweedie_predict(model, tbl, offset_col := NULL) AS TABLE
SELECT * EXCLUDE (__reg_rid__, __reg_score__),
       exp(__reg_score__) AS prediction
FROM __reg_score(model, tbl, 'tweedie_predict', offset_col)
ORDER BY __reg_rid__;

-- Negative binomial (NB2, log link) for overdispersed counts. `alpha` is the
-- fixed dispersion (variance = mu + alpha*mu^2); alpha -> 0 recovers Poisson.
-- Matches statsmodels GLM NegativeBinomial(alpha=alpha) (fixed dispersion);
-- alpha is a hyperparameter here, not estimated.
CREATE OR REPLACE MACRO nbinom_fit(tbl, outcome, alpha := 1.0, max_iter := 50000, learning_rate := NULL, tol := 1e-10, l2 := 0.0, offset_col := NULL, weights_col := NULL, l1 := 0.0, solver := 'auto') AS TABLE
SELECT * FROM __reg_fit(tbl, outcome, 'nbinom', 'nbinom_fit', max_iter, learning_rate, tol, l2, offset_col, weights_col, NULL, l1, alpha, solver);

CREATE OR REPLACE MACRO nbinom_predict(model, tbl, offset_col := NULL) AS TABLE
SELECT * EXCLUDE (__reg_rid__, __reg_score__),
       exp(__reg_score__) AS prediction
FROM __reg_score(model, tbl, 'nbinom_predict', offset_col)
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
CREATE OR REPLACE MACRO __reg_eval(model, tbl, outcome, family, caller, offset_col, power, alpha) AS TABLE
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
                       WHEN 'nbinom'   THEN exp(z.z)
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
           sum((y - yhat) * (y - yhat) / pow(yhat, power)) AS pearson_tw,
           -- Negative binomial (NB2, r = 1/alpha): log-likelihood, half-deviance,
           -- and Pearson chi-square (variance = mu + alpha*mu^2).
           sum(lgamma(y + 1.0 / alpha) - lgamma(1.0 / alpha) - lgamma(y + 1)
               + (1.0 / alpha) * ln((1.0 / alpha) / (1.0 / alpha + yhat))
               + y * ln(yhat / (1.0 / alpha + yhat))) AS ll_nb,
           sum((CASE WHEN y > 0 THEN y * ln(y / yhat) ELSE 0.0 END)
               - (y + 1.0 / alpha) * ln((y + 1.0 / alpha) / (yhat + 1.0 / alpha))) AS dev_nb_half,
           sum((y - yhat) * (y - yhat) / (yhat + alpha * yhat * yhat)) AS pearson_nb
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
               + pow(a.ybar, 2.0 - power) / (2.0 - power)) AS null_dev_tw_half,
           sum((CASE WHEN y > 0 THEN y * ln(y / a.ybar) ELSE 0.0 END)
               - (y + 1.0 / alpha) * ln((y + 1.0 / alpha) / (a.ybar + 1.0 / alpha))) AS null_dev_nb_half
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
                WHEN 'poisson'  THEN a.ll_pois
                WHEN 'nbinom'   THEN a.ll_nb END AS loglik,
    CASE family WHEN 'logistic' THEN -2.0 * a.ll_bin
                WHEN 'poisson'  THEN 2.0 * a.dev_pois_half
                WHEN 'gamma'    THEN 2.0 * a.dev_gam_half
                WHEN 'tweedie'  THEN 2.0 * a.dev_tw_half
                WHEN 'nbinom'   THEN 2.0 * a.dev_nb_half END AS deviance,
    CASE family WHEN 'logistic' THEN -2.0 * nu.ll0_bin
                WHEN 'poisson'  THEN 2.0 * nu.null_dev_pois_half
                WHEN 'gamma'    THEN 2.0 * nu.null_dev_gam_half
                WHEN 'tweedie'  THEN 2.0 * nu.null_dev_tw_half
                WHEN 'nbinom'   THEN 2.0 * nu.null_dev_nb_half END AS null_deviance,
    CASE family WHEN 'logistic' THEN 1.0 - a.ll_bin / nu.ll0_bin
                WHEN 'poisson'  THEN 1.0 - a.dev_pois_half / nu.null_dev_pois_half
                WHEN 'gamma'    THEN 1.0 - a.dev_gam_half / nu.null_dev_gam_half
                WHEN 'tweedie'  THEN 1.0 - a.dev_tw_half / nu.null_dev_tw_half
                WHEN 'nbinom'   THEN 1.0 - a.dev_nb_half / nu.null_dev_nb_half END AS pseudo_r2,
    CASE WHEN family = 'gamma'   THEN a.pearson_gam / (a.n - m.kparams)
         WHEN family = 'tweedie' THEN a.pearson_tw / (a.n - m.kparams)
         WHEN family = 'nbinom'  THEN a.pearson_nb / (a.n - m.kparams) END AS dispersion,
    CASE family WHEN 'linear'   THEN -2.0 * (-a.n / 2.0 * (ln(2 * pi()) + ln(a.sse / a.n) + 1.0)) + 2.0 * m.kparams
                WHEN 'logistic' THEN -2.0 * a.ll_bin  + 2.0 * m.kparams
                WHEN 'poisson'  THEN -2.0 * a.ll_pois + 2.0 * m.kparams
                WHEN 'nbinom'   THEN -2.0 * a.ll_nb   + 2.0 * m.kparams END AS aic,
    CASE family WHEN 'linear'   THEN -2.0 * (-a.n / 2.0 * (ln(2 * pi()) + ln(a.sse / a.n) + 1.0)) + ln(a.n) * m.kparams
                WHEN 'logistic' THEN -2.0 * a.ll_bin  + ln(a.n) * m.kparams
                WHEN 'poisson'  THEN -2.0 * a.ll_pois + ln(a.n) * m.kparams
                WHEN 'nbinom'   THEN -2.0 * a.ll_nb   + ln(a.n) * m.kparams END AS bic
FROM __reg_agg a, __reg_null nu, __reg_auc au, __reg_meta m, __reg_evalcheck ck
WHERE ck.ok;


CREATE OR REPLACE MACRO linreg_evaluate(model, tbl, outcome, offset_col := NULL) AS TABLE
SELECT n, rmse, mae, r2, adj_r2, loglik, aic, bic
FROM __reg_eval(model, tbl, outcome, 'linear', 'linreg_evaluate', offset_col, NULL, NULL);

CREATE OR REPLACE MACRO logit_evaluate(model, tbl, outcome, offset_col := NULL) AS TABLE
SELECT n, accuracy, auc, log_loss, loglik, deviance, null_deviance, pseudo_r2, aic, bic
FROM __reg_eval(model, tbl, outcome, 'logistic', 'logit_evaluate', offset_col, NULL, NULL);

CREATE OR REPLACE MACRO poisson_evaluate(model, tbl, outcome, offset_col := NULL) AS TABLE
SELECT n, rmse, mae, loglik, deviance, null_deviance, pseudo_r2, aic, bic
FROM __reg_eval(model, tbl, outcome, 'poisson', 'poisson_evaluate', offset_col, NULL, NULL);

CREATE OR REPLACE MACRO gamma_evaluate(model, tbl, outcome, offset_col := NULL) AS TABLE
SELECT n, rmse, mae, deviance, null_deviance, pseudo_r2, dispersion
FROM __reg_eval(model, tbl, outcome, 'gamma', 'gamma_evaluate', offset_col, NULL, NULL);

CREATE OR REPLACE MACRO tweedie_evaluate(model, tbl, outcome, power := 1.5, offset_col := NULL) AS TABLE
SELECT n, rmse, mae, deviance, null_deviance, pseudo_r2, dispersion
FROM __reg_eval(model, tbl, outcome, 'tweedie', 'tweedie_evaluate', offset_col, power, NULL);

CREATE OR REPLACE MACRO nbinom_evaluate(model, tbl, outcome, alpha := 1.0, offset_col := NULL) AS TABLE
SELECT n, rmse, mae, loglik, deviance, null_deviance, pseudo_r2, dispersion, aic, bic
FROM __reg_eval(model, tbl, outcome, 'nbinom', 'nbinom_evaluate', offset_col, NULL, alpha);


-- ---------------------------------------------------------------------------
-- Categorical encoding helper
--
-- dummy_encode_sql(tbl, outcome) inspects `tbl` and returns a SELECT statement
-- (as text) that one-hot / dummy encodes every VARCHAR (categorical) column
-- except the outcome, using R-style treatment contrasts: k-1 indicator columns
-- per factor, dropping the first level (alphabetical minimum) as the reference.
-- Numeric and boolean columns and the outcome pass through unchanged. The
-- result reproduces R's lm(y ~ ... + C(factor)) / patsy encoding, so it feeds
-- straight into the *_fit macros.
--
-- DuckDB macros cannot return a data-dependent set of columns, so this
-- generates the SQL rather than the data; run it as a second step:
--
--   -- from a driver (Python / R / etc.), two lines:
--   sql = con.sql("SELECT dummy_encode_sql('sales','revenue')").fetchone()[0]
--   con.sql(f"CREATE TABLE encoded AS {sql}")
--   con.sql("SELECT * FROM linreg_fit('encoded','revenue')")
--
-- `tbl` must be a table or view (resolvable in duckdb_columns), optionally
-- schema/catalog-qualified. VARCHAR columns are treated as categorical; a NULL
-- category yields NULL dummies, so that row is dropped by the fit (as R drops
-- NA rows). High-cardinality columns produce many dummies -- encode with care.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE MACRO dummy_encode_sql(tbl, outcome) AS (
  WITH __reg_enc_cat AS (
    SELECT column_name AS col FROM duckdb_columns()
    WHERE (table_name = tbl
           OR schema_name || '.' || table_name = tbl
           OR database_name || '.' || schema_name || '.' || table_name = tbl)
      AND data_type = 'VARCHAR' AND column_name != outcome
  ),
  __reg_enc_lvl AS (
    SELECT DISTINCT name AS col, val
    FROM (UNPIVOT (SELECT CAST(COLUMNS(*) AS VARCHAR) FROM query_table(tbl)) ON COLUMNS(*) INTO NAME name VALUE val)
    WHERE name IN (SELECT col FROM __reg_enc_cat) AND val IS NOT NULL
  ),
  __reg_enc_ref AS (SELECT col, min(val) AS ref FROM __reg_enc_lvl GROUP BY col),
  __reg_enc_dum AS (
    SELECT string_agg('(' || l.col || ' = ''' || replace(l.val, '''', '''''') || ''')::INT AS "'
                      || replace(l.col || '_' || l.val, '"', '""') || '"',
                      ', ' ORDER BY l.col, l.val) AS dummies
    FROM __reg_enc_lvl l JOIN __reg_enc_ref r ON r.col = l.col WHERE l.val <> r.ref
  ),
  __reg_enc_ex AS (SELECT string_agg(col, ', ' ORDER BY col) AS excl FROM __reg_enc_cat)
  SELECT CASE WHEN (SELECT count(*) FROM __reg_enc_cat) = 0 THEN 'SELECT * FROM ' || tbl
              ELSE 'SELECT * EXCLUDE (' || (SELECT excl FROM __reg_enc_ex) || ')'
                   || coalesce(', ' || (SELECT dummies FROM __reg_enc_dum), '') || ' FROM ' || tbl END
);


-- ---------------------------------------------------------------------------
-- Multinomial (softmax) logistic regression
--
-- multinom_fit(tbl, outcome, ...) fits a K-class softmax model in the
-- identifiable baseline-category parameterization: one coefficient vector per
-- class relative to a reference class (the alphabetical-minimum label, held at
-- 0), matching R's nnet::multinom and statsmodels MNLogit. The outcome is a
-- class-label column (any type); all other columns are numeric/boolean
-- features (dummy-encode categoricals first, e.g. with dummy_encode_sql).
--
--   multinom_fit -> table (class VARCHAR, feature VARCHAR, coefficient DOUBLE)
--       with an '(Intercept)' row per class; the reference class has all-zero
--       coefficients so scoring is a self-contained softmax over every class.
--   multinom_predict(model, tbl) -> input rows + pred VARCHAR (argmax class)
--       + probs MAP(VARCHAR, DOUBLE) (full class distribution; probs['label']).
--   multinom_evaluate(model, tbl, outcome) -> (n, accuracy, log_loss).
-- ---------------------------------------------------------------------------
CREATE OR REPLACE MACRO multinom_fit(tbl, outcome, max_iter := 50000, learning_rate := NULL, tol := 1e-10, l2 := 0.0, l1 := 0.0) AS TABLE
WITH RECURSIVE
__reg_mnum AS MATERIALIZED (SELECT row_number() OVER () AS rid, * FROM query_table(tbl)),
__reg_mflong AS MATERIALIZED (
  SELECT rid, name AS col, value AS v
  FROM (UNPIVOT (SELECT rid, CAST(COLUMNS(c -> c != outcome AND c != 'rid') AS DOUBLE) FROM __reg_mnum)
        ON COLUMNS(* EXCLUDE (rid)) INTO NAME name VALUE value)
),
__reg_mylab AS MATERIALIZED (
  SELECT rid, val AS lab
  FROM (UNPIVOT (SELECT rid, CAST(COLUMNS(c -> c != 'rid') AS VARCHAR) FROM __reg_mnum)
        ON COLUMNS(* EXCLUDE (rid)) INTO NAME name VALUE val)
  WHERE name = outcome
),
__reg_mnonref AS (
  SELECT list(lab ORDER BY lab) AS nrf
  FROM (SELECT DISTINCT lab FROM __reg_mylab WHERE lab <> (SELECT min(lab) FROM __reg_mylab))
),
__reg_mstats AS MATERIALIZED (
  -- j: feature position in name order. The per-row feature vector below is
  -- assembled with ORDER BY j rather than ORDER BY col; an ordered list
  -- aggregate carries its sort key with every value, so a VARCHAR key means
  -- n*d column-name strings in the sort buffers.
  SELECT col,
         row_number() OVER (ORDER BY col) AS j,
         CASE WHEN min(v) = max(v) THEN min(v) ELSE avg(v) END AS mu,
         CASE WHEN min(v) = max(v) THEN 1.0 ELSE stddev_pop(v) END AS sigma
  FROM __reg_mflong GROUP BY col
),
__reg_mfeats AS MATERIALIZED (
  SELECT list(col ORDER BY col) AS names, list(mu ORDER BY col) AS mus,
         list(sigma ORDER BY col) AS sigmas, count(*)::INT AS d
  FROM __reg_mstats
),
__reg_mchk AS (
  SELECT CASE
    WHEN (SELECT count(*) FROM (SELECT DISTINCT lab FROM __reg_mylab)) < 2
      THEN error('multinom_fit: outcome column "' || outcome || '" must have at least 2 distinct classes')
    WHEN (SELECT d FROM __reg_mfeats) = 0
      THEN error('multinom_fit: no feature columns besides the outcome')
    WHEN l2 < 0 THEN error('multinom_fit: l2 must be >= 0, got ' || l2)
    WHEN l1 < 0 THEN error('multinom_fit: l1 must be >= 0, got ' || l1)
    ELSE true END AS ok
),
__reg_mpacked AS MATERIALIZED (
  SELECT list(struct_pack(xs := xs, yv := yv)) AS rows, count(*)::DOUBLE AS n,
         any_value(len(yv)) AS K1, any_value(len(xs)) AS D1
  FROM (
    SELECT x.rid,
           [1.0::DOUBLE] || list((x.v - s.mu) / s.sigma ORDER BY s.j) AS xs,
           list_transform(any_value(nr.nrf),
             lambda cl: CASE WHEN any_value(yl.lab) = cl THEN 1.0 ELSE 0.0 END) AS yv
    FROM __reg_mflong x
    JOIN __reg_mstats s ON s.col = x.col
    JOIN __reg_mylab yl ON yl.rid = x.rid
    CROSS JOIN __reg_mnonref nr
    GROUP BY x.rid
  )
),
__reg_mcfg AS (
  -- softmax gradient is L-Lipschitz with L <= (d+1)/2 on standardized data,
  -- plus l2 from the ridge penalty
  SELECT coalesce(learning_rate, 2.0 / (D1 + 2.0 * l2)) AS step
  FROM __reg_mpacked, __reg_mchk WHERE ok
),
__reg_mgd AS (
  SELECT 0 AS it,
         list_transform(range(K1), lambda k: list_transform(range(D1), lambda j: 0.0::DOUBLE)) AS B,
         list_transform(range(K1), lambda k: list_transform(range(D1), lambda j: 0.0::DOUBLE)) AS prev,
         1e308::DOUBLE AS move
  FROM __reg_mpacked
  UNION ALL
  SELECT it + 1, newB, B,
         list_aggregate(list_transform(newB, lambda bk, k:
             list_aggregate(list_transform(bk, lambda v, j: abs(v - look[k][j])), 'max')), 'max')
  FROM (
    SELECT it, B, look,
           -- L1 prox (soft-threshold) on the L2-inclusive gradient step, per
           -- class per coefficient; the intercept (j=1) is unpenalized. No-op
           -- when l1 = 0, so unpenalized / pure-ridge fits are unchanged.
           list_transform(zstep, lambda zk, k: list_transform(zk, lambda zkj, j:
               CASE WHEN j = 1 THEN zkj
                    ELSE sign(zkj) * greatest(abs(zkj) - threshl1, 0.0) END)) AS newB
    FROM (
      SELECT it, B, look, step * l1 AS threshl1,
             -- z[k][j] = look + step*((1/n) sum_i xs_ij r_ik - l2*look [not intercept])
             list_transform(look, lambda bk, k: list_transform(bk, lambda lkj, j:
                 lkj + step * (list_sum(list_transform(res, lambda ob: ob.r[k] * ob.xs[j])) / n
                               - CASE WHEN j = 1 THEN 0.0 ELSE l2 * lkj END))) AS zstep
      FROM (
        SELECT it, B, n, step, look,
             list_transform(rows, lambda rw: struct_pack(
                 xs := rw.xs,
                 r := list_transform(rw.yv, lambda yvk, k:
                        yvk - exp(least(list_dot_product(rw.xs, look[k]), 700.0))
                              / (1.0 + list_sum(list_transform(look,
                                    lambda bj: exp(least(list_dot_product(rw.xs, bj), 700.0)))))))) AS res
      FROM (
        SELECT g.it, g.B, p.rows, p.n, c.step,
               list_transform(g.B, lambda bk, k:
                   list_transform(bk, lambda v, j: v + (g.it::DOUBLE / (g.it + 3)) * (v - g.prev[k][j]))) AS look
        FROM __reg_mgd g, __reg_mpacked p, __reg_mcfg c
        WHERE g.it < max_iter AND g.move >= tol
      )
    )
    )
  )
),
__reg_msol AS (SELECT B FROM __reg_mgd ORDER BY it DESC LIMIT 1)
SELECT class, feature, coefficient FROM (
  SELECT nr.nrf[k] AS class, '(Intercept)' AS feature,
         s.B[k][1] - coalesce(list_sum(list_transform(f.names,
             lambda nm, j: s.B[k][j + 1] * f.mus[j] / f.sigmas[j])), 0.0) AS coefficient
  FROM __reg_msol s, __reg_mfeats f, __reg_mnonref nr, range(1, len(nr.nrf) + 1) AS gk(k)
  UNION ALL
  SELECT nr.nrf[k], f.names[j], s.B[k][j + 1] / f.sigmas[j]
  FROM __reg_msol s, __reg_mfeats f, __reg_mnonref nr,
       range(1, len(nr.nrf) + 1) AS gk(k), range(1, f.d + 1) AS gj(j)
  UNION ALL
  SELECT (SELECT min(lab) FROM __reg_mylab), feat, 0.0
  FROM (SELECT '(Intercept)' AS feat UNION ALL SELECT unnest(names) FROM __reg_mfeats)
)
ORDER BY class, (feature = '(Intercept)') DESC, feature;

-- Shared softmax scorer used by multinom_predict / multinom_evaluate: returns
-- (rid, class, p) with p the class probability (NULL if a feature is missing).
CREATE OR REPLACE MACRO __reg_msoftmax(model, tbl) AS TABLE
WITH
__reg_mnum AS (SELECT row_number() OVER () AS __reg_rid__, * FROM query_table(tbl)),
__reg_mlong AS (
  SELECT __reg_rid__ AS rid, name AS col, value AS v
  FROM (UNPIVOT (SELECT __reg_rid__, TRY_CAST(COLUMNS(* EXCLUDE (__reg_rid__)) AS DOUBLE) FROM __reg_mnum)
        ON COLUMNS(* EXCLUDE (__reg_rid__)) INTO NAME name VALUE value)
),
__reg_mcoefs AS (SELECT class, feature, coefficient FROM query_table(model)),
__reg_mnf AS (SELECT count(*) AS nf FROM __reg_mcoefs
             WHERE feature != '(Intercept)' AND class = (SELECT min(class) FROM __reg_mcoefs)),
__reg_meta AS (
  SELECT n.__reg_rid__ AS rid, c.class,
    sum(CASE WHEN c.feature = '(Intercept)' THEN c.coefficient ELSE c.coefficient * l.v END) AS e,
    count(*) FILTER (WHERE c.feature != '(Intercept)' AND l.v IS NOT NULL) AS nm
  FROM __reg_mnum n CROSS JOIN __reg_mcoefs c
  LEFT JOIN __reg_mlong l ON l.rid = n.__reg_rid__ AND l.col = c.feature
  GROUP BY n.__reg_rid__, c.class
),
__reg_meta2 AS (SELECT rid, class, e, nm, max(e) OVER (PARTITION BY rid) AS maxe FROM __reg_meta)
SELECT rid, class,
       CASE WHEN nm = (SELECT nf FROM __reg_mnf)
            THEN exp(e - maxe) / sum(exp(e - maxe)) OVER (PARTITION BY rid) END AS p
FROM __reg_meta2;

CREATE OR REPLACE MACRO multinom_predict(model, tbl) AS TABLE
WITH __reg_mnum AS (SELECT row_number() OVER () AS __reg_rid__, * FROM query_table(tbl)),
__reg_mncheck AS (
  SELECT CASE WHEN coalesce(bool_or(lower(colname) IN ('pred', 'probs')), false)
              THEN error('multinom_predict: the input table already has a "pred" or "probs" column; rename or drop it first (e.g. SELECT * EXCLUDE (pred, probs))')
              ELSE true END AS ok
  FROM (SELECT * FROM (SELECT 1 AS __reg_one)
        LEFT JOIN (SELECT CAST(COLUMNS(*) AS VARCHAR) FROM query_table(tbl) LIMIT 1) ON true)
       UNPIVOT INCLUDE NULLS (v FOR colname IN (COLUMNS(* EXCLUDE (__reg_one))))
),
__reg_magg AS (
  SELECT rid, arg_max(class, p) AS pred, map(list(class ORDER BY class), list(p ORDER BY class)) AS probs
  FROM __reg_msoftmax(model, tbl) GROUP BY rid
)
SELECT n.* EXCLUDE (__reg_rid__), a.pred AS pred, a.probs AS probs
FROM __reg_mnum n LEFT JOIN __reg_magg a ON a.rid = n.__reg_rid__
WHERE (SELECT ok FROM __reg_mncheck)
ORDER BY n.__reg_rid__;

CREATE OR REPLACE MACRO multinom_evaluate(model, tbl, outcome) AS TABLE
WITH
__reg_mnum AS (SELECT row_number() OVER () AS __reg_rid__, * FROM query_table(tbl)),
__reg_mtrue AS (
  SELECT rid, val AS lab
  FROM (UNPIVOT (SELECT __reg_rid__ AS rid, CAST(COLUMNS(c -> c != '__reg_rid__') AS VARCHAR) FROM __reg_mnum)
        ON COLUMNS(* EXCLUDE (rid)) INTO NAME name VALUE val)
  WHERE name = outcome
),
__reg_mrm AS (
  SELECT s.rid, max(CASE WHEN s.class = t.lab THEN s.p END) AS p_true,
         arg_max(s.class, s.p) AS pred, any_value(t.lab) AS truelab
  FROM __reg_msoftmax(model, tbl) s JOIN __reg_mtrue t ON t.rid = s.rid
  WHERE s.p IS NOT NULL GROUP BY s.rid
)
SELECT count(*)::BIGINT AS n,
       avg(CASE WHEN pred = truelab THEN 1.0 ELSE 0.0 END) AS accuracy,
       -avg(ln(greatest(p_true, 1e-15))) AS log_loss
FROM __reg_mrm;


-- ---------------------------------------------------------------------------
-- Cross-validation for hyperparameter selection (pure SQL, all folds at once)
--
-- All k folds x |grid| models are fit SIMULTANEOUSLY in one recursive CTE (the
-- coefficient state is a list of k*|grid| vectors); each fold-model's gradient
-- sums only over rows outside its held-out fold, then held-out rows are scored.
-- Standardization is global (cv.glmnet's default); folds are (row# - 1) % k
-- (deterministic -- shuffle first if rows are ordered by the outcome). Returns
-- one row per grid value with the mean held-out deviance (squared error for
-- linear); pick the smallest cv_deviance. Cost scales with
-- k * |grid| * features * rows * iterations -- keep the grid modest.
--
--   cv_l2(tbl, outcome, family, l2_grid, k := 5)  -> (l2, cv_deviance)
--       ridge grid; family in linear/logistic/poisson/gamma
--   cv_l1(tbl, outcome, family, l1_grid, k := 5)  -> (l1, cv_deviance)
--       lasso grid; same families
--   cv_power(tbl, outcome, power_grid, k := 5)     -> (power, cv_deviance)
--       Tweedie variance power grid
--   cv_alpha(tbl, outcome, alpha_grid, k := 5)     -> (alpha, cv_deviance)
--       negative binomial dispersion grid
-- ---------------------------------------------------------------------------
CREATE OR REPLACE MACRO __reg_cv(tbl, outcome, family, grid, sweep, k, max_iter, learning_rate, tol) AS TABLE
WITH RECURSIVE
__reg_cv_chk AS (
  SELECT CASE
    WHEN family NOT IN ('linear','logistic','poisson','gamma','tweedie','nbinom')
      THEN error('cv: unsupported family ' || family)
    WHEN k < 2 THEN error('cv: k must be >= 2')
    WHEN len(grid) < 1 THEN error('cv: grid must be non-empty')
    WHEN sweep IN ('l2','l1') AND list_aggregate(grid,'min') < 0 THEN error('cv: penalty values must be >= 0')
    WHEN sweep = 'power' AND list_aggregate(grid,'min') < 1 THEN error('cv: tweedie power must be >= 1')
    WHEN sweep = 'alpha' AND list_aggregate(grid,'min') <= 0 THEN error('cv: nbinom alpha must be > 0')
    ELSE true END AS ok
),
__reg_cv_num AS MATERIALIZED (SELECT row_number() OVER () AS rid, * FROM query_table(tbl)),
__reg_cv_flong AS MATERIALIZED (
  SELECT rid, name AS col, value AS v
  FROM (UNPIVOT (SELECT rid, CAST(COLUMNS(c -> c != outcome AND c != 'rid') AS DOUBLE) FROM __reg_cv_num)
        ON COLUMNS(* EXCLUDE (rid)) INTO NAME name VALUE value)
),
__reg_cv_yraw AS MATERIALIZED (
  SELECT rid, val AS y, (rid - 1) % k AS fold
  FROM (UNPIVOT (SELECT rid, CAST(COLUMNS(c -> c != 'rid') AS DOUBLE) FROM __reg_cv_num)
        ON COLUMNS(* EXCLUDE (rid)) INTO NAME name VALUE val) WHERE name = outcome
),
__reg_cv_stats AS MATERIALIZED (
  -- j: feature position in name order. The per-row feature vector below is
  -- assembled with ORDER BY j rather than ORDER BY col; an ordered list
  -- aggregate carries its sort key with every value, so a VARCHAR key means
  -- n*d column-name strings in the sort buffers.
  SELECT col, row_number() OVER (ORDER BY col) AS j,
         CASE WHEN min(v)=max(v) THEN min(v) ELSE avg(v) END AS mu,
         CASE WHEN min(v)=max(v) THEN 1.0 ELSE stddev_pop(v) END AS sigma
  FROM __reg_cv_flong GROUP BY col
),
__reg_cv_feats AS MATERIALIZED (SELECT count(*)::INT AS d FROM __reg_cv_stats),
__reg_cv_ys AS (
  SELECT CASE WHEN family='linear' THEN avg(y) ELSE 0.0 END AS mu_y,
         CASE WHEN family='logistic' THEN 1.0
              WHEN family IN ('poisson','gamma','tweedie','nbinom') THEN (CASE WHEN avg(y)<1e-300 THEN 1.0 ELSE avg(y) END)
              WHEN coalesce(stddev_pop(y),0)<1e-300 THEN 1.0 ELSE stddev_pop(y) END AS sd_y
  FROM __reg_cv_yraw
),
__reg_cv_n AS (SELECT count(*)::DOUBLE AS n FROM __reg_cv_yraw),
__reg_cv_foldsz AS (SELECT fold AS f, count(*)::DOUBLE AS sz FROM __reg_cv_yraw GROUP BY fold),
-- per-model hyperparameters: model m = (g-1)*k + f + 1
__reg_cv_marr AS (
  SELECT list(l2 ORDER BY m) AS ml2, list(l1 ORDER BY m) AS ml1,
         list(pw ORDER BY m) AS mpow, list(alpi ORDER BY m) AS malp_int,
         list(f ORDER BY m) AS mfold, list(ntrain ORDER BY m) AS mntrain, count(*)::INT AS M
  FROM (
    SELECT (g-1)*k + f + 1 AS m, f,
           CASE WHEN sweep='l2' THEN grid[g] ELSE 0.0 END AS l2,
           CASE WHEN sweep='l1' THEN grid[g] ELSE 0.0 END AS l1,
           CASE WHEN sweep='power' THEN grid[g] ELSE 1.5 END AS pw,
           (CASE WHEN sweep='alpha' THEN grid[g] ELSE 1.0 END) * (SELECT sd_y FROM __reg_cv_ys) AS alpi,
           (SELECT n FROM __reg_cv_n) - coalesce((SELECT sz FROM __reg_cv_foldsz z WHERE z.f=t.f),0) AS ntrain
    FROM range(1,len(grid)+1) tg(g), range(0,k) t(f)
  )
),
__reg_cv_rows AS MATERIALIZED (
  SELECT x.rid, any_value(yr.fold) AS fold, any_value(yr.y) AS y,
         (any_value(yr.y) - any_value(ys.mu_y)) / any_value(ys.sd_y) AS yt,
         [1.0::DOUBLE] || list((x.v - s.mu)/s.sigma ORDER BY s.j) AS xs
  FROM __reg_cv_flong x JOIN __reg_cv_stats s ON s.col=x.col JOIN __reg_cv_yraw yr ON yr.rid=x.rid
  CROSS JOIN __reg_cv_ys ys GROUP BY x.rid
),
__reg_cv_packed AS MATERIALIZED (SELECT list(struct_pack(xs := xs, yt := yt, fold := fold)) AS rows FROM __reg_cv_rows),
__reg_cv_cfg AS (
  SELECT coalesce(learning_rate,
           CASE WHEN family='logistic' THEN 4.0/(f.d+1+4.0*list_aggregate(ma.ml2,'max'))
                ELSE 1.0/(f.d+1+list_aggregate(ma.ml2,'max')) END) AS step,
         f.d + 1 AS D1, ma.M AS M
  FROM __reg_cv_feats f, __reg_cv_marr ma, __reg_cv_chk chk WHERE chk.ok
),
__reg_cv_gd AS (
  SELECT 0 AS it,
         list_transform(range(c.M), lambda m: list_transform(range(c.D1), lambda j: 0.0::DOUBLE)) AS B,
         list_transform(range(c.M), lambda m: list_transform(range(c.D1), lambda j: 0.0::DOUBLE)) AS prev,
         1e308::DOUBLE AS move
  FROM __reg_cv_cfg c
  UNION ALL
  SELECT it+1, newB, B,
         list_aggregate(list_transform(newB, lambda bm, m:
             list_aggregate(list_transform(bm, lambda v, j: abs(v - look[m][j])), 'max')), 'max')
  FROM (
    SELECT it, B, look,
           -- L1 prox on the L2-inclusive gradient step, per model per coef
           list_transform(zstep, lambda zm, m: list_transform(zm, lambda zmj, j:
               CASE WHEN j = 1 THEN zmj
                    ELSE sign(zmj) * greatest(abs(zmj) - (step/damp)*ml1[m], 0.0) END)) AS newB
    FROM (
      SELECT it, B, look, step, damp, ml1,
             list_transform(look, lambda bm, m: list_transform(bm, lambda lmj, j:
                 lmj + (step/damp) * (
                   list_sum(list_transform(res, lambda ob:
                       CASE WHEN mfold[m] != ob.fold THEN ob.xs[j] * ob.r[m] ELSE 0.0 END)) / mntrain[m]
                   - CASE WHEN j = 1 THEN 0.0 ELSE ml2[m] * lmj END))) AS zstep
      FROM (
        SELECT it, B, look, step, mfold, ml2, ml1, mntrain, res,
               CASE WHEN family IN ('poisson','gamma','tweedie','nbinom')
                    THEN greatest(1.0, list_aggregate(list_transform(res,
                           lambda ob: list_aggregate(ob.hw, 'max')), 'max'))
                    ELSE 1.0 END AS damp
        FROM (
          SELECT it, B, look, step, mfold, ml2, ml1, mntrain, mpow, malp_int,
                 list_transform(rows, lambda rw: struct_pack(
                   xs := rw.xs, fold := rw.fold,
                   r := list_transform(look, lambda bm, m:
                     (CASE WHEN family='logistic' THEN rw.yt - 1.0/(1.0+exp(-list_dot_product(rw.xs,bm)))
                           WHEN family='poisson'  THEN rw.yt - exp(least(list_dot_product(rw.xs,bm),700.0))
                           WHEN family='gamma'    THEN rw.yt / exp(greatest(least(list_dot_product(rw.xs,bm),700.0),-700.0)) - 1.0
                           WHEN family='tweedie'  THEN (rw.yt - exp(greatest(least(list_dot_product(rw.xs,bm),700.0),-700.0)))
                                                       * pow(exp(greatest(least(list_dot_product(rw.xs,bm),700.0),-700.0)), 1.0 - mpow[m])
                           WHEN family='nbinom'   THEN (rw.yt - exp(least(list_dot_product(rw.xs,bm),700.0)))
                                                       / (1.0 + malp_int[m] * exp(least(list_dot_product(rw.xs,bm),700.0)))
                           ELSE rw.yt - list_dot_product(rw.xs,bm) END)),
                   hw := list_transform(look, lambda bm, m:
                     (CASE WHEN family='poisson' THEN exp(least(list_dot_product(rw.xs,bm),700.0))
                           WHEN family='gamma'   THEN rw.yt / exp(greatest(least(list_dot_product(rw.xs,bm),700.0),-700.0))
                           WHEN family='tweedie' THEN (2.0-mpow[m])*pow(exp(greatest(least(list_dot_product(rw.xs,bm),700.0),-700.0)),2.0-mpow[m])
                                                     + (mpow[m]-1.0)*rw.yt*pow(exp(greatest(least(list_dot_product(rw.xs,bm),700.0),-700.0)),1.0-mpow[m])
                           WHEN family='nbinom'  THEN exp(least(list_dot_product(rw.xs,bm),700.0))*(1.0+malp_int[m]*rw.yt)
                                                     / pow(1.0+malp_int[m]*exp(least(list_dot_product(rw.xs,bm),700.0)),2.0)
                           ELSE 0.0 END)))) AS res
          FROM (
            SELECT g.it, g.B, p.rows, c.step, ma.mfold, ma.ml2, ma.ml1, ma.mntrain, ma.mpow, ma.malp_int,
                   list_transform(g.B, lambda bm, m: list_transform(bm, lambda v, j:
                       v + (g.it::DOUBLE/(g.it+3)) * (v - g.prev[m][j]))) AS look
            FROM __reg_cv_gd g, __reg_cv_packed p, __reg_cv_cfg c, __reg_cv_marr ma
            WHERE g.it < max_iter AND g.move >= tol
          )
        )
      )
    )
  )
),
__reg_cv_sol AS (SELECT B FROM __reg_cv_gd ORDER BY it DESC LIMIT 1),
__reg_cv_score AS (
  SELECT gg.g AS g, r.y AS y,
         list_dot_product(r.xs, s.B[(gg.g - 1) * k + r.fold + 1]) AS eta,
         ys.mu_y AS mu_y, ys.sd_y AS sd_y,
         CASE WHEN sweep='power' THEN grid[gg.g] ELSE 1.5 END AS pw,
         CASE WHEN sweep='alpha' THEN grid[gg.g] ELSE 1.0 END AS al
  FROM __reg_cv_sol s, __reg_cv_rows r, __reg_cv_ys ys, range(1, len(grid)+1) gg(g)
)
SELECT grid[g] AS param,
       sum(CASE family
             WHEN 'linear'   THEN pow(y - (mu_y + sd_y * eta), 2)
             WHEN 'logistic' THEN -2.0 * (y * ln(greatest(1.0/(1.0+exp(-eta)),1e-15)) + (1-y)*ln(greatest(1.0-1.0/(1.0+exp(-eta)),1e-15)))
             WHEN 'poisson'  THEN 2.0 * ((CASE WHEN y>0 THEN y*ln(y/(sd_y*exp(eta))) ELSE 0.0 END) - (y - sd_y*exp(eta)))
             WHEN 'gamma'    THEN 2.0 * (-ln(y/(sd_y*exp(eta))) + (y - sd_y*exp(eta))/(sd_y*exp(eta)))
             WHEN 'tweedie'  THEN 2.0 * (pow(greatest(y,0.0),2.0-pw)/((1.0-pw)*(2.0-pw)) - y*pow(sd_y*exp(eta),1.0-pw)/(1.0-pw) + pow(sd_y*exp(eta),2.0-pw)/(2.0-pw))
             WHEN 'nbinom'   THEN 2.0 * ((CASE WHEN y>0 THEN y*ln(y/(sd_y*exp(eta))) ELSE 0.0 END) - (y + 1.0/al)*ln((y + 1.0/al)/(sd_y*exp(eta) + 1.0/al)))
           END) / (SELECT n FROM __reg_cv_n) AS cv_deviance
FROM __reg_cv_score
GROUP BY g, grid[g]
ORDER BY g;
CREATE OR REPLACE MACRO cv_l1(tbl, outcome, family, l1_grid, k := 5, max_iter := 20000, learning_rate := NULL, tol := 1e-8) AS TABLE
SELECT param AS l1, cv_deviance FROM __reg_cv(tbl, outcome, family, l1_grid, 'l1', k, max_iter, learning_rate, tol);
CREATE OR REPLACE MACRO cv_power(tbl, outcome, power_grid, k := 5, max_iter := 20000, learning_rate := NULL, tol := 1e-8) AS TABLE
SELECT param AS power, cv_deviance FROM __reg_cv(tbl, outcome, 'tweedie', power_grid, 'power', k, max_iter, learning_rate, tol);
CREATE OR REPLACE MACRO cv_alpha(tbl, outcome, alpha_grid, k := 5, max_iter := 20000, learning_rate := NULL, tol := 1e-8) AS TABLE
SELECT param AS alpha, cv_deviance FROM __reg_cv(tbl, outcome, 'nbinom', alpha_grid, 'alpha', k, max_iter, learning_rate, tol);

CREATE OR REPLACE MACRO cv_l2(tbl, outcome, family, l2_grid, k := 5, max_iter := 20000, learning_rate := NULL, tol := 1e-8) AS TABLE
SELECT param AS l2, cv_deviance FROM __reg_cv(tbl, outcome, family, l2_grid, 'l2', k, max_iter, learning_rate, tol);


-- ---------------------------------------------------------------------------
-- Negative binomial dispersion estimation (profile likelihood, pure SQL)
--
-- nbinom_dispersion(tbl, outcome, alpha_grid) fits the NB2 mean model for each
-- alpha in the grid SIMULTANEOUSLY (one model per alpha, all on the full data,
-- in a single recursive CTE) and returns (alpha, loglik) -- the profile
-- log-likelihood. The alpha with the largest loglik is the (grid-resolution)
-- maximum-likelihood dispersion estimate; feed it back into nbinom_fit. Refine
-- with a finer grid around the peak for a sharper estimate. loglik matches
-- statsmodels' fixed-alpha GLM NegativeBinomial.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE MACRO nbinom_dispersion(tbl, outcome, alpha_grid, max_iter := 20000, learning_rate := NULL, tol := 1e-8) AS TABLE
WITH RECURSIVE
__reg_nbd_chk AS (
  SELECT CASE
    WHEN len(alpha_grid) < 1 THEN error('nbinom_dispersion: alpha_grid must be non-empty')
    WHEN list_aggregate(alpha_grid,'min') <= 0 THEN error('nbinom_dispersion: alpha values must be > 0')
    ELSE true END AS ok
),
__reg_nbd_num AS MATERIALIZED (SELECT row_number() OVER () AS rid, * FROM query_table(tbl)),
__reg_nbd_flong AS MATERIALIZED (
  SELECT rid, name AS col, value AS v
  FROM (UNPIVOT (SELECT rid, CAST(COLUMNS(c -> c != outcome AND c != 'rid') AS DOUBLE) FROM __reg_nbd_num)
        ON COLUMNS(* EXCLUDE (rid)) INTO NAME name VALUE value)
),
__reg_nbd_yraw AS MATERIALIZED (
  SELECT rid, val AS y
  FROM (UNPIVOT (SELECT rid, CAST(COLUMNS(c -> c != 'rid') AS DOUBLE) FROM __reg_nbd_num)
        ON COLUMNS(* EXCLUDE (rid)) INTO NAME name VALUE val) WHERE name = outcome
),
__reg_nbd_ycheck AS (
  SELECT CASE WHEN min(y) < 0 THEN error('nbinom_dispersion: outcome must be non-negative') ELSE true END AS ok
  FROM __reg_nbd_yraw
),
__reg_nbd_stats AS MATERIALIZED (
  -- j: feature position in name order. The per-row feature vector below is
  -- assembled with ORDER BY j rather than ORDER BY col; an ordered list
  -- aggregate carries its sort key with every value, so a VARCHAR key means
  -- n*d column-name strings in the sort buffers.
  SELECT col, row_number() OVER (ORDER BY col) AS j,
         CASE WHEN min(v)=max(v) THEN min(v) ELSE avg(v) END AS mu,
         CASE WHEN min(v)=max(v) THEN 1.0 ELSE stddev_pop(v) END AS sigma
  FROM __reg_nbd_flong GROUP BY col
),
__reg_nbd_feats AS MATERIALIZED (SELECT count(*)::INT AS d FROM __reg_nbd_stats),
__reg_nbd_ys AS (SELECT CASE WHEN avg(y)<1e-300 THEN 1.0 ELSE avg(y) END AS sd_y FROM __reg_nbd_yraw),
__reg_nbd_n AS (SELECT count(*)::DOUBLE AS n FROM __reg_nbd_yraw),
-- one model per grid alpha (no folds); alpha_int = alpha * mean(y)
__reg_nbd_marr AS (
  SELECT list(alpi ORDER BY g) AS malp_int, count(*)::INT AS M
  FROM (SELECT g, alpha_grid[g] * (SELECT sd_y FROM __reg_nbd_ys) AS alpi FROM range(1,len(alpha_grid)+1) t(g))
),
__reg_nbd_rows AS MATERIALIZED (
  SELECT x.rid, any_value(yr.y) AS y, any_value(yr.y) / any_value(ys.sd_y) AS yt,
         [1.0::DOUBLE] || list((x.v - s.mu)/s.sigma ORDER BY s.j) AS xs
  FROM __reg_nbd_flong x JOIN __reg_nbd_stats s ON s.col=x.col JOIN __reg_nbd_yraw yr ON yr.rid=x.rid
  CROSS JOIN __reg_nbd_ys ys GROUP BY x.rid
),
__reg_nbd_packed AS MATERIALIZED (SELECT list(struct_pack(xs := xs, yt := yt)) AS rows, count(*)::DOUBLE AS n FROM __reg_nbd_rows),
__reg_nbd_cfg AS (
  SELECT coalesce(learning_rate, 1.0/(f.d+1)) AS step, f.d+1 AS D1, ma.M AS M
  FROM __reg_nbd_feats f, __reg_nbd_marr ma, __reg_nbd_chk c1, __reg_nbd_ycheck c2 WHERE c1.ok AND c2.ok
),
__reg_nbd_gd AS (
  SELECT 0 AS it,
         list_transform(range(c.M), lambda m: list_transform(range(c.D1), lambda j: 0.0::DOUBLE)) AS B,
         list_transform(range(c.M), lambda m: list_transform(range(c.D1), lambda j: 0.0::DOUBLE)) AS prev,
         1e308::DOUBLE AS move
  FROM __reg_nbd_cfg c
  UNION ALL
  SELECT it+1, newB, B,
         list_aggregate(list_transform(newB, lambda bm, m:
             list_aggregate(list_transform(bm, lambda v, j: abs(v - look[m][j])), 'max')), 'max')
  FROM (
    SELECT it, B, look,
           list_transform(look, lambda bm, m: list_transform(bm, lambda lmj, j:
               lmj + (step/damp) * (list_sum(list_transform(res, lambda ob: ob.xs[j] * ob.r[m])) / n))) AS newB
    FROM (
      SELECT it, B, look, step, n, res,
             greatest(1.0, list_aggregate(list_transform(res, lambda ob: list_aggregate(ob.hw,'max')), 'max')) AS damp
      FROM (
        SELECT it, B, look, step, n, malp_int,
               list_transform(rows, lambda rw: struct_pack(
                 r := list_transform(look, lambda bm, m:
                        (rw.yt - exp(least(list_dot_product(rw.xs,bm),700.0)))
                        / (1.0 + malp_int[m]*exp(least(list_dot_product(rw.xs,bm),700.0)))),
                 hw := list_transform(look, lambda bm, m:
                        exp(least(list_dot_product(rw.xs,bm),700.0))*(1.0+malp_int[m]*rw.yt)
                        / pow(1.0+malp_int[m]*exp(least(list_dot_product(rw.xs,bm),700.0)),2.0)),
                 xs := rw.xs)) AS res
        FROM (
          SELECT g.it, g.B, p.rows, p.n, c.step, ma.malp_int,
                 list_transform(g.B, lambda bm, m: list_transform(bm, lambda v, j:
                     v + (g.it::DOUBLE/(g.it+3)) * (v - g.prev[m][j]))) AS look
          FROM __reg_nbd_gd g, __reg_nbd_packed p, __reg_nbd_cfg c, __reg_nbd_marr ma
          WHERE g.it < max_iter AND g.move >= tol
        )
      )
    )
  )
),
__reg_nbd_sol AS (SELECT B FROM __reg_nbd_gd ORDER BY it DESC LIMIT 1),
-- profile NB2 log-likelihood per grid alpha (r = 1/alpha), mu = mean(y)*exp(eta)
__reg_nbd_ll AS (
  SELECT gg.g AS g, r.y AS y, alpha_grid[gg.g] AS alpha,
         ys.sd_y * exp(list_dot_product(r.xs, s.B[gg.g])) AS mu
  FROM __reg_nbd_sol s, __reg_nbd_rows r, __reg_nbd_ys ys, range(1,len(alpha_grid)+1) gg(g)
)
SELECT alpha,
       sum(lgamma(y + 1.0/alpha) - lgamma(1.0/alpha) - lgamma(y + 1)
           + (1.0/alpha) * ln((1.0/alpha)/(1.0/alpha + mu))
           + y * ln(mu/(1.0/alpha + mu))) AS loglik
FROM __reg_nbd_ll
GROUP BY alpha
ORDER BY alpha;


-- ---------------------------------------------------------------------------
-- Two-stage grid refinement (pure SQL)
--
-- reg_grid(lo, hi, n) builds a linear (or, with log_spaced := true,
-- log-spaced) grid of n points in [lo, hi] -- handy for the COARSE grid.
--
-- The *_refine wrappers do a two-stage sweep in a single call: they run the
-- coarse grid, locate the best hyperparameter, then re-sweep a finer grid of
-- n_refine points BRACKETING that best value between its two coarse-grid
-- neighbours, and return the refined (param, metric) curve. If the best value
-- sits on a grid boundary the refined grid is one-sided toward the interior.
-- Take the argmin cv_deviance (or argmax loglik) of the result as the estimate.
--
--   cv_l2_refine / cv_l1_refine / cv_power_refine / cv_alpha_refine  -> min cv_deviance
--   nbinom_dispersion_refine                                         -> max loglik
--
-- Two stages of n points resolve the optimum as finely as ~n^2 of a single
-- grid at a fraction of the cost. Assumes the coarse grid brackets the optimum
-- (widen it if the best lands on an endpoint and you expect the optimum beyond).
-- ---------------------------------------------------------------------------

-- linear or log-spaced grid of n points spanning [lo, hi]
CREATE OR REPLACE MACRO reg_grid(lo, hi, n, log_spaced := false) AS (
  CASE WHEN n < 2 THEN [lo::DOUBLE]
       WHEN log_spaced THEN list_transform(range(n), lambda i: exp(ln(lo) + (ln(hi)-ln(lo))*i/(n-1.0)))
       ELSE list_transform(range(n), lambda i: lo + (hi-lo)*i/(n-1.0)) END
);

-- a finer grid of n points bracketing `best` between its neighbours in `grid`
-- (one-sided toward the interior when `best` is a grid endpoint)
CREATE OR REPLACE MACRO __reg_refine_grid(grid, best, n) AS (
  WITH nb AS (
    SELECT coalesce((SELECT max(g) FROM unnest(grid) AS t(g) WHERE g < best), best) AS lo,
           coalesce((SELECT min(g) FROM unnest(grid) AS t(g) WHERE g > best), best) AS hi
  )
  SELECT CASE WHEN lo = hi THEN [best::DOUBLE]
              ELSE list_transform(range(n), lambda i: lo + (hi-lo)*i/(n-1.0)) END
  FROM nb
);

-- two-stage CV engine: sweep coarse grid, then a finer grid around the best
CREATE OR REPLACE MACRO __reg_cv_refine(tbl, outcome, family, grid, sweep, k, n_refine, max_iter, learning_rate, tol) AS TABLE
WITH __rr_best AS (
  SELECT param AS b FROM __reg_cv(tbl, outcome, family, grid, sweep, k, max_iter, learning_rate, tol)
  ORDER BY cv_deviance LIMIT 1
)
SELECT param, cv_deviance
FROM __reg_cv(tbl, outcome, family,
              (SELECT __reg_refine_grid(grid, b, n_refine) FROM __rr_best),
              sweep, k, max_iter, learning_rate, tol);

CREATE OR REPLACE MACRO cv_l2_refine(tbl, outcome, family, l2_grid, k := 5, n_refine := 10, max_iter := 20000, learning_rate := NULL, tol := 1e-8) AS TABLE
SELECT param AS l2, cv_deviance FROM __reg_cv_refine(tbl, outcome, family, l2_grid, 'l2', k, n_refine, max_iter, learning_rate, tol);
CREATE OR REPLACE MACRO cv_l1_refine(tbl, outcome, family, l1_grid, k := 5, n_refine := 10, max_iter := 20000, learning_rate := NULL, tol := 1e-8) AS TABLE
SELECT param AS l1, cv_deviance FROM __reg_cv_refine(tbl, outcome, family, l1_grid, 'l1', k, n_refine, max_iter, learning_rate, tol);
CREATE OR REPLACE MACRO cv_power_refine(tbl, outcome, power_grid, k := 5, n_refine := 10, max_iter := 20000, learning_rate := NULL, tol := 1e-8) AS TABLE
SELECT param AS power, cv_deviance FROM __reg_cv_refine(tbl, outcome, 'tweedie', power_grid, 'power', k, n_refine, max_iter, learning_rate, tol);
CREATE OR REPLACE MACRO cv_alpha_refine(tbl, outcome, alpha_grid, k := 5, n_refine := 10, max_iter := 20000, learning_rate := NULL, tol := 1e-8) AS TABLE
SELECT param AS alpha, cv_deviance FROM __reg_cv_refine(tbl, outcome, 'nbinom', alpha_grid, 'alpha', k, n_refine, max_iter, learning_rate, tol);

-- two-stage dispersion estimation: refine around the profile-likelihood peak
CREATE OR REPLACE MACRO nbinom_dispersion_refine(tbl, outcome, alpha_grid, n_refine := 10, max_iter := 20000, learning_rate := NULL, tol := 1e-8) AS TABLE
WITH __rr_best AS (
  SELECT alpha AS b FROM nbinom_dispersion(tbl, outcome, alpha_grid, max_iter, learning_rate, tol)
  ORDER BY loglik DESC LIMIT 1
)
SELECT alpha, loglik
FROM nbinom_dispersion(tbl, outcome,
       (SELECT __reg_refine_grid(alpha_grid, b, n_refine) FROM __rr_best),
       max_iter, learning_rate, tol);


-- ===========================================================================
-- Inference: standard errors, z/t statistics, p-values, confidence intervals
--   Cov(beta) = phi * (X'WX)^{-1}, with the expected-information (Fisher) IRLS
--   working weight per family; z (fixed dispersion) or Student-t with n-d df
--   (estimated dispersion) for p-values / CIs. All in pure SQL:
--     * norm_cdf / norm_ppf  -- West/Hart + Acklam+Halley (double precision)
--     * t_cdf   / t_ppf      -- regularized incomplete beta (Lentz CF) + Newton
--     * Gauss-Jordan matrix inversion with partial pivoting (inlined)
-- ===========================================================================

-- ---- Standard normal CDF and quantile (closed form, ~1e-15) ----------------
CREATE OR REPLACE MACRO __reg_norm_q(a) AS (          -- upper tail P(Z>a), a>=0
  CASE
    WHEN a > 37.0 THEN 0.0
    WHEN a < 7.071067811865475244 THEN
      exp(-a*a/2.0)
      * ((((((0.0352624965998911*a+0.700383064443688)*a+6.37396220353165)*a
            +33.912866078383)*a+112.079291497871)*a+221.213596169931)*a+220.206867912376)
      / (((((((0.0883883476483184*a+1.75566716318264)*a+16.064177579207)*a
            +86.7807322029461)*a+296.564248779674)*a+637.333633378831)*a
            +793.826512519948)*a+440.413735824752)
    ELSE exp(-a*a/2.0) / (a+1.0/(a+2.0/(a+3.0/(a+4.0/(a+0.65))))) / 2.506628274631
  END
);
CREATE OR REPLACE MACRO norm_cdf(z) AS (
  CASE WHEN z >= 0.0 THEN 1.0 - __reg_norm_q(z::DOUBLE) ELSE __reg_norm_q((-z)::DOUBLE) END
);
CREATE OR REPLACE MACRO __reg_acklam_tail(q) AS (
  (((((-7.784894002430293e-03*q-3.223964580411365e-01)*q-2.400758277161838e+00)*q
      -2.549732539343734e+00)*q+4.374664141464968e+00)*q+2.938163982698783e+00)
  / ((((7.784695709041462e-03*q+3.224671290700398e-01)*q+2.445134137142996e+00)*q
      +3.754408661907416e+00)*q+1.0)
);
CREATE OR REPLACE MACRO __reg_acklam_central(q) AS (
  (((((-3.969683028665376e+01*(q*q)+2.209460984245205e+02)*(q*q)-2.759285104469687e+02)*(q*q)
      +1.383577518672690e+02)*(q*q)-3.066479806614716e+01)*(q*q)+2.506628277459239e+00)*q
  / (((((-5.447609879822406e+01*(q*q)+1.615858368580409e+02)*(q*q)-1.556989798598866e+02)*(q*q)
      +6.680131188771972e+01)*(q*q)-1.328068155288572e+01)*(q*q)+1.0)
);
CREATE OR REPLACE MACRO __reg_norm_ppf_raw(p) AS (
  CASE WHEN p < 0.02425  THEN  __reg_acklam_tail(sqrt(-2.0*ln(p)))
       WHEN p <= 0.97575 THEN  __reg_acklam_central(p-0.5)
       ELSE                   -__reg_acklam_tail(sqrt(-2.0*ln(1.0-p))) END
);
CREATE OR REPLACE MACRO __reg_norm_ppf_halley(x0, p) AS (
  x0 - ((norm_cdf(x0)-p)*2.506628274631*exp(x0*x0/2.0))
       / (1.0 + x0*((norm_cdf(x0)-p)*2.506628274631*exp(x0*x0/2.0))/2.0)
);
-- Macro expansion is textual: every reference to a parameter re-expands the
-- caller's whole argument expression. __reg_norm_ppf_halley references x0 dozens
-- of times (twice through norm_cdf, which itself references its argument ~15
-- times per branch of __reg_norm_q), so writing this the obvious way --
--   __reg_norm_ppf_halley(__reg_norm_ppf_raw(p), p)
-- -- pastes the __reg_norm_ppf_raw tree in ~60 times and the caller's `p`
-- expression hundreds of times. The binder is superlinear in expression-tree
-- size, so with a compound argument that alone cost ~2.5s of query BINDING
-- (before a single row is read), which is what made *_summary and *_predict_ci
-- take seconds regardless of table size.
-- A one-element list_transform gives a genuine local binding: the argument is
-- evaluated once and the body refers to a cheap lambda variable. Same result to
-- the last bit; binding drops to ~6ms.
CREATE OR REPLACE MACRO norm_ppf(p) AS (
  list_transform([p::DOUBLE], pp ->
    list_transform([__reg_norm_ppf_raw(pp)], x0 -> __reg_norm_ppf_halley(x0, pp))[1]
  )[1]
);

-- ---- Student-t CDF and quantile via regularized incomplete beta ------------
CREATE OR REPLACE MACRO __reg_fpmin(z) AS (CASE WHEN abs(z) < 1e-30 THEN 1e-30 ELSE z END);
CREATE OR REPLACE MACRO __reg_bcf_aa(a, b, x, j) AS (   -- j-th continued-fraction coefficient
  CASE WHEN j % 2 = 1
       THEN  ((j+1)//2)::DOUBLE * (b - ((j+1)//2)::DOUBLE) * x
             / ((a - 1.0 + 2.0*((j+1)//2)::DOUBLE) * (a + 2.0*((j+1)//2)::DOUBLE))
       ELSE -(a + ((j+1)//2)::DOUBLE) * (a + b + ((j+1)//2)::DOUBLE) * x
             / ((a + 2.0*((j+1)//2)::DOUBLE) * (a + 1.0 + 2.0*((j+1)//2)::DOUBLE))
  END
);
-- Modified-Lentz CF (400 terms). The j-th CF coefficient is precomputed into the
-- term list (aa) rather than recomputed inside the fold: DuckDB does not
-- common-subexpression-eliminate inside a lambda body, so the four textual
-- __reg_bcf_aa(...) uses below would otherwise each be evaluated per term.
CREATE OR REPLACE MACRO __reg_betacf(a, b, x) AS (
  list_reduce(
    [ struct_pack(c := 1.0::DOUBLE,
                  d := 1.0 / __reg_fpmin(1.0 - (a+b)*x/(a+1.0)),
                  h := 1.0 / __reg_fpmin(1.0 - (a+b)*x/(a+1.0)), aa := 0.0::DOUBLE) ]
    || list_transform(range(1, 401),
         lambda i: struct_pack(c := 0.0::DOUBLE, d := 0.0::DOUBLE, h := 0.0::DOUBLE,
                               aa := __reg_bcf_aa(a,b,x,i)::DOUBLE)),
    (acc, e) -> struct_pack(
       c  := __reg_fpmin(1.0 + e.aa / acc.c),
       d  := 1.0 / __reg_fpmin(1.0 + e.aa * acc.d),
       h  := acc.h * (1.0 / __reg_fpmin(1.0 + e.aa * acc.d))
                   * __reg_fpmin(1.0 + e.aa / acc.c),
       aa := e.aa)
  ).h
);
-- Regularized incomplete beta I_x(a,b). Macro expansion is textual, so a
-- __reg_betacf call in each CASE branch would put two copies of the 400-term
-- fold in the plan. The prefactor is symmetric under (a,b,x) -> (b,a,1-x), so
-- only the arguments and the sign flip: one expansion covers both branches.
CREATE OR REPLACE MACRO __reg_betai(a, b, x) AS (
  CASE WHEN x <= 0.0 THEN 0.0
       WHEN x >= 1.0 THEN 1.0
       ELSE (CASE WHEN x < (a+1.0)/(a+b+2.0) THEN 0.0 ELSE 1.0 END)
          + (CASE WHEN x < (a+1.0)/(a+b+2.0) THEN 1.0 ELSE -1.0 END)
          * exp(lgamma(a+b)-lgamma(a)-lgamma(b)+a*ln(x)+b*ln(1.0-x))
          * __reg_betacf(CASE WHEN x < (a+1.0)/(a+b+2.0) THEN a ELSE b END,
                         CASE WHEN x < (a+1.0)/(a+b+2.0) THEN b ELSE a END,
                         CASE WHEN x < (a+1.0)/(a+b+2.0) THEN x ELSE 1.0-x END)
          / (CASE WHEN x < (a+1.0)/(a+b+2.0) THEN a ELSE b END)
  END
);
-- P(T>t), cancellation-free tail. Likewise folded to a single __reg_betai
-- expansion (t_ppf runs 13 Newton steps, each expanding this macro).
CREATE OR REPLACE MACRO __reg_t_sf(t, df) AS (
  (CASE WHEN t >= 0.0 THEN 0.0 ELSE 1.0 END)
  + (CASE WHEN t >= 0.0 THEN 0.5 ELSE -0.5 END)
    * __reg_betai(df/2.0, 0.5, df/(df + t*t))
);
CREATE OR REPLACE MACRO t_cdf(t, df) AS ( 1.0 - __reg_t_sf(t::DOUBLE, df::DOUBLE) );
CREATE OR REPLACE MACRO __reg_t_pdf(t, df) AS (
  exp(lgamma((df+1.0)/2.0) - lgamma(df/2.0) - 0.5*ln(df * 3.141592653589793::DOUBLE))
  * pow(1.0 + t*t/df, -(df+1.0)/2.0)
);
-- Newton from the normal quantile. p and df are bound to lambda variables for
-- the same reason as in norm_ppf above: the Newton body expands __reg_t_sf and
-- __reg_t_pdf, which reference df many times over, and each such reference would
-- otherwise paste in the caller's whole `df` expression.
CREATE OR REPLACE MACRO __reg_t_ppf(p, df) AS (
  CASE WHEN df = 1.0 THEN tan(3.141592653589793 * (p - 0.5))    -- Cauchy: exact
       ELSE list_transform([p::DOUBLE], pp ->
              list_transform([df::DOUBLE], dd ->
                list_reduce(
                  [ norm_ppf(pp) ] || list_transform(range(1, 13), lambda i: 0.0::DOUBLE),
                  (t, e) -> t - ((1.0 - __reg_t_sf(t, dd)) - pp) / __reg_t_pdf(t, dd))
              )[1]
            )[1]
  END
);
CREATE OR REPLACE MACRO t_ppf(p, df) AS ( __reg_t_ppf(p::DOUBLE, df::DOUBLE) );

-- ---- Shared coefficient-inference core -------------------------------------
CREATE OR REPLACE MACRO __reg_summary(model, tbl, outcome, family, caller,
                                      conf_level, offset_col, weights_col, power, alpha,
                                      robust, cluster_col) AS TABLE
WITH RECURSIVE
mdl AS (SELECT feature, coefficient FROM query_table(model)),
-- Feature position in name order, so the per-row feature vector below can be
-- ordered by an integer rather than by the VARCHAR column name (see __reg_stats).
mdlj AS (SELECT feature, row_number() OVER (ORDER BY feature) AS j
         FROM mdl WHERE feature != '(Intercept)'),
beta AS (
  SELECT ([ coalesce((SELECT coefficient FROM mdl WHERE feature = '(Intercept)'), 0.0) ]::DOUBLE[]
          || list(coefficient ORDER BY feature) FILTER (WHERE feature != '(Intercept)')) AS bvec,
         ([ '(Intercept)' ]
          || list(feature ORDER BY feature) FILTER (WHERE feature != '(Intercept)')) AS names,
         (count(*) FILTER (WHERE feature != '(Intercept)'))::INT AS k
  FROM mdl
),
num AS (SELECT row_number() OVER () AS rid, * FROM query_table(tbl)),
alllong AS (
  SELECT rid, name AS col, value AS v
  FROM (UNPIVOT (SELECT rid, TRY_CAST(COLUMNS(* EXCLUDE (rid)) AS DOUBLE) FROM num)
        ON COLUMNS(* EXCLUDE (rid)) INTO NAME name VALUE value)
),
yv AS (SELECT rid, v AS y  FROM alllong WHERE col = outcome),
ov AS (SELECT rid, v AS o  FROM alllong WHERE col = offset_col),
wv AS (SELECT rid, v AS wt FROM alllong WHERE col = weights_col),
clv AS (SELECT rid, v AS cl FROM alllong WHERE col = cluster_col),
feat AS (
  SELECT l.rid, [1.0::DOUBLE] || list(l.v ORDER BY m.j) AS xs, count(*)::INT AS nf
  FROM alllong l JOIN mdlj m ON m.feature = l.col
  GROUP BY l.rid
),
rows0 AS (
  SELECT f.rid, f.xs, y.y,
         CASE WHEN offset_col IS NULL THEN 0.0 ELSE o.o END AS off,
         CASE WHEN weights_col IS NULL THEN 1.0 ELSE coalesce(w.wt, 1.0) END AS wt
  FROM feat f
  JOIN yv y ON y.rid = f.rid
  LEFT JOIN ov o ON o.rid = f.rid
  LEFT JOIN wv w ON w.rid = f.rid
  CROSS JOIN beta
  WHERE f.nf = beta.k AND y.y IS NOT NULL
    AND (offset_col IS NULL OR o.o IS NOT NULL)
),
rww AS (
  SELECT r.rid, r.xs, r.y, r.wt, mu,
         r.wt * (CASE family
                   WHEN 'logistic' THEN mu*(1.0-mu)  WHEN 'linear' THEN 1.0
                   WHEN 'poisson'  THEN mu           WHEN 'gamma'  THEN 1.0
                   WHEN 'tweedie'  THEN pow(mu, 2.0-power)
                   WHEN 'nbinom'   THEN mu/(1.0+alpha*mu) END) AS w,
         r.wt * (CASE family
                   WHEN 'logistic' THEN (r.y-mu)*(r.y-mu)/(mu*(1.0-mu))
                   WHEN 'linear'   THEN (r.y-mu)*(r.y-mu)
                   WHEN 'poisson'  THEN (r.y-mu)*(r.y-mu)/mu
                   WHEN 'gamma'    THEN (r.y-mu)*(r.y-mu)/(mu*mu)
                   WHEN 'tweedie'  THEN (r.y-mu)*(r.y-mu)/pow(mu, power)
                   WHEN 'nbinom'   THEN (r.y-mu)*(r.y-mu)/(mu*(1.0+alpha*mu)) END) AS pearson
  FROM (
    -- eta clamped to [-700, 700] (as the fit does) so mu = exp(eta) never overflows
    SELECT rid, xs, y, wt,
           CASE family WHEN 'logistic' THEN 1.0/(1.0+exp(-greatest(-700.0, least(eta, 700.0))))
                       WHEN 'linear'   THEN eta
                       ELSE exp(greatest(-700.0, least(eta, 700.0))) END AS mu
    FROM (SELECT rid, xs, y, wt, off + list_dot_product(xs, (SELECT bvec FROM beta)) AS eta FROM rows0)
  ) r
),
dims AS (SELECT count(*)::INT AS n, (SELECT k FROM beta)+1 AS d FROM rww),
idx AS (SELECT unnest(range(1, (SELECT d FROM dims)+1)) AS i),
pairs AS (SELECT a.i AS a, b.i AS b FROM idx a, idx b),
xwx AS (
  SELECT list(rowlist ORDER BY a) AS A FROM (
    SELECT a, list(val ORDER BY b) AS rowlist FROM (
      SELECT p.a AS a, p.b AS b, sum(rww.w * rww.xs[p.a] * rww.xs[p.b]) AS val
      FROM rww, pairs p GROUP BY p.a, p.b
    ) GROUP BY a
  )
),
-- Scale X'WX to unit diagonal (correlation form) before inversion: makes the
-- singular-pivot test scale-invariant and squares less conditioning error.
-- dsc[j] = sqrt(diag_j); Cov = phi * D^-1 R^-1 D^-1, so SE_j = sqrt(phi*Rinv_jj)/dsc_j.
scal AS (
  SELECT A, list_transform(A, lambda row, i: CASE WHEN row[i] > 1e-300 THEN sqrt(row[i]) ELSE 1.0 END) AS dsc
  FROM xwx
),
rscaled AS (
  SELECT dsc, list_transform(A, lambda row, i: list_transform(row, lambda v, j: v/(dsc[i]*dsc[j]))) AS R
  FROM scal
),
gj(k, d, sing, M) AS (
  SELECT 0, len(R), false,
         list_transform(R, lambda row, i:
             list_concat(list_transform(row, lambda v, j: v::DOUBLE),
                         list_transform(row, lambda v, j: CASE WHEN j = i THEN 1.0 ELSE 0.0 END)))
  FROM rscaled
  UNION ALL
  SELECT col, d, sing OR abs(piv) < 1e-12,
         CASE WHEN abs(piv) < 1e-12 THEN Mswap
              ELSE list_transform(Mswap, lambda row, i:
                       CASE WHEN i = col THEN normpivot
                            ELSE list_transform(row, lambda v, j: v - row[col]*normpivot[j]) END) END
  FROM (
    SELECT col, d, sing, Mswap, Mswap[col][col] AS piv,
           list_transform(Mswap[col], lambda v, j: v / Mswap[col][col]) AS normpivot
    FROM (
      SELECT col, d, sing,
             list_transform(M, lambda row, i:
                 CASE WHEN i = col THEN M[p] WHEN i = p THEN M[col] ELSE row END) AS Mswap
      FROM (
        SELECT col, d, sing, M, list_position(pcol, list_aggregate(pcol, 'max')) AS p
        FROM (
          SELECT k, k+1 AS col, d, sing, M,
                 list_transform(M, lambda row, i:
                     CASE WHEN i >= k+1 THEN abs(row[k+1]) ELSE -1e308 END) AS pcol
          FROM gj WHERE k < d
        )
      )
    )
  )
),
covinv AS (
  SELECT CASE WHEN sing THEN NULL
              ELSE list_transform(M, lambda row, i: list_slice(row, d+1, 2*d)) END AS Rinv
  FROM gj WHERE k = d
),
disp AS (
  SELECT CASE WHEN family IN ('linear','gamma','tweedie')
              THEN (SELECT sum(pearson) FROM rww) / nullif((SELECT n-d FROM dims), 0)
              ELSE 1.0 END AS phi,
         family IN ('linear','gamma','tweedie') AS est,
         (SELECT (n-d)::DOUBLE FROM dims) AS df
),
-- ---- Robust (sandwich) covariance: Cov = A^-1 B A^-1 -----------------------
-- Computed alongside the model-based covariance; selected when robust != 'none'
-- or cluster_col is given. A = X'diag(a*hw)X uses the OBSERVED-info weight hw
-- (matches statsmodels for the non-canonical log-link families); the meat B is
-- the score-outer-product. Dispersion-free -> z inference. e_i = the GD residual.
robrow AS (
  SELECT rww.rid, rww.xs, rww.wt, cl.cl AS cl,
         rww.wt * (CASE family
                     WHEN 'logistic' THEN mu*(1.0-mu)  WHEN 'linear' THEN 1.0
                     WHEN 'poisson'  THEN mu
                     WHEN 'gamma'    THEN rww.y/mu
                     WHEN 'tweedie'  THEN (2.0-power)*pow(mu,2.0-power) + (power-1.0)*rww.y*pow(mu,1.0-power)
                     WHEN 'nbinom'   THEN mu*(1.0+alpha*rww.y)/pow(1.0+alpha*mu,2.0) END) AS hwt,
         rww.wt * (CASE family
                     WHEN 'gamma'   THEN (rww.y-mu)/mu
                     WHEN 'tweedie' THEN (rww.y-mu)*pow(mu,1.0-power)
                     WHEN 'nbinom'  THEN (rww.y-mu)/(1.0+alpha*mu)
                     ELSE rww.y-mu END) AS sc                       -- score scalar = a*r
  FROM rww LEFT JOIN clv cl ON cl.rid = rww.rid
),
rdims AS (SELECT count(DISTINCT cl)::INT AS G FROM robrow),
breadA AS (
  SELECT list(rowlist ORDER BY a) AS A FROM (
    SELECT a, list(val ORDER BY b) AS rowlist FROM (
      SELECT p.a AS a, p.b AS b, sum(robrow.hwt * robrow.xs[p.a] * robrow.xs[p.b]) AS val
      FROM robrow, pairs p GROUP BY p.a, p.b) GROUP BY a)
),
breadinv AS (
  SELECT list_transform(RAinv, lambda row, i: list_transform(row, lambda v, j: v/(dscA[i]*dscA[j]))) AS Ainv
  FROM (SELECT dscA, __reg_matinv(list_transform(A, lambda row, i: list_transform(row, lambda v, j: v/(dscA[i]*dscA[j])))) AS RAinv
        FROM (SELECT A, list_transform(A, lambda row, i: CASE WHEN row[i] > 1e-300 THEN sqrt(row[i]) ELSE 1.0 END) AS dscA FROM breadA))
),
lev AS (
  SELECT r.xs, r.sc, r.cl, r.wt,
         r.hwt * list_sum(list_transform(r.xs, lambda xa, a: xa * list_dot_product(bi.Ainv[a], r.xs))) AS h
  FROM robrow r CROSS JOIN breadinv bi
),
meat_hc AS (
  SELECT list(rowlist ORDER BY a) AS B FROM (
    SELECT a, list(val ORDER BY b) AS rowlist FROM (
      SELECT p.a AS a, p.b AS b,
             sum((CASE WHEN robust = 'hc2' THEN m.sc*m.sc/m.wt/(1.0-m.h)
                       WHEN robust = 'hc3' THEN m.sc*m.sc/m.wt/((1.0-m.h)*(1.0-m.h))
                       ELSE m.sc*m.sc/m.wt END) * m.xs[p.a] * m.xs[p.b]) AS val
      FROM lev m, pairs p GROUP BY p.a, p.b) GROUP BY a)
),
clsg AS (
  SELECT cl, list(sga ORDER BY a) AS sg FROM (
    SELECT l.cl, ix.i AS a, sum(l.sc * l.xs[ix.i]) AS sga FROM lev l, idx ix GROUP BY l.cl, ix.i) GROUP BY cl
),
meat_cl AS (
  SELECT list(rowlist ORDER BY a) AS B FROM (
    SELECT a, list(val ORDER BY b) AS rowlist FROM (
      SELECT p.a AS a, p.b AS b, sum(c.sg[p.a] * c.sg[p.b]) AS val
      FROM clsg c, pairs p GROUP BY p.a, p.b) GROUP BY a)
),
robvar AS (
  SELECT list_transform(range(1, dm.d+1), lambda j:
      (CASE WHEN cluster_col IS NOT NULL
            THEN (rd.G::DOUBLE/(rd.G-1)) * ((dm.n-1.0)/(dm.n-dm.d))
            WHEN robust = 'hc1' THEN dm.n::DOUBLE/(dm.n-dm.d) ELSE 1.0 END)
      * list_sum(list_transform(bi.Ainv[j], lambda va, a: va * list_dot_product(bm.B[a], bi.Ainv[j])))) AS rv
  FROM breadinv bi
       CROSS JOIN (SELECT CASE WHEN cluster_col IS NOT NULL THEN (SELECT B FROM meat_cl) ELSE (SELECT B FROM meat_hc) END AS B) bm
       CROSS JOIN dims dm CROSS JOIN rdims rd
),
robchk AS (
  SELECT CASE WHEN robust NOT IN ('none','hc0','hc1','hc2','hc3')
              THEN error(caller || ': robust must be one of ''none'',''hc0'',''hc1'',''hc2'',''hc3''; got ''' || robust || '''')
              ELSE true END AS ok
),
final AS (
  SELECT b.names AS names, b.bvec AS bvec, c.Rinv AS Rinv, s.dsc AS dsc, rv.rv AS rv,
         dp.phi AS phi, dp.est AS est, dp.df AS df,
         (robust != 'none' OR cluster_col IS NOT NULL) AS robactive,
         (dp.est AND dp.df > 0.0 AND robust = 'none' AND cluster_col IS NULL) AS uset,
         CASE WHEN dp.est AND dp.df > 0.0 AND robust = 'none' AND cluster_col IS NULL
                THEN t_ppf(1.0-(1.0-conf_level)/2.0, dp.df)
              ELSE norm_ppf(1.0-(1.0-conf_level)/2.0) END AS crit
  FROM beta b CROSS JOIN covinv c CROSS JOIN scal s CROSS JOIN disp dp CROSS JOIN robvar rv
       CROSS JOIN robchk rc WHERE rc.ok
),
-- per-coefficient SE with guards: NULL when the covariance is singular / non-finite / non-positive
percoef AS (
  SELECT gs.i AS i, names[gs.i] AS feature, bvec[gs.i] AS coefficient, uset, df, crit,
         CASE WHEN robactive THEN
                -- df <= 0 (saturated): robust variance is undefined; at n==d the
                -- leverage h->1 makes hc2/hc3's sc^2/(1-h)^k a 0/0 finite artifact
                CASE WHEN df > 0.0 AND isfinite(rv[gs.i]) AND rv[gs.i] > 0.0 THEN sqrt(rv[gs.i]) ELSE NULL END
              ELSE
                CASE WHEN Rinv IS NOT NULL AND isfinite(phi * Rinv[gs.i][gs.i]) AND phi * Rinv[gs.i][gs.i] > 0.0
                     THEN sqrt(phi * Rinv[gs.i][gs.i]) / dsc[gs.i] ELSE NULL END
         END AS std_error
  FROM final, unnest(range(1, len(bvec)+1)) AS gs(i)
)
SELECT feature, coefficient, std_error,
       coefficient / std_error AS statistic,
       CASE WHEN std_error IS NULL THEN NULL
            WHEN uset THEN 2.0 * __reg_t_sf(abs(coefficient / std_error), df)
            ELSE 2.0 * norm_cdf(-abs(coefficient / std_error)) END AS p_value,
       coefficient - crit * std_error AS conf_low,
       coefficient + crit * std_error AS conf_high
FROM percoef ORDER BY i;

CREATE OR REPLACE MACRO logit_summary(model, tbl, outcome, conf_level := 0.95, offset_col := NULL, weights_col := NULL, robust := 'none', cluster_col := NULL) AS TABLE
SELECT * FROM __reg_summary(model, tbl, outcome, 'logistic', 'logit_summary', conf_level, offset_col, weights_col, NULL, NULL, robust, cluster_col);
CREATE OR REPLACE MACRO linreg_summary(model, tbl, outcome, conf_level := 0.95, offset_col := NULL, weights_col := NULL, robust := 'none', cluster_col := NULL) AS TABLE
SELECT * FROM __reg_summary(model, tbl, outcome, 'linear', 'linreg_summary', conf_level, offset_col, weights_col, NULL, NULL, robust, cluster_col);
CREATE OR REPLACE MACRO poisson_summary(model, tbl, outcome, conf_level := 0.95, offset_col := NULL, weights_col := NULL, robust := 'none', cluster_col := NULL) AS TABLE
SELECT * FROM __reg_summary(model, tbl, outcome, 'poisson', 'poisson_summary', conf_level, offset_col, weights_col, NULL, NULL, robust, cluster_col);
CREATE OR REPLACE MACRO gamma_summary(model, tbl, outcome, conf_level := 0.95, offset_col := NULL, weights_col := NULL, robust := 'none', cluster_col := NULL) AS TABLE
SELECT * FROM __reg_summary(model, tbl, outcome, 'gamma', 'gamma_summary', conf_level, offset_col, weights_col, NULL, NULL, robust, cluster_col);
CREATE OR REPLACE MACRO tweedie_summary(model, tbl, outcome, power := 1.5, conf_level := 0.95, offset_col := NULL, weights_col := NULL, robust := 'none', cluster_col := NULL) AS TABLE
SELECT * FROM __reg_summary(model, tbl, outcome, 'tweedie', 'tweedie_summary', conf_level, offset_col, weights_col, power, NULL, robust, cluster_col);
CREATE OR REPLACE MACRO nbinom_summary(model, tbl, outcome, alpha := 1.0, conf_level := 0.95, offset_col := NULL, weights_col := NULL, robust := 'none', cluster_col := NULL) AS TABLE
SELECT * FROM __reg_summary(model, tbl, outcome, 'nbinom', 'nbinom_summary', conf_level, offset_col, weights_col, NULL, alpha, robust, cluster_col);

-- ---------------------------------------------------------------------------
-- Confidence intervals on the predicted MEAN response (prediction intervals).
-- Cov(beta) = phi*(X'WX)^-1 is estimated from the training data `tbl`; then for
-- each row of `newdata` (default: tbl) the linear predictor eta = x'beta+offset
-- has SE(eta) = sqrt(phi * x'(X'WX)^-1 x), and the mean CI is g^-1(eta +- crit*
-- SE(eta)) on the response scale (g^-1 monotone, so bounds keep their order).
-- z for fixed-dispersion families, Student-t(n-d) for estimated (as *_summary).
-- Returns the newdata columns plus prediction / conf_low / conf_high.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE MACRO __reg_predict_ci(model, tbl, outcome, newdata, family, caller,
                                         conf_level, offset_col, weights_col, power, alpha) AS TABLE
WITH RECURSIVE
mdl AS (SELECT feature, coefficient FROM query_table(model)),
-- Feature position in name order, so the per-row feature vector below can be
-- ordered by an integer rather than by the VARCHAR column name (see __reg_stats).
mdlj AS (SELECT feature, row_number() OVER (ORDER BY feature) AS j
         FROM mdl WHERE feature != '(Intercept)'),
beta AS (
  SELECT ([ coalesce((SELECT coefficient FROM mdl WHERE feature = '(Intercept)'), 0.0) ]::DOUBLE[]
          || list(coefficient ORDER BY feature) FILTER (WHERE feature != '(Intercept)')) AS bvec,
         (count(*) FILTER (WHERE feature != '(Intercept)'))::INT AS k
  FROM mdl
),
-- === model-based covariance from the TRAINING data (as __reg_summary) ===
num AS (SELECT row_number() OVER () AS rid, * FROM query_table(tbl)),
alllong AS (
  SELECT rid, name AS col, value AS v
  FROM (UNPIVOT (SELECT rid, TRY_CAST(COLUMNS(* EXCLUDE (rid)) AS DOUBLE) FROM num)
        ON COLUMNS(* EXCLUDE (rid)) INTO NAME name VALUE value)
),
yv AS (SELECT rid, v AS y  FROM alllong WHERE col = outcome),
ov AS (SELECT rid, v AS o  FROM alllong WHERE col = offset_col),
wv AS (SELECT rid, v AS wt FROM alllong WHERE col = weights_col),
feat AS (
  SELECT l.rid, [1.0::DOUBLE] || list(l.v ORDER BY m.j) AS xs, count(*)::INT AS nf
  FROM alllong l JOIN mdlj m ON m.feature = l.col
  GROUP BY l.rid
),
rows0 AS (
  SELECT f.rid, f.xs, y.y,
         CASE WHEN offset_col IS NULL THEN 0.0 ELSE o.o END AS off,
         CASE WHEN weights_col IS NULL THEN 1.0 ELSE coalesce(w.wt, 1.0) END AS wt
  FROM feat f JOIN yv y ON y.rid = f.rid
  LEFT JOIN ov o ON o.rid = f.rid LEFT JOIN wv w ON w.rid = f.rid CROSS JOIN beta
  WHERE f.nf = beta.k AND y.y IS NOT NULL AND (offset_col IS NULL OR o.o IS NOT NULL)
),
rww AS (
  SELECT r.xs, r.y, r.wt, mu,
         r.wt * (CASE family WHEN 'logistic' THEN mu*(1.0-mu) WHEN 'linear' THEN 1.0
                   WHEN 'poisson' THEN mu WHEN 'gamma' THEN 1.0
                   WHEN 'tweedie' THEN pow(mu, 2.0-power) WHEN 'nbinom' THEN mu/(1.0+alpha*mu) END) AS w,
         r.wt * (CASE family WHEN 'logistic' THEN (r.y-mu)*(r.y-mu)/(mu*(1.0-mu))
                   WHEN 'linear' THEN (r.y-mu)*(r.y-mu) WHEN 'poisson' THEN (r.y-mu)*(r.y-mu)/mu
                   WHEN 'gamma' THEN (r.y-mu)*(r.y-mu)/(mu*mu) WHEN 'tweedie' THEN (r.y-mu)*(r.y-mu)/pow(mu,power)
                   WHEN 'nbinom' THEN (r.y-mu)*(r.y-mu)/(mu*(1.0+alpha*mu)) END) AS pearson
  FROM (SELECT xs, y, wt, CASE family WHEN 'logistic' THEN 1.0/(1.0+exp(-greatest(-700.0,least(eta,700.0))))
                            WHEN 'linear' THEN eta ELSE exp(greatest(-700.0,least(eta,700.0))) END AS mu
        FROM (SELECT xs, y, wt, off + list_dot_product(xs, (SELECT bvec FROM beta)) AS eta FROM rows0)) r
),
dims AS (SELECT count(*)::INT AS n, (SELECT k FROM beta)+1 AS d FROM rww),
idx AS (SELECT unnest(range(1, (SELECT d FROM dims)+1)) AS i),
pairs AS (SELECT a.i AS a, b.i AS b FROM idx a, idx b),
xwx AS (
  SELECT list(rowlist ORDER BY a) AS A FROM (
    SELECT a, list(val ORDER BY b) AS rowlist FROM (
      SELECT p.a AS a, p.b AS b, sum(rww.w * rww.xs[p.a] * rww.xs[p.b]) AS val
      FROM rww, pairs p GROUP BY p.a, p.b) GROUP BY a)
),
scal AS (SELECT A, list_transform(A, lambda row, i: CASE WHEN row[i] > 1e-300 THEN sqrt(row[i]) ELSE 1.0 END) AS dsc FROM xwx),
rscaled AS (SELECT dsc, list_transform(A, lambda row, i: list_transform(row, lambda v, j: v/(dsc[i]*dsc[j]))) AS R FROM scal),
gj(k, d, sing, M) AS (
  SELECT 0, len(R), false,
         list_transform(R, lambda row, i: list_concat(list_transform(row, lambda v, j: v::DOUBLE),
             list_transform(row, lambda v, j: CASE WHEN j = i THEN 1.0 ELSE 0.0 END)))
  FROM rscaled
  UNION ALL
  SELECT col, d, sing OR abs(piv) < 1e-12,
         CASE WHEN abs(piv) < 1e-12 THEN Mswap
              ELSE list_transform(Mswap, lambda row, i: CASE WHEN i = col THEN normpivot
                       ELSE list_transform(row, lambda v, j: v - row[col]*normpivot[j]) END) END
  FROM (SELECT col, d, sing, Mswap, Mswap[col][col] AS piv,
               list_transform(Mswap[col], lambda v, j: v / Mswap[col][col]) AS normpivot
        FROM (SELECT col, d, sing, list_transform(M, lambda row, i:
                       CASE WHEN i = col THEN M[p] WHEN i = p THEN M[col] ELSE row END) AS Mswap
              FROM (SELECT col, d, sing, M, list_position(pcol, list_aggregate(pcol, 'max')) AS p
                    FROM (SELECT k, k+1 AS col, d, sing, M, list_transform(M, lambda row, i:
                                   CASE WHEN i >= k+1 THEN abs(row[k+1]) ELSE -1e308 END) AS pcol
                          FROM gj WHERE k < d))))
),
covinv AS (SELECT CASE WHEN sing THEN NULL ELSE list_transform(M, lambda row, i: list_slice(row, d+1, 2*d)) END AS Rinv FROM gj WHERE k = d),
disp AS (
  SELECT CASE WHEN family IN ('linear','gamma','tweedie')
              THEN (SELECT sum(pearson) FROM rww) / nullif((SELECT n-d FROM dims), 0) ELSE 1.0 END AS phi,
         (family IN ('linear','gamma','tweedie') AND (SELECT n-d FROM dims) > 0) AS uset,
         (SELECT (n-d)::DOUBLE FROM dims) AS df
),
cparams AS (
  SELECT c.Rinv AS Rinv, s.dsc AS dsc, dp.phi AS phi, dp.uset AS uset, dp.df AS df,
         CASE WHEN dp.uset THEN t_ppf(1.0-(1.0-conf_level)/2.0, dp.df)
              ELSE norm_ppf(1.0-(1.0-conf_level)/2.0) END AS crit
  FROM covinv c CROSS JOIN scal s CROSS JOIN disp dp
),
-- === score newdata (default = tbl) ===
snum AS (SELECT row_number() OVER () AS srid, * FROM query_table(coalesce(newdata, tbl))),
salllong AS (
  SELECT srid, name AS col, value AS v
  FROM (UNPIVOT (SELECT srid, TRY_CAST(COLUMNS(* EXCLUDE (srid)) AS DOUBLE) FROM snum)
        ON COLUMNS(* EXCLUDE (srid)) INTO NAME name VALUE value)
),
soff AS (SELECT srid, v AS o FROM salllong WHERE col = offset_col),
sfeat AS (
  SELECT l.srid, [1.0::DOUBLE] || list(l.v ORDER BY m.j) AS xs, count(*)::INT AS nf
  FROM salllong l JOIN mdlj m ON m.feature = l.col
  GROUP BY l.srid
),
scored AS (
  SELECT sn.srid,
         CASE WHEN sf.nf = (SELECT k FROM beta)
                   AND (offset_col IS NULL OR so.o IS NOT NULL)
              THEN (CASE WHEN offset_col IS NULL THEN 0.0 ELSE so.o END)
                   + list_dot_product(sf.xs, (SELECT bvec FROM beta)) END AS eta,
         CASE WHEN sf.nf = (SELECT k FROM beta) AND cp.Rinv IS NOT NULL
              THEN cp.phi * list_sum(list_transform(
                       list_transform(sf.xs, lambda v, a: v / cp.dsc[a]),
                       lambda va, a: va * list_dot_product(cp.Rinv[a], list_transform(sf.xs, lambda v2, a2: v2 / cp.dsc[a2]))))
              END AS var_eta,
         cp.crit AS crit
  FROM snum sn
  LEFT JOIN sfeat sf ON sf.srid = sn.srid
  LEFT JOIN soff so ON so.srid = sn.srid
  CROSS JOIN cparams cp
)
SELECT sn.* EXCLUDE (srid),
       CASE family WHEN 'logistic' THEN 1.0/(1.0+exp(-s.eta)) WHEN 'linear' THEN s.eta ELSE exp(s.eta) END AS prediction,
       CASE WHEN s.var_eta IS NULL OR NOT isfinite(s.var_eta) OR s.var_eta < 0.0 THEN NULL
            ELSE (CASE family WHEN 'logistic' THEN 1.0/(1.0+exp(-(s.eta - s.crit*sqrt(s.var_eta))))
                              WHEN 'linear' THEN s.eta - s.crit*sqrt(s.var_eta)
                              ELSE exp(s.eta - s.crit*sqrt(s.var_eta)) END) END AS conf_low,
       CASE WHEN s.var_eta IS NULL OR NOT isfinite(s.var_eta) OR s.var_eta < 0.0 THEN NULL
            ELSE (CASE family WHEN 'logistic' THEN 1.0/(1.0+exp(-(s.eta + s.crit*sqrt(s.var_eta))))
                              WHEN 'linear' THEN s.eta + s.crit*sqrt(s.var_eta)
                              ELSE exp(s.eta + s.crit*sqrt(s.var_eta)) END) END AS conf_high
FROM scored s JOIN snum sn ON sn.srid = s.srid
ORDER BY s.srid;

CREATE OR REPLACE MACRO logit_predict_ci(model, tbl, outcome, newdata := NULL, conf_level := 0.95, offset_col := NULL, weights_col := NULL) AS TABLE
SELECT * FROM __reg_predict_ci(model, tbl, outcome, newdata, 'logistic', 'logit_predict_ci', conf_level, offset_col, weights_col, NULL, NULL);
CREATE OR REPLACE MACRO linreg_predict_ci(model, tbl, outcome, newdata := NULL, conf_level := 0.95, offset_col := NULL, weights_col := NULL) AS TABLE
SELECT * FROM __reg_predict_ci(model, tbl, outcome, newdata, 'linear', 'linreg_predict_ci', conf_level, offset_col, weights_col, NULL, NULL);
CREATE OR REPLACE MACRO poisson_predict_ci(model, tbl, outcome, newdata := NULL, conf_level := 0.95, offset_col := NULL, weights_col := NULL) AS TABLE
SELECT * FROM __reg_predict_ci(model, tbl, outcome, newdata, 'poisson', 'poisson_predict_ci', conf_level, offset_col, weights_col, NULL, NULL);
CREATE OR REPLACE MACRO gamma_predict_ci(model, tbl, outcome, newdata := NULL, conf_level := 0.95, offset_col := NULL, weights_col := NULL) AS TABLE
SELECT * FROM __reg_predict_ci(model, tbl, outcome, newdata, 'gamma', 'gamma_predict_ci', conf_level, offset_col, weights_col, NULL, NULL);
CREATE OR REPLACE MACRO tweedie_predict_ci(model, tbl, outcome, newdata := NULL, power := 1.5, conf_level := 0.95, offset_col := NULL, weights_col := NULL) AS TABLE
SELECT * FROM __reg_predict_ci(model, tbl, outcome, newdata, 'tweedie', 'tweedie_predict_ci', conf_level, offset_col, weights_col, power, NULL);
CREATE OR REPLACE MACRO nbinom_predict_ci(model, tbl, outcome, newdata := NULL, alpha := 1.0, conf_level := 0.95, offset_col := NULL, weights_col := NULL) AS TABLE
SELECT * FROM __reg_predict_ci(model, tbl, outcome, newdata, 'nbinom', 'nbinom_predict_ci', conf_level, offset_col, weights_col, NULL, alpha);

-- ---------------------------------------------------------------------------
-- Influence diagnostics (per training observation). hat = observed-info
-- leverage h_i = a_i*hw_i * x_i'(X'diag(a*hw)X)^-1 x_i; Pearson and deviance
-- residuals; the studentized (standardized Pearson) residual r_P/sqrt(phi(1-h));
-- and Cook's distance (r_P^2/phi)*h/(d(1-h)^2). Matches statsmodels
-- GLMInfluence. Returns the input columns plus the five diagnostics.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE MACRO __reg_influence(model, tbl, outcome, family, caller, offset_col, weights_col, power, alpha) AS TABLE
WITH RECURSIVE
mdl AS (SELECT feature, coefficient FROM query_table(model)),
-- Feature position in name order, so the per-row feature vector below can be
-- ordered by an integer rather than by the VARCHAR column name (see __reg_stats).
mdlj AS (SELECT feature, row_number() OVER (ORDER BY feature) AS j
         FROM mdl WHERE feature != '(Intercept)'),
beta AS (
  SELECT ([ coalesce((SELECT coefficient FROM mdl WHERE feature = '(Intercept)'), 0.0) ]::DOUBLE[]
          || list(coefficient ORDER BY feature) FILTER (WHERE feature != '(Intercept)')) AS bvec,
         (count(*) FILTER (WHERE feature != '(Intercept)'))::INT AS k
  FROM mdl
),
num AS (SELECT row_number() OVER () AS rid, * FROM query_table(tbl)),
alllong AS (
  SELECT rid, name AS col, value AS v
  FROM (UNPIVOT (SELECT rid, TRY_CAST(COLUMNS(* EXCLUDE (rid)) AS DOUBLE) FROM num)
        ON COLUMNS(* EXCLUDE (rid)) INTO NAME name VALUE value)
),
yv AS (SELECT rid, v AS y  FROM alllong WHERE col = outcome),
ov AS (SELECT rid, v AS o  FROM alllong WHERE col = offset_col),
wv AS (SELECT rid, v AS wt FROM alllong WHERE col = weights_col),
feat AS (
  SELECT l.rid, [1.0::DOUBLE] || list(l.v ORDER BY m.j) AS xs, count(*)::INT AS nf
  FROM alllong l JOIN mdlj m ON m.feature = l.col GROUP BY l.rid
),
rows0 AS (
  SELECT f.rid, f.xs, y.y,
         CASE WHEN offset_col IS NULL THEN 0.0 ELSE o.o END AS off,
         CASE WHEN weights_col IS NULL THEN 1.0 ELSE coalesce(w.wt, 1.0) END AS wt
  FROM feat f JOIN yv y ON y.rid = f.rid
  LEFT JOIN ov o ON o.rid = f.rid LEFT JOIN wv w ON w.rid = f.rid CROSS JOIN beta
  WHERE f.nf = beta.k AND y.y IS NOT NULL AND (offset_col IS NULL OR o.o IS NOT NULL)
),
-- per row: mu, observed weight hw, variance V, residual, unit deviance
pr AS (
  SELECT rid, xs, wt, y, mu,
         wt * (CASE family WHEN 'logistic' THEN mu*(1.0-mu) WHEN 'linear' THEN 1.0
                 WHEN 'poisson' THEN mu WHEN 'gamma' THEN y/mu
                 WHEN 'tweedie' THEN (2.0-power)*pow(mu,2.0-power)+(power-1.0)*y*pow(mu,1.0-power)
                 WHEN 'nbinom' THEN mu*(1.0+alpha*y)/pow(1.0+alpha*mu,2.0) END) AS hwt,
         (y - mu) AS resid,
         (CASE family WHEN 'logistic' THEN mu*(1.0-mu) WHEN 'linear' THEN 1.0 WHEN 'poisson' THEN mu
                 WHEN 'gamma' THEN mu*mu WHEN 'tweedie' THEN pow(mu,power) WHEN 'nbinom' THEN mu*(1.0+alpha*mu) END) AS Vmu,
         (CASE family
            WHEN 'logistic' THEN 2.0*((CASE WHEN y>0 THEN y*ln(y/mu) ELSE 0.0 END) + (CASE WHEN y<1 THEN (1.0-y)*ln((1.0-y)/(1.0-mu)) ELSE 0.0 END))
            WHEN 'linear'   THEN (y-mu)*(y-mu)
            WHEN 'poisson'  THEN 2.0*((CASE WHEN y>0 THEN y*ln(y/mu) ELSE 0.0 END) - (y-mu))
            WHEN 'gamma'    THEN 2.0*(-ln(y/mu) + (y-mu)/mu)
            WHEN 'tweedie'  THEN 2.0*((CASE WHEN y>0 THEN pow(y,2.0-power)/((1.0-power)*(2.0-power)) ELSE 0.0 END) - y*pow(mu,1.0-power)/(1.0-power) + pow(mu,2.0-power)/(2.0-power))
            WHEN 'nbinom'   THEN 2.0*((CASE WHEN y>0 THEN y*ln(y/mu) ELSE 0.0 END) - (y+1.0/alpha)*ln((y+1.0/alpha)/(mu+1.0/alpha))) END) AS udev
  FROM (SELECT rid, xs, wt, y,
               CASE family WHEN 'logistic' THEN 1.0/(1.0+exp(-greatest(-700.0,least(eta,700.0))))
                           WHEN 'linear' THEN eta ELSE exp(greatest(-700.0,least(eta,700.0))) END AS mu
        FROM (SELECT rid, xs, wt, y, off + list_dot_product(xs, (SELECT bvec FROM beta)) AS eta FROM rows0))
),
dims AS (SELECT count(*)::INT AS n, (SELECT k FROM beta)+1 AS d FROM pr),
idx AS (SELECT unnest(range(1, (SELECT d FROM dims)+1)) AS i),
pairs AS (SELECT a.i AS a, b.i AS b FROM idx a, idx b),
breadA AS (
  SELECT list(rowlist ORDER BY a) AS A FROM (
    SELECT a, list(val ORDER BY b) AS rowlist FROM (
      SELECT p.a AS a, p.b AS b, sum(pr.hwt * pr.xs[p.a] * pr.xs[p.b]) AS val
      FROM pr, pairs p GROUP BY p.a, p.b) GROUP BY a)
),
breadinv AS (
  SELECT list_transform(RAinv, lambda row, i: list_transform(row, lambda v, j: v/(dscA[i]*dscA[j]))) AS Ainv
  FROM (SELECT dscA, __reg_matinv(list_transform(A, lambda row, i: list_transform(row, lambda v, j: v/(dscA[i]*dscA[j])))) AS RAinv
        FROM (SELECT A, list_transform(A, lambda row, i: CASE WHEN row[i] > 1e-300 THEN sqrt(row[i]) ELSE 1.0 END) AS dscA FROM breadA))
),
lev AS (
  SELECT p.rid, p.hwt * list_sum(list_transform(p.xs, lambda xa, a: xa * list_dot_product(bi.Ainv[a], p.xs))) AS h
  FROM pr p CROSS JOIN breadinv bi
),
disp AS (
  SELECT CASE WHEN family IN ('linear','gamma','tweedie')
              THEN (SELECT sum(wt*resid*resid/Vmu) FROM pr) / nullif((SELECT n-d FROM dims), 0) ELSE 1.0 END AS phi,
         (SELECT d FROM dims) AS d
),
diag AS (
  SELECT p.rid,
         CASE WHEN isfinite(l.h) THEN l.h ELSE NULL END AS hat,  -- NULL (not NaN) on singular bread
         p.resid * sqrt(p.wt) / sqrt(p.Vmu) AS pearson_resid,
         sign(p.resid) * sqrt(p.wt * greatest(p.udev, 0.0)) AS deviance_resid,
         CASE WHEN isfinite(l.h) AND l.h < 1.0 THEN (p.resid*sqrt(p.wt)/sqrt(p.Vmu)) / sqrt(dp.phi*(1.0-l.h)) END AS std_resid,
         CASE WHEN isfinite(l.h) AND l.h < 1.0 THEN (p.resid*p.resid*p.wt/p.Vmu/dp.phi) * l.h / (dp.d*(1.0-l.h)*(1.0-l.h)) END AS cooks_distance
  FROM pr p JOIN lev l ON l.rid = p.rid CROSS JOIN disp dp
)
SELECT n.* EXCLUDE (rid), d.hat, d.pearson_resid, d.deviance_resid, d.std_resid, d.cooks_distance
FROM num n JOIN diag d ON d.rid = n.rid ORDER BY n.rid;

CREATE OR REPLACE MACRO logit_influence(model, tbl, outcome, offset_col := NULL, weights_col := NULL) AS TABLE
SELECT * FROM __reg_influence(model, tbl, outcome, 'logistic', 'logit_influence', offset_col, weights_col, NULL, NULL);
CREATE OR REPLACE MACRO linreg_influence(model, tbl, outcome, offset_col := NULL, weights_col := NULL) AS TABLE
SELECT * FROM __reg_influence(model, tbl, outcome, 'linear', 'linreg_influence', offset_col, weights_col, NULL, NULL);
CREATE OR REPLACE MACRO poisson_influence(model, tbl, outcome, offset_col := NULL, weights_col := NULL) AS TABLE
SELECT * FROM __reg_influence(model, tbl, outcome, 'poisson', 'poisson_influence', offset_col, weights_col, NULL, NULL);
CREATE OR REPLACE MACRO gamma_influence(model, tbl, outcome, offset_col := NULL, weights_col := NULL) AS TABLE
SELECT * FROM __reg_influence(model, tbl, outcome, 'gamma', 'gamma_influence', offset_col, weights_col, NULL, NULL);
CREATE OR REPLACE MACRO tweedie_influence(model, tbl, outcome, power := 1.5, offset_col := NULL, weights_col := NULL) AS TABLE
SELECT * FROM __reg_influence(model, tbl, outcome, 'tweedie', 'tweedie_influence', offset_col, weights_col, power, NULL);
CREATE OR REPLACE MACRO nbinom_influence(model, tbl, outcome, alpha := 1.0, offset_col := NULL, weights_col := NULL) AS TABLE
SELECT * FROM __reg_influence(model, tbl, outcome, 'nbinom', 'nbinom_influence', offset_col, weights_col, NULL, alpha);

-- Multinomial (softmax) coefficient inference. Baseline-category Fisher
-- information I[(c,j),(c',k)] = sum_i p_ic(delta_cc' - p_ic') x_ij x_ik over the
-- K-1 non-reference classes; Cov = I^-1, dispersion fixed = 1, z inference.
-- Returns one row per estimated (class, feature); the reference (alphabetical-
-- min) class is the fixed baseline and is not reported.
CREATE OR REPLACE MACRO multinom_summary(model, tbl, outcome, conf_level := 0.95) AS TABLE
WITH RECURSIVE
mdl AS (SELECT class, feature, coefficient FROM query_table(model)),
refc AS (SELECT min(class) AS ref FROM mdl),
featnames AS (
  SELECT [ '(Intercept)' ] || list(DISTINCT feature ORDER BY feature) FILTER (WHERE feature != '(Intercept)') AS fn
  FROM mdl
),
-- beta vector per class ([intercept, features sorted]); non-reference classes ordered
bpc AS (
  SELECT class,
         [ coalesce(max(coefficient) FILTER (WHERE feature = '(Intercept)'), 0.0) ]::DOUBLE[]
         || list(coefficient ORDER BY feature) FILTER (WHERE feature != '(Intercept)') AS bvec
  FROM mdl GROUP BY class
),
Bmat AS (
  SELECT list(bvec ORDER BY class) AS B, list(class ORDER BY class) AS cls
  FROM bpc WHERE class != (SELECT ref FROM refc)
),
num AS (SELECT row_number() OVER () AS rid, * FROM query_table(tbl)),
alllong AS (
  SELECT rid, name AS col, value AS v
  FROM (UNPIVOT (SELECT rid, TRY_CAST(COLUMNS(* EXCLUDE (rid)) AS DOUBLE) FROM num)
        ON COLUMNS(* EXCLUDE (rid)) INTO NAME name VALUE value)
),
kfeat AS (SELECT count(DISTINCT feature) FILTER (WHERE feature != '(Intercept)') AS k FROM mdl),
mdlj AS (
  SELECT feature, row_number() OVER (ORDER BY feature) AS j
  FROM (SELECT DISTINCT feature FROM mdl WHERE feature != '(Intercept)')
),
feat AS (
  SELECT l.rid, [1.0::DOUBLE] || list(l.v ORDER BY m.j) AS xs, count(*)::INT AS nf
  FROM alllong l JOIN mdlj m ON m.feature = l.col
  GROUP BY l.rid
),
-- per-row softmax probabilities for the non-reference classes
probs AS (
  SELECT rid, xs, list_transform(ee, lambda e: e/(1.0 + list_sum(ee))) AS p
  FROM (
    SELECT f.rid, f.xs,
           list_transform((SELECT B FROM Bmat), lambda bc: exp(least(list_dot_product(f.xs, bc), 700.0))) AS ee
    FROM feat f, kfeat WHERE f.nf = kfeat.k
  )
),
dims AS (
  SELECT (SELECT len(B) FROM Bmat) AS km1,
         (SELECT k FROM kfeat) + 1 AS d,
         ((SELECT len(B) FROM Bmat)) * ((SELECT k FROM kfeat) + 1) AS M
),
-- block-structured information matrix via flat index pairs
idx AS (SELECT unnest(range(1, (SELECT M FROM dims)+1)) AS i),
pairs AS (
  SELECT a.i AS a, b.i AS b,
         (a.i-1)//(SELECT d FROM dims) AS ca, (a.i-1)%(SELECT d FROM dims) AS ja,
         (b.i-1)//(SELECT d FROM dims) AS cb, (b.i-1)%(SELECT d FROM dims) AS kb
  FROM idx a, idx b
),
info AS (
  SELECT list(rowlist ORDER BY a) AS A FROM (
    SELECT a, list(val ORDER BY b) AS rowlist FROM (
      SELECT p.a AS a, p.b AS b,
             sum(pr.p[p.ca+1] * ((CASE WHEN p.ca = p.cb THEN 1.0 ELSE 0.0 END) - pr.p[p.cb+1])
                 * pr.xs[p.ja+1] * pr.xs[p.kb+1]) AS val
      FROM probs pr, pairs p GROUP BY p.a, p.b
    ) GROUP BY a
  )
),
-- diagonal scaling -> unit-diagonal correlation form
scal AS (SELECT A, list_transform(A, lambda row, i: CASE WHEN row[i] > 1e-300 THEN sqrt(row[i]) ELSE 1.0 END) AS dsc FROM info),
rscaled AS (SELECT dsc, list_transform(A, lambda row, i: list_transform(row, lambda v, j: v/(dsc[i]*dsc[j]))) AS R FROM scal),
gj(k, d, sing, M) AS (
  SELECT 0, len(R), false,
         list_transform(R, lambda row, i:
             list_concat(list_transform(row, lambda v, j: v::DOUBLE),
                         list_transform(row, lambda v, j: CASE WHEN j = i THEN 1.0 ELSE 0.0 END)))
  FROM rscaled
  UNION ALL
  SELECT col, d, sing OR abs(piv) < 1e-12,
         CASE WHEN abs(piv) < 1e-12 THEN Mswap
              ELSE list_transform(Mswap, lambda row, i:
                       CASE WHEN i = col THEN normpivot
                            ELSE list_transform(row, lambda v, j: v - row[col]*normpivot[j]) END) END
  FROM (
    SELECT col, d, sing, Mswap, Mswap[col][col] AS piv,
           list_transform(Mswap[col], lambda v, j: v / Mswap[col][col]) AS normpivot
    FROM (
      SELECT col, d, sing,
             list_transform(M, lambda row, i:
                 CASE WHEN i = col THEN M[p] WHEN i = p THEN M[col] ELSE row END) AS Mswap
      FROM (
        SELECT col, d, sing, M, list_position(pcol, list_aggregate(pcol, 'max')) AS p
        FROM (
          SELECT k, k+1 AS col, d, sing, M,
                 list_transform(M, lambda row, i: CASE WHEN i >= k+1 THEN abs(row[k+1]) ELSE -1e308 END) AS pcol
          FROM gj WHERE k < d
        )
      )
    )
  )
),
covinv AS (
  SELECT CASE WHEN sing THEN NULL ELSE list_transform(M, lambda row, i: list_slice(row, d+1, 2*d)) END AS Rinv
  FROM gj WHERE k = d
),
final AS (
  SELECT bm.B AS B, bm.cls AS cls, fn.fn AS fn, c.Rinv AS Rinv, s.dsc AS dsc,
         dm.d AS d, norm_ppf(1.0-(1.0-conf_level)/2.0) AS crit
  FROM Bmat bm CROSS JOIN featnames fn CROSS JOIN covinv c CROSS JOIN scal s CROSS JOIN dims dm
),
percoef AS (
  SELECT gs.a AS a, cls[(gs.a-1)//d + 1] AS class, fn[(gs.a-1)%d + 1] AS feature,
         B[(gs.a-1)//d + 1][(gs.a-1)%d + 1] AS coefficient, crit,
         CASE WHEN Rinv IS NOT NULL AND isfinite(Rinv[gs.a][gs.a]) AND Rinv[gs.a][gs.a] > 0.0
              THEN sqrt(Rinv[gs.a][gs.a]) / dsc[gs.a] ELSE NULL END AS std_error
  FROM final, unnest(range(1, len(B)*d + 1)) AS gs(a)
)
SELECT class, feature, coefficient, std_error,
       coefficient / std_error AS statistic,
       CASE WHEN std_error IS NULL THEN NULL ELSE 2.0 * norm_cdf(-abs(coefficient / std_error)) END AS p_value,
       coefficient - crit * std_error AS conf_low,
       coefficient + crit * std_error AS conf_high
FROM percoef ORDER BY class, a;
