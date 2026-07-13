"""Deterministic test suite for the duckLM DuckDB regression macros.

Every fit/predict is checked against an independent scikit-learn reference on
the same data (fixed seeds), so a failure means the macros disagree with a
trusted implementation -- not merely that some previously-recorded number
changed. Mirrors the adversarial verification the macros were developed under,
in a form anyone can reproduce with `pytest`.

Run from the repo root:  pytest -q
"""

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import (
    ElasticNet,
    GammaRegressor,
    Lasso,
    LinearRegression,
    LogisticRegression,
    PoissonRegressor,
    Ridge,
)
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import (
    accuracy_score,
    d2_tweedie_score,
    log_loss,
    mean_absolute_error,
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_squared_error,
    mean_tweedie_deviance,
    r2_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")  # silence sklearn solver/deprecation chatter

MACRO_FILE = Path(__file__).resolve().parents[1] / "regression_macros.sql"
DuckDBError = getattr(duckdb, "Error", Exception)


# --------------------------------------------------------------------------- #
# Fixtures & helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def con():
    c = duckdb.connect()
    c.execute(MACRO_FILE.read_text())
    return c


def _load(con, name, df):
    con.register(f"_src_{name}", df)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _src_{name}")
    con.unregister(f"_src_{name}")


def _params(kw):
    return "".join(f", {k} := {v}" for k, v in kw.items())


def fit(con, macro, df, outcome="y", **kw):
    """Return {feature: coefficient} from a *_fit macro."""
    _load(con, "traindata", df)
    rows = con.execute(
        f"SELECT feature, coefficient FROM {macro}('traindata', '{outcome}'{_params(kw)})"
    ).fetchall()
    return {f: c for f, c in rows}


def predict(con, macro, coefs, data, **kw):
    """Run a *_predict macro given a coefficient dict and a scoring DataFrame."""
    model = pd.DataFrame({"feature": list(coefs), "coefficient": list(coefs.values())})
    _load(con, "modeltbl", model)
    _load(con, "scoredata", data)
    return con.execute(
        f"SELECT * FROM {macro}('modeltbl', 'scoredata'{_params(kw)})"
    ).df()


def evaluate(con, macro, coefs, data, outcome="y", **kw):
    """Run a *_evaluate macro and return its single metrics row as a Series."""
    model = pd.DataFrame({"feature": list(coefs), "coefficient": list(coefs.values())})
    _load(con, "modeltbl", model)
    _load(con, "scoredata", data)
    return con.execute(
        f"SELECT * FROM {macro}('modeltbl', 'scoredata', '{outcome}'{_params(kw)})"
    ).df().iloc[0]


def zscore(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)  # population sd (ddof=0), matching the macro
    return (X - mu) / sd, mu, sd


def mixed_features(seed, n=800):
    """Three features with deliberately different means and scales."""
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.normal(2, 3, n),
        rng.normal(-1, 0.5, n),
        rng.normal(10, 4, n),
    ])


def frame(X, y):
    df = pd.DataFrame(X, columns=[f"x{i + 1}" for i in range(X.shape[1])])
    df["y"] = y
    return df


NAMES = ["x1", "x2", "x3"]


def assert_coefs(coefs, intercept, slopes, *, atol, rtol, names=NAMES):
    assert coefs["(Intercept)"] == pytest.approx(intercept, abs=atol, rel=rtol)
    for name, ref in zip(names, slopes):
        assert coefs[name] == pytest.approx(ref, abs=atol, rel=rtol)


def logreg_unpenalized(X, y):
    return LogisticRegression(
        C=np.inf, solver="newton-cholesky", max_iter=10000, tol=1e-11
    ).fit(X, y)


# --------------------------------------------------------------------------- #
# Statistical correctness vs scikit-learn (unpenalized)
# --------------------------------------------------------------------------- #
class TestStatisticalCorrectness:
    def test_linreg_matches_sklearn(self, con):
        X = mixed_features(11)
        rng = np.random.default_rng(101)
        y = 3 + X @ [1.5, -2.0, 0.4] + rng.normal(0, 2, len(X))
        coefs = fit(con, "linreg_fit", frame(X, y))
        ref = LinearRegression().fit(X, y)
        assert_coefs(coefs, ref.intercept_, ref.coef_, atol=1e-6, rtol=1e-6)

    def test_linreg_noiseless_exact_recovery(self, con):
        X = mixed_features(14, n=1000)
        y = 2.5 + X @ [-1.2, 0.3, 0.05]  # no noise
        coefs = fit(con, "linreg_fit", frame(X, y))
        assert_coefs(coefs, 2.5, [-1.2, 0.3, 0.05], atol=1e-7, rtol=0)

    def test_logit_matches_sklearn(self, con):
        X = mixed_features(1)
        Xs, _, _ = zscore(X)
        rng = np.random.default_rng(1)
        y = rng.binomial(1, 1 / (1 + np.exp(-(0.5 + Xs @ [1.0, -0.8, 0.3]))))
        coefs = fit(con, "logit_fit", frame(X, y))
        ref = logreg_unpenalized(X, y)
        assert_coefs(coefs, ref.intercept_[0], ref.coef_[0], atol=1e-4, rtol=1e-4)

    def test_poisson_matches_sklearn(self, con):
        X = mixed_features(2)
        Xs, _, _ = zscore(X)
        rng = np.random.default_rng(2)
        y = rng.poisson(np.exp(0.8 + Xs @ [0.4, -0.3, 0.2])).astype(float)
        coefs = fit(con, "poisson_fit", frame(X, y))
        ref = PoissonRegressor(alpha=0, max_iter=20000, tol=1e-12).fit(X, y)
        assert_coefs(coefs, ref.intercept_, ref.coef_, atol=1e-5, rtol=1e-4)

    def test_gamma_matches_sklearn(self, con):
        X = mixed_features(3)
        Xs, _, _ = zscore(X)
        rng = np.random.default_rng(3)
        mu = np.exp(1.0 + Xs @ [0.5, -0.2, 0.3])
        y = rng.gamma(2.0, mu / 2.0)  # positive, mean mu
        coefs = fit(con, "gamma_fit", frame(X, y))
        ref = GammaRegressor(alpha=0, max_iter=20000, tol=1e-12).fit(X, y)
        assert_coefs(coefs, ref.intercept_, ref.coef_, atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- #
# Ridge (L2) regularization
# --------------------------------------------------------------------------- #
class TestRidge:
    L2 = 0.5

    def test_ridge_linear(self, con):
        X = mixed_features(21)
        rng = np.random.default_rng(201)
        y = 3 + X @ [1.5, -2.0, 0.5] + rng.normal(0, 2, len(X))
        n = len(X)
        _, _, sd = zscore(X)
        coefs = fit(con, "linreg_fit", frame(X, y), l2=self.L2)
        Xs, _, _ = zscore(X)
        ref = Ridge(alpha=n * self.L2).fit(Xs, (y - y.mean()) / y.std())
        got = np.array([coefs[c] for c in NAMES]) * sd / y.std()
        assert got == pytest.approx(ref.coef_, abs=1e-7)

    def test_ridge_logistic(self, con):
        X = mixed_features(22)
        Xs, _, sd = zscore(X)
        rng = np.random.default_rng(202)
        y = rng.binomial(1, 1 / (1 + np.exp(-(0.4 + Xs @ [1.1, -0.9, 0.5]))))
        n = len(X)
        coefs = fit(con, "logit_fit", frame(X, y), l2=self.L2)
        ref = LogisticRegression(
            C=1 / (n * self.L2), solver="newton-cholesky", max_iter=20000, tol=1e-12
        ).fit(Xs, y)
        got = np.array([coefs[c] for c in NAMES]) * sd
        assert got == pytest.approx(ref.coef_[0], abs=1e-7)

    def test_ridge_poisson(self, con):
        X = mixed_features(23)
        Xs, _, sd = zscore(X)
        rng = np.random.default_rng(203)
        y = rng.poisson(np.exp(0.5 + Xs @ [0.4, -0.3, 0.2])).astype(float)
        coefs = fit(con, "poisson_fit", frame(X, y), l2=self.L2)
        ref = PoissonRegressor(alpha=self.L2, max_iter=20000, tol=1e-12).fit(
            Xs, y / y.mean()
        )
        got = np.array([coefs[c] for c in NAMES]) * sd
        assert got == pytest.approx(ref.coef_, abs=1e-7)

    def test_ridge_gamma(self, con):
        X = mixed_features(24)
        Xs, _, sd = zscore(X)
        rng = np.random.default_rng(204)
        y = rng.gamma(2.0, np.exp(0.6 + Xs @ [0.3, -0.2, 0.25]) / 2.0)
        coefs = fit(con, "gamma_fit", frame(X, y), l2=self.L2)
        ref = GammaRegressor(alpha=self.L2, max_iter=20000, tol=1e-12).fit(
            Xs, y / y.mean()
        )
        got = np.array([coefs[c] for c in NAMES]) * sd
        assert got == pytest.approx(ref.coef_, abs=1e-7)

    def test_l2_zero_equals_unpenalized(self, con):
        X = mixed_features(25)
        rng = np.random.default_rng(205)
        y = 1 + X @ [0.7, -1.1, 0.2] + rng.normal(0, 1, len(X))
        base = fit(con, "linreg_fit", frame(X, y))
        zero = fit(con, "linreg_fit", frame(X, y), l2=0.0)
        for k in base:
            assert zero[k] == pytest.approx(base[k], abs=1e-12)

    def test_l2_monotonic_shrinkage(self, con):
        X = mixed_features(26)
        Xs, _, sd = zscore(X)
        rng = np.random.default_rng(206)
        y = rng.binomial(1, 1 / (1 + np.exp(-(0.3 + Xs @ [1.2, -1.0, 0.6]))))
        df = frame(X, y)
        norms = []
        for l2 in [0.01, 0.1, 1.0, 10.0, 100.0]:
            coefs = fit(con, "logit_fit", df, l2=l2)
            std_slopes = np.array([coefs[c] for c in NAMES]) * sd
            norms.append(np.linalg.norm(std_slopes))
        assert all(a > b for a, b in zip(norms, norms[1:])), norms

    def test_ridge_matches_drop_null_reference(self, con):
        """Ridge standardizes over complete rows: matches drop-NULLs-then-fit."""
        X = mixed_features(27)
        rng = np.random.default_rng(207)
        y = 2 + X @ [1.0, -1.5, 0.3] + rng.normal(0, 1.5, len(X))
        df = frame(X, y)
        mask = rng.random(len(X)) < 0.12
        df.loc[mask, "x1"] = np.nan  # NULLs in one column only
        coefs = fit(con, "linreg_fit", df, l2=self.L2)
        comp = ~mask
        Xc = X[comp]
        _, _, sdc = zscore(Xc)
        yc = y[comp]
        Xsc, _, _ = zscore(Xc)
        ref = Ridge(alpha=comp.sum() * self.L2).fit(Xsc, (yc - yc.mean()) / yc.std())
        got = np.array([coefs[c] for c in NAMES]) * sdc / yc.std()
        assert got == pytest.approx(ref.coef_, abs=1e-7)


# --------------------------------------------------------------------------- #
# Predict semantics
# --------------------------------------------------------------------------- #
class TestPredict:
    def _simple_model(self, con):
        X = mixed_features(31, n=400)
        rng = np.random.default_rng(301)
        y = rng.binomial(1, 1 / (1 + np.exp(-(0.2 + zscore(X)[0] @ [1.0, -0.7, 0.4]))))
        return fit(con, "logit_fit", frame(X, y)), frame(X, y)

    def test_logit_predict_prob_and_pred(self, con):
        coefs, df = self._simple_model(con)
        out = predict(con, "logit_predict", coefs, df.drop(columns="y"))
        z = coefs["(Intercept)"] + df[NAMES].to_numpy() @ [coefs[c] for c in NAMES]
        assert out["prob"].to_numpy() == pytest.approx(1 / (1 + np.exp(-z)), abs=1e-12)
        assert (out["pred"] == (out["prob"] >= 0.5)).all()
        assert ((out["prob"] >= 0) & (out["prob"] <= 1)).all()

    def test_threshold(self, con):
        coefs, df = self._simple_model(con)
        default = predict(con, "logit_predict", coefs, df.drop(columns="y"))
        high = predict(con, "logit_predict", coefs, df.drop(columns="y"), threshold=0.9)
        assert (default["prob"].to_numpy() == pytest.approx(high["prob"].to_numpy()))
        assert (high["pred"] == (high["prob"] >= 0.9)).all()

    def test_linreg_predict_is_score(self, con):
        X = mixed_features(32, n=300)
        rng = np.random.default_rng(302)
        y = 1 + X @ [0.5, -0.3, 0.2] + rng.normal(0, 1, len(X))
        coefs = fit(con, "linreg_fit", frame(X, y))
        out = predict(con, "linreg_predict", coefs, frame(X, y).drop(columns="y"))
        expected = coefs["(Intercept)"] + X @ [coefs[c] for c in NAMES]
        assert out["prediction"].to_numpy() == pytest.approx(expected, abs=1e-10)

    def test_poisson_predict_is_exp_score(self, con):
        X = mixed_features(33, n=300)
        rng = np.random.default_rng(303)
        y = rng.poisson(np.exp(0.5 + zscore(X)[0] @ [0.3, -0.2, 0.1])).astype(float)
        coefs = fit(con, "poisson_fit", frame(X, y))
        out = predict(con, "poisson_predict", coefs, frame(X, y).drop(columns="y"))
        z = coefs["(Intercept)"] + X @ [coefs[c] for c in NAMES]
        assert out["prediction"].to_numpy() == pytest.approx(np.exp(z), rel=1e-10)
        assert (out["prediction"] > 0).all()

    def test_gamma_predict_is_exp_score(self, con):
        X = mixed_features(34, n=300)
        rng = np.random.default_rng(304)
        y = rng.gamma(2.0, np.exp(0.6 + zscore(X)[0] @ [0.3, -0.2, 0.2]) / 2.0)
        coefs = fit(con, "gamma_fit", frame(X, y))
        out = predict(con, "gamma_predict", coefs, frame(X, y).drop(columns="y"))
        z = coefs["(Intercept)"] + X @ [coefs[c] for c in NAMES]
        assert out["prediction"].to_numpy() == pytest.approx(np.exp(z), rel=1e-10)

    def test_null_feature_gives_null_prediction(self, con):
        coefs, df = self._simple_model(con)
        data = df.drop(columns="y").copy()
        data.loc[[0, 5, 9], "x1"] = np.nan
        out = predict(con, "logit_predict", coefs, data)
        assert out["prob"].isna().to_numpy().nonzero()[0].tolist() == [0, 5, 9]

    def test_missing_column_gives_null(self, con):
        coefs, df = self._simple_model(con)
        out = predict(con, "logit_predict", coefs, df.drop(columns=["y", "x3"]))
        assert out["prob"].isna().all()

    def test_extra_columns_passthrough_and_order(self, con):
        coefs, df = self._simple_model(con)
        data = df.drop(columns="y").copy()
        data.insert(0, "id", range(len(data)))
        data["label"] = ["row%d" % i for i in range(len(data))]
        out = predict(con, "logit_predict", coefs, data)
        assert out["id"].tolist() == list(range(len(data)))  # order preserved
        assert out["label"].tolist() == data["label"].tolist()
        base = predict(con, "logit_predict", coefs, df.drop(columns="y"))
        assert out["prob"].to_numpy() == pytest.approx(base["prob"].to_numpy())


# --------------------------------------------------------------------------- #
# Goodness-of-fit metrics
# --------------------------------------------------------------------------- #
class TestEvaluate:
    def test_linreg_evaluate_matches_sklearn(self, con):
        X = mixed_features(51)
        rng = np.random.default_rng(501)
        y = 2 + X @ [1.0, -1.5, 0.3] + rng.normal(0, 2, len(X))
        df = frame(X, y)
        coefs = fit(con, "linreg_fit", df)
        pred = predict(con, "linreg_predict", coefs, df)["prediction"].to_numpy()
        m = evaluate(con, "linreg_evaluate", coefs, df)
        assert m["n"] == len(X)
        assert m["r2"] == pytest.approx(r2_score(y, pred), abs=1e-9)
        assert m["rmse"] == pytest.approx(np.sqrt(mean_squared_error(y, pred)), rel=1e-9)
        assert m["mae"] == pytest.approx(mean_absolute_error(y, pred), rel=1e-9)
        # log-likelihood / AIC / BIC from the standard Gaussian-MLE formula
        n, k = len(X), len(coefs)
        sse = float(((y - pred) ** 2).sum())
        ll = -n / 2 * (np.log(2 * np.pi) + np.log(sse / n) + 1)
        assert m["loglik"] == pytest.approx(ll, rel=1e-9)
        assert m["aic"] == pytest.approx(-2 * ll + 2 * k, rel=1e-9)
        assert m["bic"] == pytest.approx(-2 * ll + np.log(n) * k, rel=1e-9)

    def test_logit_evaluate_matches_sklearn(self, con):
        X = mixed_features(52)
        Xs, _, _ = zscore(X)
        rng = np.random.default_rng(502)
        y = rng.binomial(1, 1 / (1 + np.exp(-(0.4 + Xs @ [1.0, -0.8, 0.4]))))
        df = frame(X, y)
        coefs = fit(con, "logit_fit", df)
        prob = predict(con, "logit_predict", coefs, df)["prob"].to_numpy()
        m = evaluate(con, "logit_evaluate", coefs, df)
        assert m["auc"] == pytest.approx(roc_auc_score(y, prob), abs=1e-9)
        assert m["log_loss"] == pytest.approx(log_loss(y, prob), rel=1e-9)
        assert m["accuracy"] == pytest.approx(accuracy_score(y, (prob >= 0.5).astype(int)))
        ll = float((y * np.log(prob) + (1 - y) * np.log(1 - prob)).sum())
        assert m["loglik"] == pytest.approx(ll, rel=1e-9)
        assert m["deviance"] == pytest.approx(-2 * ll, rel=1e-9)
        n, k = len(X), len(coefs)
        assert m["aic"] == pytest.approx(-2 * ll + 2 * k, rel=1e-9)

    def test_poisson_evaluate_matches_sklearn(self, con):
        X = mixed_features(53)
        Xs, _, _ = zscore(X)
        rng = np.random.default_rng(503)
        y = rng.poisson(np.exp(0.7 + Xs @ [0.4, -0.3, 0.2])).astype(float)
        df = frame(X, y)
        coefs = fit(con, "poisson_fit", df)
        mu = predict(con, "poisson_predict", coefs, df)["prediction"].to_numpy()
        m = evaluate(con, "poisson_evaluate", coefs, df)
        n = len(X)
        assert m["deviance"] == pytest.approx(n * mean_poisson_deviance(y, mu), rel=1e-8)
        assert m["pseudo_r2"] == pytest.approx(d2_tweedie_score(y, mu, power=1), rel=1e-8)

    def test_gamma_evaluate_matches_sklearn(self, con):
        X = mixed_features(54)
        Xs, _, _ = zscore(X)
        rng = np.random.default_rng(504)
        y = rng.gamma(2.0, np.exp(0.8 + Xs @ [0.4, -0.2, 0.3]) / 2.0)
        df = frame(X, y)
        coefs = fit(con, "gamma_fit", df)
        mu = predict(con, "gamma_predict", coefs, df)["prediction"].to_numpy()
        m = evaluate(con, "gamma_evaluate", coefs, df)
        n = len(X)
        assert m["deviance"] == pytest.approx(n * mean_gamma_deviance(y, mu), rel=1e-8)
        assert m["pseudo_r2"] == pytest.approx(d2_tweedie_score(y, mu, power=2), rel=1e-8)
        disp = float((((y - mu) / mu) ** 2).sum()) / (n - len(coefs))
        assert m["dispersion"] == pytest.approx(disp, rel=1e-8)

    def test_evaluate_on_holdout(self, con):
        Xtr = mixed_features(55, n=600)
        rng = np.random.default_rng(505)
        ytr = 1 + Xtr @ [0.5, -1.0, 0.2] + rng.normal(0, 1, len(Xtr))
        coefs = fit(con, "linreg_fit", frame(Xtr, ytr))
        Xte = mixed_features(56, n=200)
        yte = 1 + Xte @ [0.5, -1.0, 0.2] + rng.normal(0, 1, len(Xte))
        m = evaluate(con, "linreg_evaluate", coefs, frame(Xte, yte))
        pred = predict(con, "linreg_predict", coefs, frame(Xte, yte))["prediction"].to_numpy()
        assert m["n"] == 200
        assert m["r2"] == pytest.approx(r2_score(yte, pred), abs=1e-9)

    def test_evaluate_drops_null_rows(self, con):
        X = mixed_features(57, n=300)
        rng = np.random.default_rng(507)
        y = 1 + X @ [0.5, -0.5, 0.2] + rng.normal(0, 1, len(X))
        df = frame(X, y)
        coefs = fit(con, "linreg_fit", df)
        df.loc[[1, 2, 3], "x1"] = np.nan  # 3 rows become incomplete
        m = evaluate(con, "linreg_evaluate", coefs, df)
        assert m["n"] == 297

    def test_evaluate_errors_on_no_rows(self, con):
        X = mixed_features(58, n=100)
        rng = np.random.default_rng(508)
        y = 1 + X @ [0.5, -0.5, 0.2] + rng.normal(0, 1, len(X))
        coefs = fit(con, "linreg_fit", frame(X, y))
        with pytest.raises(DuckDBError) as e:
            evaluate(con, "linreg_evaluate", coefs, frame(X, y), outcome="not_a_column")
        assert "no rows" in str(e.value)


# --------------------------------------------------------------------------- #
# Offset / exposure
# --------------------------------------------------------------------------- #
class TestOffset:
    def _poisson_exposure(self, seed, n=1500):
        rng = np.random.default_rng(seed)
        x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
        expo = rng.uniform(0.5, 5, n)
        logo = np.log(expo)
        y = rng.poisson(expo * np.exp(-0.4 + 0.7 * x1 - 0.3 * x2)).astype(float)
        return pd.DataFrame({"x1": x1, "x2": x2, "logexp": logo, "y": y}), x1, x2, logo, y

    def test_poisson_offset_matches_glm(self, con):
        # unpenalized Poisson with log-exposure offset == statsmodels-style GLM;
        # cross-checked here against sklearn PoissonRegressor on the residualised
        # target is awkward, so verify the score-equation optimality instead:
        # sum((y - exposure*exp(xb)) * x_j) ~ 0.
        df, x1, x2, logo, y = self._poisson_exposure(61)
        coefs = fit(con, "poisson_fit", df, offset_col="'logexp'")
        z = coefs["(Intercept)"] + coefs["x1"] * x1 + coefs["x2"] * x2 + logo
        resid = y - np.exp(z)
        n = len(y)
        for col, xv in [("intercept", np.ones(n)), ("x1", x1), ("x2", x2)]:
            assert abs((resid * xv).sum()) / n < 1e-6, col

    def test_offset_changes_fit(self, con):
        df, *_ = self._poisson_exposure(62)
        with_off = fit(con, "poisson_fit", df, offset_col="'logexp'")
        # dropping exposure entirely forces the intercept to absorb mean exposure
        ignored = fit(con, "poisson_fit", df.drop(columns="logexp"))
        assert abs(with_off["(Intercept)"] - ignored["(Intercept)"]) > 0.5

    def test_predict_with_offset(self, con):
        df, x1, x2, logo, y = self._poisson_exposure(63)
        coefs = fit(con, "poisson_fit", df, offset_col="'logexp'")
        out = predict(con, "poisson_predict", coefs, df, offset_col="'logexp'")
        z = coefs["(Intercept)"] + coefs["x1"] * x1 + coefs["x2"] * x2 + logo
        assert out["prediction"].to_numpy() == pytest.approx(np.exp(z), rel=1e-9)

    def test_null_offset_row_gives_null_prediction(self, con):
        df, *_ = self._poisson_exposure(64)
        coefs = fit(con, "poisson_fit", df, offset_col="'logexp'")
        data = df.drop(columns="y").copy()
        data.loc[[0, 7], "logexp"] = np.nan
        out = predict(con, "poisson_predict", coefs, data, offset_col="'logexp'")
        assert out["prediction"].isna().to_numpy().nonzero()[0].tolist() == [0, 7]

    def test_evaluate_with_offset(self, con):
        df, x1, x2, logo, y = self._poisson_exposure(65)
        coefs = fit(con, "poisson_fit", df, offset_col="'logexp'")
        mu = predict(con, "poisson_predict", coefs, df, offset_col="'logexp'")["prediction"].to_numpy()
        m = evaluate(con, "poisson_evaluate", coefs, df, offset_col="'logexp'")
        from sklearn.metrics import mean_poisson_deviance
        assert m["deviance"] == pytest.approx(len(y) * mean_poisson_deviance(y, mu), rel=1e-8)

    def test_missing_offset_column_errors(self, con):
        df, *_ = self._poisson_exposure(66)
        _load(con, "traindata", df)
        with pytest.raises(DuckDBError) as e:
            con.execute(
                "SELECT * FROM poisson_fit('traindata', 'y', offset_col := 'nope')"
            ).fetchall()
        assert "offset column" in str(e.value) and "nope" in str(e.value)


# --------------------------------------------------------------------------- #
# L1 / elastic-net (sparse feature selection)
# --------------------------------------------------------------------------- #
class TestL1:
    def _sparse(self, seed, n=2000):
        """6 features on different scales; true coefficient vector is sparse."""
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (n, 6)) * [1, 2, 0.5, 3, 1, 0.2] + [2, -1, 5, 0, 3, -2]
        beta = np.array([1.5, 0.0, -2.0, 0.0, 0.8, 0.0])  # x2, x4, x6 truly zero
        return X, beta

    def _cols(self, X):
        return pd.DataFrame(X, columns=[f"x{i + 1}" for i in range(X.shape[1])])

    def test_linear_lasso_matches_sklearn(self, con):
        X, beta = self._sparse(101)
        rng = np.random.default_rng(1010)
        y = 3 + X @ beta + rng.normal(0, 1.5, len(X))
        Xs, _, sd = zscore(X)
        names = [f"x{i + 1}" for i in range(6)]
        coefs = fit(con, "linreg_fit", self._cols(X).assign(y=y), l1=0.05)
        got = np.array([coefs[c] for c in names]) * sd / y.std()
        ref = Lasso(alpha=0.05, max_iter=200000, tol=1e-12).fit(Xs, (y - y.mean()) / y.std())
        assert got == pytest.approx(ref.coef_, abs=1e-4)

    def test_linear_elasticnet_matches_sklearn(self, con):
        X, beta = self._sparse(102)
        rng = np.random.default_rng(1020)
        y = 3 + X @ beta + rng.normal(0, 1.5, len(X))
        Xs, _, sd = zscore(X)
        names = [f"x{i + 1}" for i in range(6)]
        l1, l2 = 0.03, 0.04
        coefs = fit(con, "linreg_fit", self._cols(X).assign(y=y), l1=l1, l2=l2)
        got = np.array([coefs[c] for c in names]) * sd / y.std()
        ref = ElasticNet(alpha=l1 + l2, l1_ratio=l1 / (l1 + l2),
                         max_iter=200000, tol=1e-12).fit(Xs, (y - y.mean()) / y.std())
        assert got == pytest.approx(ref.coef_, abs=1e-4)

    def test_lasso_selects_true_zeros(self, con):
        X, beta = self._sparse(103)
        rng = np.random.default_rng(1030)
        y = 3 + X @ beta + rng.normal(0, 1.5, len(X))
        coefs = fit(con, "linreg_fit", self._cols(X).assign(y=y), l1=0.15)
        # the three genuinely-zero features get coefficient exactly 0
        assert coefs["x2"] == 0.0 and coefs["x4"] == 0.0 and coefs["x6"] == 0.0
        # a genuinely-nonzero feature survives
        assert coefs["x1"] != 0.0 and coefs["x3"] != 0.0

    def test_l1_zero_equals_unpenalized(self, con):
        X, beta = self._sparse(104)
        rng = np.random.default_rng(1040)
        y = 1 + X @ beta + rng.normal(0, 1, len(X))
        with_l1 = fit(con, "linreg_fit", self._cols(X).assign(y=y), l1=0.0)
        plain = fit(con, "linreg_fit", self._cols(X).assign(y=y))
        for k in plain:
            assert with_l1[k] == pytest.approx(plain[k], abs=1e-9)

    def test_logistic_l1_matches_sklearn(self, con):
        X, beta = self._sparse(105)
        Xs, _, sd = zscore(X)
        rng = np.random.default_rng(1050)
        yb = rng.binomial(1, 1 / (1 + np.exp(-(0.3 + Xs @ beta))))
        names = [f"x{i + 1}" for i in range(6)]
        l1 = 0.02
        coefs = fit(con, "logit_fit", self._cols(X).assign(y=yb), l1=l1)
        got = np.array([coefs[c] for c in names]) * sd
        ref = LogisticRegression(penalty="l1", solver="saga", C=1 / (len(X) * l1),
                                 max_iter=500000, tol=1e-10).fit(Xs, yb)
        assert got == pytest.approx(ref.coef_[0], abs=5e-3)

    def test_poisson_l1_kkt_optimality(self, con):
        # sklearn has no L1 Poisson, so check the subgradient KKT conditions:
        # active coords have |grad| = l1, zeroed coords have |grad| <= l1.
        X, beta = self._sparse(106)
        Xs, mu, sd = zscore(X)
        rng = np.random.default_rng(1060)
        y = rng.poisson(np.exp(0.4 + Xs @ (beta * 0.5))).astype(float)
        l1, n = 0.02, len(X)
        coefs = fit(con, "poisson_fit", self._cols(X).assign(y=y), l1=l1)
        bstd = np.array([coefs[f"x{i + 1}"] for i in range(6)]) * sd
        z = np.log(y.mean()) - np.log(y.mean()) + Xs @ bstd  # standardized eta (intercept cancels below)
        # rebuild standardized intercept so mean(mu)~mean(ytil): use macro intercept
        b0_std = coefs["(Intercept)"] - np.log(y.mean()) + (np.array([coefs[f"x{i+1}"] for i in range(6)]) * mu).sum()
        muhat = np.exp(b0_std + Xs @ bstd)
        g = -(Xs.T @ (y / y.mean() - muhat)) / n
        for j in range(6):
            if abs(bstd[j]) > 1e-8:
                assert abs(abs(g[j]) - l1) < 1e-4
            else:
                assert abs(g[j]) <= l1 + 1e-4

    def test_negative_l1_errors(self, con):
        X, beta = self._sparse(107, n=200)
        rng = np.random.default_rng(1070)
        y = 1 + X @ beta + rng.normal(0, 1, len(X))
        _load(con, "traindata", self._cols(X).assign(y=y))
        with pytest.raises(DuckDBError) as e:
            con.execute("SELECT * FROM linreg_fit('traindata', 'y', l1 := -1.0)").fetchall()
        assert "l1 must be >= 0" in str(e.value)


# --------------------------------------------------------------------------- #
# Tweedie regression
# --------------------------------------------------------------------------- #
class TestTweedie:
    def _compound(self, seed, n=3000, zero_frac=0.3):
        """Compound Poisson-Gamma: a fraction of exact zeros plus positive."""
        rng = np.random.default_rng(seed)
        X = np.column_stack([rng.normal(0, 1, n), rng.normal(0, 1, n)])
        mu = np.exp(0.5 + 0.6 * X[:, 0] - 0.3 * X[:, 1])
        y = np.where(rng.random(n) < zero_frac, 0.0, rng.gamma(2.0, mu / 2.0))
        return pd.DataFrame(X, columns=["x1", "x2"]).assign(y=y), X, y

    @pytest.mark.parametrize("power", [1.3, 1.5, 1.7])
    def test_tweedie_matches_sklearn(self, con, power):
        df, X, y = self._compound(81 + int(power * 10))
        coefs = fit(con, "tweedie_fit", df, power=power)
        ref = TweedieRegressor(power=power, alpha=0, link="log",
                               max_iter=100000, tol=1e-10).fit(X, y)
        assert coefs["(Intercept)"] == pytest.approx(ref.intercept_, abs=1e-4, rel=1e-4)
        assert np.array([coefs["x1"], coefs["x2"]]) == pytest.approx(ref.coef_, abs=1e-4, rel=1e-4)

    def test_tweedie_p1_equals_poisson(self, con):
        rng = np.random.default_rng(85)
        X = np.column_stack([rng.normal(0, 1, 2000), rng.normal(0, 1, 2000)])
        y = rng.poisson(np.exp(0.5 + 0.5 * X[:, 0] - 0.2 * X[:, 1])).astype(float)
        df = pd.DataFrame(X, columns=["x1", "x2"]).assign(y=y)
        tw = fit(con, "tweedie_fit", df, power=1.0)
        po = fit(con, "poisson_fit", df)
        for k in po:
            assert tw[k] == pytest.approx(po[k], abs=1e-6)

    def test_tweedie_p2_equals_gamma(self, con):
        rng = np.random.default_rng(86)
        X = np.column_stack([rng.normal(0, 1, 2000), rng.normal(0, 1, 2000)])
        y = rng.gamma(2.0, np.exp(0.5 + 0.4 * X[:, 0]) / 2.0)
        df = pd.DataFrame(X, columns=["x1", "x2"]).assign(y=y)
        tw = fit(con, "tweedie_fit", df, power=2.0)
        ga = fit(con, "gamma_fit", df)
        for k in ga:
            assert tw[k] == pytest.approx(ga[k], abs=1e-6)

    def test_tweedie_handles_zeros(self, con):
        df, X, y = self._compound(87, zero_frac=0.45)
        assert (y == 0).mean() > 0.3  # genuinely zero-inflated
        coefs = fit(con, "tweedie_fit", df, power=1.5)
        assert all(np.isfinite(v) for v in coefs.values())

    def test_tweedie_predict_is_exp_score(self, con):
        df, X, y = self._compound(88)
        coefs = fit(con, "tweedie_fit", df, power=1.5)
        out = predict(con, "tweedie_predict", coefs, df)
        z = coefs["(Intercept)"] + X @ [coefs["x1"], coefs["x2"]]
        assert out["prediction"].to_numpy() == pytest.approx(np.exp(z), rel=1e-9)

    @pytest.mark.parametrize("power", [1.3, 1.6])
    def test_tweedie_evaluate_matches_sklearn(self, con, power):
        df, X, y = self._compound(89 + int(power * 10))
        coefs = fit(con, "tweedie_fit", df, power=power)
        mu = predict(con, "tweedie_predict", coefs, df)["prediction"].to_numpy()
        m = evaluate(con, "tweedie_evaluate", coefs, df, power=power)
        n = len(y)
        assert m["deviance"] == pytest.approx(n * mean_tweedie_deviance(y, mu, power=power), rel=1e-7)
        assert m["pseudo_r2"] == pytest.approx(d2_tweedie_score(y, mu, power=power), rel=1e-7)

    def test_tweedie_power_below_one_errors(self, con):
        df, X, y = self._compound(91)
        _load(con, "traindata", df)
        with pytest.raises(DuckDBError) as e:
            con.execute("SELECT * FROM tweedie_fit('traindata', 'y', power := 0.5)").fetchall()
        assert "power must be >= 1" in str(e.value)

    def test_tweedie_negative_outcome_errors(self, con):
        df, X, y = self._compound(92)
        df.loc[0, "y"] = -1.0
        _load(con, "traindata", df)
        with pytest.raises(DuckDBError) as e:
            con.execute("SELECT * FROM tweedie_fit('traindata', 'y', power := 1.5)").fetchall()
        assert "non-negative" in str(e.value)


# --------------------------------------------------------------------------- #
# Sample weights
# --------------------------------------------------------------------------- #
class TestWeights:
    def test_linear_weights_match_sklearn(self, con):
        X = mixed_features(71)
        rng = np.random.default_rng(701)
        w = rng.uniform(0.1, 5, len(X))
        y = 2 + X @ [1.5, -1.0, 0.4] + rng.normal(0, 2, len(X))
        df = frame(X, y).assign(w=w)
        coefs = fit(con, "linreg_fit", df, weights_col="'w'")
        ref = LinearRegression().fit(X, y, sample_weight=w)
        assert_coefs(coefs, ref.intercept_, ref.coef_, atol=1e-6, rtol=1e-6)

    def test_logistic_weights_match_sklearn(self, con):
        X = mixed_features(72)
        Xs, _, _ = zscore(X)
        rng = np.random.default_rng(702)
        w = rng.uniform(0.1, 5, len(X))
        y = rng.binomial(1, 1 / (1 + np.exp(-(0.3 + Xs @ [0.9, -0.6, 0.4]))))
        df = frame(X, y).assign(w=w)
        coefs = fit(con, "logit_fit", df, weights_col="'w'")
        ref = logreg_unpenalized(X, y).fit(X, y, sample_weight=w)
        assert_coefs(coefs, ref.intercept_[0], ref.coef_[0], atol=1e-4, rtol=1e-4)

    def test_poisson_weights_match_sklearn(self, con):
        X = mixed_features(73)
        Xs, _, _ = zscore(X)
        rng = np.random.default_rng(703)
        w = rng.uniform(0.1, 5, len(X))
        y = rng.poisson(np.exp(0.5 + Xs @ [0.4, -0.3, 0.2])).astype(float)
        df = frame(X, y).assign(w=w)
        coefs = fit(con, "poisson_fit", df, weights_col="'w'")
        ref = PoissonRegressor(alpha=0, max_iter=20000, tol=1e-12).fit(X, y, sample_weight=w)
        assert_coefs(coefs, ref.intercept_, ref.coef_, atol=1e-4, rtol=1e-4)

    def test_gamma_weights_match_sklearn(self, con):
        X = mixed_features(74)
        Xs, _, _ = zscore(X)
        rng = np.random.default_rng(704)
        w = rng.uniform(0.1, 5, len(X))
        y = rng.gamma(2.0, np.exp(0.6 + Xs @ [0.4, -0.2, 0.3]) / 2.0)
        df = frame(X, y).assign(w=w)
        coefs = fit(con, "gamma_fit", df, weights_col="'w'")
        ref = GammaRegressor(alpha=0, max_iter=20000, tol=1e-12).fit(X, y, sample_weight=w)
        assert_coefs(coefs, ref.intercept_, ref.coef_, atol=1e-4, rtol=1e-4)

    def test_equal_weights_equal_unweighted(self, con):
        X = mixed_features(75)
        rng = np.random.default_rng(705)
        y = 1 + X @ [0.5, -1.0, 0.3] + rng.normal(0, 1, len(X))
        weighted = fit(con, "linreg_fit", frame(X, y).assign(w=3.0), weights_col="'w'")
        plain = fit(con, "linreg_fit", frame(X, y))
        for k in plain:
            assert weighted[k] == pytest.approx(plain[k], abs=1e-9)

    def test_integer_weights_equal_replicated_rows(self, con):
        X = mixed_features(76, n=300)
        rng = np.random.default_rng(706)
        freq = rng.integers(1, 5, len(X))
        y = 1 + X @ [0.8, -0.5, 0.2] + rng.normal(0, 1, len(X))
        weighted = fit(con, "linreg_fit", frame(X, y).assign(w=freq.astype(float)), weights_col="'w'")
        rep = frame(np.repeat(X, freq, axis=0), np.repeat(y, freq))
        replicated = fit(con, "linreg_fit", rep)
        for k in replicated:
            assert weighted[k] == pytest.approx(replicated[k], abs=1e-7)

    def test_weights_with_offset(self, con):
        # score-equation optimality of the weighted, offset Poisson fit
        X = mixed_features(77)
        rng = np.random.default_rng(707)
        w = rng.uniform(0.2, 4, len(X))
        logo = np.log(rng.uniform(0.5, 4, len(X)))
        y = rng.poisson(np.exp(logo - 0.3 + zscore(X)[0] @ [0.5, -0.2, 0.3])).astype(float)
        df = frame(X, y).assign(w=w, logexp=logo)
        coefs = fit(con, "poisson_fit", df, weights_col="'w'", offset_col="'logexp'")
        z = coefs["(Intercept)"] + X @ [coefs[c] for c in NAMES] + logo
        resid = w * (y - np.exp(z))  # weighted score equations
        n = len(X)
        for xv in [np.ones(n), X[:, 0], X[:, 1], X[:, 2]]:
            assert abs((resid * xv).sum()) / w.sum() < 1e-6

    def test_negative_weight_errors(self, con):
        X = mixed_features(78, n=100)
        rng = np.random.default_rng(708)
        w = rng.uniform(0.1, 3, len(X)); w[0] = -1.0
        y = 1 + X @ [0.5, -0.5, 0.2] + rng.normal(0, 1, len(X))
        _load(con, "traindata", frame(X, y).assign(w=w))
        with pytest.raises(DuckDBError) as e:
            con.execute("SELECT * FROM linreg_fit('traindata', 'y', weights_col := 'w')").fetchall()
        assert "non-negative" in str(e.value)

    def test_missing_weights_column_errors(self, con):
        X = mixed_features(79, n=100)
        rng = np.random.default_rng(709)
        y = 1 + X @ [0.5, -0.5, 0.2] + rng.normal(0, 1, len(X))
        _load(con, "traindata", frame(X, y))
        with pytest.raises(DuckDBError) as e:
            con.execute("SELECT * FROM linreg_fit('traindata', 'y', weights_col := 'nope')").fetchall()
        assert "weights column" in str(e.value) and "nope" in str(e.value)


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
class TestEdgeCases:
    @pytest.mark.parametrize("value", [4.0, 5.0, 4.2, 0.1, np.pi, -7.3, 1e6])
    def test_constant_feature_coefficient_exactly_zero(self, con, value):
        X = mixed_features(41, n=300)[:, :2]
        rng = np.random.default_rng(401)
        y = 1 + X @ [1.2, -0.8] + rng.normal(0, 1, len(X))
        df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "cst": value, "y": y})
        coefs = fit(con, "linreg_fit", df)
        assert coefs["cst"] == 0.0  # exactly, not merely near 0

    def test_null_rows_dropped_matches_complete_fit(self, con):
        X = mixed_features(42, n=500)
        rng = np.random.default_rng(402)
        y = 1 + X @ [0.8, -1.2, 0.4] + rng.normal(0, 1, len(X))
        df = frame(X, y)
        fmask = rng.random(len(X)) < 0.1
        ymask = rng.random(len(X)) < 0.05
        df.loc[fmask, "x1"] = np.nan
        df.loc[ymask, "y"] = np.nan
        coefs = fit(con, "linreg_fit", df)
        comp = ~(fmask | ymask)
        ref = LinearRegression().fit(X[comp], y[comp])
        assert_coefs(coefs, ref.intercept_, ref.coef_, atol=1e-4, rtol=1e-4)

    def test_boolean_outcome(self, con):
        X = mixed_features(43, n=300)
        rng = np.random.default_rng(403)
        yb = rng.binomial(1, 1 / (1 + np.exp(-(0.3 + zscore(X)[0] @ [1.0, -0.6, 0.3]))))
        df = frame(X, yb)
        df["y"] = df["y"].astype(bool)
        coefs = fit(con, "logit_fit", df)
        ref = logreg_unpenalized(X, yb)
        assert_coefs(coefs, ref.intercept_[0], ref.coef_[0], atol=1e-4, rtol=1e-4)

    def test_integer_and_decimal_features(self, con):
        rng = np.random.default_rng(404)
        n = 400
        xi = rng.integers(-10, 10, n)
        xd = np.round(rng.normal(0, 1, n), 3)
        y = 1 + 0.2 * xi - 0.5 * xd + rng.normal(0, 1, n)
        con.execute("CREATE OR REPLACE TABLE traindata AS SELECT * FROM (VALUES "
                    + ",".join(f"({int(a)}, {b}, {c})" for a, b, c in zip(xi, xd, y))
                    + ") AS t(xi, xd, y)")
        rows = con.execute(
            "SELECT feature, coefficient FROM linreg_fit('traindata', 'y')"
        ).fetchall()
        coefs = {f: c for f, c in rows}
        ref = LinearRegression().fit(np.column_stack([xi, xd]), y)
        assert coefs["xi"] == pytest.approx(ref.coef_[0], abs=1e-6)
        assert coefs["xd"] == pytest.approx(ref.coef_[1], abs=1e-6)


# --------------------------------------------------------------------------- #
# Categorical encoding helper (dummy_encode_sql)
# --------------------------------------------------------------------------- #
class TestDummyEncode:
    def _generate_and_fit(self, con, df, outcome="y"):
        _load(con, "rawdata", df)
        sql = con.execute("SELECT dummy_encode_sql('rawdata', %r)" % outcome).fetchone()[0]
        con.execute(f"CREATE OR REPLACE TABLE encdata AS {sql}")
        rows = con.execute(
            f"SELECT feature, coefficient FROM linreg_fit('encdata', '{outcome}')"
        ).fetchall()
        return {f: c for f, c in rows}, sql

    def test_matches_manual_drop_first_onehot(self, con):
        # R's C(factor) == pandas get_dummies(drop_first=True) on sorted levels;
        # verify the generated encoding reproduces sklearn on that design.
        rng = np.random.default_rng(201)
        n = 4000
        num = rng.normal(0, 1, n)
        promo = rng.random(n) < 0.4
        region = rng.choice(["North", "South", "East", "West"], n)
        eff = {"North": 0.6, "South": 1.7, "East": 0.0, "West": 0.9}
        y = 1 + 2 * num - 1.5 * promo + np.array([eff[r] for r in region]) + rng.normal(0, 1, n)
        df = pd.DataFrame({"y": y, "num": num, "promo": promo, "region": region})
        coefs, _ = self._generate_and_fit(con, df)
        # reference design: drop the alphabetically-first level (East)
        dummies = pd.get_dummies(df["region"], prefix="region").drop(columns="region_East")
        X = np.column_stack([num, promo.astype(float), dummies.to_numpy().astype(float)])
        ref = LinearRegression().fit(X, y)
        names = ["num", "promo"] + list(dummies.columns)
        assert coefs["(Intercept)"] == pytest.approx(ref.intercept_, abs=1e-4)
        for name, c in zip(names, ref.coef_):
            assert coefs[name] == pytest.approx(c, abs=1e-4), name

    def test_reference_is_first_level_and_kminus1_dummies(self, con):
        df = pd.DataFrame({"y": [1.0, 2, 3, 4], "g": ["b", "a", "c", "a"]})
        _load(con, "rawdata", df)
        sql = con.execute("SELECT dummy_encode_sql('rawdata', 'y')").fetchone()[0]
        # 3 levels -> 2 dummies; reference 'a' (min) omitted
        assert '"g_b"' in sql and '"g_c"' in sql and '"g_a"' not in sql
        assert "EXCLUDE (g)" in sql

    def test_no_categoricals_passthrough(self, con):
        df = pd.DataFrame({"y": [1.0, 2, 3], "x": [0.1, 0.2, 0.3], "flag": [True, False, True]})
        _load(con, "rawdata", df)
        sql = con.execute("SELECT dummy_encode_sql('rawdata', 'y')").fetchone()[0]
        assert sql.strip() == "SELECT * FROM rawdata"  # boolean/numeric untouched

    def test_null_category_yields_null_dummy(self, con):
        df = pd.DataFrame({"y": [1.0, 2, 3, 4], "g": ["a", "b", None, "b"]})
        _load(con, "rawdata", df)
        sql = con.execute("SELECT dummy_encode_sql('rawdata', 'y')").fetchone()[0]
        con.execute(f"CREATE OR REPLACE TABLE encdata AS {sql}")
        # the NULL-category row gets a NULL dummy (so the fit will drop it, as R
        # drops NA rows); the other three are non-NULL
        non_null = con.execute("SELECT count(g_b) FROM encdata").fetchone()[0]
        assert non_null == 3


# --------------------------------------------------------------------------- #
# Error handling & reserved names
# --------------------------------------------------------------------------- #
class TestErrors:
    def _tiny(self, con, cols="x DOUBLE, y INTEGER", rows="(1.0, 1), (2.0, 0)"):
        con.execute(f"CREATE OR REPLACE TABLE traindata AS SELECT * FROM (VALUES {rows}) "
                    f"AS t({', '.join(c.split()[0] for c in cols.split(','))})")

    def _err(self, con, sql):
        with pytest.raises(DuckDBError) as e:
            con.execute(sql).fetchall()
        return str(e.value)

    def test_logit_nonbinary(self, con):
        con.execute("CREATE OR REPLACE TABLE t AS SELECT * FROM (VALUES (1.0,1),(2.0,2)) v(x,y)")
        assert "binary" in self._err(con, "SELECT * FROM logit_fit('t','y')")

    def test_poisson_negative(self, con):
        con.execute("CREATE OR REPLACE TABLE t AS SELECT * FROM (VALUES (1.0,-1.0),(2.0,3.0)) v(x,y)")
        assert "non-negative" in self._err(con, "SELECT * FROM poisson_fit('t','y')")

    def test_gamma_nonpositive(self, con):
        con.execute("CREATE OR REPLACE TABLE t AS SELECT * FROM (VALUES (1.0,0.0),(2.0,3.0)) v(x,y)")
        assert "strictly positive" in self._err(con, "SELECT * FROM gamma_fit('t','y')")

    def test_entirely_null_feature(self, con):
        con.execute("CREATE OR REPLACE TABLE t AS SELECT NULL::DOUBLE x1, 2.0 x2, 1 y "
                    "UNION ALL SELECT NULL, 3.0, 0")
        msg = self._err(con, "SELECT * FROM logit_fit('t','y')")
        assert "entirely NULL" in msg and "x1" in msg

    def test_no_feature_columns(self, con):
        con.execute("CREATE OR REPLACE TABLE t AS SELECT * FROM (VALUES (1),(0)) v(y)")
        assert "no feature columns" in self._err(con, "SELECT * FROM linreg_fit('t','y')")

    def test_no_complete_rows(self, con):
        con.execute("CREATE OR REPLACE TABLE t AS SELECT NULL::DOUBLE x1, 2.0 x2, 1 y "
                    "UNION ALL SELECT 3.0, NULL, 0")
        assert "no complete" in self._err(con, "SELECT * FROM logit_fit('t','y')")

    def test_negative_l2(self, con):
        con.execute("CREATE OR REPLACE TABLE t AS SELECT * FROM (VALUES (1.0,1),(2.0,0)) v(x,y)")
        assert "l2 must be >= 0" in self._err(con, "SELECT * FROM logit_fit('t','y', l2 := -1.0)")

    def test_reserved_column_name(self, con):
        con.execute('CREATE OR REPLACE TABLE t AS SELECT 1.0 "__reg_rid__", 1 y '
                    "UNION ALL SELECT 2.0, 0")
        assert "reserved" in self._err(con, "SELECT * FROM logit_fit('t','y')")

    def test_intercept_feature_name(self, con):
        con.execute('CREATE OR REPLACE TABLE t AS SELECT 1.0 "(Intercept)", 1 y '
                    "UNION ALL SELECT 2.0, 0")
        assert "(Intercept)" in self._err(con, "SELECT * FROM logit_fit('t','y')")

    def test_prob_pred_collision(self, con):
        con.execute("CREATE OR REPLACE TABLE m AS SELECT 'x' feature, 1.0 coefficient")
        con.execute("CREATE OR REPLACE TABLE d AS SELECT 1.0 x, 0.5 prob")
        assert "prob" in self._err(con, "SELECT * FROM logit_predict('m','d')")

    def test_prediction_collision(self, con):
        con.execute("CREATE OR REPLACE TABLE m AS SELECT 'x' feature, 1.0 coefficient")
        con.execute("CREATE OR REPLACE TABLE d AS SELECT 1.0 x, 0.5 prediction")
        assert "prediction" in self._err(con, "SELECT * FROM linreg_predict('m','d')")

    def test_nonexistent_table(self, con):
        assert "no_such_table" in self._err(
            con, "SELECT * FROM linreg_fit('no_such_table','y')"
        )
