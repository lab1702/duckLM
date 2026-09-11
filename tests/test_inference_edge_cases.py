"""Inference regression cases for names, sample membership, and weights."""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def con():
    connection = duckdb.connect()
    connection.execute((Path(__file__).resolve().parents[1] / "regression_macros.sql").read_text())
    yield connection
    connection.close()


def load(con, name, data):
    con.register("edge_source", data)
    con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM edge_source')
    con.unregister("edge_source")


def model(con, feature="x", name="edge_model", multinomial=False):
    data = pd.DataFrame({"feature": ["(Intercept)", feature], "coefficient": [0.3, 0.2]})
    if multinomial:
        data = pd.concat([data.assign(coefficient=0.0, **{"class": "a"}), data.assign(**{"class": "b"}),
                          data.assign(coefficient=[-0.1, -0.15], **{"class": "c"})], ignore_index=True)
    load(con, name, data)


def training():
    rng = np.random.default_rng(8401)
    x = rng.normal(size=48)
    return pd.DataFrame({"x": x, "y": rng.poisson(np.exp(0.3 + 0.2 * x)).astype(float)})


@pytest.mark.parametrize("macro", ["linreg_summary", "linreg_predict_ci", "linreg_influence", "multinom_summary"])
@pytest.mark.parametrize("table_name", ["mdl", "beta", "num", "final"])
@pytest.mark.parametrize("table_role", ["training", "model"])
def test_inference_does_not_shadow_user_tables(con, macro, table_name, table_role):
    model(con, multinomial=macro == "multinom_summary")
    data = training()
    if macro == "multinom_summary":
        data["y"] = np.resize(["a", "b", "c"], len(data))
    load(con, "edge_train", data)
    expected = con.execute(f"SELECT * FROM {macro}('edge_model', 'edge_train', 'y')").df()
    if table_role == "training":
        load(con, table_name, data)
        actual = con.execute(f"SELECT * FROM {macro}('edge_model', '{table_name}', 'y')").df()
    else:
        model(con, name=table_name, multinomial=macro == "multinom_summary")
        actual = con.execute(f"SELECT * FROM {macro}('{table_name}', 'edge_train', 'y')").df()
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("macro", ["linreg_summary", "linreg_predict_ci", "linreg_influence", "multinom_summary"])
@pytest.mark.parametrize("feature", ["rid", "srid"])
def test_inference_preserves_row_identifier_features(con, macro, feature):
    data = training()
    multinomial = macro == "multinom_summary"
    if multinomial:
        data["y"] = np.resize(["a", "b", "c"], len(data))
    model(con, multinomial=multinomial)
    load(con, "edge_train", data)
    expected = con.execute(f"SELECT * FROM {macro}('edge_model', 'edge_train', 'y')").df()
    model(con, feature=feature, multinomial=multinomial)
    load(con, "edge_train", data.rename(columns={"x": feature}))
    actual = con.execute(f"SELECT * FROM {macro}('edge_model', 'edge_train', 'y')").df()
    if "feature" in actual:
        actual["feature"] = actual["feature"].replace({feature: "x"})
    else:
        actual = actual.rename(columns={feature: "x"})
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("macro", ["linreg_summary", "linreg_predict_ci", "linreg_influence"])
def test_inference_drops_null_requested_weights(con, macro):
    model(con)
    data = training().assign(w=1.0)
    data.loc[3, ["y", "w"]] = [1000.0, np.nan]
    load(con, "edge_train", data)
    load(con, "edge_complete", data.dropna())
    extra = ", newdata := 'edge_complete'" if macro.endswith("predict_ci") else ""
    actual = con.execute(f"SELECT * FROM {macro}('edge_model', 'edge_train', 'y', weights_col := 'w'{extra})").df()
    expected = con.execute(f"SELECT * FROM {macro}('edge_model', 'edge_complete', 'y', weights_col := 'w'{extra})").df()
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("macro", ["linreg_summary", "linreg_predict_ci", "linreg_influence"])
def test_inference_rejects_missing_weights_column(con, macro):
    model(con)
    load(con, "edge_train", training())
    with pytest.raises(duckdb.Error, match="weights column.*not found"):
        con.execute(f"SELECT * FROM {macro}('edge_model', 'edge_train', 'y', weights_col := 'missing')").fetchall()


@pytest.mark.parametrize("labels", [["north", "south", "east", "west"], ["01", "1", "1.0", "2"]])
def test_cluster_labels_keep_exact_identity(con, labels):
    model(con)
    data = training().assign(grp=np.resize(np.arange(4), 48))
    load(con, "edge_train", data)
    expected = con.execute("SELECT * FROM poisson_summary('edge_model', 'edge_train', 'y', cluster_col := 'grp')").df()
    data["grp"] = np.resize(labels, len(data))
    load(con, "edge_train", data)
    actual = con.execute("SELECT * FROM poisson_summary('edge_model', 'edge_train', 'y', cluster_col := 'grp')").df()
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("robust", ["hc0", "hc1", "hc2", "hc3"])
@pytest.mark.parametrize("zero_weight", [False, True])
def test_analytic_weight_robust_covariance_matches_sandwich(con, robust, zero_weight):
    # statsmodels GLM.score_factor includes var_weights in the observation score;
    # sandwich_covariance only divides by freq_weights, not analytic var_weights.
    # Reference: https://www.statsmodels.org/stable/_modules/statsmodels/stats/sandwich_covariance.html
    model(con)
    data = training().assign(w=np.linspace(0.25, 3.0, 48))
    if zero_weight:
        data.loc[0, "w"] = 0.0
    x = np.column_stack([np.ones(len(data)), data.x])
    mu = np.exp(x @ np.array([0.3, 0.2]))
    a = data.w.to_numpy()
    bread = np.linalg.inv(x.T @ ((a * mu)[:, None] * x))
    h = np.einsum("ij,jk,ik->i", x, bread, x) * a * mu
    meat_weights = (a * (data.y.to_numpy() - mu)) ** 2
    if robust == "hc2":
        meat_weights /= 1.0 - h
    elif robust == "hc3":
        meat_weights /= (1.0 - h) ** 2
    cov = bread @ (x.T @ (meat_weights[:, None] * x)) @ bread
    if robust == "hc1":
        cov *= len(data) / (len(data) - x.shape[1])
    expected = np.sqrt(np.diag(cov))
    for scale in [1.0, 7.0]:
        load(con, "edge_train", data.assign(w=a * scale))
        actual = con.execute(f"SELECT std_error FROM poisson_summary('edge_model', 'edge_train', 'y', weights_col := 'w', robust := '{robust}')").df()["std_error"].to_numpy()
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=1e-10)


@pytest.mark.parametrize("power, family", [(1.0, "poisson"), (2.0, "gamma")])
def test_tweedie_deviance_residual_endpoints(con, power, family):
    model(con)
    data = training()
    if family == "gamma":
        data["y"] += 0.2
    load(con, "edge_train", data)
    actual = con.execute(f"SELECT deviance_resid FROM tweedie_influence('edge_model', 'edge_train', 'y', power := {power})").df()["deviance_resid"].to_numpy()
    expected = con.execute(f"SELECT deviance_resid FROM {family}_influence('edge_model', 'edge_train', 'y')").df()["deviance_resid"].to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


def test_multinomial_summary_excludes_null_outcomes(con):
    model(con, multinomial=True)
    data = training()
    data["y"] = np.resize(["a", "b", "c"], len(data)).astype(object)
    data.loc[0:10, "y"] = None
    load(con, "edge_train", data)
    load(con, "edge_complete", data.dropna())
    actual = con.execute("SELECT * FROM multinom_summary('edge_model', 'edge_train', 'y')").df()
    expected = con.execute("SELECT * FROM multinom_summary('edge_model', 'edge_complete', 'y')").df()
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("macro", ["linreg_summary", "linreg_predict_ci", "linreg_influence", "multinom_summary"])
@pytest.mark.parametrize("reserved", ["__reg_rid__", "__REG_srid__"])
def test_inference_rejects_reserved_input_columns(con, macro, reserved):
    model(con, multinomial=macro == "multinom_summary")
    data = training()
    data[reserved] = 1.0
    if macro == "multinom_summary":
        data["y"] = np.resize(["a", "b", "c"], len(data))
    load(con, "edge_train", data)
    with pytest.raises(duckdb.Error, match="reserved"):
        con.execute(f"SELECT * FROM {macro}('edge_model', 'edge_train', 'y')").fetchall()


def test_prediction_ci_rejects_reserved_newdata_columns(con):
    model(con)
    load(con, "edge_train", training())
    load(con, "edge_score", training().assign(__reg_srid__=1.0))
    with pytest.raises(duckdb.Error, match="reserved"):
        con.execute("SELECT * FROM linreg_predict_ci('edge_model', 'edge_train', 'y', newdata := 'edge_score')").fetchall()


@pytest.mark.parametrize("missing_column", [False, True])
def test_cluster_summary_rejects_unknown_group_membership(con, missing_column):
    model(con)
    data = training()
    if not missing_column:
        data["grp"] = np.resize(["north", "south", "east"], len(data)).astype(object)
        data.loc[0, "grp"] = None
    load(con, "edge_train", data)
    message = "cluster column.*not found" if missing_column else "cluster column.*contains NULL"
    with pytest.raises(duckdb.Error, match=message):
        con.execute("SELECT * FROM poisson_summary('edge_model', 'edge_train', 'y', cluster_col := 'grp')").fetchall()


def test_cluster_summary_ignores_unknown_groups_on_excluded_rows(con):
    model(con)
    data = training()
    data["grp"] = np.resize(["north", "south", "east"], len(data)).astype(object)
    data.loc[0, ["y", "grp"]] = [np.nan, None]
    load(con, "edge_train", data)
    load(con, "edge_complete", data.dropna())
    actual = con.execute("SELECT * FROM poisson_summary('edge_model', 'edge_train', 'y', cluster_col := 'grp')").df()
    expected = con.execute("SELECT * FROM poisson_summary('edge_model', 'edge_complete', 'y', cluster_col := 'grp')").df()
    pd.testing.assert_frame_equal(actual, expected)
