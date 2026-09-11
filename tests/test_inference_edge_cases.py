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


def rank_deficient_data(con):
    i = np.arange(20)
    x = i / 10.0
    z = ((i * 7) % 11) / 10.0
    data = pd.DataFrame({"x": x, "z": z, "u": x + z,
                         "y": 1.0 + 2.0 * x + 3.0 * z + 0.1 * (i % 3)})
    coefficients = pd.DataFrame({"feature": ["(Intercept)", "u", "x", "z"],
                                 "coefficient": [1.0, 0.5, 1.5, 2.5]})
    load(con, "edge_train", data)
    load(con, "edge_model", coefficients)
    return data, coefficients


@pytest.mark.parametrize("robust", ["hc0", "hc1", "hc2", "hc3", "cluster"])
def test_rank_deficiency_nulls_robust_inference(con, robust):
    data, coefficients = rank_deficient_data(con)
    if robust == "cluster":
        load(con, "edge_train", data.assign(grp=np.arange(len(data)) % 4))
        extra = "cluster_col := 'grp'"
    else:
        extra = f"robust := '{robust}'"
    result = con.execute(f"SELECT * FROM linreg_summary('edge_model', 'edge_train', 'y', {extra})").df()
    assert result["feature"].tolist() == coefficients["feature"].tolist()
    np.testing.assert_array_equal(result["coefficient"], coefficients["coefficient"])
    assert result[["std_error", "statistic", "p_value", "conf_low", "conf_high"]].isna().all().all()


def test_rank_deficiency_nulls_leverage_but_preserves_residuals_and_predictions(con):
    data, _ = rank_deficient_data(con)
    result = con.execute("SELECT * FROM linreg_influence('edge_model', 'edge_train', 'y')").df()
    assert len(result) == len(data)
    assert result[["hat", "std_resid", "cooks_distance"]].isna().all().all()
    assert np.isfinite(result[["pearson_resid", "deviance_resid"]].to_numpy()).all()
    predicted = con.execute("SELECT * FROM linreg_predict_ci('edge_model', 'edge_train', 'y')").df()
    np.testing.assert_allclose(predicted["prediction"], 1 + 2 * data.x + 3 * data.z)
    assert predicted[["conf_low", "conf_high"]].isna().all().all()


@pytest.mark.parametrize("column", ["prediction", "conf_low", "conf_high"])
@pytest.mark.parametrize("uppercase", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("newdata", [False, True])
def test_prediction_ci_rejects_output_column_collisions(con, column, uppercase, empty, newdata):
    model(con)
    load(con, "edge_train", training())
    name = column.upper() if uppercase else column
    data = training().assign(**{name: 999.0})
    if empty:
        data = data.iloc[:0]
    load(con, "edge_score" if newdata else "edge_train", data)
    extra = ", newdata := 'edge_score'" if newdata else ""
    with pytest.raises(duckdb.Error, match="collides with the output"):
        con.execute(f"SELECT * FROM linreg_predict_ci('edge_model', 'edge_train', 'y'{extra})").fetchall()


@pytest.mark.parametrize("column", ["hat", "pearson_resid", "deviance_resid", "std_resid", "cooks_distance"])
@pytest.mark.parametrize("uppercase", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_influence_rejects_output_column_collisions(con, column, uppercase, empty):
    model(con)
    name = column.upper() if uppercase else column
    data = training().assign(**{name: 999.0})
    if empty:
        data = data.iloc[:0]
    load(con, "edge_train", data)
    with pytest.raises(duckdb.Error, match="collides with the output"):
        con.execute("SELECT * FROM linreg_influence('edge_model', 'edge_train', 'y')").fetchall()


@pytest.mark.parametrize('df', [0.1, 1.0, 2.0, 5.0, 30.0])
@pytest.mark.parametrize('probability', [1e-6, 0.025, 0.5, 0.975, 1-1e-6])
def test_student_t_quantiles_in_heavy_tails(con, df, probability):
    from scipy.stats import t
    actual = con.execute('SELECT t_ppf(?,?)', [probability, df]).fetchone()[0]
    assert actual == pytest.approx(t.ppf(probability, df), rel=1e-10, abs=1e-12)


def test_student_t_quantile_boundaries(con):
    from scipy.stats import norm
    assert con.execute('SELECT t_ppf(NULL,2), t_ppf(.5,NULL)').fetchone() == (None,None)
    assert con.execute("SELECT isnan(t_ppf(-.1,2)),isnan(t_ppf(.5,0)),isnan(t_ppf(.5,'NaN'::DOUBLE))").fetchone() == (True,True,True)
    assert con.execute('SELECT t_ppf(0,2),t_ppf(1,2)').fetchone() == (-float('inf'),float('inf'))
    assert con.execute("SELECT t_ppf(.975,'Infinity'::DOUBLE)").fetchone()[0] == pytest.approx(norm.ppf(.975))


@pytest.mark.parametrize('family,expected_se', [('linreg',1/6),('logit',np.sqrt(.4)),('poisson',1/np.sqrt(20)),('gamma',1/6),('tweedie',1/6),('nbinom',np.sqrt(3/20))])
def test_intercept_only_inference_retains_observations(con,family,expected_se):
    from scipy.stats import norm,t
    b0 = .5 if family=='linreg' else 0.0 if family=='logit' else np.log(2)
    con.execute("CREATE TABLE only_intercept AS SELECT '(Intercept)' feature,?::DOUBLE coefficient",[b0])
    yexpr = 'i%2' if family in ('linreg','logit') else '1+2*(i%2)'
    con.execute(f'CREATE TABLE intercept_data AS SELECT i x,({yexpr})::DOUBLE y FROM range(10)t(i)')
    summary = con.execute(f"SELECT * FROM {family}_summary('only_intercept','intercept_data','y')").fetchall()
    assert len(summary)==1
    assert summary[0][2] == pytest.approx(expected_se)
    critical=t.ppf(.975,9) if family in ('linreg','gamma','tweedie') else norm.ppf(.975)
    assert summary[0][-2:] == pytest.approx((b0-critical*expected_se,b0+critical*expected_se))
    ci=con.execute(f"SELECT prediction,conf_low,conf_high FROM {family}_predict_ci('only_intercept','intercept_data','y')").fetchall()
    assert len(ci)==10
    inverse=(lambda z:z) if family=='linreg' else (lambda z:1/(1+np.exp(-z))) if family=='logit' else np.exp
    assert ci[0] == pytest.approx(tuple(inverse(z) for z in [b0,b0-critical*expected_se,b0+critical*expected_se]))
    leverage=con.execute(f"SELECT hat FROM {family}_influence('only_intercept','intercept_data','y')").fetchnumpy()['hat']
    assert len(leverage)==10
    assert leverage.sum() == pytest.approx(1.0)


def test_normal_quantile_boundaries(con):
    assert con.execute('SELECT norm_ppf(0),norm_ppf(1),norm_ppf(NULL)').fetchone()==(-float('inf'),float('inf'),None)
    assert con.execute("SELECT isnan(norm_ppf(-.1)),isnan(norm_ppf(1.1)),isnan(norm_ppf('NaN'::DOUBLE))").fetchone()==(True,True,True)
    assert con.execute("SELECT t_ppf(0,'Infinity'::DOUBLE),t_ppf(1,'Infinity'::DOUBLE)").fetchone()==(-float('inf'),float('inf'))


def test_multinomial_intercept_only_summary(con):
    con.execute("CREATE TABLE class_intercepts AS SELECT * FROM (VALUES ('a','(Intercept)',0.),('b','(Intercept)',0.)) t(class,feature,coefficient)")
    con.execute("CREATE TABLE class_data AS SELECT i x,CASE WHEN i%2=0 THEN 'a' ELSE 'b' END y FROM range(10)t(i)")
    rows=con.execute("SELECT class,feature,std_error FROM multinom_summary('class_intercepts','class_data','y')").fetchall()
    assert len(rows)==1
    assert rows[0][:2]==('b','(Intercept)')
    assert rows[0][2]==pytest.approx(np.sqrt(.4))


@pytest.mark.parametrize('family',['linreg','logit','poisson','gamma','tweedie','nbinom'])
@pytest.mark.parametrize('missing_rows',[False,True])
def test_inference_preserves_model_and_predictions_without_training_rows(con,family,missing_rows):
    con.execute("CREATE TABLE empty_training_model AS SELECT * FROM (VALUES ('(Intercept)',1.),('x',2.))t(feature,coefficient)")
    con.execute('CREATE TABLE empty_training(x DOUBLE,y DOUBLE)')
    if missing_rows:
        con.execute('INSERT INTO empty_training VALUES (NULL,1),(1,NULL)')
    con.execute('CREATE TABLE prediction_rows AS SELECT 2.0 x')
    summary=con.execute(f"SELECT * FROM {family}_summary('empty_training_model','empty_training','y')").fetchall()
    assert len(summary)==2
    assert [row[:2] for row in summary]==[('(Intercept)',1.0),('x',2.0)]
    assert all(all(value is None for value in row[2:]) for row in summary)
    prediction=con.execute(f"SELECT prediction,conf_low,conf_high FROM {family}_predict_ci('empty_training_model','empty_training','y',newdata:='prediction_rows')").fetchall()
    expected=5.0 if family=='linreg' else 1/(1+np.exp(-5.0)) if family=='logit' else np.exp(5.0)
    assert len(prediction)==1
    assert prediction[0][0]==pytest.approx(expected)
    assert prediction[0][1:]==(None,None)


@pytest.mark.parametrize('eta,outcome',[(40.,0.),(-40.,1.),(-800.,1.)])
def test_logistic_influence_deviance_uses_finite_logit(con,eta,outcome):
    con.execute("CREATE TABLE residual_model AS SELECT '(Intercept)' feature,?::DOUBLE coefficient",[eta])
    con.execute('CREATE TABLE residual_data AS SELECT ?::DOUBLE y',[outcome])
    actual=con.execute("SELECT deviance_resid FROM logit_influence('residual_model','residual_data','y')").fetchone()[0]
    expected=np.sign(outcome-.5)*np.sqrt(2*(outcome*np.logaddexp(0,-eta)+(1-outcome)*np.logaddexp(0,eta)))
    assert actual==pytest.approx(expected)


@pytest.mark.parametrize('alpha',[1e-12,1e-16,1e-20])
def test_nb_influence_deviance_has_poisson_limit(con,alpha):
    con.execute("CREATE TABLE nb_residual_model AS SELECT '(Intercept)' feature,ln(2.0) coefficient")
    con.execute('CREATE TABLE nb_residual_data AS SELECT 1.0 y')
    actual=con.execute(f"SELECT deviance_resid FROM nbinom_influence('nb_residual_model','nb_residual_data','y',alpha:={alpha})").fetchone()[0]
    expected=-np.sqrt(2*(-np.log(2)-(1+1/alpha)*np.log1p(-alpha/(1+2*alpha))))
    assert actual==pytest.approx(expected,abs=1e-10)


@pytest.mark.parametrize('confidence',['-0.95','0','1','1.1','NULL',"'NaN'::DOUBLE", "'Infinity'::DOUBLE"])
@pytest.mark.parametrize('kind',['summary','predict_ci','multinom_summary'])
def test_inference_rejects_invalid_confidence_levels(con,confidence,kind):
    con.execute("CREATE TABLE bad_conf_data AS SELECT i::DOUBLE x,1.0+i y FROM range(4)t(i)")
    if kind=='multinom_summary':
        con.execute("CREATE TABLE bad_conf_model AS SELECT * FROM (VALUES ('a','(Intercept)',0.),('b','(Intercept)',0.)) t(class,feature,coefficient)")
        call='multinom_summary'
    else:
        con.execute("CREATE TABLE bad_conf_model AS SELECT '(Intercept)' feature,1.0 coefficient")
        call='linreg_'+kind
    with pytest.raises(duckdb.Error,match='conf_level must be finite and strictly between 0 and 1'):
        con.execute(f"SELECT * FROM {call}('bad_conf_model','bad_conf_data','y',conf_level:={confidence})").fetchall()


@pytest.mark.parametrize('family,parameter',[('nbinom','alpha'),('tweedie','power')])
@pytest.mark.parametrize('kind',['summary','predict_ci','influence'])
@pytest.mark.parametrize('value',['NULL','-0.1',"'NaN'::DOUBLE","'Infinity'::DOUBLE"])
def test_inference_rejects_invalid_distribution_parameters(con,family,parameter,kind,value):
    con.execute("CREATE TABLE distribution_model AS SELECT '(Intercept)' feature,0.0 coefficient")
    con.execute('CREATE TABLE distribution_data AS SELECT i::DOUBLE x,1.0+i%3 y FROM range(10)t(i)')
    with pytest.raises(duckdb.Error,match=parameter+' must be finite'):
        con.execute(f"SELECT * FROM {family}_{kind}('distribution_model','distribution_data','y',{parameter}:={value})").fetchall()
