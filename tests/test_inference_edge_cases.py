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


@pytest.mark.parametrize('robust', ['none','hc0','hc1','hc2','hc3','cluster'])
def test_exact_linear_fit_preserves_zero_coefficient_uncertainty(con, robust):
    con.execute('CREATE TABLE exact_fit AS SELECT i::DOUBLE x,3.0::DOUBLE y FROM range(6)t(i)')
    con.execute("CREATE TABLE exact_model AS SELECT * FROM linreg_fit('exact_fit','y')")
    con.execute('ALTER TABLE exact_fit ADD COLUMN cl INTEGER')
    con.execute('UPDATE exact_fit SET cl=x::INTEGER%2')
    extra = "cluster_col:='cl'" if robust == 'cluster' else f"robust:='{robust}'"
    rows = con.execute(f"SELECT coefficient,std_error,conf_low,conf_high FROM linreg_summary('exact_model','exact_fit','y',{extra})").fetchall()
    assert rows == [(3.,0.,3.,3.),(0.,0.,0.,0.)]


@pytest.mark.parametrize('family', ['poisson','nbinom'])
@pytest.mark.parametrize('eta,y', [(-1500.,1e-300),(-1000.,0.),(1000.,0.),(710.,1e308)])
def test_count_diagnostics_preserve_representable_residual_roots(con,family,eta,y):
    from decimal import Decimal, localcontext
    con.execute("CREATE TABLE root_model AS SELECT '(Intercept)' feature,?::DOUBLE coefficient",[eta])
    con.execute('CREATE TABLE root_data AS SELECT ?::DOUBLE y FROM range(3)',[y])
    with localcontext() as context:
        context.prec=800
        yy=Decimal.from_float(y); mu=Decimal.from_float(eta).exp()
        variance=mu if family=='poisson' else mu+mu*mu
        expected_pearson=float((yy-mu)/variance.sqrt())
        if family=='poisson':
            halfdev=(yy*(yy/mu).ln() if yy else 0)-yy+mu
        else:
            halfdev=(yy*(yy/mu).ln() if yy else 0)-(yy+1)*((yy+1)/(mu+1)).ln()
        expected_deviance=float((2*halfdev).sqrt())*(1 if yy>mu else -1)
    rows=con.execute(f"SELECT hat,pearson_resid,deviance_resid,std_resid FROM {family}_influence('root_model','root_data','y')").fetchall()
    for hat,pearson,deviance,standardized in rows:
        np.testing.assert_allclose([pearson,deviance],
            [expected_pearson,expected_deviance],rtol=1e-10,atol=5e-324)
        if family=='nbinom' and eta==1000.:
            # Observed information for all-zero NB outcomes underflows here;
            # raw residuals remain defined independently of that covariance.
            assert hat is None and standardized is None
        else:
            assert hat==pytest.approx(1/3,rel=1e-12)
            assert standardized==pytest.approx(expected_pearson/np.sqrt(2/3),rel=1e-10,abs=5e-324)
    if family=='nbinom':
        dispersion=con.execute("SELECT dispersion FROM nbinom_evaluate('root_model','root_data','y')").fetchone()[0]
        assert dispersion==pytest.approx(1.5*expected_pearson**2,rel=1e-10,abs=5e-324)


@pytest.mark.parametrize('family', ['poisson','nbinom'])
@pytest.mark.parametrize('eta,y,weight', [(0.,1e160,1e-100),(-710.,1.,1e-100),(-750.,1.,1e-300)])
def test_count_cooks_distance_restores_weight_before_squaring(con,family,eta,y,weight):
    con.execute("CREATE TABLE cook_model AS SELECT '(Intercept)' feature,0.0::DOUBLE coefficient")
    con.execute('CREATE TABLE cook_data AS SELECT ?::DOUBLE y,?::DOUBLE expo,?::DOUBLE w FROM range(3)',[y,eta,weight])
    rows=con.execute(f"SELECT hat,pearson_resid,cooks_distance FROM {family}_influence('cook_model','cook_data','y',offset_col:='expo',weights_col:='w')").fetchall()
    # The identical intercept-only rows have leverage 1/3. Compute the
    # weighted Pearson square independently in log space to avoid overflow.
    log_variance=eta+(np.logaddexp(0.,eta) if family=='nbinom' else 0.)
    expected=.75*np.exp(np.log(weight)+2*np.log(y-np.exp(eta))-log_variance)
    for hat,pearson,cook in rows:
        assert hat==pytest.approx(1/3,rel=1e-12)
        assert np.isfinite(pearson) and np.isfinite(cook)
        assert cook==pytest.approx(expected,rel=1e-10)


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


@pytest.mark.parametrize('score', [-710., -720., -740.])
def test_logistic_prediction_intervals_preserve_negative_tail_probabilities(con, score):
    from scipy.stats import norm

    con.execute("CREATE TABLE tail_model AS SELECT * FROM (VALUES ('(Intercept)',0.),('x',1.))t(feature,coefficient)")
    con.execute('CREATE TABLE tail_train AS SELECT i::DOUBLE x,(i>0)::DOUBLE y,1e12 w FROM range(-1,2)t(i)')
    con.execute('CREATE TABLE tail_score AS SELECT ?::DOUBLE x',[score])
    actual = np.array(con.execute("SELECT prediction,conf_low,conf_high FROM logit_predict_ci('tail_model','tail_train','y',newdata:='tail_score',weights_col:='w')").fetchone())
    x = np.column_stack([np.ones(3),np.arange(-1.,2.)])
    p = 1/(1+np.exp(-x[:,1]))
    covariance = np.linalg.inv(x.T@((1e12*p*(1-p))[:,None]*x))
    row = np.array([1.,score])
    margin = norm.ppf(.975)*np.sqrt(row@covariance@row)
    logits = np.array([score,score-margin,score+margin])
    expected = np.exp(logits)/(1+np.exp(logits))
    assert (actual > 0).all()
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=0.)


@pytest.mark.parametrize('family', ['linreg', 'logit', 'poisson', 'gamma', 'tweedie', 'nbinom'])
@pytest.mark.parametrize('extreme_column', ['x', 'expo', 'y'])
def test_zero_weight_extremes_cannot_contaminate_inference(con, family, extreme_column):
    model(con)
    outcome = 'i%2' if family == 'logit' else '1.0+i%3'
    con.execute(f'CREATE TABLE positive AS SELECT i/10.0 x,({outcome})::DOUBLE y,0.0::DOUBLE expo,1.0 w,i%3 grp FROM range(12)t(i)')
    con.execute('CREATE TABLE benign AS SELECT * FROM positive UNION ALL SELECT 0,1,0,0,0')
    con.execute('CREATE TABLE extreme AS SELECT * FROM benign')
    # Logistic outcomes stay binary; the other families allow large positive y.
    value = 1.0 if family == 'logit' and extreme_column == 'y' else 1e308
    con.execute(f'UPDATE extreme SET {extreme_column}=? WHERE w=0', [value])
    args = "weights_col:='w',offset_col:='expo'"
    # Compare the same analytic-weight row count and cluster membership.
    for extra in ["robust:='none'", "robust:='hc0'", "robust:='hc1'", "robust:='hc2'", "robust:='hc3'", "cluster_col:='grp'"]:
        expected = con.execute(f"SELECT * FROM {family}_summary('edge_model','benign','y',{args},{extra})").df()
        actual = con.execute(f"SELECT * FROM {family}_summary('edge_model','extreme','y',{args},{extra})").df()
        assert np.isfinite(actual['std_error']).all()
        pd.testing.assert_frame_equal(actual, expected)
    intervals = []
    diagnostics = []
    for table in ['benign', 'extreme']:
        intervals.append(con.execute(f"SELECT prediction,conf_low,conf_high FROM {family}_predict_ci('edge_model','{table}','y',newdata:='positive',{args})").df())
        diagnostics.append(con.execute(f"SELECT hat,pearson_resid,deviance_resid,std_resid,cooks_distance FROM {family}_influence('edge_model','{table}','y',{args})").df())
    pd.testing.assert_frame_equal(intervals[0], intervals[1])
    pd.testing.assert_frame_equal(diagnostics[0], diagnostics[1])
    assert np.isfinite(intervals[1].to_numpy()).all()
    assert np.isfinite(diagnostics[1].to_numpy()).all()
    assert (diagnostics[1].iloc[-1] == 0.0).all()


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
    for scale in [1.0, 7.0, 1e160, 1e-170]:
        load(con, "edge_train", data.assign(w=a * scale))
        actual = con.execute(f"SELECT std_error FROM poisson_summary('edge_model', 'edge_train', 'y', weights_col := 'w', robust := '{robust}')").df()["std_error"].to_numpy()
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=1e-10)


@pytest.mark.parametrize('family', ['linreg', 'logit', 'poisson', 'gamma', 'tweedie', 'nbinom'])
@pytest.mark.parametrize('robust', ['hc0', 'hc1', 'hc2', 'hc3', 'cluster'])
def test_robust_summary_preserves_extreme_common_weight_scale(con, family, robust):
    model(con)
    data = training().assign(w=np.linspace(0.25, 1.0, 48), grp=np.resize(np.arange(4), 48))
    if family == 'logit':
        data['y'] = (data['y'] > 1).astype(float)
    elif family in ['gamma', 'tweedie']:
        data['y'] += 0.2
    extra = "cluster_col:='grp'" if robust == 'cluster' else f"robust:='{robust}'"
    values = []
    for scale in [1.0, 1e308, 1e-308]:
        load(con, 'edge_train', data.assign(w=data.w * scale))
        result = con.execute(f"""
            SELECT std_error FROM {family}_summary(
                'edge_model', 'edge_train', 'y', weights_col:='w', {extra})
        """).fetchnumpy()['std_error']
        assert not np.ma.getmaskarray(result).any()
        assert np.isfinite(result).all()
        values.append(result)
    np.testing.assert_allclose(values[1:], np.tile(values[0], (2, 1)), rtol=1e-10)


@pytest.mark.parametrize('family', ['linreg', 'gamma', 'tweedie'])
def test_estimated_dispersion_inference_preserves_extreme_weight_scale(con, family):
    model(con)
    data = training().assign(w=np.linspace(0.25, 1.0, 48))
    data['y'] += 0.2
    x = np.column_stack([np.ones(len(data)), data.x])
    eta = x @ np.array([0.3, 0.2])
    mu = eta if family == 'linreg' else np.exp(eta)
    power = 2.0 if family == 'gamma' else 1.5
    variance = np.ones(len(data)) if family == 'linreg' else mu**power
    information = data.w.to_numpy() * (1.0 if family == 'linreg' else mu**(2-power))
    phi = np.sum(data.w.to_numpy()*(data.y.to_numpy()-mu)**2/variance)/(len(data)-2)
    expected = np.sqrt(phi*np.diag(np.linalg.inv(x.T @ (information[:,None]*x))))
    intervals = []
    for scale in [1.0, 1e-308, 1e308]:
        load(con, 'edge_train', data.assign(w=data.w*scale))
        actual = con.execute(f"SELECT std_error FROM {family}_summary('edge_model','edge_train','y',weights_col:='w')").df()['std_error'].to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-10)
        intervals.append(con.execute(f"SELECT conf_low,conf_high FROM {family}_predict_ci('edge_model','edge_train','y',weights_col:='w')").df().to_numpy())
    assert np.isfinite(intervals).all()
    np.testing.assert_allclose(intervals[1:], np.stack([intervals[0],intervals[0]]), rtol=1e-10)


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


@pytest.mark.parametrize('probability', [1e-20,1e-100,1e-300,1e-320,5e-324,np.nextafter(1.,0.)])
def test_normal_quantiles_preserve_extreme_valid_probabilities(con, probability):
    from scipy.stats import norm
    actual=con.execute('SELECT norm_ppf(?)',[float(probability)]).fetchone()[0]
    assert actual==pytest.approx(norm.ppf(probability),abs=2e-12)


@pytest.mark.parametrize('value', [-8.,-2.,0.,2.,8.])
def test_student_t_cdf_has_normal_limit(con,value):
    from scipy.stats import norm
    actual=con.execute("SELECT t_cdf(?,'Infinity'::DOUBLE)",[value]).fetchone()[0]
    assert actual==pytest.approx(norm.cdf(value),rel=1e-12,abs=0)
    p=.975
    assert con.execute("SELECT t_cdf(t_ppf(?,'Infinity'::DOUBLE),'Infinity'::DOUBLE)",[p]).fetchone()[0]==pytest.approx(p)


@pytest.mark.parametrize('offset', [40.0,1000.0])
def test_logistic_influence_preserves_confident_observation_diagnostics(con,offset):
    con.execute(f"CREATE TABLE confident_data AS SELECT * FROM "
                f"(VALUES (-1.,0.,0.),(-1.,1.,0.),(1.,0.,0.),(1.,1.,0.),(0.,0.,{offset}))t(x,y,expo)")
    con.execute("CREATE TABLE confident_model AS SELECT * FROM logit_fit('confident_data','y',offset_col:='expo')")
    actual=con.execute("SELECT hat,pearson_resid,std_resid,cooks_distance FROM logit_influence('confident_model','confident_data','y',offset_col:='expo') WHERE expo>0").fetchone()
    # The first four rows have mu=1/4 and the surprising fifth row has mu~1.
    eta=offset-np.log(3)
    variance=np.exp(-eta)/(1+np.exp(-eta))**2
    hat=variance/(.75+variance)
    pearson=-np.exp(eta/2)
    np.testing.assert_allclose(actual,[hat,pearson,pearson/np.sqrt(1-hat),2/3],rtol=1e-8,atol=0)


@pytest.mark.parametrize('eta', [40.0,1000.0,-40.0,-1000.0])
def test_logistic_influence_retains_tiny_correct_prediction_residuals(con,eta):
    con.execute("CREATE TABLE tail_model AS SELECT '(Intercept)' feature,0.0 coefficient UNION ALL SELECT 'x',1.0")
    con.execute(f"CREATE TABLE tail_data AS SELECT * FROM (VALUES (0.,0.),(1.,1.),(-1.,0.),({eta},{int(eta>0)}))t(x,y)")
    actual=con.execute(f"SELECT pearson_resid,deviance_resid FROM logit_influence('tail_model','tail_data','y') WHERE x={eta}").fetchone()
    sign=1 if eta>0 else -1
    expected=sign*np.exp(-abs(eta)/2)
    np.testing.assert_allclose(actual,[expected,np.sqrt(2)*expected],rtol=1e-12,atol=0)


@pytest.mark.parametrize('scale', [1e-170,1e170])
@pytest.mark.parametrize('family', ['gamma','tweedie'])
def test_gamma_inference_is_invariant_to_outcome_scale(con,scale,family):
    extra=',power:=2.0' if family=='tweedie' else ''
    con.execute('CREATE TABLE base_gamma AS SELECT (i%5)::DOUBLE x,(1+i%7)::DOUBLE y FROM range(30)t(i)')
    con.execute(f'CREATE TABLE scaled_gamma AS SELECT x,y*{scale} y FROM base_gamma')
    results=[]
    for table in ['base_gamma','scaled_gamma']:
        con.execute(f"CREATE OR REPLACE TABLE gamma_model AS SELECT * FROM {family}_fit('{table}','y'{extra})")
        summary=con.execute(f"SELECT std_error,conf_low-coefficient,conf_high-coefficient FROM {family}_summary('gamma_model','{table}','y'{extra})").fetchnumpy()
        ci=con.execute(f"SELECT prediction,conf_low,conf_high FROM {family}_predict_ci('gamma_model','{table}','y'{extra})").fetchnumpy()
        influence=con.execute(f"SELECT hat,pearson_resid,deviance_resid,std_resid,cooks_distance FROM {family}_influence('gamma_model','{table}','y'{extra})").fetchnumpy()
        evaluation=con.execute(f"SELECT dispersion FROM {family}_evaluate('gamma_model','{table}','y'{extra})").fetchnumpy()
        results.append((summary,ci,influence,evaluation))
    for kind in range(4):
        for key,expected in results[0][kind].items():
            actual=results[1][kind][key]
            assert not np.ma.is_masked(actual)
            if kind==1:
                actual=actual/scale
            np.testing.assert_allclose(actual,expected,rtol=1e-8,atol=1e-10)


@pytest.mark.parametrize('family', ['poisson','linreg','gamma','tweedie'])
def test_intervals_preserve_confidence_levels_immediately_below_one(con,family):
    from scipy.stats import norm,t
    level=float(np.nextafter(1.,0.))
    normal=family=='poisson'
    critical=-norm.ppf((1-level)/2) if normal else -t.ppf((1-level)/2,11)
    usual=norm.ppf(.975) if normal else t.ppf(.975,11)
    coefficient=2. if family=='linreg' else np.log(2.)
    con.execute('CREATE TABLE wide_data AS SELECT (1+i%3)::DOUBLE y FROM range(12)t(i)')
    con.execute("CREATE TABLE wide_model AS SELECT '(Intercept)' feature,?::DOUBLE coefficient",[float(coefficient)])
    beta,se,lo,hi=con.execute(f"SELECT coefficient,std_error,conf_low,conf_high FROM {family}_summary('wide_model','wide_data','y',conf_level:=?)",[level]).fetchone()
    np.testing.assert_allclose([lo,hi],[beta-critical*se,beta+critical*se],rtol=1e-10)
    intervals=[]
    for confidence in [.95,level]:
        row=con.execute(f"SELECT prediction,conf_low,conf_high FROM {family}_predict_ci('wide_model','wide_data','y',conf_level:=?) LIMIT 1",[confidence]).fetchone()
        intervals.append(np.array(row) if family=='linreg' else np.log(row))
    prediction,low,high=intervals[0]
    width=(high-low)/2/usual*critical
    np.testing.assert_allclose(intervals[1],[prediction,prediction-width,prediction+width],rtol=1e-10)


def test_multinomial_summary_preserves_confidence_level_immediately_below_one(con):
    from scipy.stats import norm
    level=float(np.nextafter(1.,0.))
    con.execute("CREATE TABLE wide_multi_data AS SELECT CASE WHEN i%2=0 THEN 'a' ELSE 'b' END y FROM range(12)t(i)")
    con.execute("CREATE TABLE wide_multi_model AS SELECT * FROM (VALUES ('a','(Intercept)',0.),('b','(Intercept)',0.))t(class,feature,coefficient)")
    beta,se,lo,hi=con.execute("SELECT coefficient,std_error,conf_low,conf_high FROM multinom_summary('wide_multi_model','wide_multi_data','y',conf_level:=?)",[level]).fetchone()
    critical=-norm.ppf((1-level)/2)
    np.testing.assert_allclose([lo,hi],[beta-critical*se,beta+critical*se],rtol=1e-10)


@pytest.mark.parametrize('df', [.1,.5,1.,2.])
@pytest.mark.parametrize('magnitude', [1e160,1e200])
def test_student_t_retains_extreme_representable_tails(con,df,magnitude):
    import math
    log_constant=math.lgamma((df+1)/2)-math.lgamma(df/2)-.5*math.log(math.pi)+(df/2-1)*math.log(df)
    # The next power-tail term is O(df/t^2), far below double precision here.
    expected=math.exp(log_constant-df*math.log(magnitude))
    actual=con.execute('SELECT t_cdf(?,?)',[-magnitude,df]).fetchone()[0]
    assert actual==pytest.approx(expected,rel=1e-12,abs=5e-324)


@pytest.mark.parametrize('df,probability', [(.1,1e-20),(.5,1e-100),(.01,.01)])
def test_student_t_quantiles_reach_large_finite_values(con,df,probability):
    import math
    log_constant=math.lgamma((df+1)/2)-math.lgamma(df/2)-.5*math.log(math.pi)+(df/2-1)*math.log(df)
    expected=-math.exp((log_constant-math.log(probability))/df)
    actual=con.execute('SELECT t_ppf(?,?)',[probability,df]).fetchone()[0]
    assert actual==pytest.approx(expected,rel=1e-11)
    assert con.execute('SELECT t_cdf(?,?)',[actual,df]).fetchone()[0]==pytest.approx(probability,rel=1e-12)


@pytest.mark.parametrize('df', [.1,.5])
@pytest.mark.parametrize('magnitude', [1e308,1.6e308])
def test_student_t_quantiles_keep_finite_brackets_near_double_limit(con,df,magnitude):
    import math
    log_constant=math.lgamma((df+1)/2)-math.lgamma(df/2)-.5*math.log(math.pi)+(df/2-1)*math.log(df)
    probability=math.exp(log_constant-df*math.log(magnitude))
    actual=con.execute('SELECT t_ppf(?,?)',[probability,df]).fetchone()[0]
    assert np.isfinite(actual)
    assert actual/magnitude==pytest.approx(-1.,rel=1e-11)


def test_student_t_quantiles_beyond_double_range_remain_infinite(con):
    assert con.execute('SELECT t_ppf(1e-200,.5)').fetchone()[0] == -float('inf')


@pytest.mark.parametrize('df', [.1,1.,30.,1000.,1e8,1e12,1e15])
@pytest.mark.parametrize('difference', [-1e-6,-1e-9,1e-9,1e-6])
def test_student_t_quantiles_preserve_probabilities_near_median(con,df,difference):
    from scipy.stats import t
    probability=.5+difference
    actual=con.execute('SELECT t_ppf(?,?)',[probability,df]).fetchone()[0]
    assert actual==pytest.approx(t.ppf(probability,df),rel=1e-7,abs=5e-16)
    assert con.execute('SELECT t_cdf(?,?)',[actual,df]).fetchone()[0]==pytest.approx(probability,abs=2e-16)


@pytest.mark.parametrize('df', [30.,1000.,1e8,1e12,1e15])
@pytest.mark.parametrize('value', [-8.,-2.,-1e-8,1e-8,1.,2.,8.])
def test_student_t_cdf_matches_central_and_large_df_reference(con,df,value):
    from scipy.stats import t
    actual=con.execute('SELECT t_cdf(?,?)',[value,df]).fetchone()[0]
    assert actual==pytest.approx(t.cdf(value,df),rel=1e-12,abs=1e-16)


@pytest.mark.parametrize('df', [1e8,1e12,1e15])
@pytest.mark.parametrize('probability', [1e-100,1e-300])
def test_large_df_student_t_quantiles_preserve_extreme_tails(con,df,probability):
    from scipy.stats import t
    actual=con.execute('SELECT t_ppf(?,?)',[probability,df]).fetchone()[0]
    assert actual==pytest.approx(t.ppf(probability,df),rel=1e-12)
    assert con.execute('SELECT t_cdf(?,?)',[actual,df]).fetchone()[0]==pytest.approx(probability,rel=1e-11,abs=5e-324)


@pytest.mark.parametrize('extreme_x', [-1e9,1e9])
def test_multinomial_information_matches_binary_logistic_at_saturation(con,extreme_x):
    con.execute(f"CREATE TABLE binary_data AS SELECT * FROM (VALUES (-1.,'0'),(1.,'1'),(0.,'1'),({extreme_x},'1'))t(x,y)")
    con.execute("CREATE TABLE binary_model AS SELECT * FROM (VALUES ('(Intercept)',0.),('x',4e-8))t(feature,coefficient)")
    con.execute("CREATE TABLE multi_model AS SELECT label AS class,feature,CASE WHEN label='0' THEN 0. ELSE coefficient END AS coefficient FROM binary_model,(VALUES ('0'),('1'))t(label)")
    binary=con.execute("SELECT std_error FROM logit_summary('binary_model','binary_data','y') ORDER BY (feature='(Intercept)') DESC").fetchnumpy()['std_error']
    multi=con.execute("SELECT std_error FROM multinom_summary('multi_model','binary_data','y') ORDER BY (feature='(Intercept)') DESC").fetchnumpy()['std_error']
    assert not np.ma.is_masked(multi)
    np.testing.assert_allclose(multi,binary,rtol=1e-10)


@pytest.mark.parametrize('family', ['linreg','logit','poisson','gamma','tweedie','nbinom'])
@pytest.mark.parametrize('offset_kind', ['null','absent','invalid'])
def test_prediction_intervals_require_usable_scoring_offsets(con, family, offset_kind):
    con.execute("CREATE TABLE offset_model AS SELECT * FROM (VALUES ('(Intercept)',1.),('x',.25))t(feature,coefficient)")
    outcome = 'i%2' if family == 'logit' else '1+i%3'
    con.execute(f'CREATE TABLE offset_train AS SELECT i/10.0 x,{outcome} y,0.0 expo FROM range(10)t(i)')
    offset = {'null': ',CASE WHEN i<2 THEN NULL ELSE 0.0 END expo',
              'absent': '',
              'invalid': ",CASE WHEN i<2 THEN 'invalid' ELSE '0' END expo"}[offset_kind]
    con.execute(f'CREATE TABLE offset_score AS SELECT i/10.0 x{offset} FROM range(3)t(i)')
    rows = con.execute(f"SELECT prediction,conf_low,conf_high FROM {family}_predict_ci('offset_model','offset_train','y',newdata:='offset_score',offset_col:='expo') ORDER BY x").fetchall()
    assert rows[:2] == [(None,None,None)]*2
    if offset_kind == 'absent':
        assert rows[2] == (None,None,None)
    else:
        assert np.isfinite(rows[2]).all()
        assert rows[2][1] <= rows[2][0] <= rows[2][2]


@pytest.mark.parametrize('logit', [800.,1200.])
@pytest.mark.parametrize('feature_scale', [1e100,1e200,1e300])
@pytest.mark.parametrize('classes', [2,3])
def test_multinomial_information_combines_underflowed_probabilities_with_features(con, logit, feature_scale, classes):
    con.execute('CREATE TABLE extreme_classes(x DOUBLE,y VARCHAR)')
    con.executemany('INSERT INTO extreme_classes VALUES (?,?)',
                    [(-feature_scale,'0'),(feature_scale,'1')]+[(0.,str(i)) for i in range(classes)])
    con.execute('CREATE TABLE extreme_model(class VARCHAR,feature VARCHAR,coefficient DOUBLE)')
    coefficients = [('0','(Intercept)',0.),('0','x',0.),('1','(Intercept)',0.),('1','x',logit/feature_scale)]
    if classes == 3:
        coefficients += [('2','(Intercept)',0.),('2','x',-logit/feature_scale)]
    con.executemany('INSERT INTO extreme_model VALUES (?,?,?)',coefficients)
    rows = con.execute("SELECT class,feature,std_error FROM multinom_summary('extreme_model','extreme_classes','y')").fetchall()
    # Central rows identify intercepts; the extreme rows retain slope
    # information proportional to exp(-logit)*feature_scale**2.
    expected_slope = np.exp(logit/2-np.log(feature_scale))/np.sqrt(2 if classes == 2 else 1)
    for _,feature,error in rows:
        expected = np.sqrt(2) if feature == '(Intercept)' else expected_slope
        assert error is not None and np.isfinite(error)
        assert error/expected == pytest.approx(1.,rel=1e-10)


def test_multinomial_information_preserves_large_logit_differences(con):
    from scipy.special import softmax
    xs=np.array([-1.,0.,1.,1000.])
    con.execute("CREATE TABLE multi_data AS SELECT * FROM (VALUES (-1.,'a'),(0.,'b'),(1.,'c'),(1000.,'b'))t(x,y)")
    con.execute("CREATE TABLE multi_model AS SELECT * FROM (VALUES ('a','(Intercept)',0.),('a','x',0.),('b','(Intercept)',0.),('b','x',1.),('c','(Intercept)',0.),('c','x',.9))t(class,feature,coefficient)")
    information=np.zeros((4,4))
    for x in xs:
        p=softmax([0.,x,.9*x])
        covariance=-np.outer(p[1:],p[1:])
        for j in range(2):
            covariance[j,j]=p[j+1]*sum(p[k] for k in range(3) if k!=j+1)
        information+=np.kron(covariance,np.outer([1.,x],[1.,x]))
    actual=con.execute("SELECT std_error FROM multinom_summary('multi_model','multi_data','y') ORDER BY class,(feature='(Intercept)') DESC").fetchnumpy()['std_error']
    assert not np.ma.is_masked(actual)
    np.testing.assert_allclose(actual,np.sqrt(np.diag(np.linalg.inv(information))),rtol=1e-10)


@pytest.mark.parametrize('family', ['linreg','logit','poisson','gamma','tweedie','nbinom'])
@pytest.mark.parametrize('scale', [1e-160, 1e160])
@pytest.mark.parametrize('robust', ['none', 'hc0', 'cluster'])
def test_inference_preserves_extreme_feature_units(con, family, scale, robust):
    model(con)
    model(con, name='scaled_model')
    con.execute("UPDATE scaled_model SET coefficient=coefficient/? WHERE feature='x'", [scale])
    data = training().assign(w=np.linspace(0.25, 1.0, 48), grp=np.resize(np.arange(4), 48))
    data['y'] = (data.y > 1).astype(float) if family == 'logit' else data.y + 0.2
    load(con, 'base_units', data)
    load(con, 'scaled_units', data.assign(x=data.x*scale))
    extra = "cluster_col:='grp'" if robust == 'cluster' else f"robust:='{robust}'"
    summaries, intervals, diagnostics = [], [], []
    for mdl, table in [('edge_model','base_units'), ('scaled_model','scaled_units')]:
        summary = con.execute(f"SELECT * FROM {family}_summary('{mdl}','{table}','y',weights_col:='w',{extra})").df()
        if table == 'scaled_units':
            summary.loc[summary.feature=='x', ['coefficient','std_error','conf_low','conf_high']] *= scale
        summaries.append(summary.drop(columns='feature').to_numpy())
        if robust == 'none':
            intervals.append(con.execute(f"SELECT prediction,conf_low,conf_high FROM {family}_predict_ci('{mdl}','{table}','y',weights_col:='w')").df().to_numpy())
            diagnostics.append(con.execute(f"SELECT hat,pearson_resid,deviance_resid,std_resid,cooks_distance FROM {family}_influence('{mdl}','{table}','y',weights_col:='w')").df().to_numpy())
    assert np.isfinite(summaries).all()
    np.testing.assert_allclose(summaries[1], summaries[0], rtol=1e-9, atol=1e-12)
    if robust == 'none':
        assert np.isfinite(intervals).all()
        assert np.isfinite(diagnostics).all()
        np.testing.assert_allclose(intervals[1], intervals[0], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(diagnostics[1], diagnostics[0], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize('scale', [1e-160, 1e160])
def test_multinomial_inference_preserves_extreme_feature_units(con, scale):
    model(con, multinomial=True)
    model(con, name='scaled_model', multinomial=True)
    con.execute("UPDATE scaled_model SET coefficient=coefficient/? WHERE feature='x'", [scale])
    data = training().assign(y=np.resize(['a','b','c'], 48))
    load(con, 'base_units', data)
    load(con, 'scaled_units', data.assign(x=data.x*scale))
    summaries = []
    for mdl, table in [('edge_model','base_units'), ('scaled_model','scaled_units')]:
        summary = con.execute(f"SELECT * FROM multinom_summary('{mdl}','{table}','y') ORDER BY class,feature").df()
        if table == 'scaled_units':
            summary.loc[summary.feature=='x', ['coefficient','std_error','conf_low','conf_high']] *= scale
        summaries.append(summary.drop(columns=['class','feature']).to_numpy())
    assert np.isfinite(summaries).all()
    np.testing.assert_allclose(summaries[1], summaries[0], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('family', ['linreg','logit','poisson','gamma','tweedie','nbinom'])
@pytest.mark.parametrize('operation', ['summary','predict_ci','influence'])
@pytest.mark.parametrize('weight', ['-1.0', "'NaN'::DOUBLE", "'Infinity'::DOUBLE", "'-Infinity'::DOUBLE"])
def test_inference_rejects_invalid_retained_weights(con, family, operation, weight):
    model(con)
    data = training()
    data['y'] = (data.y > 1).astype(float) if family == 'logit' else data.y + 0.2
    load(con, 'invalid_weights', data)
    con.execute(f'ALTER TABLE invalid_weights ADD COLUMN w DOUBLE DEFAULT {weight}')
    with pytest.raises(duckdb.Error, match='weights must be (finite|non-negative)'):
        con.execute(f"SELECT * FROM {family}_{operation}('edge_model','invalid_weights','y',weights_col:='w')").fetchall()


@pytest.mark.parametrize('operation', ['summary','predict_ci','influence'])
def test_inference_weight_validation_ignores_incomplete_rows(con, operation):
    model(con)
    load(con, 'retained_weights', training().assign(w=1.0))
    call = f"SELECT * FROM poisson_{operation}('edge_model','retained_weights','y',weights_col:='w')"
    expected = con.execute(call).df()
    con.execute("INSERT INTO retained_weights VALUES (NULL,1.0,'NaN'::DOUBLE)")
    actual = con.execute(call).df()
    if operation == 'predict_ci':
        assert actual.iloc[-1][['prediction','conf_low','conf_high']].isna().all()
        actual = actual.iloc[:-1]
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize('family', ['logit','poisson','nbinom'])
def test_fixed_dispersion_inference_preserves_extreme_weight_scale(con, family):
    model(con)
    data = training().assign(w=np.linspace(0.25, 1.0, 48))
    if family == 'logit':
        data['y'] = (data.y > 1).astype(float)
    errors, diagnostics = [], []
    for scale in [1.0, 1e-308, 1e308]:
        load(con, 'fixed_weights', data.assign(w=data.w*scale))
        se = con.execute(f"SELECT std_error FROM {family}_summary('edge_model','fixed_weights','y',weights_col:='w')").df()['std_error'].to_numpy()
        assert np.isfinite(se).all()
        errors.append(se*np.sqrt(scale))
        ci = con.execute(f"SELECT prediction,conf_low,conf_high FROM {family}_predict_ci('edge_model','fixed_weights','y',weights_col:='w')").df()
        assert not ci.isna().any().any()
        if scale < 1:
            assert (ci.conf_low == 0.0).all()
            assert (ci.conf_high == (1.0 if family=='logit' else np.inf)).all()
        elif scale > 1:
            np.testing.assert_array_equal(ci.conf_low, ci.prediction)
            np.testing.assert_array_equal(ci.conf_high, ci.prediction)
        diag = con.execute(f"SELECT hat,pearson_resid,deviance_resid,std_resid,cooks_distance FROM {family}_influence('edge_model','fixed_weights','y',weights_col:='w')").df()
        assert np.isfinite(diag.to_numpy()).all()
        diag[['pearson_resid','deviance_resid','std_resid']] /= np.sqrt(scale)
        diag['cooks_distance'] /= scale
        diagnostics.append(diag.to_numpy())
    np.testing.assert_allclose(errors[1:], np.stack([errors[0],errors[0]]), rtol=1e-10)
    np.testing.assert_allclose(diagnostics[1:], np.stack([diagnostics[0],diagnostics[0]]), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('family,power', [('gamma',2.0), ('tweedie',1.5)])
@pytest.mark.parametrize('scale', [1e-305, 1e305])
def test_log_link_inference_preserves_finite_means_beyond_old_clipping_range(con, family, power, scale):
    model(con)
    model(con, name='scaled_model')
    con.execute("UPDATE scaled_model SET coefficient=coefficient+ln(?) WHERE feature='(Intercept)'", [scale])
    data = training().assign(y=training().y+0.2)
    load(con, 'base_response', data)
    load(con, 'scaled_response', data.assign(y=data.y*scale))
    errors, intervals, diagnostics = [], [], []
    for mdl, table, factor in [('edge_model','base_response',1.0), ('scaled_model','scaled_response',scale)]:
        errors.append(con.execute(f"SELECT std_error FROM {family}_summary('{mdl}','{table}','y')").df()['std_error'].to_numpy())
        intervals.append(con.execute(f"SELECT prediction,conf_low,conf_high FROM {family}_predict_ci('{mdl}','{table}','y')").df().to_numpy()/factor)
        diag = con.execute(f"SELECT hat,pearson_resid,deviance_resid,std_resid,cooks_distance FROM {family}_influence('{mdl}','{table}','y')").df()
        diag[['pearson_resid','deviance_resid']] /= factor**(1-power/2)
        diagnostics.append(diag.to_numpy())
    for observed in [errors, intervals, diagnostics]:
        assert np.isfinite(observed).all()
        np.testing.assert_allclose(observed[1], observed[0], rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize('power,scale', [(4.,1e-200),(4.,1e200),(6.,1e-100),(6.,1e100),(3.,1e-305),(3.,1e305)])
def test_tweedie_inference_cancels_extreme_information_and_dispersion_units(con, power, scale):
    con.execute('CREATE TABLE base_response AS SELECT i::DOUBLE/5 x,exp(.1*i/5)*(1.+.1*(i%3)) y,i%4 grp FROM range(20)t(i)')
    con.execute('CREATE TABLE scaled_response AS SELECT x,y*? y,grp FROM base_response',[scale])
    results = []
    for table,factor in [('base_response',1.),('scaled_response',scale)]:
        con.execute(f"CREATE OR REPLACE TABLE fit_input AS SELECT x,y FROM {table}")
        con.execute(f"CREATE OR REPLACE TABLE inference_model AS SELECT * FROM tweedie_fit('fit_input','y',power:={power})")
        errors = []
        for robust in ['none','hc0','hc1','hc2','hc3','cluster']:
            extra = ",cluster_col:='grp'" if robust == 'cluster' else f",robust:='{robust}'"
            errors.append(con.execute(f"SELECT std_error FROM tweedie_summary('inference_model','{table}','y',power:={power}{extra})").df().to_numpy())
        intervals = con.execute(f"SELECT prediction,conf_low,conf_high FROM tweedie_predict_ci('inference_model','{table}','y',power:={power})").df().to_numpy()/factor
        diagnostics = con.execute(f"SELECT hat,pearson_resid,deviance_resid,std_resid,cooks_distance FROM tweedie_influence('inference_model','{table}','y',power:={power})").df()
        diagnostics[['pearson_resid','deviance_resid']] /= np.exp(np.log(factor)*(1-power/2))
        results.append([np.asarray(errors),intervals,diagnostics.to_numpy()])
    for baseline,scaled in zip(*results):
        assert np.isfinite(scaled).all()
        np.testing.assert_allclose(scaled,baseline,rtol=1e-8,atol=1e-10)


@pytest.mark.parametrize('exponent', [100,200,300])
@pytest.mark.parametrize('weight_scale', [1.,1e100])
@pytest.mark.parametrize('feature_scale', [1e-100,1.,1e100])
def test_poisson_inference_preserves_joint_information_and_feature_scales(con, exponent, weight_scale, feature_scale):
    from scipy.stats import norm

    small,big = 10.**(-exponent),10.**exponent
    x = feature_scale/np.sqrt(small)
    con.execute("CREATE TABLE joint_model AS SELECT * FROM (VALUES ('(Intercept)',0.),('x',0.))t(feature,coefficient)")
    con.execute('CREATE TABLE joint_data(x DOUBLE,y DOUBLE,w DOUBLE,expo DOUBLE)')
    con.executemany('INSERT INTO joint_data VALUES (?,?,?,?)',
                    [(-x,small,weight_scale,np.log(small)),(x,small,weight_scale,np.log(small)),(0.,big,weight_scale/big,np.log(big))])
    args = "'joint_model','joint_data','y',weights_col:='w',offset_col:='expo'"
    errors = dict(con.execute(f'SELECT feature,std_error FROM poisson_summary({args})').fetchall())
    # Joint Fisher information is diag(weight_scale,2*weight_scale*feature_scale^2).
    assert errors['(Intercept)']*np.sqrt(weight_scale) == pytest.approx(1.,rel=1e-10)
    assert errors['x']*feature_scale*np.sqrt(weight_scale) == pytest.approx(1/np.sqrt(2),rel=1e-10)
    hats = con.execute(f'SELECT hat FROM poisson_influence({args}) ORDER BY x').fetchnumpy()['hat']
    np.testing.assert_allclose(hats,[.5,1.,.5],rtol=1e-10,atol=1e-12)
    con.execute('CREATE TABLE joint_score AS SELECT 0.0 x,0.0 expo')
    interval = con.execute(f"SELECT prediction,conf_low,conf_high FROM poisson_predict_ci({args},newdata:='joint_score')").fetchone()
    margin = norm.ppf(.975)/np.sqrt(weight_scale)
    np.testing.assert_allclose(interval,[1.,np.exp(-margin),np.exp(margin)],rtol=1e-10,atol=0.)


@pytest.mark.parametrize('exponent', [100,200,300])
@pytest.mark.parametrize('robust', ['hc0','hc1'])
def test_poisson_sandwich_preserves_joint_information_scales(con, exponent, robust):
    small,big = 10.**(-exponent),10.**exponent
    x = 1/np.sqrt(small)
    con.execute("CREATE TABLE joint_model AS SELECT * FROM (VALUES ('(Intercept)',0.),('x',0.))t(feature,coefficient)")
    con.execute('CREATE TABLE joint_data(x DOUBLE,y DOUBLE,w DOUBLE,expo DOUBLE)')
    con.executemany('INSERT INTO joint_data VALUES (?,?,?,?)',
                    [(-x,1.1*small,1.,np.log(small)),(x,.9*small,1.,np.log(small)),(0.,1.2*big,1/big,np.log(big))])
    errors = dict(con.execute(f"SELECT feature,std_error FROM poisson_summary('joint_model','joint_data','y',weights_col:='w',offset_col:='expo',robust:='{robust}')").fetchall())
    correction = np.sqrt(3) if robust == 'hc1' else 1.
    assert errors['(Intercept)']/correction == pytest.approx(.2,rel=1e-9)
    assert errors['x']/np.sqrt(small)/correction == pytest.approx(.1/np.sqrt(2),rel=1e-9)


@pytest.mark.parametrize('offset', [1000.,2000.])
def test_logistic_score_coordinates_survive_vanishing_information(con, offset):
    con.execute("CREATE TABLE confident_model AS SELECT * FROM (VALUES ('(Intercept)',-ln(3.)),('x',0.))t(feature,coefficient)")
    con.execute(f"CREATE TABLE confident_data AS SELECT * FROM (VALUES (-1.,0.,0.),(-1.,1.,0.),(1.,0.,0.),(1.,1.,0.),(0.,0.,{offset}))t(x,y,expo)")
    # Ordinary rows contribute Fisher diag(.75,.75); score outer products
    # including the confidently wrong final row give diag(2.25,1.25).
    errors = dict(con.execute("SELECT feature,std_error FROM logit_summary('confident_model','confident_data','y',offset_col:='expo',robust:='hc0')").fetchall())
    assert errors == pytest.approx({'(Intercept)':2.,'x':np.sqrt(20)/3},rel=1e-10)
    cook = con.execute("SELECT cooks_distance FROM logit_influence('confident_model','confident_data','y',offset_col:='expo') WHERE expo>0").fetchone()[0]
    assert cook == pytest.approx(2/3,rel=1e-10)


@pytest.mark.parametrize('robust', ['hc0','hc1','hc2','hc3','cluster'])
@pytest.mark.parametrize('repeats', [1,3])
@pytest.mark.parametrize('weight', [1e-100,1.,1e100])
def test_poisson_robust_covariance_preserves_finite_large_count_errors(con, robust, repeats, weight):
    con.execute("CREATE TABLE count_model AS SELECT * FROM (VALUES ('(Intercept)',ln(8e307)),('x',0.))t(feature,coefficient)")
    con.execute(f'CREATE TABLE large_counts AS SELECT i%4//2 AS x,CASE WHEN i%2=0 THEN 0. ELSE 1.6e308 END y,{weight} w,i grp FROM range({4*repeats})t(i)')
    extra = "cluster_col:='grp'" if robust == 'cluster' else f"robust:='{robust}'"
    errors = dict(con.execute(f"SELECT feature,std_error FROM poisson_summary('count_model','large_counts','y',weights_col:='w',{extra})").fetchall())
    n = 4*repeats
    correction = {'hc0':1.,'hc1':np.sqrt(n/(n-2)),
                  'hc2':1/np.sqrt(1-2/n),'hc3':1/(1-2/n),
                  'cluster':np.sqrt(n/(n-2))}[robust]
    assert errors == pytest.approx({'(Intercept)':correction/np.sqrt(2*repeats),
                                    'x':correction/np.sqrt(repeats)},rel=1e-10)


@pytest.mark.parametrize('direction', [-1.,1.])
@pytest.mark.parametrize('magnitude', [1e308,1.6e308])
@pytest.mark.parametrize('weight', [1e-100,1.,1e100])
def test_logistic_deviance_residual_takes_root_before_doubling_loss(con, direction, magnitude, weight):
    outcome = 0. if direction > 0 else 1.
    con.execute(f"CREATE TABLE loss_model AS SELECT * FROM (VALUES ('(Intercept)',{-direction}*ln(3.)),('x',0.))t(feature,coefficient)")
    con.execute(f'CREATE TABLE loss_rows AS SELECT *,{weight} w FROM (VALUES (-1.,0.,0.),(-1.,1.,0.),(1.,0.,0.),(1.,1.,0.),(0.,{outcome},{direction*magnitude}))t(x,y,expo)')
    residual,cook = con.execute("SELECT deviance_resid,cooks_distance FROM logit_influence('loss_model','loss_rows','y',offset_col:='expo',weights_col:='w') WHERE expo!=0").fetchone()
    expected = -direction*np.sqrt(2)*np.sqrt(magnitude)*np.sqrt(weight)
    assert np.isfinite(residual)
    assert residual/expected == pytest.approx(1.,rel=1e-12)
    assert cook/weight == pytest.approx(2/3,rel=1e-10)


@pytest.mark.parametrize('family', ['poisson','gamma','nbinom'])
def test_log_link_deviance_residual_takes_root_before_doubling_loss(con, family):
    mean,outcome = (1.,1.6e308) if family == 'gamma' else (1.6e308,0.)
    con.execute("CREATE TABLE halfdev_model AS SELECT '(Intercept)' feature,ln(?) coefficient",[mean])
    con.execute('CREATE TABLE halfdev_rows AS SELECT ?::DOUBLE y FROM range(3)',[outcome])
    extra = ',alpha:=1e-308' if family == 'nbinom' else ''
    rows = con.execute(f"SELECT deviance_resid FROM {family}_influence('halfdev_model','halfdev_rows','y'{extra})").fetchall()
    halfdev = np.log1p(1e-308*mean)/1e-308 if family == 'nbinom' else 1.6e308
    expected = (1 if family == 'gamma' else -1)*np.sqrt(2)*np.sqrt(halfdev)
    for residual, in rows:
        assert np.isfinite(residual)
        assert residual/expected == pytest.approx(1.,rel=1e-10)


@pytest.mark.parametrize('scale', [1e-305, 1e-170, 1e170, 1e305])
@pytest.mark.parametrize('robust', ['none', 'hc0', 'hc1', 'hc2', 'hc3', 'cluster'])
def test_linear_inference_preserves_extreme_response_units(con, scale, robust):
    model(con)
    model(con, name='scaled_model')
    con.execute('UPDATE scaled_model SET coefficient=coefficient*?', [scale])
    data = training().assign(w=np.linspace(.25, 1, 48), expo=.1, grp=np.arange(48)%4)
    data.loc[47, 'w'] = 0
    load(con, 'base_response', data)
    load(con, 'scaled_response', data.assign(y=data.y*scale, expo=data.expo*scale))
    extra = "cluster_col:='grp'" if robust == 'cluster' else f"robust:='{robust}'"
    summaries, intervals, diagnostics = [], [], []
    for mdl, table, factor in [('edge_model', 'base_response', 1), ('scaled_model', 'scaled_response', scale)]:
        args = f"'{mdl}','{table}','y',weights_col:='w',offset_col:='expo'"
        summary = con.execute(f'SELECT * FROM linreg_summary({args},{extra})').df()
        summary[['coefficient', 'std_error', 'conf_low', 'conf_high']] /= factor
        summaries.append(summary.drop(columns='feature').to_numpy())
        intervals.append(con.execute(f'SELECT prediction,conf_low,conf_high FROM linreg_predict_ci({args})').df().to_numpy()/factor)
        diag = con.execute(f'SELECT hat,pearson_resid,deviance_resid,std_resid,cooks_distance FROM linreg_influence({args})').df()
        diag[['pearson_resid', 'deviance_resid']] /= factor
        diagnostics.append(diag.to_numpy())
    for results in [summaries, intervals, diagnostics]:
        assert np.isfinite(results).all()
        np.testing.assert_allclose(results[1], results[0], rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize('scale', [1e160, 1e300])
@pytest.mark.parametrize('alpha', [.1, 1., 5.])
def test_negative_binomial_large_mean_diagnostics_match_finite_ratio_reference(con, scale, alpha):
    from scipy.stats import norm
    con.execute('CREATE TABLE large_nb AS SELECT (i%5)::DOUBLE x,?*(1+i%3) y FROM range(30)t(i)', [scale])
    con.execute(f"CREATE TABLE nb_model AS SELECT * FROM nbinom_fit('large_nb','y',alpha:={alpha})")
    design = np.column_stack([np.ones(30), np.arange(30)%5])
    ratio = (1+np.arange(30)%3)/2
    # At these means, 1/mu is negligible relative to alpha in double precision.
    hweights = ratio/alpha
    scores = (ratio-1)/alpha
    bread_inv = np.linalg.inv(design.T @ (hweights[:, None]*design))
    meat = design.T @ ((scores**2)[:, None]*design)
    expected_se = np.sqrt(np.diag(bread_inv @ meat @ bread_inv))
    hat = hweights*np.einsum('ij,jk,ik->i', design, bread_inv, design)
    pearson = (ratio-1)/np.sqrt(alpha)
    deviance = np.sign(ratio-1)*np.sqrt(2*(ratio-1-np.log(ratio))/alpha)
    expected_diag = np.column_stack([hat, pearson, deviance, pearson/np.sqrt(1-hat), pearson**2*hat/(2*(1-hat)**2)])
    args = f"'nb_model','large_nb','y',alpha:={alpha}"
    actual_se = con.execute(f"SELECT std_error FROM nbinom_summary({args},robust:='hc0')").df()['std_error'].to_numpy()
    np.testing.assert_allclose(actual_se, expected_se, rtol=1e-10)
    actual_diag = con.execute(f'SELECT hat,pearson_resid,deviance_resid,std_resid,cooks_distance FROM nbinom_influence({args})').df().to_numpy()
    np.testing.assert_allclose(actual_diag, expected_diag, rtol=1e-9, atol=1e-11)
    dispersion = con.execute(f'SELECT dispersion FROM nbinom_evaluate({args})').fetchone()[0]
    assert dispersion == pytest.approx(np.sum(pearson**2)/28, rel=1e-10)
    variance = alpha*np.einsum('ij,jk,ik->i', design, np.linalg.inv(design.T@design), design)
    expected_ci = 2*np.exp(np.column_stack([-np.sqrt(variance), np.sqrt(variance)])*norm.ppf(.975))
    actual_ci = con.execute(f'SELECT conf_low,conf_high FROM nbinom_predict_ci({args})').df().to_numpy()/scale
    np.testing.assert_allclose(actual_ci, expected_ci, rtol=1e-10)


@pytest.mark.parametrize('family', ['poisson', 'nbinom'])
@pytest.mark.parametrize('scale', [1e-302, 1e-310])
def test_tiny_positive_information_preserves_uncertainty_and_leverage(con, family, scale):
    x = np.arange(-10, 11)/10
    mu = np.exp(.3*x)
    ratio = 1+.1*(np.arange(21)%3)
    design = np.column_stack([np.ones(21), x])
    load(con, 'tiny_information', pd.DataFrame({'x': x, 'y': scale*mu*ratio}))
    con.execute("CREATE TABLE tiny_model AS SELECT '(Intercept)' feature,ln(?) coefficient UNION ALL SELECT 'x',.3", [scale])
    inverse = np.linalg.inv(design.T@(mu[:, None]*design))
    # NB tends to Poisson here: alpha*mu is far below machine precision.
    se = con.execute(f"SELECT std_error FROM {family}_summary('tiny_model','tiny_information','y')").df()['std_error'].to_numpy()
    np.testing.assert_allclose(se*np.sqrt(scale), np.sqrt(np.diag(inverse)), rtol=1e-10)
    scores = mu*(ratio-1)
    robust_cov = inverse@(design.T@((scores**2)[:, None]*design))@inverse
    robust = con.execute(f"SELECT std_error FROM {family}_summary('tiny_model','tiny_information','y',robust:='hc0')").df()['std_error'].to_numpy()
    np.testing.assert_allclose(robust, np.sqrt(np.diag(robust_cov)), rtol=1e-9)
    hat = con.execute(f"SELECT hat FROM {family}_influence('tiny_model','tiny_information','y')").df()['hat'].to_numpy()
    np.testing.assert_allclose(hat, mu*np.einsum('ij,jk,ik->i', design, inverse, design), rtol=1e-10)
    ci = con.execute(f"SELECT conf_low,conf_high FROM {family}_predict_ci('tiny_model','tiny_information','y')").fetchall()
    # The link-scale standard error is finite; exponentiating its enormous
    # normal interval legitimately reaches zero/infinity, rather than NULL.
    assert all(low == 0.0 and high == np.inf for low, high in ci)


@pytest.mark.parametrize('scale', [1e-302, 1e-310])
def test_multinomial_tiny_class_information_has_finite_standard_errors(con, scale):
    con.execute("CREATE TABLE tiny_classes AS SELECT i::DOUBLE x,CASE WHEN i%2=0 THEN 'a' ELSE 'b' END y FROM range(6)t(i)")
    con.execute("CREATE TABLE tiny_model AS SELECT 'a' AS class,'(Intercept)' feature,0.0 coefficient UNION ALL SELECT 'a','x',0.0 UNION ALL SELECT 'b','(Intercept)',ln(?) UNION ALL SELECT 'b','x',0.0", [scale])
    design = np.column_stack([np.ones(6), np.arange(6)])
    se = con.execute("SELECT std_error FROM multinom_summary('tiny_model','tiny_classes','y')").df()['std_error'].to_numpy()
    np.testing.assert_allclose(se*np.sqrt(scale), np.sqrt(np.diag(np.linalg.inv(design.T@design))), rtol=1e-10)


@pytest.mark.parametrize('feature_scale,response_scale,new_x', [(1.,1.,1e160), (1.,1.,1e300), (1e-100,1e-170,1e160), (1e100,1e100,1e160), (1e-200,1e-200,1e150), (1e-310,1e-310,1e150), (1e-310,1e-200,1e150)])
def test_prediction_intervals_scale_newdata_before_covariance_products(con, feature_scale, response_scale, new_x):
    from scipy.stats import t
    x = np.arange(10, dtype=float)
    y = x+x%2
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(design,y,rcond=None)[0]
    covariance = np.linalg.inv(design.T@design)*np.sum((y-design@beta)**2)/8
    # Compute in units of the scoring value so no reference variance overflows.
    scoring = np.array([feature_scale/new_x,1.])
    expected_prediction = scoring@beta
    width = t.ppf(.975,8)*np.sqrt(scoring@covariance@scoring)
    con.execute('CREATE TABLE extrap_train AS SELECT i::DOUBLE*? x,(i+i%2)::DOUBLE*? y FROM range(10)t(i)', [feature_scale,response_scale])
    con.execute("CREATE TABLE extrap_model AS SELECT * FROM linreg_fit('extrap_train','y')")
    con.execute('CREATE TABLE extrap_score AS SELECT ?::DOUBLE x', [new_x])
    actual = np.array(con.execute("SELECT prediction,conf_low,conf_high FROM linreg_predict_ci('extrap_model','extrap_train','y',newdata:='extrap_score')").fetchone())
    units = new_x*(response_scale/feature_scale)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual/units, [expected_prediction,expected_prediction-width,expected_prediction+width], rtol=1e-9)


@pytest.mark.parametrize('robust', ['none', 'hc0', 'hc1', 'hc2', 'hc3', 'cluster'])
@pytest.mark.parametrize('scale', [1e-310, 1e-305])
def test_linear_standard_errors_combine_response_and_feature_units(con, robust, scale):
    x = np.arange(10, dtype=float)
    y = x+x%2
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(design,y,rcond=None)[0]
    inverse = np.linalg.inv(design.T@design)
    residual = y-design@beta
    if robust == 'none':
        covariance = inverse*(residual@residual)/8
    elif robust == 'cluster':
        scores = design*residual[:,None]
        groups = np.array([scores[x.astype(int)//2 == group].sum(axis=0) for group in range(5)])
        covariance = inverse@(groups.T@groups)@inverse*(5/4)*(9/8)
    else:
        meat_weights = residual**2
        hat = np.einsum('ij,jk,ik->i',design,inverse,design)
        if robust == 'hc2': meat_weights /= 1-hat
        if robust == 'hc3': meat_weights /= (1-hat)**2
        covariance = inverse@(design.T@(meat_weights[:,None]*design))@inverse
        if robust == 'hc1': covariance *= 10/8
    con.execute('CREATE TABLE tiny_units AS SELECT i::DOUBLE*? x,(i+i%2)::DOUBLE*? y FROM range(10)t(i)', [scale,scale])
    con.execute("CREATE TABLE tiny_units_model AS SELECT * FROM linreg_fit('tiny_units','y')")
    if robust == 'cluster':
        con.execute('ALTER TABLE tiny_units ADD COLUMN grp INTEGER')
        con.execute('UPDATE tiny_units SET grp = CAST(round(x/?) AS INTEGER)//2', [scale])
        options = ",cluster_col:='grp'"
    else:
        options = f",robust:='{robust}'"
    actual = con.execute(f"SELECT feature,std_error,statistic,p_value,conf_low,conf_high FROM linreg_summary('tiny_units_model','tiny_units','y'{options})").fetchall()
    assert np.isfinite(np.array([row[1:] for row in actual], dtype=float)).all()
    standard_errors = np.array([row[1] for row in actual])
    standard_errors[0] /= scale
    np.testing.assert_allclose(standard_errors,np.sqrt(np.diag(covariance)),rtol=1e-9)


@pytest.mark.parametrize('family', ['linreg','logit','poisson','gamma','tweedie','nbinom'])
@pytest.mark.parametrize('scale', [1e200,1e300])
@pytest.mark.parametrize('robust', ['none','hc0','hc1','hc2','hc3'])
def test_inference_preserves_positive_weights_below_relative_double_range(con, family, scale, robust):
    sw = np.array([1.,1.,1/scale])
    design = np.array([[1.,0.],[1.,1.],[1/scale,1.]])
    linear = family == 'linreg'
    binary = family == 'logit'
    y = np.array([0.,1.,0.]) if linear or binary else np.array([.75,1.5,1.])
    beta = [1/3,1/3] if linear else [0.,0.]
    residual = np.array([-1/3,1/3,-1/3]) if linear else sw*(y-(.5 if binary else 1.))
    variance = .25 if binary else 2. if family == 'nbinom' else 1.
    fisher = .25 if binary else .5 if family == 'nbinom' else 1.
    observed = y if family == 'gamma' else 2*y-1 if family == 'tweedie' else (1+y)/4 if family == 'nbinom' else np.full(3,fisher)
    bread = np.linalg.inv(design.T@(observed[:,None]*design))
    hat = observed*np.einsum('ij,jk,ik->i',design,bread,design)
    estimated = family in ['linreg','gamma','tweedie']
    phi = np.sum(residual**2/variance) if estimated else 1.
    if robust == 'none':
        covariance = phi*np.linalg.inv(fisher*(design.T@design))
        expected_se = np.sqrt(np.diag(covariance))/(1. if estimated else np.sqrt(scale))
    else:
        score = residual/(2. if family == 'nbinom' else 1.)
        meat = score**2
        if robust == 'hc2': meat /= 1-hat
        if robust == 'hc3': meat /= (1-hat)**2
        covariance = bread@(design.T@(meat[:,None]*design))@bread
        if robust == 'hc1': covariance *= 3.
        expected_se = np.sqrt(np.diag(covariance))
    con.execute('CREATE TABLE wide_weights(x DOUBLE,y DOUBLE,w DOUBLE)')
    con.executemany('INSERT INTO wide_weights VALUES (?,?,?)', [(0.,float(y[0]),scale),(1.,float(y[1]),scale),(scale,float(y[2]),1/scale)])
    con.execute('CREATE TABLE weight_model(feature VARCHAR,coefficient DOUBLE)')
    con.executemany('INSERT INTO weight_model VALUES (?,?)',[('(Intercept)',beta[0]),('x',beta[1])])
    extra = ',power:=3.' if family == 'tweedie' else ''
    args = f"'weight_model','wide_weights','y',weights_col:='w'{extra}"
    actual_se = np.array(con.execute(f"SELECT std_error FROM {family}_summary({args},robust:='{robust}')").fetchall()).ravel()
    np.testing.assert_allclose(actual_se,expected_se,rtol=1e-9)
    if robust == 'none':
        diagnostics = np.array(con.execute(f'SELECT hat,pearson_resid,std_resid,cooks_distance FROM {family}_influence({args})').fetchall())
        pearson = residual/np.sqrt(variance)
        expected_std = pearson/np.sqrt(phi*(1-hat))
        expected_cook = (pearson**2/phi)*hat/(2*(1-hat)**2)
        np.testing.assert_allclose(diagnostics[:,0],hat,rtol=1e-9)
        np.testing.assert_allclose(diagnostics[:,1]/np.sqrt(scale),pearson,rtol=1e-9,atol=1e-310)
        np.testing.assert_allclose(diagnostics[:,2]/(1. if estimated else np.sqrt(scale)),expected_std,rtol=1e-9,atol=1e-310)
        np.testing.assert_allclose(diagnostics[:,3]/(1. if estimated else scale),expected_cook,rtol=1e-9,atol=1e-310)
        if linear:
            intervals = np.array(con.execute(f'SELECT prediction,conf_low,conf_high FROM linreg_predict_ci({args})').fetchall())
            assert np.isfinite(intervals).all()


@pytest.mark.parametrize('power,scale', [(3.,1e-170),(3.,1e170),(2.,1e-310)])
@pytest.mark.parametrize('robust', ['none','hc0','hc1','hc2','hc3'])
def test_tweedie_inference_combines_mean_powers_before_scaling(con, power, scale, robust):
    summaries, hats = [], []
    for factor in [1.,scale]:
        con.execute('CREATE OR REPLACE TABLE power_units AS SELECT i::DOUBLE/10 x,(1.0+i%3)*? y FROM range(12)t(i)',[factor])
        con.execute(f"CREATE OR REPLACE TABLE power_model AS SELECT * FROM tweedie_fit('power_units','y',power:={power})")
        summaries.append(con.execute(f"SELECT std_error FROM tweedie_summary('power_model','power_units','y',power:={power},robust:='{robust}')").fetchnumpy()['std_error'])
        hats.append(con.execute(f"SELECT hat FROM tweedie_influence('power_model','power_units','y',power:={power})").fetchnumpy()['hat'])
    assert np.isfinite(summaries).all() and np.isfinite(hats).all()
    np.testing.assert_allclose(summaries[1],summaries[0],rtol=1e-8)
    np.testing.assert_allclose(hats[1],hats[0],rtol=1e-8)


@pytest.mark.parametrize('family,power', [('gamma', 2.0), ('tweedie', 1.5)])
@pytest.mark.parametrize('robust', ['none', 'hc0', 'hc2', 'hc3'])
def test_inference_retains_underflowed_log_link_means(con, family, power, robust):
    from scipy.stats import t

    x = np.arange(4.)
    y = np.array([1e-320, 2e-320, 3e-320, 4e-320])
    design = np.column_stack([np.ones(4), x])
    eta = -750.
    load(con, 'underflow_data', pd.DataFrame({'x': x, 'y': y}))
    con.execute("CREATE TABLE underflow_model AS SELECT '(Intercept)' feature,-750.::DOUBLE coefficient UNION ALL SELECT 'x',0.")
    ratio = np.exp(np.log(y) - eta)
    root_info = np.exp((1-power/2)*eta)
    pearson = (ratio-1)*root_info
    phi = pearson @ pearson / 2
    fisher_inv = np.linalg.inv(design.T @ design)
    observed = 1 + (power-1)*(ratio-1)
    bread_inv = np.linalg.inv(design.T @ (observed[:, None]*design))
    hat = observed*np.einsum('ij,jk,ik->i', design, bread_inv, design)
    if robust == 'none':
        covariance = (phi/root_info/root_info)*fisher_inv
    else:
        meat = (ratio-1)**2
        if robust == 'hc2': meat /= 1-hat
        if robust == 'hc3': meat /= (1-hat)**2
        covariance = bread_inv @ (design.T @ (meat[:, None]*design)) @ bread_inv
    args = "'underflow_model','underflow_data','y'"
    actual = con.execute(f"SELECT std_error FROM {family}_summary({args},robust:='{robust}')").fetchnumpy()['std_error']
    np.testing.assert_allclose(actual, np.sqrt(np.diag(covariance)), rtol=1e-10)
    diag = con.execute(f"SELECT hat,pearson_resid,std_resid,cooks_distance FROM {family}_influence({args})").fetchall()
    standardized = pearson/np.sqrt(phi*(1-hat))
    cooks = (pearson**2/phi)*hat/(2*(1-hat)**2)
    np.testing.assert_allclose(diag, np.column_stack([hat, pearson, standardized, cooks]), rtol=1e-10, atol=0)
    # Score at a representable mean so NULL or incorrect training covariance
    # cannot hide behind endpoints that both round to zero.
    if robust == 'none':
        con.execute('ALTER TABLE underflow_data ADD COLUMN expo DOUBLE DEFAULT 0')
        con.execute('CREATE TABLE underflow_new AS SELECT 0.::DOUBLE x,750.::DOUBLE expo')
        low, high = con.execute(f"SELECT conf_low,conf_high FROM {family}_predict_ci({args},offset_col:='expo',newdata:='underflow_new')").fetchone()
        radius = t.ppf(.975, 2)*np.sqrt(covariance[0,0])
        # The large residuals make this interval extend beyond DOUBLE; the
        # lower endpoint must still be zero rather than missing.
        assert low == np.exp(-radius)
        assert np.isinf(high)


def test_fitted_gamma_inference_with_underflowed_offset_mean(con):
    con.execute("""CREATE TABLE underflow_fit AS SELECT * FROM (VALUES
        (0.,1e-320,-750.,1e-20),(1.,exp(-350.),-350.,1.),
        (2.,exp(50.),50.,1.),(3.,exp(50.),50.,1.)) t(x,y,expo,w)""")
    con.execute("CREATE TABLE fitted_underflow AS SELECT * FROM gamma_fit('underflow_fit','y',offset_col:='expo',weights_col:='w',max_iter:=2000)")
    data = con.sql('FROM underflow_fit').df()
    beta = con.sql('SELECT coefficient FROM fitted_underflow').fetchnumpy()['coefficient']
    design = np.column_stack([np.ones(4), data.x])
    eta = design @ beta + data.expo.to_numpy()
    ratio = np.exp(np.log(data.y.to_numpy())-eta)
    pearson = np.sqrt(data.w.to_numpy())*(ratio-1)
    phi = pearson @ pearson/2
    inv = np.linalg.inv(design.T @ (data.w.to_numpy()[:,None]*design))
    args = "'fitted_underflow','underflow_fit','y',offset_col:='expo',weights_col:='w'"
    actual = con.execute(f'SELECT std_error FROM gamma_summary({args})').fetchnumpy()['std_error']
    np.testing.assert_allclose(actual, np.sqrt(phi*np.diag(inv)), rtol=1e-9)
    actual_pearson = con.execute(f'SELECT pearson_resid FROM gamma_influence({args})').fetchnumpy()['pearson_resid']
    np.testing.assert_allclose(actual_pearson, pearson, rtol=1e-9, atol=1e-13)
    from scipy.stats import t
    intervals = con.execute(f'SELECT conf_low,conf_high FROM gamma_predict_ci({args})').fetchall()
    radius = t.ppf(.975, 2)*np.sqrt(phi*np.einsum('ij,jk,ik->i', design, inv, design))
    np.testing.assert_allclose(intervals, np.column_stack([np.exp(eta-radius), np.exp(eta+radius)]), rtol=1e-10, atol=0)
