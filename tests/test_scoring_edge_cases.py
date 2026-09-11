"""Regression coverage for scoring boundaries and generated categorical SQL."""

from pathlib import Path
from decimal import Decimal, localcontext

import duckdb
import numpy as np
import pytest
from sklearn.metrics import d2_tweedie_score, mean_tweedie_deviance


@pytest.fixture
def con():
    connection = duckdb.connect()
    connection.execute(
        (Path(__file__).resolve().parents[1] / "regression_macros.sql").read_text()
    )
    yield connection
    connection.close()


def _metrics(con, call):
    result = con.execute(f"SELECT * FROM {call}")
    return dict(zip([column[0] for column in result.description], result.fetchone()))


@pytest.mark.parametrize("power", [1.0, 2.0])
def test_tweedie_evaluation_at_poisson_and_gamma_endpoints(con, power):
    con.execute(
        "CREATE TABLE model AS SELECT '(Intercept)' feature, 0.5 coefficient "
        "UNION ALL SELECT 'x', 0.2"
    )
    y = np.array([0.0, 1.0, 3.0, 2.0, 5.0]) if power == 1 else np.array([1, 2, 4, 3, 6])
    con.execute("CREATE TABLE observations(x DOUBLE, y DOUBLE)")
    con.executemany("INSERT INTO observations VALUES (?, ?)", list(enumerate(y.tolist())))
    mu = np.exp(0.5 + 0.2 * np.arange(len(y)))
    metrics = _metrics(
        con, f"tweedie_evaluate('model', 'observations', 'y', power := {power})"
    )
    assert metrics["deviance"] == pytest.approx(
        len(y) * mean_tweedie_deviance(y, mu, power=power)
    )
    assert metrics["null_deviance"] == pytest.approx(
        len(y) * mean_tweedie_deviance(y, np.full(len(y), y.mean()), power=power)
    )
    assert metrics["pseudo_r2"] == pytest.approx(d2_tweedie_score(y, mu, power=power))


def _decimal_tweedie_deviance(y, mu, power):
    """Evaluate the defining expression with enough precision for cancellation."""
    if y == mu:
        return 0.0
    with localcontext() as context:
        context.prec = 80
        y, mu, power = (Decimal.from_float(float(v)) for v in (y, mu, power))
        if power == 1:
            half = (y * (y / mu).ln() if y else Decimal(0)) - y + mu
        elif power == 2:
            half = -(y / mu).ln() + y / mu - 1
        else:
            half = (
                y ** (2 - power) / ((1 - power) * (2 - power))
                - y * mu ** (1 - power) / (1 - power)
                + mu ** (2 - power) / (2 - power)
            )
        return float(2 * half)


@pytest.mark.parametrize("power, with_zero", [
    (float(power), with_zero)
    for power in [
        1.0, np.nextafter(1.0, 2.0), 1.0 + 1e-10, 1.01, 1.5,
        2.0 - 1e-10, np.nextafter(2.0, 1.0), 2.0,
        np.nextafter(2.0, 3.0), 2.0 + 1e-10, 2.01, 3.0,
    ]
    for with_zero in [False, True]
    if not with_zero or power < 2  # Zero outcomes require power < 2.
])
def test_tweedie_deviance_is_continuous_near_endpoints(con, power, with_zero):
    con.execute("CREATE TABLE model AS SELECT * FROM "
                "(VALUES ('(Intercept)', 0.0), ('x', 1.0)) t(feature, coefficient)")
    y = np.array([0.0 if with_zero else 1.0, 3.0, 5.0])
    x = np.array([0.0, 0.5, 1.0])
    con.execute("CREATE TABLE observations(x DOUBLE, y DOUBLE)")
    con.executemany("INSERT INTO observations VALUES (?, ?)", list(zip(x.tolist(), y.tolist())))
    mu = np.exp(x)
    unit_deviance = np.array([
        _decimal_tweedie_deviance(yi, mui, power) for yi, mui in zip(y, mu)
    ])
    null_deviance = sum(_decimal_tweedie_deviance(yi, y.mean(), power) for yi in y)
    metrics = _metrics(con, f"tweedie_evaluate('model', 'observations', 'y', power := {power!r}::DOUBLE)")
    assert metrics["deviance"] == pytest.approx(unit_deviance.sum(), rel=1e-11, abs=1e-12)
    assert metrics["null_deviance"] == pytest.approx(null_deviance, rel=1e-11, abs=1e-12)
    assert metrics["pseudo_r2"] == pytest.approx(1 - unit_deviance.sum() / null_deviance,
                                                 rel=1e-11, abs=1e-12)
    actual = con.execute(f"SELECT deviance_resid FROM tweedie_influence("
                         f"'model', 'observations', 'y', power := {power!r}::DOUBLE)").fetchnumpy()["deviance_resid"]
    expected = np.sign(y - mu) * np.sqrt(unit_deviance)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("power", [1.0000000001, 1.5, 1.9999999999, 2.0000000001, 3.0])
@pytest.mark.parametrize("outcomes", [[1e-300, 1e300], [1 - 1e-7, 1 + 1e-7]])
def test_tweedie_deviance_handles_extreme_and_nearly_exact_means(con, power, outcomes):
    con.execute("CREATE TABLE model AS SELECT '(Intercept)' feature, 0.0 coefficient")
    con.execute("CREATE TABLE observations(y DOUBLE)")
    con.executemany("INSERT INTO observations VALUES (?)", [(y,) for y in outcomes])
    metrics = _metrics(con, f"tweedie_evaluate('model', 'observations', 'y', power := {power}::DOUBLE)")
    expected = sum(_decimal_tweedie_deviance(y, 1.0, power) for y in outcomes)
    assert np.isfinite(metrics["deviance"])
    assert metrics["deviance"] == pytest.approx(expected, rel=1e-10, abs=0)


@pytest.mark.parametrize("outcome", [0, 1])
def test_logit_evaluation_on_single_class_holdout(con, outcome):
    con.execute(
        "CREATE TABLE model AS SELECT '(Intercept)' feature, 0.5 coefficient "
        "UNION ALL SELECT 'x', 0.2"
    )
    con.execute(f"CREATE TABLE observations AS SELECT i x, {outcome} y FROM range(5) t(i)")
    metrics = _metrics(con, "logit_evaluate('model', 'observations', 'y')")
    scores = 0.5 + 0.2 * np.arange(5)
    expected_loss = np.mean(np.logaddexp(0, scores) - outcome * scores)
    assert metrics["n"] == 5
    assert metrics["log_loss"] == pytest.approx(expected_loss)
    assert metrics["loglik"] == pytest.approx(-5 * expected_loss)
    assert metrics["null_deviance"] == 0
    assert metrics["auc"] is None
    assert metrics["pseudo_r2"] is None


def test_linear_evaluation_of_exact_fit_retains_error_metrics(con):
    con.execute(
        "CREATE TABLE model AS SELECT '(Intercept)' feature, 1.0 coefficient "
        "UNION ALL SELECT 'x', 2.0"
    )
    con.execute("CREATE TABLE observations AS SELECT i x, 1 + 2*i y FROM range(5) t(i)")
    metrics = _metrics(con, "linreg_evaluate('model', 'observations', 'y')")
    assert metrics["rmse"] == 0
    assert metrics["mae"] == 0
    assert metrics["r2"] == 1
    assert metrics["adj_r2"] == 1
    assert metrics["loglik"] == float("inf")
    assert metrics["aic"] == float("-inf")
    assert metrics["bic"] == float("-inf")


@pytest.mark.parametrize("family", ["linreg", "logit", "poisson", "gamma", "tweedie", "nbinom"])
@pytest.mark.parametrize("with_offset", [False, True])
def test_intercept_only_evaluation_matches_predictions(con, family, with_offset):
    intercept = 0.4 if family == "logit" else 1.0
    con.execute(
        "CREATE TABLE model AS SELECT '(Intercept)' feature, ?::DOUBLE coefficient",
        [intercept],
    )
    con.execute(
        "CREATE TABLE observations AS "
        "SELECT i x, (i % 2)::DOUBLE binary_y, (i + 1)::DOUBLE y, "
        "CASE WHEN i = 2 THEN NULL ELSE 0.1 * i END exposure_offset FROM range(5) t(i)"
    )
    offset_arg = ", offset_col := 'exposure_offset'" if with_offset else ""
    outcome = "binary_y" if family == "logit" else "y"
    prediction = "prob" if family == "logit" else "prediction"
    scored = con.execute(
        f"SELECT {outcome}, {prediction} FROM {family}_predict('model', 'observations'{offset_arg}) "
        f"WHERE {prediction} IS NOT NULL"
    ).fetchnumpy()
    metrics = _metrics(con, f"{family}_evaluate('model', 'observations', '{outcome}'{offset_arg})")
    assert metrics["n"] == (4 if with_offset else 5)
    y, mu = scored[outcome], scored[prediction]
    if family == "logit":
        assert metrics["log_loss"] == pytest.approx(-np.mean(y * np.log(mu) + (1 - y) * np.log1p(-mu)))
    else:
        assert metrics["rmse"] == pytest.approx(np.sqrt(np.mean((y - mu) ** 2)))
        assert metrics["mae"] == pytest.approx(np.mean(np.abs(y - mu)))


@pytest.mark.parametrize("macro", ["tweedie_predict", "nbinom_predict"])
@pytest.mark.parametrize("column", ["prediction", "PrEdIcTiOn"])
@pytest.mark.parametrize("empty", [False, True])
def test_count_family_predictions_reject_output_collisions(con, macro, column, empty):
    con.execute("CREATE TABLE model AS SELECT '(Intercept)' feature, 0.0 coefficient")
    con.execute(f'CREATE TABLE observations(x DOUBLE, "{column}" DOUBLE)')
    if not empty:
        con.execute("INSERT INTO observations VALUES (1, 123)")
    with pytest.raises(duckdb.InvalidInputException, match="collides with the output column"):
        con.execute(f"SELECT * FROM {macro}('model', 'observations')").fetchall()


@pytest.mark.parametrize("qualified", [False, True])
def test_dummy_encoding_quotes_source_names_and_values(con, qualified):
    con.execute('CREATE SCHEMA "schema space"')
    relation = '"schema space"."sales.\'table"' if qualified else '"sales.\'table"'
    con.execute(f'CREATE TABLE {relation}("select" DOUBLE, "cat\"\" label" VARCHAR, y DOUBLE)')
    con.executemany(
        f"INSERT INTO {relation} VALUES (?, ?, ?)",
        [(1, "a", 2), (2, "b's", 5), (3, None, 8)],
    )
    sql = con.execute("SELECT dummy_encode_sql(?, 'y')", [relation]).fetchone()[0]
    encoded = con.execute(sql)
    assert [column[0] for column in encoded.description] == ["select", "y", 'cat" label_b\'s']
    assert encoded.fetchall() == [(1.0, 2.0, 0), (2.0, 5.0, 1), (3.0, 8.0, None)]


def test_dummy_encoding_resolves_same_named_relations_in_distinct_schemas(con):
    con.execute("CREATE SCHEMA other")
    con.execute("CREATE TABLE rawdata(x DOUBLE, y DOUBLE)")
    con.execute("INSERT INTO rawdata VALUES (1, 2)")
    con.execute("CREATE TABLE other.rawdata(category VARCHAR, y DOUBLE)")
    con.execute("INSERT INTO other.rawdata VALUES ('a', 1), ('b', 2)")
    sql = con.execute("SELECT dummy_encode_sql('rawdata', 'y')").fetchone()[0]
    assert con.execute(sql).fetchall() == [(1.0, 2.0)]


@pytest.mark.parametrize("level", ["only", None])
def test_dummy_encoding_preserves_null_exclusion_without_dummy_columns(con, level):
    con.execute("CREATE TABLE rawdata(x DOUBLE, y DOUBLE, category VARCHAR)")
    con.executemany(
        "INSERT INTO rawdata VALUES (?, ?, ?)",
        [(i, 1 + 2 * i, level) for i in range(4)] + [(4, 999, None)],
    )
    sql = con.execute("SELECT dummy_encode_sql('rawdata', 'y')").fetchone()[0]
    con.execute("CREATE TABLE encoded AS " + sql)
    assert con.execute("SELECT count(*) FROM encoded").fetchone()[0] == (0 if level is None else 4)
    if level is not None:
        coefficients = dict(con.execute("SELECT * FROM linreg_fit('encoded', 'y')").fetchall())
        assert coefficients["(Intercept)"] == pytest.approx(1)
        assert coefficients["x"] == pytest.approx(2)


def test_multinomial_evaluation_preserves_rid_outcome(con):
    con.execute("""
        CREATE TABLE observations AS
        SELECT i AS x, CASE WHEN i%2=0 THEN 'a' ELSE 'b' END AS rid
        FROM range(8) q(i)
    """)
    con.execute("CREATE TABLE model AS SELECT * FROM multinom_fit('observations', 'rid')")
    scores = con.execute("SELECT rid, pred, probs FROM multinom_predict('model', 'observations')").fetchall()
    actual = _metrics(con, "multinom_evaluate('model', 'observations', 'rid')")
    assert actual["n"] == len(scores) == 8
    assert actual["accuracy"] == pytest.approx(np.mean([label == pred for label, pred, _ in scores]))
    assert actual["log_loss"] == pytest.approx(-np.mean([np.log(probs[label]) for label, _, probs in scores]))


@pytest.mark.parametrize('family,power', [('poisson',1.0),('gamma',2.0),('tweedie',1.5),('nbinom',None),('logit',None)])
def test_offset_null_deviance_uses_intercept_only_fit(con, family, power):
    from scipy.optimize import brentq
    y = np.array([0.,1.,0.,1.,1.,0.]) if family == 'logit' else np.array([1.,3.,2.,8.,5.,7.])
    offsets = np.array([-1.,0.2,0.7,1.3,-0.4,0.8])
    con.execute("CREATE TABLE offset_null_data(y DOUBLE, expo DOUBLE)")
    con.executemany('INSERT INTO offset_null_data VALUES (?,?)', list(zip(y.tolist(), offsets.tolist())))
    def inverse(b):
        return 1/(1+np.exp(-b-offsets)) if family == 'logit' else np.exp(b+offsets)
    def score(b):
        mu = inverse(b)
        if family == 'nbinom': return np.sum((y-mu)/(1+mu))
        if family == 'logit': return np.sum(y-mu)
        return np.sum((y-mu)*mu**(1-power))
    intercept = brentq(score,-10,10,xtol=1e-14)
    mu = inverse(intercept)
    con.execute("CREATE TABLE offset_null_model AS SELECT '(Intercept)' feature, ?::DOUBLE coefficient", [intercept])
    metrics = _metrics(con, f"{family}_evaluate('offset_null_model','offset_null_data','y',offset_col:='expo')")
    if family == 'logit': expected = -2*np.sum(y*np.log(mu)+(1-y)*np.log1p(-mu))
    elif family == 'nbinom': expected = 2*np.sum(y*np.log(y/mu)-(y+1)*np.log((y+1)/(mu+1)))
    else: expected = len(y)*mean_tweedie_deviance(y,mu,power=power)
    assert metrics['null_deviance'] == pytest.approx(expected,abs=1e-10)
    assert metrics['deviance'] == pytest.approx(expected,abs=1e-10)
    assert metrics['pseudo_r2'] == pytest.approx(0.0,abs=1e-10)


def test_perfect_offset_null_model_has_undefined_pseudo_r2(con):
    con.execute("CREATE TABLE perfect_null_model AS SELECT '(Intercept)' feature, 0.0 coefficient")
    con.execute('CREATE TABLE perfect_null_data AS SELECT pow(2,i) y, ln(pow(2,i)) expo FROM range(4)t(i)')
    result = _metrics(con,"poisson_evaluate('perfect_null_model','perfect_null_data','y',offset_col:='expo')")
    assert result['null_deviance'] == 0
    assert result['pseudo_r2'] is None


def test_logit_null_likelihood_stays_finite_with_extreme_offsets(con):
    con.execute("CREATE TABLE extreme_null_model AS SELECT '(Intercept)' feature, 0.0 coefficient")
    con.execute('CREATE TABLE extreme_null_data AS SELECT * FROM (VALUES (0.0,100.0),(1.0,-100.0)) t(y,expo)')
    result = _metrics(con,"logit_evaluate('extreme_null_model','extreme_null_data','y',offset_col:='expo')")
    assert result['null_deviance'] == pytest.approx(400.0)
    assert result['deviance'] == pytest.approx(400.0)
    assert result['loglik'] == pytest.approx(-200.0)
    assert result['log_loss'] == pytest.approx(100.0)
    assert result['pseudo_r2'] == pytest.approx(0.0)
    assert result['aic'] == pytest.approx(402.0)
    assert result['bic'] == pytest.approx(400.0+np.log(2))


def test_multinomial_scoring_reuses_one_snapshot(con):
    con.execute('SELECT setseed(.42)')
    con.execute("CREATE TABLE snapshot_model AS SELECT * FROM (VALUES ('a','(Intercept)',0.),('a','x',0.),('b','(Intercept)',0.),('b','x',1.))t(class,feature,coefficient)")
    con.execute("CREATE VIEW changing_order AS SELECT i::DOUBLE-10 x, CASE WHEN i>=10 THEN 'b' ELSE 'a' END y FROM range(20)t(i) ORDER BY random()")
    scored = con.execute("SELECT x,probs['b'] FROM multinom_predict('snapshot_model','changing_order')").fetchall()
    for x, probability in scored:
        assert probability == pytest.approx(1/(1+np.exp(-x)),abs=1e-14)
    metrics = _metrics(con,"multinom_evaluate('snapshot_model','changing_order','y')")
    assert metrics['n'] == 20
    # The x=0 tie may choose either class; every other label must match.
    assert metrics['accuracy'] in (.95,1.0)
    assert metrics['log_loss'] == pytest.approx(np.logaddexp(0,-np.abs(np.arange(20)-10)).mean())


@pytest.mark.parametrize('power',[1.0,1.2,1.5,1.9])
@pytest.mark.parametrize('with_offset',[False,True])
def test_tweedie_zero_holdout_has_zero_null_deviance(con,power,with_offset):
    con.execute("CREATE TABLE zero_holdout_model AS SELECT '(Intercept)' feature,0.0 coefficient")
    con.execute('CREATE TABLE zero_holdout AS SELECT i x,0.0 y,i/10.0 expo FROM range(5)t(i)')
    offset=",offset_col:='expo'" if with_offset else ''
    result=_metrics(con,f"tweedie_evaluate('zero_holdout_model','zero_holdout','y',power:={power}{offset})")
    assert result['null_deviance']==0
    assert result['pseudo_r2'] is None
    mu=np.exp(np.arange(5)/10) if with_offset else np.ones(5)
    assert result['deviance']==pytest.approx(2*np.sum(mu**(2-power))/(2-power))


@pytest.mark.parametrize('family',['poisson','nbinom'])
@pytest.mark.parametrize('outcome',[0.0,1.0])
def test_count_evaluation_retains_finite_log_likelihood_after_underflow(con,family,outcome):
    con.execute("CREATE TABLE small_mean_model AS SELECT '(Intercept)' feature,-800.0 coefficient")
    con.execute('CREATE TABLE small_mean_data AS SELECT ?::DOUBLE y',[outcome])
    result=_metrics(con,f"{family}_evaluate('small_mean_model','small_mean_data','y')")
    assert result['loglik']==pytest.approx(-800*outcome)
    expected=0 if outcome==0 else 1598 if family=='poisson' else 1600-4*np.log(2)
    assert result['deviance']==pytest.approx(expected)
    assert result['aic']==pytest.approx(1600*outcome+2)


@pytest.mark.parametrize('alpha',[1e-5,1e-6,1e-12,1e-16])
@pytest.mark.parametrize('outcome',[0,1,5,100])
def test_negative_binomial_metrics_near_poisson_limit(con,alpha,outcome):
    import math
    mu=2.5
    con.execute("CREATE TABLE limit_model AS SELECT '(Intercept)' feature,?::DOUBLE coefficient",[np.log(mu)])
    con.execute('CREATE TABLE limit_data AS SELECT ?::DOUBLE y',[outcome])
    result=_metrics(con,f"nbinom_evaluate('limit_model','limit_data','y',alpha:={alpha})")
    # Integer-count gamma ratio evaluated as a stable finite product.
    expected=sum(math.log1p(j*alpha) for j in range(outcome))-math.lgamma(outcome+1)+outcome*math.log(mu)-(1/alpha+outcome)*math.log1p(alpha*mu)
    expected_dev=2*((outcome*math.log(outcome/mu) if outcome else 0)-(outcome+1/alpha)*math.log1p(alpha*(outcome-mu)/(1+alpha*mu)))
    assert result['loglik']==pytest.approx(expected,abs=1e-8)
    assert result['deviance']==pytest.approx(expected_dev,abs=1e-8)


def test_binary_evaluation_rejects_nonbinary_holdout(con):
    con.execute("CREATE TABLE binary_model AS SELECT * FROM (VALUES ('(Intercept)',0.),('x',1.))t(feature,coefficient)")
    con.execute('CREATE TABLE bad_holdout AS SELECT i::DOUBLE x,i::DOUBLE y FROM range(6)t(i)')
    with pytest.raises(duckdb.Error,match='outcome must be binary'):
        con.execute("SELECT * FROM logit_evaluate('binary_model','bad_holdout','y')").fetchall()


@pytest.mark.parametrize('family,parameter,values',[
    ('nbinom','alpha',['0','-1','NULL',"'NaN'::DOUBLE","'Infinity'::DOUBLE"]),
    ('tweedie','power',['.5','-1','NULL',"'NaN'::DOUBLE","'Infinity'::DOUBLE"]),
])
def test_evaluation_rejects_invalid_distribution_parameters(con,family,parameter,values):
    con.execute("CREATE TABLE invalid_parameter_model AS SELECT '(Intercept)' feature,0.0 coefficient")
    con.execute('CREATE TABLE invalid_parameter_data AS SELECT 1.0 y')
    for value in values:
        with pytest.raises(duckdb.Error,match=parameter+' must be finite'):
            con.execute(f"SELECT * FROM {family}_evaluate('invalid_parameter_model','invalid_parameter_data','y',{parameter}:={value})").fetchall()
