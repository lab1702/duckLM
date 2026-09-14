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


@pytest.mark.parametrize('family', ['linreg', 'logit', 'poisson', 'gamma', 'tweedie', 'nbinom'])
@pytest.mark.parametrize('empty', [False, True])
def test_evaluation_filters_incomplete_scores_before_outcome_validation(con, family, empty):
    con.execute("CREATE TABLE filter_model AS SELECT * FROM "
                "(VALUES ('(Intercept)', .3), ('x1', .2), ('x2', -.1)) t(feature, coefficient)")
    con.execute('CREATE TABLE filter_rows(x1 DOUBLE, x2 DOUBLE, expo DOUBLE, y DOUBLE)')
    # Invalid outcomes on unscorable rows must not trigger domain validation.
    con.execute('INSERT INTO filter_rows VALUES '
                '(NULL, 1., .1, -99.), (1., NULL, .1, -99.), '
                '(1., 2., NULL, -99.), (1., 2., .1, NULL)')
    call = f"{family}_evaluate('filter_model','filter_rows','y',offset_col:='expo')"
    if empty:
        with pytest.raises(duckdb.Error, match='no rows with a non-NULL prediction and outcome'):
            _metrics(con, call)
        return

    outcomes = [0., 1., 1., 0., 1.] if family == 'logit' else [1., 2., 3., 2., 4.]
    con.executemany('INSERT INTO filter_rows VALUES (?, ?, ?, ?)',
                    [(i / 5., i / 7., i / 10., y) for i, y in enumerate(outcomes)])
    con.execute('CREATE TABLE filter_complete AS SELECT * FROM filter_rows '
                'WHERE x1 IS NOT NULL AND x2 IS NOT NULL AND expo IS NOT NULL AND y IS NOT NULL')
    actual = _metrics(con, call)
    expected = _metrics(con, call.replace("'filter_rows'", "'filter_complete'"))
    assert actual['n'] == 5
    for metric in expected:
        if expected[metric] is None:
            assert actual[metric] is None
        else:
            assert actual[metric] == pytest.approx(expected[metric], rel=1e-12, abs=1e-12)


@pytest.mark.parametrize('logit', [40.0, 100.0, 700.0])
def test_logistic_tail_loss_matches_offset_null(con, logit):
    con.execute("CREATE TABLE tail_model AS SELECT '(Intercept)' feature,0.::DOUBLE coefficient")
    con.execute(f'CREATE TABLE tail_rows AS SELECT * FROM (VALUES ({logit},1.),(-{logit},0.))t(expo,y)')
    metrics = _metrics(con, "logit_evaluate('tail_model','tail_rows','y',offset_col:='expo')")
    expected_loss = np.logaddexp(0.0, -logit)
    assert metrics['log_loss'] == pytest.approx(expected_loss, rel=1e-12, abs=0.0)
    assert metrics['deviance'] == pytest.approx(4*expected_loss, rel=1e-12, abs=0.0)
    assert metrics['null_deviance'] == pytest.approx(4*expected_loss, rel=1e-12, abs=0.0)
    assert metrics['pseudo_r2'] == pytest.approx(0.0, abs=1e-12)


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


@pytest.mark.parametrize('power,y', [(4.,1e-155),(6.,2e-78)])
@pytest.mark.parametrize('delta', [-.01,-.001,-1e-4,0.,1e-4,.001,.01])
def test_tweedie_deviance_combines_large_response_scale_with_small_residual(con,power,y,delta):
    eta = con.execute('SELECT ln(?)+?', [y,delta]).fetchone()[0]
    con.execute("CREATE TABLE model AS SELECT '(Intercept)' feature,?::DOUBLE coefficient",[eta])
    con.execute('CREATE TABLE observations AS SELECT ?::DOUBLE y FROM range(3)',[y])
    actual = _metrics(con,f"tweedie_evaluate('model','observations','y',power:={power})")['deviance']
    if delta == 0:
        assert actual == 0.0
    else:
        expected = 3*_decimal_tweedie_deviance(y,np.exp(eta),power)
        assert np.isfinite(actual)
        assert actual == pytest.approx(expected,rel=2e-8)


@pytest.mark.parametrize('power', [1e155,1e308])
def test_tweedie_exact_deviance_remains_zero_at_large_finite_powers(con,power):
    con.execute("CREATE TABLE model AS SELECT '(Intercept)' feature,0.0::DOUBLE coefficient")
    con.execute('CREATE TABLE observations AS SELECT 1.0::DOUBLE y FROM range(4)')
    metrics=_metrics(con,f"tweedie_evaluate('model','observations','y',power:={power})")
    assert metrics['deviance']==0.0
    assert metrics['null_deviance']==0.0
    assert metrics['pseudo_r2'] is None
    residuals=con.execute(f"SELECT deviance_resid FROM tweedie_influence('model','observations','y',power:={power})").fetchall()
    assert residuals==[(0.0,)]*4


@pytest.mark.parametrize('family', ['gamma','tweedie','nbinom'])
@pytest.mark.parametrize('sparse', [False,True])
def test_dispersion_scales_before_pearson_squares_and_sums(con,family,sparse):
    con.execute("CREATE TABLE model AS SELECT '(Intercept)' feature,0.0::DOUBLE coefficient")
    if sparse:
        con.execute('CREATE TABLE observations AS SELECT CASE WHEN i=0 THEN 1e155 ELSE 1.0 END y FROM range(1000)t(i)')
        count=1000; large_y=Decimal.from_float(1e155); copies=1
    else:
        con.execute('CREATE TABLE observations AS SELECT 1e154::DOUBLE y FROM range(4)')
        count=4; large_y=Decimal.from_float(1e154); copies=4
    with localcontext() as context:
        context.prec=80
        expected=float(copies*(large_y-1)**2/(2 if family=='nbinom' else 1)/(count-1))
    actual=_metrics(con,f"{family}_evaluate('model','observations','y')")['dispersion']
    assert np.isfinite(actual)
    assert actual==pytest.approx(expected,rel=1e-12)


def test_tweedie_dispersion_preserves_tiny_mean_residual_scale(con):
    eta=float(np.log(1e-320))
    con.execute("CREATE TABLE model AS SELECT '(Intercept)' feature,?::DOUBLE coefficient",[eta])
    con.execute('CREATE TABLE observations AS SELECT 1e-10::DOUBLE y FROM range(4)')
    expected=4/3*np.exp(2*np.log(1e-10)-eta)
    actual=_metrics(con,"tweedie_evaluate('model','observations','y',power:=1)")['dispersion']
    assert np.isfinite(actual)
    assert actual==pytest.approx(expected,rel=1e-10)


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


@pytest.mark.parametrize('family,positive_outcome', [('logit', 1), ('nbinom', 2)])
@pytest.mark.parametrize('offset', [-1e30, -1e300])
def test_offset_null_root_converges_with_widely_spaced_offsets(con, family, positive_outcome, offset):
    con.execute("CREATE TABLE wide_null_model AS SELECT '(Intercept)' feature, 0.::DOUBLE coefficient")
    con.execute('CREATE TABLE wide_null_data(y DOUBLE,expo DOUBLE)')
    con.executemany('INSERT INTO wide_null_data VALUES (?,?)', [(0, offset), (0, 0), (positive_outcome, 0)])
    result = _metrics(con, f"{family}_evaluate('wide_null_model','wide_null_data','y',offset_col:='expo')")
    assert np.isfinite(result['null_deviance'])
    assert result['null_deviance'] == pytest.approx(result['deviance'], rel=1e-12)
    assert result['pseudo_r2'] == pytest.approx(0, abs=1e-12)


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


@pytest.mark.parametrize('magnitude', [100.0, 1000.0])
def test_multinomial_loss_preserves_finite_extreme_logits(con, magnitude):
    con.execute("CREATE TABLE multiclass_model AS SELECT * FROM "
                "(VALUES ('a','(Intercept)',0.),('a','x',0.),"
                "('b','(Intercept)',0.),('b','x',1.))t(class,feature,coefficient)")
    con.execute(f"CREATE TABLE confident_errors AS SELECT * FROM "
                f"(VALUES ({magnitude},'a'),(-{magnitude},'b'))t(x,y)")
    metrics = _metrics(con, "multinom_evaluate('multiclass_model','confident_errors','y')")
    assert metrics['n'] == 2
    assert metrics['accuracy'] == 0
    assert metrics['log_loss'] == pytest.approx(np.logaddexp(0, magnitude), rel=1e-12)


@pytest.mark.parametrize('correct_rows', [1, 3, 9])
def test_multinomial_mean_loss_stays_finite_when_one_logit_difference_overflows(con, correct_rows):
    con.execute("CREATE TABLE wide_logits AS SELECT * FROM (VALUES "
                "('a','(Intercept)',0.::DOUBLE),('b','(Intercept)',1e308),"
                "('c','(Intercept)',-1e308))t(class,feature,coefficient)")
    con.execute(f"CREATE TABLE wide_labels AS SELECT 'c' y UNION ALL SELECT 'b' FROM range({correct_rows})")
    metrics = _metrics(con, "multinom_evaluate('wide_logits','wide_labels','y')")
    assert metrics['n'] == correct_rows+1
    assert metrics['accuracy'] == pytest.approx(correct_rows/(correct_rows+1))
    assert metrics['log_loss'] == pytest.approx((2/(correct_rows+1))*1e308, rel=1e-12)


def test_multinomial_loss_is_infinite_for_an_unseen_class(con):
    con.execute("CREATE TABLE multiclass_model AS SELECT * FROM "
                "(VALUES ('a','(Intercept)',0.),('b','(Intercept)',0.))t(class,feature,coefficient)")
    con.execute("CREATE TABLE unseen_class AS SELECT 'c' y")
    metrics = _metrics(con, "multinom_evaluate('multiclass_model','unseen_class','y')")
    assert metrics['n'] == 1
    assert metrics['accuracy'] == 0
    assert metrics['log_loss'] == np.inf


@pytest.mark.parametrize('power', [1.0, 1.5, 2.0, 3.0, 4.0])
@pytest.mark.parametrize('offset_shift', [-100.0, 0.0, 100.0])
def test_tweedie_offset_null_deviance_matches_analytic_optimum(con, power, offset_shift):
    from scipy.special import logsumexp

    y = np.array([.01, 10.])
    offset = np.array([0.,4.]) + offset_shift
    intercept = logsumexp(np.log(y)+(1-power)*offset) - logsumexp((2-power)*offset)
    con.execute("CREATE TABLE model AS SELECT '(Intercept)' feature,?::DOUBLE coefficient", [float(intercept)])
    con.execute('CREATE TABLE observations(y DOUBLE, expo DOUBLE)')
    con.executemany('INSERT INTO observations VALUES (?,?)', list(zip(y.tolist(), offset.tolist())))
    metrics = _metrics(con, f"tweedie_evaluate('model','observations','y',power:={power},offset_col:='expo')")
    mu = np.exp(intercept+offset)
    expected = len(y)*mean_tweedie_deviance(y,mu,power=power)
    assert metrics['deviance'] == pytest.approx(expected, rel=1e-9)
    assert metrics['null_deviance'] == pytest.approx(expected, rel=1e-9)
    assert metrics['pseudo_r2'] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize('scores', [(40.,50.),(-800.,-750.),(800.,850.)])
@pytest.mark.parametrize('labels,expected', [((0,1),1.),((1,0),0.),((0,1),.5)])
def test_logistic_auc_preserves_extreme_score_order(con,scores,labels,expected):
    if expected==.5:
        scores=(scores[0],scores[0])
    con.execute("CREATE TABLE model AS SELECT * FROM (VALUES ('(Intercept)',0.),('x',1.))t(feature,coefficient)")
    con.execute('CREATE TABLE observations(x DOUBLE,y DOUBLE)')
    con.executemany('INSERT INTO observations VALUES (?,?)',list(zip(scores,labels)))
    assert _metrics(con,"logit_evaluate('model','observations','y')")['auc']==expected


@pytest.mark.parametrize('family', ['gamma','tweedie','nbinom'])
@pytest.mark.parametrize('n', [1,2])
def test_holdout_dispersion_is_undefined_without_residual_degrees_of_freedom(con,family,n):
    con.execute("CREATE TABLE model AS SELECT * FROM (VALUES ('(Intercept)',ln(2.)),('x',0.))t(feature,coefficient)")
    con.execute(f'CREATE TABLE observations AS SELECT i::DOUBLE x,1.0 y FROM range({n})t(i)')
    metrics=_metrics(con,f"{family}_evaluate('model','observations','y')")
    assert metrics['dispersion'] is None
    assert np.isfinite(metrics['deviance'])


@pytest.mark.parametrize('mean', [1e12,1e16])
@pytest.mark.parametrize('relative_error', [0.,1e-7])
def test_poisson_deviance_is_stable_near_large_means(con,mean,relative_error):
    eta=float(np.log(mean))
    y=float(mean*(1+relative_error))
    con.execute("CREATE TABLE model AS SELECT '(Intercept)' feature,?::DOUBLE coefficient",[eta])
    con.execute('CREATE TABLE observations AS SELECT ?::DOUBLE y',[y])
    with localcontext() as context:
        context.prec=80
        yy,zz=Decimal.from_float(y),Decimal.from_float(eta)
        expected=float(2*(yy*(yy.ln()-zz)-yy+zz.exp()))
    actual=_metrics(con,"poisson_evaluate('model','observations','y')")['deviance']
    assert actual>=0
    assert actual==pytest.approx(expected,rel=1e-7,abs=1e-12)
    residual=con.execute("SELECT deviance_resid FROM poisson_influence('model','observations','y')").fetchone()[0]
    assert residual*residual==pytest.approx(expected,rel=1e-7,abs=1e-12)


@pytest.mark.parametrize('family,power', [('poisson',None),('gamma',None),('nbinom',None),('tweedie',1.),('tweedie',2.)])
def test_offset_null_deviance_preserves_extreme_log_scores(con,family,power):
    y=.1 if family=='nbinom' else 1.
    intercept=np.log(2*y/(1-y)) if family=='nbinom' else 800-np.log(2) if family=='gamma' or power==2 else np.log(2)
    con.execute("CREATE TABLE model AS SELECT '(Intercept)' feature,?::DOUBLE coefficient",[float(intercept)])
    con.execute('CREATE TABLE observations AS SELECT ?::DOUBLE y,expo FROM (VALUES (-800.),(0.))t(expo)',[y])
    extra='' if power is None else f',power:={power}'
    metrics=_metrics(con,f"{family}_evaluate('model','observations','y',offset_col:='expo'{extra})")
    assert np.isfinite(metrics['deviance'])
    assert metrics['null_deviance']==pytest.approx(metrics['deviance'],rel=1e-10)
    assert metrics['pseudo_r2']==pytest.approx(0.,abs=1e-10)


@pytest.mark.parametrize('y', [1e8,1e12,1e16,1e100])
@pytest.mark.parametrize('r', [1,2,10])
@pytest.mark.parametrize('log_ratio', [-1.,0.,1.])
def test_large_count_negative_binomial_likelihood_retains_normalization(con,y,r,log_ratio):
    eta=float(np.log(y)+log_ratio)
    con.execute("CREATE TABLE model AS SELECT '(Intercept)' feature,?::DOUBLE coefficient",[eta])
    con.execute('CREATE TABLE observations AS SELECT ?::DOUBLE y',[y])
    with localcontext() as context:
        context.prec=160
        yy,rr,zz=Decimal.from_float(y),Decimal(r),Decimal.from_float(eta)
        # For integer r, the gamma ratio is an exact short product even at huge y.
        logcomb=sum(((yy+j)/j).ln() for j in range(1,r))
        expected=float(logcomb+rr*(rr.ln()-(rr+zz.exp()).ln())+yy*(zz-(rr+zz.exp()).ln()))
    metrics=_metrics(con,f"nbinom_evaluate('model','observations','y',alpha:={1/r})")
    assert metrics['loglik']==pytest.approx(expected,rel=1e-12,abs=1e-10)
    assert metrics['aic']==pytest.approx(-2*expected+2,rel=1e-12,abs=1e-10)


@pytest.mark.parametrize('mean', [1e8,1e12,1e16])
@pytest.mark.parametrize('relative_error', [0.,1e-7])
def test_large_count_poisson_likelihood_retains_normalization(con,mean,relative_error):
    eta=float(np.log(mean));y=float(mean*(1+relative_error))
    con.execute("CREATE TABLE model AS SELECT '(Intercept)' feature,?::DOUBLE coefficient",[eta])
    con.execute('CREATE TABLE observations AS SELECT ?::DOUBLE y',[y])
    with localcontext() as context:
        context.prec=90
        yy,zz=Decimal.from_float(y),Decimal.from_float(eta)
        log2pi=Decimal('1.837877066409345483560659472811235279722794947275566825634303081')
        logfactorial=(yy+Decimal('.5'))*yy.ln()-yy+log2pi/2+1/(12*yy)-1/(360*yy**3)
        expected=float(yy*zz-zz.exp()-logfactorial)
    metrics=_metrics(con,"poisson_evaluate('model','observations','y')")
    assert metrics['loglik']==pytest.approx(expected,rel=1e-7,abs=1e-10)
    assert metrics['aic']==pytest.approx(-2*expected+2,rel=1e-7,abs=1e-10)
    assert metrics['bic']==pytest.approx(-2*expected,rel=1e-7,abs=1e-10)


@pytest.mark.parametrize('call', ['predict','evaluate'])
@pytest.mark.parametrize('column', ['__reg_rid__','__REG_RID__','__reg_extra'])
@pytest.mark.parametrize('n', [0,2])
def test_multinomial_scoring_rejects_reserved_columns(con,call,column,n):
    con.execute("CREATE TABLE model AS SELECT * FROM (VALUES ('a','(Intercept)',0.),('b','(Intercept)',0.))t(class,feature,coefficient)")
    con.execute(f'CREATE TABLE observations AS SELECT i x,\'a\' y,42 "{column}" FROM range({n})t(i)')
    outcome=",'y'" if call=='evaluate' else ''
    with pytest.raises(duckdb.Error,match='column names beginning with.*__reg_.*reserved'):
        con.execute(f"SELECT * FROM multinom_{call}('model','observations'{outcome})").fetchall()


@pytest.mark.parametrize('call', ['predict','evaluate'])
@pytest.mark.parametrize('reserved_table', ['model','observations'])
def test_multinomial_scoring_rejects_reserved_tables(con,call,reserved_table):
    model='__REG_model' if reserved_table=='model' else 'model'
    table='__REG_observations' if reserved_table=='observations' else 'observations'
    con.execute(f"CREATE TABLE {model} AS SELECT * FROM (VALUES ('a','(Intercept)',0.),('b','(Intercept)',0.))t(class,feature,coefficient)")
    con.execute(f"CREATE TABLE {table} AS SELECT 1.0 x,'a' y")
    outcome=",'y'" if call=='evaluate' else ''
    with pytest.raises(duckdb.Error,match='table names beginning with.*__reg_.*reserved'):
        con.execute(f"SELECT * FROM multinom_{call}('{model}','{table}'{outcome})").fetchall()


@pytest.mark.parametrize('alpha', [1e-310,5e-324])
@pytest.mark.parametrize('eta', [-800.,0.,2.])
def test_subnormal_nb_dispersion_preserves_poisson_scoring_limit(con,alpha,eta):
    con.execute("CREATE TABLE model AS SELECT '(Intercept)' feature,?::DOUBLE coefficient",[eta])
    con.execute('CREATE TABLE observations AS SELECT i::DOUBLE y FROM range(5)t(i)')
    expected=_metrics(con,"poisson_evaluate('model','observations','y')")
    actual=_metrics(con,f"nbinom_evaluate('model','observations','y',alpha:={alpha})")
    for name in ['loglik','deviance','null_deviance','aic','bic']:
        assert np.isfinite(actual[name])
        assert actual[name]==pytest.approx(expected[name],rel=1e-12,abs=1e-11)


@pytest.mark.parametrize('scale', [1e160, 1e-170, 1e-305, 4e307])
def test_linear_metrics_preserve_extreme_finite_residual_units(con, scale):
    con.execute("CREATE TABLE model AS SELECT '(Intercept)' feature,0.0 coefficient UNION ALL SELECT 'x',0.0")
    con.execute('CREATE TABLE observations AS SELECT i::DOUBLE x,(1+i)*? y FROM range(4)t(i)', [scale])
    metrics = _metrics(con, "linreg_evaluate('model','observations','y')")
    expected_ll = -2 * (np.log(2*np.pi) + 2*np.log(scale) + np.log(7.5) + 1)
    assert metrics['rmse']/scale == pytest.approx(np.sqrt(7.5), rel=1e-12)
    assert metrics['mae']/scale == pytest.approx(2.5, rel=1e-12)
    assert metrics['r2'] == pytest.approx(-5.0, abs=1e-12)
    assert metrics['adj_r2'] == pytest.approx(-8.0, abs=1e-12)
    assert metrics['loglik'] == pytest.approx(expected_ll, abs=1e-10)
    assert metrics['aic'] == pytest.approx(-2*expected_ll + 4, abs=1e-10)
    assert metrics['bic'] == pytest.approx(-2*expected_ll + 2*np.log(4), abs=1e-10)


@pytest.mark.parametrize('scale', [1e160, 1e-170, 1e-305])
def test_exact_linear_fit_retains_r_squared_at_extreme_units(con, scale):
    con.execute("CREATE TABLE model AS SELECT '(Intercept)' feature,0.0 coefficient UNION ALL SELECT 'x',1.0")
    con.execute('CREATE TABLE observations AS SELECT i*? x,i*? y FROM range(4)t(i)', [scale, scale])
    metrics = _metrics(con, "linreg_evaluate('model','observations','y')")
    assert metrics['rmse'] == 0.0
    assert metrics['r2'] == 1.0
    assert metrics['adj_r2'] == 1.0
    assert metrics['loglik'] == np.inf


@pytest.mark.parametrize('family', ['linreg','logit','poisson','gamma','tweedie','nbinom'])
@pytest.mark.parametrize('with_offset', [False,True])
def test_scoring_cancels_overflowing_products_before_restoring_units(con, family, with_offset):
    expected = .5 if family == 'logit' else 0. if family == 'linreg' else 1.
    outcome = 0. if family == 'logit' else expected
    offset = ', -1e308 expo' if with_offset else ''
    argument = ",offset_col:='expo'" if with_offset else ''
    con.execute(f'CREATE TABLE cancel_data AS SELECT 2.0 x,2.0 z,{outcome} y{offset} FROM range(2)')
    con.execute('CREATE TABLE cancel_model(feature VARCHAR,coefficient DOUBLE)')
    con.executemany('INSERT INTO cancel_model VALUES (?,?)', [('(Intercept)',1e308 if with_offset else 0.),('x',1e308),('z',-1e308)])
    field = 'prob' if family == 'logit' else 'prediction'
    actual = np.array(con.execute(f"SELECT {field} FROM {family}_predict('cancel_model','cancel_data'{argument})").fetchall()).ravel()
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    metrics = _metrics(con,f"{family}_evaluate('cancel_model','cancel_data','y'{argument})")
    assert metrics['n'] == 2
    assert metrics['log_loss' if family == 'logit' else 'rmse'] == pytest.approx(np.log(2.) if family == 'logit' else 0.)


def test_multinomial_scoring_cancels_overflowing_products(con):
    con.execute("CREATE TABLE cancel_data AS SELECT 2.0 x,2.0 z,'b' y FROM range(2)")
    con.execute("CREATE TABLE cancel_model AS SELECT * FROM (VALUES ('a','(Intercept)',0.),('a','x',0.),('a','z',0.),('b','(Intercept)',0.),('b','x',1e308),('b','z',-1e308))t(class,feature,coefficient)")
    probabilities = con.execute("SELECT probs FROM multinom_predict('cancel_model','cancel_data')").fetchall()
    assert all(p == {'a':.5,'b':.5} for p, in probabilities)
    assert _metrics(con,"multinom_evaluate('cancel_model','cancel_data','y')")['log_loss'] == pytest.approx(np.log(2.))


@pytest.mark.parametrize('outcome',[1e-310,1e-170,1e170,1e307])
@pytest.mark.parametrize('alpha',[1e-10,1.,1e100])
@pytest.mark.parametrize('offset_shift',[-1000.,0.,1000.])
def test_negative_binomial_offset_null_model_uses_unclipped_score(con, outcome, alpha, offset_shift):
    # Equal outcomes with exposures 1 and 2 give the quadratic
    # 4*a*r^2+3*(1-a)*r-2=0, where a=alpha*y and exp(intercept)=r*y.
    log_a = np.log(alpha)+np.log(outcome)
    if log_a >= 0:
        inv_a = np.exp(-log_a)
        ratio = (3*(1-inv_a)+np.sqrt(9*(1-inv_a)**2+32*inv_a))/8
    else:
        a = np.exp(log_a)
        ratio = 4/(3*(1-a)+np.sqrt(9*(1-a)**2+32*a))
    intercept = np.log(outcome)+np.log(ratio)-offset_shift
    con.execute('CREATE TABLE nb_null_data(y DOUBLE,expo DOUBLE)')
    con.executemany('INSERT INTO nb_null_data VALUES (?,?)',[(outcome,offset_shift),(outcome,offset_shift+np.log(2.))])
    con.execute("CREATE TABLE nb_null_model AS SELECT '(Intercept)' feature,?::DOUBLE coefficient",[intercept])
    metrics = _metrics(con,f"nbinom_evaluate('nb_null_model','nb_null_data','y',offset_col:='expo',alpha:={alpha})")
    assert np.isfinite([metrics['deviance'],metrics['null_deviance'],metrics['pseudo_r2']]).all()
    assert metrics['null_deviance'] == pytest.approx(metrics['deviance'],rel=1e-7,abs=0.)
    assert metrics['pseudo_r2'] == pytest.approx(0.,abs=1e-7)


@pytest.mark.parametrize('coordinate', [1., 2., 1e308])
@pytest.mark.parametrize('intercept', [1., 1e-170])
def test_scoring_keeps_small_terms_when_large_products_cancel(con, coordinate, intercept):
    con.execute("CREATE TABLE cancelling_model AS SELECT * FROM (VALUES ('(Intercept)',?),('x',1e308),('z',-1e308))t(feature,coefficient)", [intercept])
    con.execute('CREATE TABLE cancelling_data AS SELECT ?::DOUBLE x,?::DOUBLE z,?::DOUBLE y', [coordinate, coordinate, intercept])
    prediction = con.execute("SELECT prediction FROM linreg_predict('cancelling_model','cancelling_data')").fetchone()[0]
    assert prediction == pytest.approx(intercept, rel=1e-12, abs=0.)
    assert _metrics(con,"linreg_evaluate('cancelling_model','cancelling_data','y')")['rmse'] == 0.


def test_multinomial_scoring_keeps_intercept_after_overflowing_slope_cancellation(con):
    con.execute("CREATE TABLE cancelling_multinomial AS SELECT * FROM (VALUES ('a','(Intercept)',0.),('a','x',0.),('a','z',0.),('b','(Intercept)',1.),('b','x',1e308),('b','z',-1e308))t(class,feature,coefficient)")
    con.execute('CREATE TABLE cancelling_multidata AS SELECT 2.0 x,2.0 z')
    probabilities = con.execute("SELECT probs FROM multinom_predict('cancelling_multinomial','cancelling_multidata')").fetchone()[0]
    assert probabilities['b'] == pytest.approx(1/(1+np.exp(-1)), abs=1e-12)


@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_adjusted_r_squared_requires_positive_residual_degrees_of_freedom(con, n):
    con.execute("CREATE TABLE model AS SELECT * FROM (VALUES ('(Intercept)',0.),('x',0.),('z',0.))t(feature,coefficient)")
    con.execute(f'CREATE TABLE observations AS SELECT i::DOUBLE x,0.0 z,i+1.0 y FROM range({n})t(i)')
    # Excluded rows must not create apparent residual degrees of freedom.
    con.execute('INSERT INTO observations VALUES (NULL,0,5),(5,0,NULL)')
    metrics = _metrics(con, "linreg_evaluate('model','observations','y')")
    y = np.arange(1., n+1.)
    assert metrics['n'] == n
    assert metrics['rmse'] == pytest.approx(np.sqrt(np.mean(y*y)))
    assert metrics['mae'] == pytest.approx(y.mean())
    assert np.isfinite(metrics['loglik'])
    if n > 1:
        r2 = 1 - np.sum(y*y)/np.sum((y-y.mean())**2)
        assert metrics['r2'] == pytest.approx(r2)
    if n <= 3:
        assert metrics['adj_r2'] is None
    else:
        assert metrics['adj_r2'] == pytest.approx(1-(1-r2)*(n-1)/(n-3))


@pytest.mark.parametrize('pairs', [2, 4])
@pytest.mark.parametrize('family,coordinate', [('linreg',1e308),('linreg',1e138),('logit',1e308),('poisson',1e308)])
def test_scoring_preserves_small_products_when_finite_product_sum_overflows(con, pairs, family, coordinate):
    coefficients = [('(Intercept)',0.)] + [(f'p{i}',1e308) for i in range(pairs)] + [(f'n{i}',-1e308) for i in range(pairs)] + [('small',1e-308)]
    con.execute('CREATE TABLE model(feature VARCHAR,coefficient DOUBLE)')
    con.executemany('INSERT INTO model VALUES (?,?)',coefficients)
    columns = ','.join(f'1.0 {feature}' for feature,_ in coefficients[1:-1])
    con.execute(f'CREATE TABLE observations AS SELECT {columns},?::DOUBLE small',[coordinate])
    expected = coordinate*1e-308
    if family == 'logit':
        expected = 1/(1+np.exp(-expected))
    elif family == 'poisson':
        expected = np.exp(expected)
    column = 'prob' if family == 'logit' else 'prediction'
    prediction = con.execute(f"SELECT {column} FROM {family}_predict('model','observations')").fetchone()[0]
    assert prediction == pytest.approx(expected,rel=1e-12,abs=0.)


@pytest.mark.parametrize('score', [-710., -720., -740.])
def test_logistic_predictions_preserve_representable_negative_tail_probabilities(con, score):
    con.execute("CREATE TABLE model AS SELECT * FROM (VALUES ('(Intercept)',0.),('x',1.))t(feature,coefficient)")
    con.execute('CREATE TABLE observations AS SELECT ?::DOUBLE x',[score])
    expected = np.exp(score)/(1+np.exp(score))
    for threshold,predicted_class in [(expected/2,True),(expected*2,False)]:
        probability,classification = con.execute("SELECT prob,pred FROM logit_predict('model','observations',threshold:=?)",[threshold]).fetchone()
        assert probability > 0.
        assert probability == pytest.approx(expected,rel=1e-12,abs=0.)
        assert classification == predicted_class


@pytest.mark.parametrize('family', ['logit','multinom'])
@pytest.mark.parametrize('n', [1,2,3,8])
@pytest.mark.parametrize('all_errors', [False,True])
def test_mean_log_loss_preserves_finite_extreme_losses_when_rows_repeat(con, family, n, all_errors):
    if family == 'logit':
        con.execute("CREATE TABLE model AS SELECT * FROM (VALUES ('(Intercept)',0.),('x',1.))t(feature,coefficient)")
        label = '0.0'
    else:
        con.execute("CREATE TABLE model AS SELECT * FROM (VALUES ('a','(Intercept)',0.),('a','x',0.),('b','(Intercept)',0.),('b','x',1.))t(class,feature,coefficient)")
        label = "'a'"
    score = '1e308::DOUBLE' if all_errors else 'CASE WHEN i=0 THEN 1e308 ELSE 0.0 END'
    con.execute(f'CREATE TABLE observations AS SELECT {score} x,{label} y FROM range({n})t(i)')
    loss = con.execute(f"SELECT log_loss FROM {family}_evaluate('model','observations','y')").fetchone()[0]
    assert np.isfinite(loss)
    assert loss/1e308 == pytest.approx(1. if all_errors else 1/n,rel=1e-12)


@pytest.mark.parametrize('baseline', [1e16, -1e16, 1e100, -1e100])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('pattern', [[0,1,2], [0,1,1]])
def test_linear_r2_centers_large_outcome_means(con, baseline, reverse, pattern):
    step = abs(np.spacing(baseline))
    values = [baseline+step*pattern[i % 3] for i in range(6)]
    if reverse:
        values.reverse()
    con.execute('CREATE TABLE mean_rows(x DOUBLE,y DOUBLE)')
    con.executemany('INSERT INTO mean_rows VALUES (0,?)', [(y,) for y in values])
    con.execute("CREATE TABLE mean_model AS SELECT * FROM linreg_fit('mean_rows','y')")
    prediction = con.execute("SELECT coefficient FROM mean_model WHERE feature='(Intercept)'").fetchone()[0]
    # Decimal computes the exact mean of the input doubles independently of
    # both DuckDB's AVG and the centering strategy used by the implementation.
    with localcontext() as context:
        context.prec = 150
        exact = [Decimal.from_float(y) for y in values]
        mean = sum(exact)/len(exact)
        sse = sum((y-Decimal.from_float(prediction))**2 for y in exact)
        sst = sum((y-mean)**2 for y in exact)
        expected = float(1-sse/sst)
    metrics = _metrics(con, "linreg_evaluate('mean_model','mean_rows','y')")
    assert metrics['r2'] == pytest.approx(expected, abs=1e-14)
    assert metrics['adj_r2'] == pytest.approx(1-(1-expected)*5/4, abs=1e-14)


def test_negative_binomial_offset_null_score_retains_overflowing_mean_ratio(con):
    con.execute("CREATE TABLE nb_null_model AS SELECT '(Intercept)' feature,ln(1e308) coefficient UNION ALL SELECT 'x',0.")
    con.execute('CREATE TABLE nb_null_rows AS SELECT * FROM (VALUES (0.,1e308,0.),(1.,1e308,2.))t(x,y,o)')
    # At this scale NB(alpha=1) agrees with the Gamma limit to DOUBLE
    # precision. Its offset null mean has an independent closed-form optimum.
    offset = np.array([0., 2.])
    log_mean = float(np.log(1e308))
    null_intercept = log_mean+np.log(np.exp(-offset).mean())
    null_log_ratio = log_mean-null_intercept-offset
    model_log_ratio = -offset
    expected_null = 2*np.sum(np.exp(null_log_ratio)-1-null_log_ratio)
    expected_model = 2*np.sum(np.exp(model_log_ratio)-1-model_log_ratio)
    actual = _metrics(con, "nbinom_evaluate('nb_null_model','nb_null_rows','y',offset_col:='o')")
    assert actual['null_deviance'] == pytest.approx(expected_null, rel=1e-12)
    assert actual['pseudo_r2'] == pytest.approx(1-expected_model/expected_null, rel=1e-12)


@pytest.mark.parametrize('outcome,alpha', [(1e308,1e-310), (1e-300,1e-310), (1e308,1e308)])
def test_nb_offset_null_root_is_invariant_to_score_accumulation_order(con, outcome, alpha):
    with localcontext() as context:
        context.prec = 800
        yy = Decimal.from_float(outcome)
        scaled_alpha = Decimal.from_float(alpha)*yy
        offsets = [Decimal.from_float(-.1), Decimal.from_float(.1)]
        low, high = offsets
        for _ in range(120):
            mid = (low+high)/2
            means = [(mid+offset).exp() for offset in offsets]
            score = sum((1-q)/(1+scaled_alpha*q) for q in means)
            if score>0:
                low = mid
            else:
                high = mid
        root = (low+high)/2
        inverse_alpha = 1/scaled_alpha
        halfdev = sum(-(root+offset)-(1+inverse_alpha)*
                      ((1+inverse_alpha)/((root+offset).exp()+inverse_alpha)).ln()
                      for offset in offsets)
        expected_null = float(100*yy*halfdev)  # 50 copies of each offset, 2*halfdev
    con.execute("CREATE TABLE order_model AS SELECT '(Intercept)' feature,ln(?::DOUBLE) coefficient", [outcome])
    results = []
    for offsets in [[-.1]*50+[.1]*50, [.1]*50+[-.1]*50, [-.1,.1]*50]:
        con.execute('CREATE OR REPLACE TABLE order_rows(y DOUBLE,o DOUBLE)')
        con.executemany('INSERT INTO order_rows VALUES (?,?)', [(outcome, offset) for offset in offsets])
        result = _metrics(con, f"nbinom_evaluate('order_model','order_rows','y',alpha:={alpha},offset_col:='o')")
        assert result['null_deviance'] == pytest.approx(expected_null, rel=1e-9, abs=5e-324)
        assert result['pseudo_r2'] == pytest.approx(1-result['deviance']/expected_null, rel=1e-8, abs=1e-10)
        results.append(result['null_deviance'])
    np.testing.assert_allclose(results, results[0], rtol=1e-12, atol=5e-324)
