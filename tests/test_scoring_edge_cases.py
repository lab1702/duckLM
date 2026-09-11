"""Regression coverage for scoring boundaries and generated categorical SQL."""

from pathlib import Path

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
