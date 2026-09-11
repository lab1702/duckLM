"""Regression coverage for stable scaling and batch fit row preparation."""

from pathlib import Path

import duckdb
import numpy as np
import pytest


@pytest.fixture(scope="module")
def con():
    connection = duckdb.connect()
    connection.execute((Path(__file__).resolve().parents[1] / "regression_macros.sql").read_text())
    yield connection
    connection.close()


@pytest.mark.parametrize("shift_outcome", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_ridge_is_stable_when_mean_dwarfs_spread(con, shift_outcome, weighted):
    # A standardized univariate ridge with l2=1 halves the ordinary slope.
    # The feature and outcome can both have means much larger than their SD.
    yshift = 1e9 if shift_outcome else 0
    con.execute(f"""
        CREATE OR REPLACE TABLE large_mean AS
        SELECT 1e9+i AS x, {yshift}+1+2*i AS y,
               {'i+1' if weighted else '1'}::DOUBLE AS w
        FROM range(6) t(i)
    """)
    model = dict(con.execute("""
        SELECT feature, coefficient FROM linreg_fit(
            'large_mean', 'y', weights_col := 'w', l2 := 1.0, max_iter := 20)
    """).fetchall())
    weights = np.arange(1, 7) if weighted else np.ones(6)
    mean_i = np.average(np.arange(6), weights=weights)
    assert model["x"] == pytest.approx(1.0, abs=2e-8)
    assert model["(Intercept)"] == pytest.approx(yshift + 1 + mean_i - 1e9, abs=2e-6)


def test_zero_weight_rows_do_not_change_constant_features(con):
    con.execute("""
        CREATE OR REPLACE TABLE zero_weight AS
        SELECT * FROM (VALUES (5.,1.,1.), (5.,2.,1.), (5.,4.,1.), (9.,99.,0.)) t(x,y,w)
    """)
    model = dict(con.execute("""
        SELECT feature, coefficient FROM linreg_fit(
            'zero_weight', 'y', weights_col := 'w', max_iter := 100)
    """).fetchall())
    assert model["x"] == 0.0
    assert model["(Intercept)"] == pytest.approx(7 / 3, abs=1e-9)


BATCH_CALLS = [
    "multinom_fit('{table}', 'y', l2 := 0.2, max_iter := 200)",
    "cv_l2('{table}', 'y', 'linear', [0.2, 0.8], k := 3, max_iter := 100)",
    "nbinom_dispersion('{table}', 'y', [0.5, 1.0], max_iter := 200)",
]


def batch_values(con, call, table):
    rows = con.execute("SELECT * FROM " + call.format(table=table) + " ORDER BY ALL").fetchall()
    return {tuple(row[:-1]): row[-1] for row in rows}


@pytest.mark.parametrize("call", BATCH_CALLS)
def test_batch_fit_drops_incomplete_rows_before_scaling_and_counting(con, call):
    # Extreme values in incomplete rows must affect neither the optimizer's
    # scaling/counts nor (for multinomial regression) its set of classes.
    con.execute("""
        CREATE OR REPLACE TABLE partial_rows AS
        SELECT i::DOUBLE AS x, ((i * 7) % 11)::DOUBLE AS z, (i % 3)::DOUBLE AS y
        FROM range(24) t(i)
        UNION ALL SELECT NULL, 1e9, 99
        UNION ALL SELECT 1e9, NULL, 99
        UNION ALL SELECT 1e9, 1e9, NULL
    """)
    # Put a missing row between complete rows too, so CV must assign folds
    # over complete observations rather than leave holes in its row numbers.
    con.execute("""
        CREATE OR REPLACE TABLE partial_rows AS
        SELECT * FROM partial_rows ORDER BY coalesce(x, 8.5), coalesce(z, 0)
    """)
    con.execute("""
        CREATE OR REPLACE TABLE complete_rows AS
        SELECT * FROM partial_rows WHERE x IS NOT NULL AND z IS NOT NULL AND y IS NOT NULL
    """)
    observed = batch_values(con, call, "partial_rows")
    expected = batch_values(con, call, "complete_rows")
    assert observed.keys() == expected.keys()
    for key in observed:
        assert observed[key] == pytest.approx(expected[key], abs=1e-8)


@pytest.mark.parametrize("call", BATCH_CALLS)
def test_batch_fit_rejects_entirely_null_features(con, call):
    con.execute("""
        CREATE OR REPLACE TABLE null_feature AS
        SELECT i::DOUBLE AS x, NULL::DOUBLE AS z, (i%3)::DOUBLE AS y FROM range(12) t(i)
    """)
    with pytest.raises(duckdb.Error, match='feature column\\(s\\) entirely NULL: "z"'):
        batch_values(con, call, "null_feature")


@pytest.mark.parametrize("call", BATCH_CALLS)
def test_batch_fit_rejects_no_complete_observations(con, call):
    con.execute("""
        CREATE OR REPLACE TABLE no_complete AS
        SELECT * FROM (VALUES (1.0, NULL::DOUBLE, 0.), (NULL::DOUBLE, 2.0, 1.)) t(x,z,y)
    """)
    with pytest.raises(duckdb.Error, match="no complete"):
        batch_values(con, call, "no_complete")


@pytest.mark.parametrize("call", BATCH_CALLS)
def test_batch_fit_preserves_real_rid_feature(con, call):
    con.execute("""
        CREATE OR REPLACE TABLE rid_feature AS
        SELECT i::DOUBLE AS rid, (i % 3)::DOUBLE AS y FROM range(24) t(i)
    """)
    con.execute("CREATE OR REPLACE TABLE renamed_feature AS SELECT rid AS x, y FROM rid_feature")
    actual = batch_values(con, call, "rid_feature")
    expected = batch_values(con, call, "renamed_feature")
    actual = {tuple("x" if part == "rid" else part for part in key): value for key, value in actual.items()}
    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key] == pytest.approx(expected[key], abs=1e-8)


@pytest.mark.parametrize("call", BATCH_CALLS)
def test_batch_fit_rejects_reserved_internal_columns(con, call):
    con.execute("""
        CREATE OR REPLACE TABLE reserved_feature AS
        SELECT i::DOUBLE AS __reg_rid__, (i % 3)::DOUBLE AS y FROM range(12) t(i)
    """)
    with pytest.raises(duckdb.Error, match="reserved"):
        batch_values(con, call, "reserved_feature")
