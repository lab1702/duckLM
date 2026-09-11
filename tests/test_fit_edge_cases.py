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


@pytest.mark.parametrize('family',['logit','poisson','gamma','tweedie','nbinom'])
@pytest.mark.parametrize('missing',['x','expo','wt'])
def test_outcome_validation_ignores_incomplete_training_rows(con,family,missing):
    y='i%2' if family=='logit' else 'exp(.2+.1*i)'
    con.execute(f'CREATE OR REPLACE TABLE retained_rows AS SELECT i::DOUBLE x,({y})::DOUBLE y,0.0 AS expo,1.0 wt FROM range(10)t(i)')
    con.execute('CREATE OR REPLACE TABLE with_invalid_dropped_row AS SELECT * FROM retained_rows')
    row={'x':'0.0','y':'-1.0','expo':'0.0','wt':'1.0'};row[missing]='NULL'
    con.execute('INSERT INTO with_invalid_dropped_row VALUES ('+','.join(row.values())+')')
    fitted=[]
    for table in ['retained_rows','with_invalid_dropped_row']:
        fitted.append(dict(con.execute(f"SELECT * FROM {family}_fit('{table}','y',offset_col:='expo',weights_col:='wt')").fetchall()))
    assert fitted[1]==pytest.approx(fitted[0],abs=1e-10)


@pytest.mark.parametrize('family,parameter',[('nbinom','alpha'),('tweedie','power')])
@pytest.mark.parametrize('value',['NULL',"'NaN'::DOUBLE","'Infinity'::DOUBLE"])
def test_fit_rejects_missing_or_nonfinite_distribution_parameters(con,family,parameter,value):
    con.execute('CREATE OR REPLACE TABLE distribution_input AS SELECT i::DOUBLE x,1.0+i%3 y FROM range(10)t(i)')
    with pytest.raises(duckdb.Error,match=parameter+'.*finite'):
        con.execute(f"SELECT * FROM {family}_fit('distribution_input','y',{parameter}:={value})").fetchall()


@pytest.mark.parametrize('offset_col,weights_col', [('w','w'),('y','w'),('w','y'),('y','y')])
@pytest.mark.parametrize('family', ['linreg','poisson','gamma','tweedie','nbinom'])
def test_fit_accepts_shared_outcome_offset_and_weight_columns(con, family, offset_col, weights_col):
    con.execute('CREATE OR REPLACE TABLE shared_roles AS SELECT i::DOUBLE/10 x,1.0+i%2 w,exp(.3*i/10) y FROM range(10)t(i)')
    # A distinct column for each role expresses the identical statistical problem.
    con.execute(f'CREATE OR REPLACE TABLE separate_roles AS SELECT x,y,{offset_col} offset_value,{weights_col} wt'
                + (',w' if offset_col == weights_col == 'y' else '') + ' FROM shared_roles')
    actual = dict(con.execute(f"SELECT * FROM {family}_fit('shared_roles','y',offset_col:='{offset_col}',weights_col:='{weights_col}')").fetchall())
    expected = dict(con.execute(f"SELECT * FROM {family}_fit('separate_roles','y',offset_col:='offset_value',weights_col:='wt')").fetchall())
    assert actual == pytest.approx(expected, rel=1e-8, abs=1e-8)


@pytest.mark.parametrize('family', ['linreg','logit','poisson','gamma','tweedie','nbinom','multinom'])
@pytest.mark.parametrize('penalty', ['l1','l2'])
@pytest.mark.parametrize('value', ['NULL',"'NaN'::DOUBLE","'Infinity'::DOUBLE"])
def test_fit_rejects_missing_and_nonfinite_penalties(con,family,penalty,value):
    outcome='i%2' if family=='logit' else '1.0+2*i'
    con.execute(f'CREATE OR REPLACE TABLE penalty_data AS SELECT i::DOUBLE x,{outcome} y FROM range(6)t(i)')
    with pytest.raises(duckdb.Error,match=penalty+' must be.*finite'):
        con.execute(f"SELECT * FROM {family}_fit('penalty_data','y',{penalty}:={value})").fetchall()


@pytest.mark.parametrize('family', ['poisson','gamma','tweedie','nbinom'])
@pytest.mark.parametrize('solver', ['auto','irls'])
@pytest.mark.parametrize('offset', [-40.,-100.])
@pytest.mark.parametrize('l1', [0.,.1])
def test_irls_does_not_false_converge_after_negative_offset_overshoot(con,family,solver,offset,l1):
    con.execute('CREATE OR REPLACE TABLE offset_base AS SELECT i::DOUBLE x,1.0+i y,0.0 expo FROM range(4)t(i)')
    con.execute(f'CREATE OR REPLACE TABLE offset_shifted AS SELECT x,y,{offset} expo FROM offset_base')
    coefficients=[]
    for table in ['offset_base','offset_shifted']:
        con.execute(f"CREATE OR REPLACE TABLE offset_model AS SELECT * FROM {family}_fit('{table}','y',offset_col:='expo',solver:='{solver}',l1:={l1},max_iter:=300)")
        coefficients.append(dict(con.execute('SELECT * FROM offset_model').fetchall()))
        predictions=con.execute(f"SELECT prediction FROM {family}_predict('offset_model','{table}',offset_col:='expo')").fetchnumpy()['prediction']
        assert np.isfinite(predictions).all()
    coefficients[1]['(Intercept)']+=offset
    assert coefficients[1]==pytest.approx(coefficients[0],rel=1e-7,abs=1e-7)


@pytest.mark.parametrize('family', ['gamma','tweedie','nbinom'])
def test_zero_weight_outlier_does_not_limit_irls_steps(con,family):
    con.execute('CREATE OR REPLACE TABLE weighted_offsets AS SELECT i::DOUBLE x,1.0+i y,-40.0 expo,1.0 wt FROM range(4)t(i)')
    call=f"SELECT * FROM {family}_fit('weighted_offsets','y',offset_col:='expo',weights_col:='wt')"
    expected=dict(con.execute(call).fetchall())
    con.execute('INSERT INTO weighted_offsets VALUES (1e100,1.,-40.,0.)')
    assert dict(con.execute(call).fetchall())==pytest.approx(expected,rel=1e-10,abs=1e-10)
