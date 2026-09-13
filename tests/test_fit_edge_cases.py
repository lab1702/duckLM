"""Regression coverage for stable scaling and batch fit row preparation."""

from pathlib import Path

import duckdb
import numpy as np
import pytest


def test_multinomial_high_leverage_logits_match_stable_likelihood():
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    counts = [(-1., 0, 7), (-1., 1, 3), (-1., 2, 1),
              (0., 0, 4), (0., 1, 4), (0., 2, 4),
              (1., 0, 1), (1., 1, 3), (1., 2, 7)]
    observations = [(x, cl) for x, cl, n in counts for _ in range(n)] + [(1000., 2)]
    x = np.column_stack([np.ones(len(observations)), [row[0] for row in observations]])
    y = np.array([row[1] for row in observations])

    def objective(beta):
        logits = np.column_stack([np.zeros(len(y)), x @ beta.reshape(2, 2).T])
        log_prob = logits - logsumexp(logits, axis=1, keepdims=True)
        residual = np.exp(log_prob) - np.eye(3)[y]
        return -log_prob[np.arange(len(y)), y].sum(), (residual[:, 1:].T @ x).ravel()

    reference = minimize(objective, np.zeros(4), jac=True, method='BFGS', options={'gtol': 1e-9})
    assert np.max(np.abs(reference.jac)) < 1e-6
    with duckdb.connect() as connection:
        connection.execute((Path(__file__).resolve().parents[1] / 'regression_macros.sql').read_text())
        connection.execute('CREATE TABLE large_logits(x DOUBLE,y VARCHAR)')
        connection.executemany('INSERT INTO large_logits VALUES (?,?)', [(v, str(cl)) for v, cl in observations])
        rows = connection.execute("SELECT class,feature,coefficient FROM multinom_fit('large_logits','y')").fetchall()
    coefficients = {(cl, feature): value for cl, feature, value in rows}
    actual = np.array([coefficients[cl, feature] for cl in ['1', '2'] for feature in ['(Intercept)', 'x']])
    # The default iteration cap can leave a small optimization error on this
    # ill-conditioned sample, but class probabilities must use the true logits.
    assert objective(actual)[0] <= reference.fun + 1e-5
    assert np.max(np.abs(objective(actual)[1])) < .01


@pytest.mark.parametrize('solver', ['auto', 'gd'])
@pytest.mark.parametrize('penalty', [1e8, 1e308])
def test_large_ridge_penalties_preserve_the_free_logistic_intercept(con, solver, penalty):
    con.execute('CREATE OR REPLACE TABLE ridge_intercept AS SELECT (i%3)::DOUBLE x,(i%5=0)::DOUBLE y FROM range(60)t(i)')
    coefficients = dict(con.execute(f"SELECT * FROM logit_fit('ridge_intercept','y',l2:={penalty},solver:='{solver}')").fetchall())
    assert coefficients['(Intercept)'] == pytest.approx(np.log(.2/.8),abs=1e-8)
    assert coefficients['x'] == pytest.approx(0.,abs=1e-9)


@pytest.mark.parametrize('penalty', [1e8, 1e308])
def test_large_ridge_penalties_preserve_multinomial_class_frequencies(con, penalty):
    con.execute("CREATE OR REPLACE TABLE ridge_classes AS SELECT (i%3)::DOUBLE x,CASE WHEN i%5=0 THEN 'a' WHEN i%5 IN (1,2) THEN 'b' ELSE 'c' END y FROM range(60)t(i)")
    rows = con.execute(f"SELECT class,feature,coefficient FROM multinom_fit('ridge_classes','y',l2:={penalty})").fetchall()
    for label, feature, coefficient in rows:
        expected = np.log(2.) if label != 'a' and feature == '(Intercept)' else 0.
        assert coefficient == pytest.approx(expected,abs=1e-8)


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


@pytest.mark.parametrize('solver', ['auto', 'irls'])
def test_irls_damping_does_not_false_converge_with_tiny_weight_outlier(con, solver):
    con.execute('CREATE OR REPLACE TABLE tiny_weight(x DOUBLE,y DOUBLE,w DOUBLE)')
    rows = [(-1, 0, 1)] * 3 + [(-1, 1, 1)] + [(1, 1, 1)] * 3 + [(1, 0, 1)]
    con.executemany('INSERT INTO tiny_weight VALUES (?,?,?)', rows + [(1e100, 1, 1e-250)])
    call = f"SELECT * FROM logit_fit('tiny_weight','y',weights_col:='w',solver:='{solver}')"
    if solver == 'irls':
        with pytest.raises(duckdb.Error, match='irls solver did not converge'):
            con.execute(call).fetchall()
    else:
        coefficients = dict(con.execute(call).fetchall())
        assert coefficients['x'] == pytest.approx(np.log(3), rel=1e-7)
        assert coefficients['(Intercept)'] == pytest.approx(0, abs=1e-8)


@pytest.mark.parametrize('family', ['linreg', 'logit', 'poisson', 'gamma', 'tweedie', 'nbinom'])
@pytest.mark.parametrize('weight', ['NaN', 'Infinity', '-Infinity'])
def test_fit_rejects_nonfinite_sample_weights(con, family, weight):
    outcome = 'i%2' if family == 'logit' else '1.0+i'
    con.execute(f"""
        CREATE OR REPLACE TABLE invalid_weights AS
        SELECT i::DOUBLE x, {outcome} y,
               CASE WHEN i=1 THEN '{weight}'::DOUBLE ELSE 1.0 END w
        FROM range(8)t(i)
    """)
    with pytest.raises(duckdb.Error, match='weights must be finite'):
        con.execute(f"SELECT * FROM {family}_fit('invalid_weights', 'y', weights_col:='w', max_iter:=3)").fetchall()


@pytest.mark.parametrize('solver', ['auto','irls','gd'])
def test_gamma_fit_preserves_representable_means_below_exp_minus_700(con, solver):
    con.execute('''CREATE OR REPLACE TABLE gamma_tail AS SELECT (i%2)::DOUBLE x,
        CASE WHEN i%2=0 THEN 1e-310 ELSE 1.0 END y FROM range(4)t(i)''')
    con.execute(f"CREATE OR REPLACE TABLE gamma_tail_model AS SELECT * FROM gamma_fit('gamma_tail','y',solver:='{solver}',max_iter:=2000)")
    beta = con.execute('SELECT coefficient FROM gamma_tail_model').fetchnumpy()['coefficient']
    np.testing.assert_allclose(beta,[np.log(1e-310),-np.log(1e-310)],rtol=1e-9)
    predictions = con.execute("SELECT prediction FROM gamma_predict('gamma_tail_model','gamma_tail')").fetchnumpy()['prediction']
    np.testing.assert_allclose(predictions,[1e-310,1.,1e-310,1.],rtol=1e-7,atol=0)


@pytest.mark.parametrize('family', ['poisson','gamma','nbinom'])
@pytest.mark.parametrize('offset', [-1e20,1e20])
@pytest.mark.parametrize('solver', ['auto','irls','gd'])
def test_log_link_fit_centers_large_common_offsets(con, family, offset, solver):
    con.execute('CREATE OR REPLACE TABLE common_offset AS SELECT i::DOUBLE x,exp(.3*i) y,?::DOUBLE o FROM range(6)t(i)',[offset])
    con.execute(f"CREATE OR REPLACE TABLE common_model AS SELECT * FROM {family}_fit('common_offset','y',offset_col:='o',solver:='{solver}',max_iter:=2000)")
    beta = con.execute('SELECT coefficient FROM common_model').fetchnumpy()['coefficient']
    np.testing.assert_allclose(beta,[-offset,.3],rtol=1e-7)
    prediction = con.execute(f"SELECT prediction FROM {family}_predict('common_model','common_offset',offset_col:='o')").fetchnumpy()['prediction']
    np.testing.assert_allclose(prediction,np.exp(.3*np.arange(6)),rtol=1e-7)


@pytest.mark.parametrize('solver', ['auto','irls','gd'])
@pytest.mark.parametrize('offset_scale', [-1000.,1000.])
def test_gamma_fit_initialization_survives_varying_extreme_offsets(con, solver, offset_scale):
    con.execute('CREATE OR REPLACE TABLE varying_offset AS SELECT i::DOUBLE x,1.0 y,i*?::DOUBLE o FROM range(-1,2)t(i)',[offset_scale])
    con.execute(f"CREATE OR REPLACE TABLE varying_model AS SELECT * FROM gamma_fit('varying_offset','y',offset_col:='o',solver:='{solver}',max_iter:=5000)")
    beta = con.execute('SELECT coefficient FROM varying_model').fetchnumpy()['coefficient']
    np.testing.assert_allclose(beta,[0.,-offset_scale],rtol=1e-9,atol=1e-6)
    prediction = con.execute("SELECT prediction FROM gamma_predict('varying_model','varying_offset',offset_col:='o')").fetchnumpy()['prediction']
    np.testing.assert_allclose(prediction,np.ones(3),rtol=1e-6)


@pytest.mark.parametrize('solver', ['auto','irls','gd'])
@pytest.mark.parametrize('power', [1.5,2.,3.])
def test_tweedie_fit_retains_extreme_offset_scores(con, solver, power):
    con.execute('CREATE OR REPLACE TABLE tweedie_tail AS SELECT x,o,exp(o) y FROM (VALUES(-1.),(1.))a(x),(VALUES(-720.),(0.))b(o)')
    con.execute(f"CREATE OR REPLACE TABLE tweedie_tail_model AS SELECT * FROM tweedie_fit('tweedie_tail','y',power:={power},offset_col:='o',solver:='{solver}',max_iter:=2000)")
    beta = con.execute('SELECT coefficient FROM tweedie_tail_model').fetchnumpy()['coefficient']
    np.testing.assert_allclose(beta,[0.,0.],atol=1e-6)
    rows = con.execute("SELECT y,prediction FROM tweedie_predict('tweedie_tail_model','tweedie_tail',offset_col:='o')").fetchall()
    for actual, prediction in rows:
        assert prediction == pytest.approx(actual,rel=1e-6,abs=0)


@pytest.mark.parametrize('solver', ['auto', 'gd', 'irls'])
@pytest.mark.parametrize('family,alpha', [('poisson', None), ('nbinom', 1e-310), ('nbinom', 1e-20), ('nbinom', 1.)])
def test_count_fit_preserves_large_internal_means_with_tiny_weights(con, solver, family, alpha):
    con.execute('CREATE OR REPLACE TABLE poisson_weighted_tail(x DOUBLE,y DOUBLE,w DOUBLE,o DOUBLE)')
    con.executemany('INSERT INTO poisson_weighted_tail VALUES (?,?,?,?)', [
        (-1, 1e-300, 1, np.log(1e-300)), (1, 1e-300, 1, np.log(1e-300)),
        (0, 1e300, 1e-320, np.log(1e300))])
    extra = '' if alpha is None else f',alpha:={alpha}'
    con.execute(f"CREATE OR REPLACE TABLE poisson_tail_model AS SELECT * FROM {family}_fit('poisson_weighted_tail','y',weights_col:='w',offset_col:='o',solver:='{solver}'{extra})")
    beta = con.execute('SELECT coefficient FROM poisson_tail_model').fetchnumpy()['coefficient']
    np.testing.assert_allclose(beta, [0, 0], atol=1e-9)
    rows = con.execute(f"SELECT y,prediction FROM {family}_predict('poisson_tail_model','poisson_weighted_tail',offset_col:='o')").fetchall()
    for actual, prediction in rows:
        assert prediction == pytest.approx(actual, rel=1e-9, abs=0)


@pytest.mark.parametrize('solver', ['auto', 'gd', 'irls'])
def test_tweedie_power_one_uses_stable_poisson_offset_fit(con, solver):
    con.execute('CREATE OR REPLACE TABLE power_one_offsets AS SELECT i::DOUBLE x,1.0 y,1000.0*i o FROM range(-1,2)t(i)')
    call = f"SELECT * FROM tweedie_fit('power_one_offsets','y',power:=1,offset_col:='o',solver:='{solver}',max_iter:=1000)"
    if solver == 'irls':
        # The explicit solver rejects this ill-conditioned initial information;
        # auto must use the same stable GD fallback as the Poisson fit.
        with pytest.raises(duckdb.Error, match='irls solver did not converge'):
            con.execute(call).fetchall()
        return
    coefficients = dict(con.execute(call).fetchall())
    assert coefficients['(Intercept)'] == pytest.approx(0, abs=1e-7)
    assert coefficients['x'] == pytest.approx(-1000, abs=1e-7)


@pytest.mark.parametrize('solver', ['auto', 'irls'])
@pytest.mark.parametrize('minority_weight', [1e-18, 1e-100])
@pytest.mark.parametrize('majority_label', [0, 1])
def test_logistic_fit_keeps_information_after_sigmoid_rounding(con, solver, minority_weight, majority_label):
    con.execute('CREATE OR REPLACE TABLE imbalanced_fit(x DOUBLE,y DOUBLE,w DOUBLE)')
    con.executemany('INSERT INTO imbalanced_fit VALUES (?,?,?)',
                    [(x,majority_label,1.) for x in [0,1]]+
                    [(x,1-majority_label,minority_weight) for x in [0,1]])
    coefficients = dict(con.execute(f"SELECT * FROM logit_fit('imbalanced_fit','y',weights_col:='w',solver:='{solver}',max_iter:=1000)").fetchall())
    expected = (2*majority_label-1)*-np.log(minority_weight)
    assert coefficients['(Intercept)'] == pytest.approx(expected, abs=1e-8)
    assert coefficients['x'] == pytest.approx(0, abs=1e-8)
