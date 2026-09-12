"""Rank-deficient solver fallback and zero-weight optimizer regressions."""

from pathlib import Path

import duckdb
import numpy as np
import pytest


@pytest.fixture(scope="module")
def con():
    connection = duckdb.connect()
    connection.execute((Path(__file__).resolve().parents[1] / "regression_macros.sql").read_text())
    connection.execute("SET threads=1")
    yield connection
    connection.close()


@pytest.mark.parametrize('scale', [1e160, 1e300, 1e305])
@pytest.mark.parametrize('solver', ['auto', 'gd'])
def test_negative_binomial_curvature_preserves_large_count_fallback(con, scale, solver):
    con.execute('CREATE OR REPLACE TABLE large_counts AS SELECT i::DOUBLE/10 x,?*exp(.3*i/10) y,1.0 constant FROM range(-10,11)t(i)', [scale])
    coefficients = dict(con.execute(f"SELECT * FROM nbinom_fit('large_counts','y',solver:='{solver}',max_iter:=1000)").fetchall())
    assert coefficients['(Intercept)'] == pytest.approx(np.log(scale), abs=1e-8)
    assert coefficients['x'] == pytest.approx(.3, abs=1e-8)
    assert coefficients['constant'] == 0.0


@pytest.mark.parametrize('scale', [1e160, 1e305])
def test_negative_binomial_batch_curvature_preserves_exact_large_count_means(con, scale):
    con.execute('CREATE OR REPLACE TABLE large_counts AS SELECT i::DOUBLE/10 x,?*exp(.3*i/10) y,1.0 constant FROM range(-10,11)t(i)', [scale])
    con.execute("CREATE OR REPLACE TABLE exact_nb AS SELECT '(Intercept)' feature,ln(?) coefficient UNION ALL SELECT 'x',.3 UNION ALL SELECT 'constant',0.0", [scale])
    scores = con.execute("SELECT cv_deviance FROM cv_alpha('large_counts','y',[.5,1.],k:=3,max_iter:=1000)").fetchnumpy()['cv_deviance']
    np.testing.assert_allclose(scores, 0.0, atol=1e-12)
    profile = con.execute("SELECT * FROM nbinom_dispersion('large_counts','y',[.5,1.],max_iter:=1000)").fetchall()
    for alpha, loglik in profile:
        expected = con.execute(f"SELECT loglik FROM nbinom_evaluate('exact_nb','large_counts','y',alpha:={alpha})").fetchone()[0]
        assert loglik == pytest.approx(expected, abs=1e-8)


def test_negative_binomial_batch_internal_dispersion_can_exceed_double(con):
    con.execute('CREATE OR REPLACE TABLE large_counts AS SELECT i::DOUBLE/10 x,1e307*exp(.3*i/10) y FROM range(-10,11)t(i)')
    con.execute("CREATE OR REPLACE TABLE exact_nb AS SELECT '(Intercept)' feature,ln(1e307) coefficient UNION ALL SELECT 'x',.3")
    grid = '[10.,100.,1e300]'
    scores = con.execute(f"SELECT cv_deviance FROM cv_alpha('large_counts','y',{grid},k:=3,max_iter:=1000)").fetchnumpy()['cv_deviance']
    np.testing.assert_allclose(scores, 0.0, atol=1e-12)
    profile = con.execute(f"SELECT * FROM nbinom_dispersion('large_counts','y',{grid},max_iter:=1000)").fetchall()
    for alpha, loglik in profile:
        expected = con.execute(f"SELECT loglik FROM nbinom_evaluate('exact_nb','large_counts','y',alpha:={alpha})").fetchone()[0]
        assert np.isfinite(loglik)
        assert loglik == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("scales", [[1, 1, 1], [1e-6, 1e3, 1e6]])
def test_matrix_inverse_rejects_general_dependency_at_any_scale(con, scales):
    matrix = np.array([[1., 2., 3.], [2., 5., 7.], [3., 7., 10.]])
    matrix *= np.outer(scales, scales)
    assert con.execute("SELECT __reg_matinv(?::DOUBLE[][])", [matrix.tolist()]).fetchone()[0] is None


@pytest.mark.parametrize("scales", [[1, 1, 1], [1e-6, 1e3, 1e6]])
def test_matrix_inverse_preserves_full_rank_at_any_scale(con, scales):
    matrix = np.array([[2., .25, .4], [.25, 3., .5], [.4, .5, 4.]])
    matrix *= np.outer(scales, scales)
    actual = con.execute("SELECT __reg_matinv(?::DOUBLE[][])", [matrix.tolist()]).fetchone()[0]
    np.testing.assert_allclose(actual, np.linalg.inv(matrix), rtol=1e-12, atol=0)


def dependent_data(con):
    con.execute("""
        CREATE OR REPLACE TABLE dependent AS
        SELECT x, z, x+z AS u, 1+2*x+3*z AS y
        FROM (SELECT i::DOUBLE/10 AS x, ((i*7)%11)::DOUBLE/10 AS z FROM range(20) q(i))
    """)


def test_auto_falls_back_for_general_linear_dependency(con):
    dependent_data(con)
    con.execute("CREATE OR REPLACE TABLE rank_model AS SELECT * FROM linreg_fit('dependent', 'y')")
    rmse = con.execute("""
        SELECT sqrt(avg((y-prediction)^2)) FROM linreg_predict('rank_model', 'dependent')
    """).fetchone()[0]
    assert rmse < 1e-7


def test_explicit_irls_reports_general_linear_dependency(con):
    dependent_data(con)
    with pytest.raises(duckdb.Error, match="X'WX is singular"):
        con.execute("SELECT * FROM linreg_fit('dependent', 'y', solver := 'irls')").fetchall()


def test_cv_falls_back_for_general_linear_dependency(con):
    dependent_data(con)
    con.execute("CREATE OR REPLACE TABLE independent AS SELECT x,z,y FROM dependent")
    # Every fold's full-rank reference can interpolate the same exact plane.
    for table in ("dependent", "independent"):
        deviance = con.execute(f"""
            SELECT cv_deviance FROM cv_l2('{table}', 'y', 'linear', [0.0], k := 4)
        """).fetchone()[0]
        assert deviance < 1e-10


@pytest.mark.parametrize("macro", ["poisson_fit", "gamma_fit", "tweedie_fit", "nbinom_fit"])
@pytest.mark.parametrize("solver", ["auto", "gd"])
def test_zero_weight_offsets_do_not_damp_log_link_fits(con, macro, solver):
    con.execute("""
        CREATE OR REPLACE TABLE clean_weighted AS
        SELECT i::DOUBLE AS x, 1.0 AS constant, 0.0 AS off, 1.0 AS wt, exp(.2+.3*i) AS y
        FROM range(10) t(i)
    """)
    con.execute("""
        CREATE OR REPLACE TABLE dirty_weighted AS
        SELECT * FROM clean_weighted UNION ALL SELECT 0,1,50,0,1
    """)
    models = []
    for table in ("clean_weighted", "dirty_weighted"):
        models.append(dict(con.execute(f"""
            SELECT feature, coefficient FROM {macro}(
                '{table}', 'y', offset_col := 'off', weights_col := 'wt',
                solver := '{solver}', max_iter := 3000)
        """).fetchall()))
    assert models[0]["x"] == pytest.approx(.3, abs=1e-6)
    assert models[0]["(Intercept)"] == pytest.approx(.2, abs=1e-6)
    assert models[1] == pytest.approx(models[0], abs=1e-10)


def test_cv_rejects_null_model_when_only_one_fold_is_singular(con):
    con.execute("""
        CREATE OR REPLACE TABLE mixed_rank AS
        SELECT x,z,1+2*x+3*z AS y FROM (
            SELECT i/10.0 AS x,
                   CASE WHEN i%3=0 THEN (i*7)%11/10.0 ELSE i/10.0 END AS z
            FROM range(12) t(i))
    """)
    data = con.execute("SELECT x,z,y FROM mixed_rank").fetchnumpy()
    features = np.column_stack((data["x"], data["z"]))
    outcome = data["y"]
    design = np.column_stack((np.ones(len(outcome)), (features-features.mean(0))/features.std(0)))
    scaled_outcome = (outcome-outcome.mean())/outcome.std()
    folds = np.arange(len(outcome)) % 3
    errors = []
    for fold in range(3):
        train = folds != fold
        coefficients = np.linalg.lstsq(design[train], scaled_outcome[train], rcond=None)[0]
        prediction = design[~train] @ coefficients * outcome.std() + outcome.mean()
        errors.extend((prediction-outcome[~train])**2)
    actual = con.execute("""
        SELECT cv_deviance FROM cv_l2('mixed_rank', 'y', 'linear', [0.0], k := 3)
    """).fetchone()[0]
    assert actual == pytest.approx(np.mean(errors), abs=1e-8)


def test_bounded_gamma_irls_converges_and_still_rejects_iteration_limit(con):
    con.execute("CREATE OR REPLACE TABLE overshoot AS SELECT i::DOUBLE x,1.0 y,-8.0 expo FROM range(4)t(i)")
    for solver in ['auto','irls']:
        coefficients = dict(con.execute(f"""
            SELECT * FROM gamma_fit('overshoot','y',offset_col:='expo',max_iter:=100,solver:='{solver}')
        """).fetchall())
        assert coefficients['(Intercept)'] == pytest.approx(8.0, abs=1e-7)
        assert coefficients['x'] == pytest.approx(0.0, abs=1e-7)
    # Centering a constant offset now solves the constant response at
    # initialization. Retain the iteration-limit contract on a real slope.
    con.execute('UPDATE overshoot SET y=exp(.3*x)')
    with pytest.raises(duckdb.Error, match='iteration limit reached'):
        con.execute("""
            SELECT * FROM gamma_fit('overshoot','y',offset_col:='expo',max_iter:=1,solver:='irls')
        """).fetchall()
    limited=[]
    for solver in ['auto','gd']:
        limited.append(dict(con.execute(f"""
            SELECT * FROM gamma_fit('overshoot','y',offset_col:='expo',max_iter:=1,solver:='{solver}')
        """).fetchall()))
    assert limited[0]==pytest.approx(limited[1],abs=1e-10)


def test_cv_gd_step_accounts_for_training_fold_curvature(con):
    con.execute('''
        CREATE OR REPLACE TABLE fold_curvature AS
        SELECT x,x AS x2,x AS x3,x AS y FROM (
          SELECT CASE WHEN i%2=0 THEN 1.0 ELSE .1 END
               * CASE WHEN i%4<2 THEN -1 ELSE 1 END AS x FROM range(40)t(i))
    ''')
    deviance=con.execute("SELECT cv_deviance FROM cv_l2('fold_curvature','y','linear',[0.0],k:=2)").fetchone()[0]
    assert np.isfinite(deviance)
    assert deviance < 1e-9


def test_cv_curvature_ignores_held_out_outcomes(con):
    con.execute('''CREATE OR REPLACE TABLE different_fold_scales AS
        SELECT 1.0 x,CASE WHEN i%2=0 THEN 1e8 ELSE 1.0 END y FROM range(10)t(i)''')
    actual=con.execute("SELECT cv_deviance FROM cv_l2('different_fold_scales','y','gamma',[0.0],k:=2)").fetchone()[0]
    # Each intercept-only training fold predicts its own constant outcome.
    ratios=np.array([1e8,1e-8])
    expected=np.mean(2*(-np.log(ratios)+ratios-1))
    assert actual==pytest.approx(expected,rel=1e-6)


@pytest.mark.parametrize('scale', [1e8, 1e10])
@pytest.mark.parametrize('solver', ['auto', 'gd'])
@pytest.mark.parametrize('constant', [False, True])
def test_negative_binomial_gd_recovers_large_count_trends(con, scale, solver, constant):
    extra = ',1.0 c' if constant else ''
    con.execute(f'CREATE OR REPLACE TABLE large_counts AS SELECT i::DOUBLE/10 x,{scale}*exp(.3*i/10) y{extra} FROM range(-10,11)t(i)')
    coefs = dict(con.execute(f"SELECT * FROM nbinom_fit('large_counts','y',solver:='{solver}',max_iter:=2000)").fetchall())
    assert coefs['x'] == pytest.approx(.3, abs=1e-6)
    assert coefs['(Intercept)'] == pytest.approx(np.log(scale), abs=1e-6)
    if constant:
        assert coefs['c'] == 0


@pytest.mark.parametrize('l1', [0.0, 1e-12])
def test_negative_binomial_curvature_preserves_penalty_objective(con, l1):
    con.execute('CREATE OR REPLACE TABLE penalized_counts AS SELECT i::DOUBLE/10 x,1e10*exp(.3*i/10) y FROM range(-10,11)t(i)')
    fits = [dict(con.execute(f"SELECT * FROM nbinom_fit('penalized_counts','y',solver:='{solver}',l2:=1e-11,l1:={l1},max_iter:=2000)").fetchall()) for solver in ['irls','gd']]
    assert fits[1] == pytest.approx(fits[0], rel=1e-6, abs=1e-6)


def test_negative_binomial_cv_conditions_each_dispersion_candidate(con):
    con.execute('CREATE OR REPLACE TABLE cv_large_counts AS SELECT i::DOUBLE/10 x,1e8*exp(.3*i/10) y,1.0 c FROM range(-10,11)t(i)')
    scores = con.execute("SELECT cv_deviance FROM cv_alpha('cv_large_counts','y',[1e-4,1.,1e4],k:=3,max_iter:=2000)").fetchnumpy()['cv_deviance']
    np.testing.assert_allclose(scores, 0.0, atol=1e-6)


def test_negative_binomial_dispersion_profiles_large_count_trends(con):
    con.execute('CREATE OR REPLACE TABLE profile_counts AS SELECT i::DOUBLE/10 x,1e8*exp(.3*i/10) y FROM range(-10,11)t(i)')
    # This noiseless log-linear relationship has known fitted means for every alpha.
    con.execute("CREATE OR REPLACE TABLE exact_count_model AS SELECT '(Intercept)' feature,ln(1e8) coefficient UNION ALL SELECT 'x',.3")
    actual = dict(con.execute("SELECT * FROM nbinom_dispersion('profile_counts','y',[1e-4,1.,1e4],max_iter:=2000)").fetchall())
    for alpha, loglik in actual.items():
        expected = con.execute(f"SELECT loglik FROM nbinom_evaluate('exact_count_model','profile_counts','y',alpha:={alpha})").fetchone()[0]
        assert loglik == pytest.approx(expected, rel=1e-8, abs=1e-5)


@pytest.mark.parametrize('offset', [-8.0, 4.0, 20.0])
@pytest.mark.parametrize('scale', [1.0, 1e10])
def test_negative_binomial_gd_remains_stable_with_offsets(con, offset, scale):
    con.execute(f'CREATE OR REPLACE TABLE offset_counts AS SELECT i::DOUBLE x,{scale}*(1.+i%3) y,{offset} expo FROM range(12)t(i)')
    con.execute('CREATE OR REPLACE TABLE constant_offset_counts AS SELECT *,1.0 c FROM offset_counts')
    # Changing a constant offset changes only the intercept, providing a stable
    # independent reference even when IRLS initialization is far from the mean.
    con.execute('CREATE OR REPLACE TABLE no_offset_counts AS SELECT x,y FROM offset_counts')
    expected = dict(con.execute("SELECT * FROM nbinom_fit('no_offset_counts','y',solver:='irls')").fetchall())
    expected['(Intercept)'] -= offset
    for table, solver in [('offset_counts','gd'), ('constant_offset_counts','auto')]:
        actual = dict(con.execute(f"SELECT * FROM nbinom_fit('{table}','y',offset_col:='expo',solver:='{solver}',max_iter:=2000)").fetchall())
        if table == 'constant_offset_counts':
            assert actual.pop('c') == 0
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_negative_binomial_cv_handles_different_fold_means(con):
    con.execute('CREATE OR REPLACE TABLE nb_fold_means AS SELECT 1.0 x,CASE WHEN i%2=0 THEN 100.0 ELSE 1.0 END y FROM range(10)t(i)')
    actual = con.execute("SELECT cv_deviance FROM cv_alpha('nb_fold_means','y',[.5,1.],k:=2,max_iter:=2000)").fetchnumpy()['cv_deviance']
    y, mu = np.array([100.,1.]), np.array([1.,100.])
    expected = np.mean(2 * (y*np.log(y/mu) - (y+1)*np.log((y+1)/(mu+1))))
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
