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


def test_auto_rejects_finite_unconverged_gamma_irls(con):
    con.execute("CREATE OR REPLACE TABLE overshoot AS SELECT i::DOUBLE x,1.0 y,-8.0 expo FROM range(4)t(i)")
    coefficients = dict(con.execute("""
        SELECT * FROM gamma_fit('overshoot','y',offset_col:='expo',max_iter:=100)
    """).fetchall())
    assert coefficients['(Intercept)'] == pytest.approx(8.0, abs=1e-7)
    assert coefficients['x'] == pytest.approx(0.0, abs=1e-7)
    with pytest.raises(duckdb.Error, match='iteration limit reached'):
        con.execute("""
            SELECT * FROM gamma_fit('overshoot','y',offset_col:='expo',max_iter:=100,solver:='irls')
        """).fetchall()


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
