"""Cross-validation must compare candidate mean predictions on one loss scale."""

from pathlib import Path

import duckdb
import numpy as np
import pytest
from sklearn.metrics import mean_tweedie_deviance


@pytest.fixture
def con():
    with duckdb.connect() as c:
        c.execute((Path(__file__).resolve().parents[1] / "regression_macros.sql").read_text())
        # Every feature value in every fold has the same balanced y distribution.
        # All candidates therefore predict the same exact held-out mean, 4.
        c.execute("CREATE TABLE balanced AS SELECT i//35 AS x, i%7+1 AS y FROM range(105) t(i)")
        yield c


@pytest.mark.parametrize('macro', ['cv_l2', 'cv_l1', 'cv_l2_refine', 'cv_l1_refine'])
def test_linear_cv_keeps_finite_large_scores_and_candidate_ranking(con, macro):
    con.execute('CREATE TABLE base AS SELECT (i%7)::DOUBLE x,(1+2*(i%7)+3*(i%3))::DOUBLE y FROM range(30)t(i)')
    con.execute('CREATE TABLE scaled AS SELECT x,y*1e153 y FROM base')
    extra = ',n_refine:=4' if macro.endswith('_refine') else ''
    results = [np.asarray(con.execute(f"SELECT * FROM {macro}('{table}','y','linear',[0.,1.,10.],k:=3{extra})").fetchall(), dtype=float)
               for table in ['base', 'scaled']]
    results[1][:, 1] /= 1e306
    assert np.isfinite(results[1]).all()
    np.testing.assert_allclose(results[1], results[0], rtol=1e-9, atol=1e-10)


def test_linear_cv_scales_before_squaring_individual_residuals(con):
    con.execute('CREATE TABLE base AS SELECT (i%7)::DOUBLE x,CASE WHEN i=0 THEN 1.0 ELSE 0.0 END y FROM range(300)t(i)')
    con.execute('CREATE TABLE scaled AS SELECT x,y*1e155 y FROM base')
    baseline = con.execute("SELECT cv_deviance FROM cv_l2('base','y','linear',[0.],k:=3)").fetchone()[0]
    actual = con.execute("SELECT cv_deviance FROM cv_l2('scaled','y','linear',[0.],k:=3)").fetchone()[0]
    assert np.isfinite(actual)
    assert actual/1e155/1e155 == pytest.approx(baseline, rel=1e-10)


@pytest.mark.parametrize('sweep', ['l1', 'l2'])
@pytest.mark.parametrize('scale', [1e153, 1e308])
def test_linear_cv_does_not_reconstruct_overflowing_centered_predictions(con, sweep, scale):
    con.execute('CREATE TABLE mixed_extremes AS SELECT CASE WHEN i%3=0 THEN -1. ELSE 1. END x,CASE WHEN i%3=0 THEN -1.5*? ELSE 1.5*? END y FROM range(12)t(i)', [scale, scale])
    score = con.execute(f"SELECT cv_deviance FROM cv_{sweep}('mixed_extremes','y','linear',[0.],k:=2)").fetchone()[0]
    # Squared roundoff residuals at 1e308 may exceed DOUBLE range. They may
    # produce +Infinity, but must never turn a valid non-negative MSE into NaN.
    assert score >= 0.0
    if scale == 1e153:
        assert np.isfinite(score)
        assert score/scale/scale < 1e-28


@pytest.mark.parametrize('lo,hi', [(0., 1e308), (-1e308, 1e308), (1e308, 0.)])
@pytest.mark.parametrize('n', [3, 5, 7])
def test_grid_interpolation_preserves_finite_extreme_bounds(con, lo, hi, n):
    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = 80
        expected = [float(Decimal(str(lo))*(1-Decimal(i)/(n-1)) + Decimal(str(hi))*Decimal(i)/(n-1)) for i in range(n)]
    actual = con.execute('SELECT reg_grid(?,?,?)', [lo, hi, n]).fetchone()[0]
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(np.array(actual)/1e308, np.array(expected)/1e308, atol=1e-14)
    assert actual[0] == lo and actual[-1] == hi


@pytest.mark.parametrize('grid,winner,expected', [
    ([0.,1e308], 0., [0.,2.5e307,5e307,7.5e307,1e308]),
    ([-1e308,0.,1e308], 0., [-1e308,-5e307,0.,5e307,1e308]),
])
def test_refinement_interpolates_large_neighbors_and_retains_winner(con, grid, winner, expected):
    actual = con.execute('SELECT __reg_refine_grid(?,?,5)', [grid, winner]).fetchone()[0]
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(np.array(actual)/1e308, np.array(expected)/1e308, atol=1e-14)
    assert winner in actual


@pytest.mark.parametrize('penalty', [1e8, 1e308])
@pytest.mark.parametrize('macro', ['cv_l2', 'cv_l2_refine'])
def test_ridge_candidates_cannot_change_other_candidates_in_singular_fallback(con, penalty, macro):
    con.execute('CREATE TABLE singular_cv AS SELECT (i%3)::DOUBLE x,(i%3)::DOUBLE z,(i%5=0)::DOUBLE y FROM range(60)t(i)')
    baseline = con.execute("SELECT cv_deviance FROM cv_l2('singular_cv','y','logistic',[0.],k:=2)").fetchone()[0]
    extra = ',n_refine:=3' if macro.endswith('_refine') else ''
    scores = con.execute(f"SELECT cv_deviance FROM {macro}('singular_cv','y','logistic',[0.,{penalty}],k:=2{extra})").fetchnumpy()['cv_deviance']
    expected = -2*(.2*np.log(.2)+.8*np.log(.8))
    assert baseline == pytest.approx(expected,abs=1e-10)
    np.testing.assert_allclose(scores,expected,atol=1e-9,rtol=0)


def test_cv_power_equal_predictions_have_equal_scores_including_endpoints(con):
    rows = con.execute(
        "SELECT * FROM cv_power('balanced','y',[1.0,1.3,1.5,1.7,2.0])"
    ).fetchall()
    y = np.tile(np.arange(1.0, 8.0), 15)
    expected = mean_tweedie_deviance(y, np.full(len(y), 4.0), power=1.5)
    assert len(rows) == 5
    assert [score for _, score in rows] == pytest.approx([expected] * 5, rel=1e-9)


def test_cv_alpha_equal_predictions_have_equal_scores(con):
    rows = con.execute(
        "SELECT * FROM cv_alpha('balanced','y',[0.01,0.1,1.0,10.0,100.0])"
    ).fetchall()
    y = np.tile(np.arange(1.0, 8.0), 15)
    expected = np.mean(2 * (y * np.log(y / 4) - (y + 1) * np.log((y + 1) / 5)))
    assert len(rows) == 5
    assert [score for _, score in rows] == pytest.approx([expected] * 5, rel=1e-9)


@pytest.mark.parametrize('family', ['poisson','gamma','tweedie','nbinom'])
@pytest.mark.parametrize('sweep', ['l1','l2'])
def test_cv_deviance_stays_zero_for_perfect_large_count_predictions(con,family,sweep):
    con.execute('CREATE TABLE large_constant AS SELECT (i%4)::DOUBLE x,1e16::DOUBLE y FROM range(16)t(i)')
    rows=con.execute(f"SELECT cv_deviance FROM cv_{sweep}('large_constant','y','{family}',[0.,1.],k:=2)").fetchall()
    np.testing.assert_allclose(rows,0.,atol=1e-10)


@pytest.mark.parametrize("sweep,grid", [("power", "[1.0,1.5,2.0]"), ("alpha", "[0.1,1.0,10.0]")])
def test_refinement_retains_the_same_scoring_scale(con, sweep, grid):
    coarse = con.execute(f"SELECT cv_deviance FROM cv_{sweep}('balanced','y',{grid})").fetchall()
    refined = con.execute(
        f"SELECT cv_deviance FROM cv_{sweep}_refine('balanced','y',{grid},n_refine:=4)"
    ).fetchall()
    assert len(refined) == 4
    assert [row[0] for row in refined] == pytest.approx([coarse[0][0]] * 4, rel=1e-9)


@pytest.mark.parametrize('threads', [1, 4, 24])
@pytest.mark.parametrize('call,outcome', [
    ("cv_l2('bad_domain','y','logistic',[0.,1.],k:=3,max_iter:=50)",'1+i%2'),
    ("cv_l1('bad_domain','y','poisson',[0.,1.],k:=3,max_iter:=50)",'-1+i%2'),
    ("cv_l2('bad_domain','y','gamma',[0.,1.],k:=3,max_iter:=50)",'i%2'),
    ("cv_alpha('bad_domain','y',[.5,1.],k:=3,max_iter:=50)",'-1+i%2'),
    ("cv_power('bad_domain','y',[1.5,2.],k:=3,max_iter:=50)",'i%2'),
    ("cv_power('bad_domain','y',[1.,1.5],k:=3,max_iter:=50)",'-1+i%2'),
])
def test_cv_rejects_invalid_family_outcomes(con,call,outcome,threads):
    con.execute(f'SET threads={threads}')
    con.execute(f'CREATE TABLE bad_domain AS SELECT i::DOUBLE x,({outcome})::DOUBLE y FROM range(12)t(i)')
    with pytest.raises(duckdb.Error,match='cv: outcome must'):
        con.execute(f'SELECT * FROM {call}').fetchall()


def test_duplicate_dispersion_candidates_preserve_likelihood(con):
    con.execute('CREATE TABLE dispersion_counts AS SELECT i::DOUBLE x,i%7+1.0 y FROM range(30)t(i)')
    unique=con.execute("SELECT * FROM nbinom_dispersion('dispersion_counts','y',[.5,1.])").fetchall()
    repeated=con.execute("SELECT * FROM nbinom_dispersion('dispersion_counts','y',[.5,.5,1.])").fetchall()
    assert len(repeated)==len(unique)==2
    np.testing.assert_allclose(np.asarray(repeated,dtype=float),np.asarray(unique,dtype=float),rtol=1e-12)


def test_single_point_refinement_keeps_best_candidate(con):
    coarse=con.execute("SELECT * FROM cv_l2('balanced','y','linear',[0.,1.]) ORDER BY cv_deviance,l2 LIMIT 1").fetchone()
    refined=con.execute("SELECT * FROM cv_l2_refine('balanced','y','linear',[0.,1.],n_refine:=1)").fetchall()
    assert len(refined)==1
    assert np.isfinite(float(refined[0][0])) and np.isfinite(refined[0][1])
    assert refined[0][1]==pytest.approx(coarse[1])
    assert con.execute('SELECT __reg_refine_grid([0.,1.,2.],1.,1)').fetchone()[0]==[1.0]


@pytest.mark.parametrize('macro',['cv_l2_refine','nbinom_dispersion_refine'])
def test_refinement_does_not_shadow_legal_table_names(con,macro):
    con.execute('CREATE TABLE __rr_best AS SELECT * FROM balanced')
    args=",'linear',[0.,1.]" if macro=='cv_l2_refine' else ',[.5,1.]'
    expected=con.execute(f"SELECT * FROM {macro}('balanced','y'{args},n_refine:=2)").fetchall()
    actual=con.execute(f"SELECT * FROM {macro}('__rr_best','y'{args},n_refine:=2)").fetchall()
    np.testing.assert_allclose(np.asarray(actual,dtype=float),np.asarray(expected,dtype=float),rtol=1e-10,atol=1e-12)


@pytest.mark.parametrize('family',['poisson','nbinom','tweedie'])
def test_cv_deviance_uses_finite_log_predictors_when_means_underflow(con,family):
    con.execute('CREATE TABLE extreme_holdout AS SELECT * FROM (VALUES (1000.,1.),(0.,1.),(0.,1.),(1.,exp(-1.)))t(x,y)')
    call=f"cv_l2('extreme_holdout','y','{family}',[0.],k:=2)"
    actual=con.execute(f'SELECT cv_deviance FROM {call}').fetchone()[0]
    assert np.isfinite(actual)
    y=np.array([1.,1.,1.,np.exp(-1.)]); z=np.array([-1000.,0.,0.,0.])
    if family=='poisson':
        expected=np.mean(2*(y*(np.log(y)-z)-y+np.exp(z)))
    elif family=='nbinom':
        expected=np.mean(2*(y*(np.log(y)-z)-(y+1)*(np.log1p(y)-np.logaddexp(0,z))))
    else:
        expected=np.mean(4*(-2*np.sqrt(y)+y*np.exp(-.5*z)+np.exp(.5*z)))
    assert actual==pytest.approx(expected,rel=1e-6)


@pytest.mark.parametrize('incomplete',[False,True])
@pytest.mark.parametrize('call',["cv_l2('too_small','y','linear',[0.,1.])","cv_l2_refine('too_small','y','linear',[0.,1.])"])
def test_cv_rejects_empty_training_folds_after_filtering(con,incomplete,call):
    con.execute('CREATE TABLE too_small AS SELECT 2.0 x,4.0 y')
    if incomplete:
        con.execute('INSERT INTO too_small VALUES (NULL,1),(1,NULL)')
    with pytest.raises(duckdb.Error,match='at least two complete rows'):
        con.execute('SELECT * FROM '+call).fetchall()


@pytest.mark.parametrize('n',[1,2,3,10,11])
def test_refinement_keeps_incumbent_on_uneven_grid(con,n):
    points=con.execute(f'SELECT __reg_refine_grid([0.,.5,100.],.5,{n})').fetchone()[0]
    assert len(points)==n
    assert .5 in points
    assert points==sorted(points)
    if n>=3:
        assert points[0]==0 and points[-1]==100


def test_refinement_cannot_skip_better_coarse_candidate(con):
    # Reuse the independent fixed-seed interior-optimum regression fixture.
    from test_regression_macros import TestGridRefinement
    data=TestGridRefinement()._ridge_interior(7)
    con.register('refinement_source',data)
    con.execute('CREATE TABLE uneven_refinement AS SELECT * FROM refinement_source')
    coarse=con.execute("SELECT min(cv_deviance) FROM cv_l2('uneven_refinement','y','linear',[0.,.5,100.])").fetchone()[0]
    refined=con.execute("SELECT min(cv_deviance) FROM cv_l2_refine('uneven_refinement','y','linear',[0.,.5,100.])").fetchone()[0]
    assert refined <= coarse+1e-10


@pytest.mark.parametrize('bad',['NULL',"'NaN'::DOUBLE","'Infinity'::DOUBLE"])
@pytest.mark.parametrize('kind',['alpha','power','l1','l2','dispersion'])
def test_tuning_rejects_invalid_grid_candidates(con,bad,kind):
    grid=f'[{bad},1.5]::DOUBLE[]'
    if kind=='dispersion':call=f"nbinom_dispersion('balanced','y',{grid})"
    elif kind in ('alpha','power'):call=f"cv_{kind}('balanced','y',{grid})"
    else:call=f"cv_{kind}('balanced','y','linear',{grid})"
    with pytest.raises(duckdb.Error,match='non-NULL and finite'):
        con.execute('SELECT * FROM '+call).fetchall()


@pytest.mark.parametrize('threads', [1, 4, 24])
@pytest.mark.parametrize('macro', ['nbinom_dispersion','nbinom_dispersion_refine'])
@pytest.mark.parametrize('alpha', [-1., 0.])
def test_dispersion_validates_alpha_before_logarithms(con, threads, macro, alpha):
    con.execute(f'SET threads={threads}')
    with pytest.raises(duckdb.Error,match='nbinom_dispersion: alpha values must be > 0'):
        con.execute(f"SELECT * FROM {macro}('balanced','y',[{alpha},1.])").fetchall()


@pytest.mark.parametrize('threads', [1, 4, 24])
@pytest.mark.parametrize('macro', ['nbinom_dispersion','nbinom_dispersion_refine'])
def test_dispersion_validates_outcomes_before_logarithms(con, threads, macro):
    con.execute(f'SET threads={threads}')
    con.execute('CREATE TABLE bad_domain AS SELECT i::DOUBLE x,-1.0 y FROM range(12)t(i)')
    with pytest.raises(duckdb.Error,match='nbinom_dispersion: outcome must'):
        con.execute(f"SELECT * FROM {macro}('bad_domain','y',[.5,1.])").fetchall()
