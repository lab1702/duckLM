"""Changing finite feature units must preserve fitted models and model selection."""

from pathlib import Path

import duckdb
import numpy as np
import pytest


@pytest.fixture
def con():
    with duckdb.connect() as connection:
        connection.execute((Path(__file__).resolve().parents[1]/'regression_macros.sql').read_text())
        yield connection


@pytest.mark.parametrize('scale', [1e-305, 1e-310])
@pytest.mark.parametrize('solver', ['auto', 'gd'])
@pytest.mark.parametrize('weighted', [False, True])
def test_tiny_linear_outcomes_remain_standardized_in_gradient_fallback(con, scale, solver, weighted):
    weight = ',(i+1)::DOUBLE w' if weighted else ''
    argument = ",weights_col:='w'" if weighted else ''
    con.execute(f'CREATE TABLE tiny AS SELECT i::DOUBLE x,1.0 constant,(1.0+2*i)*? y{weight} FROM range(4)t(i)', [scale])
    con.execute(f"CREATE TABLE tiny_model AS SELECT * FROM linreg_fit('tiny','y',solver:='{solver}'{argument})")
    predictions = con.execute("SELECT prediction/? AS scaled_prediction FROM linreg_predict('tiny_model','tiny') ORDER BY x", [scale]).fetchnumpy()['scaled_prediction']
    np.testing.assert_allclose(predictions, [1, 3, 5, 7], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize('family', ['poisson', 'gamma', 'tweedie', 'nbinom'])
@pytest.mark.parametrize('scale', [1e-305, 1e-310])
def test_tiny_log_link_outcomes_keep_positive_mean_scaling(con, family, scale):
    con.execute('CREATE TABLE tiny AS SELECT i::DOUBLE x,exp(.2+.1*i)*? y FROM range(6)t(i)', [scale])
    coefficients = dict(con.execute(f"SELECT * FROM {family}_fit('tiny','y',max_iter:=1000)").fetchall())
    assert coefficients['x'] == pytest.approx(.1, abs=1e-9)
    assert coefficients['(Intercept)'] == pytest.approx(.2+np.log(scale), abs=1e-9)


@pytest.mark.parametrize('solver', ['auto', 'irls', 'gd'])
def test_linear_back_transform_preserves_finite_coefficients_when_product_overflows(con, solver):
    con.execute('CREATE TABLE large_response AS SELECT * FROM (VALUES (-40.,-50.,1e308),(-20.,-10.,-1e308),(0.,-10.,1e308),(20.,30.,-1e308),(40.,30.,1e308))t(x,z,y)')
    coefficients = dict(con.execute(f"SELECT * FROM linreg_fit('large_response','y',solver:='{solver}')").fetchall())
    assert np.isfinite(list(coefficients.values())).all()
    assert coefficients['x']/1e307 == pytest.approx(1.0, abs=1e-8)
    assert coefficients['z']/1e307 == pytest.approx(-1.0, abs=1e-8)
    assert abs(coefficients['(Intercept)']/1e308) < 1e-8


@pytest.mark.parametrize('alpha', [100., 1e300])
@pytest.mark.parametrize('solver', ['auto', 'irls', 'gd'])
@pytest.mark.parametrize('weighted_offset', [False, True])
def test_negative_binomial_internal_dispersion_can_exceed_double(con, alpha, solver, weighted_offset):
    expo = '.1*(i%3)' if weighted_offset else '0.0'
    weight = '1.0+(i+10)%3' if weighted_offset else '1.0'
    con.execute(f'CREATE TABLE large_nb AS SELECT i::DOUBLE/10 x,1e307*exp(.3*i/10+({expo})) y,({expo})::DOUBLE expo,({weight})::DOUBLE wt FROM range(-10,11)t(i)')
    coefficients = dict(con.execute(f"SELECT * FROM nbinom_fit('large_nb','y',alpha:={alpha},solver:='{solver}',offset_col:='expo',weights_col:='wt',max_iter:=1000)").fetchall())
    assert np.isfinite(list(coefficients.values())).all()
    assert coefficients['x'] == pytest.approx(.3, abs=1e-8)
    assert coefficients['(Intercept)'] == pytest.approx(np.log(1e307), abs=1e-8)


def test_negative_binomial_log_dispersion_preserves_ridge_strength(con):
    from scipy.optimize import minimize
    x = np.arange(-10, 11)/10
    y = 1e307*np.exp(.3*x)
    mean_y = y[0]+np.mean(y-y[0])
    design = np.column_stack([np.ones(len(x)), (x-x.mean())/x.std()])
    scaled_y = y/mean_y
    # Large-dispersion NB has the Gamma mean objective divided by alpha*mean(y).
    penalty = (1e-310*100)*mean_y
    def objective(beta):
        eta = design@beta
        return np.mean(scaled_y*np.exp(-eta)+eta)+penalty*beta[1]**2/2
    def gradient(beta):
        return design.T@(1-scaled_y*np.exp(-design@beta))/len(x)+[0, penalty*beta[1]]
    reference = minimize(objective, [0., 0.], jac=gradient, method='BFGS', tol=1e-12).x
    assert np.max(np.abs(gradient(reference))) < 1e-9
    con.execute('CREATE TABLE large_nb AS SELECT i::DOUBLE/10 x,1e307*exp(.3*i/10) y FROM range(-10,11)t(i)')
    coefficients = dict(con.execute("SELECT * FROM nbinom_fit('large_nb','y',alpha:=100,l2:=1e-310,max_iter:=5000)").fetchall())
    assert coefficients['x'] == pytest.approx(reference[1]/x.std(), abs=1e-8)
    assert coefficients['(Intercept)'] == pytest.approx(np.log(mean_y)+reference[0]-reference[1]*x.mean()/x.std(), abs=1e-8)


@pytest.mark.parametrize('scale', [1e-170,1e170])
@pytest.mark.parametrize('family', ['linreg','logit','poisson','gamma','tweedie','nbinom'])
@pytest.mark.parametrize('weighted', [False,True])
def test_single_outcome_fits_preserve_extreme_feature_units(con,scale,family,weighted):
    outcome='i%2' if family=='logit' else '1.0+i/10.0'
    weight=',(1+i%4)::DOUBLE wt' if weighted else ''
    argument=",weights_col:='wt'" if weighted else ''
    con.execute(f'CREATE TABLE base AS SELECT i/10.0 x,{outcome} y{weight} FROM range(12)t(i)')
    con.execute(f'CREATE TABLE scaled AS SELECT * REPLACE(x*{scale} AS x) FROM base')
    models=[]
    for table in ['base','scaled']:
        models.append(dict(con.execute(f"SELECT * FROM {family}_fit('{table}','y',l2:=.1,max_iter:=300{argument})").fetchall()))
    assert np.isfinite(list(models[1].values())).all()
    models[1]['x']*=scale
    assert models[1]==pytest.approx(models[0],rel=1e-7,abs=1e-8)


@pytest.mark.parametrize('scale', [1e-170,1e170])
@pytest.mark.parametrize('weighted', [False,True])
def test_linear_fit_preserves_extreme_outcome_units(con,scale,weighted):
    weight=',(1+i%4)::DOUBLE wt' if weighted else ''
    argument=",weights_col:='wt'" if weighted else ''
    con.execute(f'CREATE TABLE base AS SELECT i/10.0 x,1.0+i/5.0+.1*(i%3) y{weight} FROM range(12)t(i)')
    con.execute(f'CREATE TABLE scaled AS SELECT * REPLACE(y*{scale} AS y) FROM base')
    models=[]
    for table in ['base','scaled']:
        models.append(dict(con.execute(f"SELECT * FROM linreg_fit('{table}','y',l2:=.1,max_iter:=300{argument})").fetchall()))
    assert {key:value/scale for key,value in models[1].items()}==pytest.approx(models[0],rel=1e-7,abs=1e-8)


@pytest.mark.parametrize('scale', [1e-170,1e170,1e308])
def test_multinomial_fit_preserves_extreme_feature_units(con,scale):
    con.execute('CREATE TABLE base AS SELECT i/10.0 x,(i%3)::VARCHAR y FROM range(15)t(i)')
    con.execute(f'CREATE TABLE scaled AS SELECT * REPLACE(x*{scale} AS x) FROM base')
    models=[]
    for table in ['base','scaled']:
        rows=con.execute(f"SELECT * FROM multinom_fit('{table}','y',l2:=.1,max_iter:=1000)").fetchall()
        models.append({(label,feature):value for label,feature,value in rows})
    actual={key:value*scale if key[1]=='x' else value for key,value in models[1].items()}
    assert actual==pytest.approx(models[0],rel=1e-7,abs=1e-8)


@pytest.mark.parametrize('scale', [1e-170,1e170,1e308])
@pytest.mark.parametrize('family', ['linear','logistic','poisson','gamma','tweedie','nbinom'])
def test_cv_preserves_extreme_feature_units(con,scale,family):
    outcome='i%2' if family=='logistic' else '1.0+i/10.0'
    con.execute(f'CREATE TABLE base AS SELECT i/10.0 x,{outcome} y FROM range(12)t(i)')
    con.execute(f'CREATE TABLE scaled AS SELECT * REPLACE(x*{scale} AS x) FROM base')
    scores=[]
    for table in ['base','scaled']:
        scores.append(con.execute(f"SELECT cv_deviance FROM cv_l2('{table}','y','{family}',[.1],k:=3,max_iter:=300)").fetchone()[0])
    assert np.isfinite(scores[1])
    assert scores[1]==pytest.approx(scores[0],rel=1e-7,abs=1e-9)


@pytest.mark.parametrize('scale', [1e-170,1e170,1e308])
def test_dispersion_profile_preserves_extreme_feature_units(con,scale):
    con.execute('CREATE TABLE base AS SELECT i/10.0 x,1.0+i%3 y FROM range(12)t(i)')
    con.execute(f'CREATE TABLE scaled AS SELECT * REPLACE(x*{scale} AS x) FROM base')
    profiles=[]
    for table in ['base','scaled']:
        profiles.append(con.execute(f"SELECT * FROM nbinom_dispersion('{table}','y',alpha_grid:=[.5,1.],max_iter:=300)").fetchall())
    np.testing.assert_allclose(np.array(profiles[1],dtype=float),np.array(profiles[0],dtype=float),rtol=1e-7,atol=1e-8)


@pytest.mark.parametrize('scale', [1e-308,1e308])
@pytest.mark.parametrize('family', ['linreg','logit','poisson','gamma','tweedie','nbinom'])
def test_fits_preserve_a_common_finite_weight_scale(con,scale,family):
    outcome='i%2' if family=='logit' else '1.0+i'
    con.execute(f'CREATE TABLE base AS SELECT i::DOUBLE x,{outcome} y,(i+1)/4.0 wt FROM range(4)t(i)')
    con.execute(f'CREATE TABLE scaled AS SELECT * REPLACE(wt*{scale} AS wt) FROM base')
    models=[]
    for table in ['base','scaled']:
        models.append(dict(con.execute(f"SELECT * FROM {family}_fit('{table}','y',weights_col:='wt',l2:=.1,max_iter:=300)").fetchall()))
    assert models[1]==pytest.approx(models[0],rel=1e-7,abs=1e-8)


@pytest.mark.parametrize('weighted', [False, True])
def test_linear_fit_handles_finite_observations_whose_sum_overflows(con, weighted):
    weight = ',(i+1)/4.0 w' if weighted else ''
    argument = ",weights_col:='w'" if weighted else ''
    con.execute(f'CREATE TABLE huge_features AS SELECT (1.0+i/10.0)*1e308 x,1.0+i y{weight} FROM range(4)t(i)')
    coefficients = dict(con.execute(f"SELECT * FROM linreg_fit('huge_features','y'{argument})").fetchall())
    assert coefficients['(Intercept)'] == pytest.approx(-9.0, abs=1e-10)
    assert coefficients['x'] * 1e308 == pytest.approx(10.0, abs=1e-10)

    con.execute(f'CREATE TABLE huge_outcomes AS SELECT i::DOUBLE x,1e308 y{weight} FROM range(4)t(i)')
    coefficients = dict(con.execute(f"SELECT * FROM linreg_fit('huge_outcomes','y'{argument})").fetchall())
    assert coefficients['(Intercept)'] == 1e308
    assert coefficients['x'] == 0.0

    con.execute('UPDATE huge_outcomes SET y=(1.0+x/10.0)*1e308')
    coefficients = dict(con.execute(f"SELECT * FROM linreg_fit('huge_outcomes','y'{argument})").fetchall())
    assert coefficients['(Intercept)'] / 1e308 == pytest.approx(1.0, abs=1e-10)
    assert coefficients['x'] / 1e307 == pytest.approx(1.0, abs=1e-10)


def test_batch_outcome_means_remain_finite_when_sum_overflows(con):
    con.execute('CREATE TABLE huge_outcomes AS SELECT i::DOUBLE x,1e308 y FROM range(6)t(i)')
    score = con.execute("SELECT cv_deviance FROM cv_l2('huge_outcomes','y','linear',[.1],k:=3)").fetchone()[0]
    assert score == 0.0
    loglik = con.execute("SELECT loglik FROM nbinom_dispersion('huge_outcomes','y',[1e-308])").fetchone()[0]
    # At this enormous count the NB2 density at its mean agrees with its
    # local normal approximation beyond double precision; variance is 2*y.
    expected = -3 * (np.log(2*np.pi) + np.log(1e308) + np.log(2.0))
    assert np.isfinite(loglik)
    assert loglik == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize('family', ['linreg','logit','poisson','gamma','tweedie','nbinom'])
@pytest.mark.parametrize('l2', [0.0, 0.1])
def test_fits_preserve_extreme_mixed_sign_feature_units(con, family, l2):
    outcome = 'i%2' if family == 'logit' else '1.0+i%3'
    con.execute(f'CREATE TABLE base AS SELECT CASE WHEN i%3=0 THEN -1.5 ELSE 1.5 END x,{outcome} y FROM range(12)t(i)')
    con.execute('CREATE TABLE scaled AS SELECT x*1e308 x,y FROM base')
    models = []
    for table in ['base', 'scaled']:
        coefficients = dict(con.execute(f"SELECT * FROM {family}_fit('{table}','y',l2:={l2},max_iter:=300)").fetchall())
        assert np.isfinite(list(coefficients.values())).all()
        models.append(coefficients)
    models[1]['x'] *= 1e308
    assert models[1] == pytest.approx(models[0], rel=1e-8, abs=1e-9)


def test_linear_fit_preserves_extreme_mixed_sign_outcome_units(con):
    con.execute('CREATE TABLE extremes AS SELECT * FROM (VALUES (-1.,-1.5e308),(1.,1.5e308),(1.,1.5e308))t(x,y)')
    coefficients = dict(con.execute("SELECT * FROM linreg_fit('extremes','y')").fetchall())
    assert coefficients['(Intercept)'] / 1e308 == pytest.approx(0.0, abs=1e-12)
    assert coefficients['x'] / 1e308 == pytest.approx(1.5, abs=1e-12)


@pytest.mark.parametrize('kind', ['multinomial', 'cv', 'dispersion'])
def test_batch_fits_preserve_extreme_mixed_sign_feature_units(con, kind):
    con.execute('CREATE TABLE base AS SELECT CASE WHEN i%3=0 THEN -1.5 ELSE 1.5 END x,1.0+i%3 y FROM range(12)t(i)')
    con.execute('CREATE TABLE scaled AS SELECT x*1e308 x,y FROM base')
    outputs = []
    for table in ['base', 'scaled']:
        if kind == 'multinomial':
            call = f"multinom_fit('{table}','y',l2:=.1,max_iter:=300)"
        elif kind == 'cv':
            call = f"cv_l2('{table}','y','linear',[.1],k:=4,max_iter:=300)"
        else:
            call = f"nbinom_dispersion('{table}','y',[.5,1.],max_iter:=300)"
        rows = con.execute('SELECT * FROM '+call+' ORDER BY ALL').fetchall()
        outputs.append(np.array([row[-1]*(1e308 if kind=='multinomial' and table=='scaled' and row[-2]=='x' else 1.0) for row in rows]))
    assert np.isfinite(outputs[1]).all()
    np.testing.assert_allclose(outputs[1], outputs[0], rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize('scale', [1e160, 1e200, 1e300])
@pytest.mark.parametrize('solver', ['auto', 'irls', 'gd'])
@pytest.mark.parametrize('l1,l2', [(0.,0.), (0.,.2), (.1,.2)])
@pytest.mark.parametrize('rare_outcome', [0., 2.])
def test_positive_weights_below_relative_double_range_still_affect_fit(con, scale, solver, l1, l2, rare_outcome):
    con.execute('CREATE TABLE varied_weights AS SELECT * FROM (VALUES (0.,0.,?),(1.,1.,?),(?,?,?))t(x,y,w)', [scale,scale,scale,rare_outcome*scale,1/scale])
    coefficients = dict(con.execute(f"SELECT * FROM linreg_fit('varied_weights','y',weights_col:='w',solver:='{solver}',l1:={l1},l2:={l2})").fetchall())
    # The root-weighted raw design tends to [[1,0],[1,1],[0,1]], with y=[0,1,rare_outcome].
    # Weighted mean(x)=mean(y)=1/2 and var(x)=3/4.
    covariance = .25+rare_outcome/2
    response_sd = np.sqrt(.25+rare_outcome**2/2)
    slope = (covariance/.75-l1*response_sd/np.sqrt(.75))/(1+l2)
    assert coefficients == pytest.approx({'(Intercept)': .5-.5*slope, 'x': slope}, abs=1e-8)
