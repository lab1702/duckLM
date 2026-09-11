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
    con.execute(f"CREATE TABLE large_model AS SELECT * FROM linreg_fit('large_response','y',solver:='{solver}')")
    coefficients = dict(con.execute("SELECT * FROM large_model").fetchall())
    assert np.isfinite(list(coefficients.values())).all()
    assert coefficients['x']/1e307 == pytest.approx(1.0, abs=1e-8)
    assert coefficients['z']/1e307 == pytest.approx(-1.0, abs=1e-8)
    assert abs(coefficients['(Intercept)']/1e308) < 1e-8
    predictions = np.array(con.execute("SELECT prediction/1e308 FROM linreg_predict('large_model','large_response')").fetchall()).ravel()
    np.testing.assert_allclose(predictions, [1.,-1.,1.,-1.,1.], atol=1e-8)
    rmse, r2 = con.execute("SELECT rmse/1e308,r2 FROM linreg_evaluate('large_model','large_response','y')").fetchone()
    assert rmse < 1e-8 and r2 == pytest.approx(1.)
    intervals = np.array(con.execute("SELECT prediction,conf_low,conf_high FROM linreg_predict_ci('large_model','large_response','y')").fetchall())
    assert np.isfinite(intervals).all()



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


@pytest.mark.parametrize('solver',['auto','irls','gd'])
@pytest.mark.parametrize('l2',[0.,.2])
@pytest.mark.parametrize('with_offset',[False,True])
def test_linear_fit_uses_weighted_coordinates_when_raw_standardization_overflows(con, solver, l2, with_offset):
    target_slope = -.5 if with_offset else 1.
    con.execute('CREATE TABLE weighted_coordinates(x DOUBLE,y DOUBLE,w DOUBLE,expo DOUBLE)')
    rows = [(x,(target_slope+(1. if with_offset else 0.))*x,w,x if with_offset else 0.) for x,w in [(0.,1e308),(1.,1e308),(1e308,1e-310)]]
    con.executemany('INSERT INTO weighted_coordinates VALUES (?,?,?,?)',rows)
    con.execute(f"CREATE TABLE weighted_model AS SELECT * FROM linreg_fit('weighted_coordinates','y',weights_col:='w',offset_col:='expo',solver:='{solver}',l2:={l2},max_iter:=1000)")
    slope = target_slope/(1+l2)
    intercept = .5*(target_slope-slope)
    coefficients = dict(con.execute('SELECT * FROM weighted_model').fetchall())
    assert coefficients == pytest.approx({'(Intercept)':intercept,'x':slope},abs=1e-8)
    predictions = np.array(con.execute("SELECT prediction FROM linreg_predict('weighted_model','weighted_coordinates',offset_col:='expo')").fetchall()).ravel()
    expected = [intercept+(slope+(1. if with_offset else 0.))*r[0] for r in rows]
    np.testing.assert_allclose(predictions,expected,rtol=1e-8,atol=1e-8)


@pytest.mark.parametrize('solver', ['auto', 'gd'])
@pytest.mark.parametrize('direction', [-1., 1.])
@pytest.mark.parametrize('offset', [0., 1.5])
def test_weighted_logistic_fit_handles_overflowing_unweighted_coordinates(con, solver, direction, offset):
    from scipy.optimize import brentq
    from scipy.special import expit

    con.execute('CREATE TABLE weighted_coordinates(x DOUBLE,y DOUBLE,w DOUBLE,expo DOUBLE)')
    con.executemany('INSERT INTO weighted_coordinates VALUES (?,?,?,?)',
                    [(0.,0.,1e308,offset),(direction,1.,1e308,offset),(direction*1e308,1.,1e-310,offset)])
    coefficients = dict(con.execute(f"SELECT * FROM logit_fit('weighted_coordinates','y',weights_col:='w',offset_col:='expo',solver:='{solver}',l2:=.2,max_iter:=100)").fetchall())
    # The tiny row contributes .005 to weighted feature variance, but its
    # likelihood gradient vanishes at the positive-margin ridge optimum.
    slope = brentq(lambda s: .5*(expit(s/2)-1)+.2*.255*s,0.,10.)
    assert coefficients == pytest.approx({'(Intercept)':-slope/2-offset,'x':direction*slope},abs=1e-8)


@pytest.mark.parametrize('solver', ['auto', 'irls', 'gd'])
@pytest.mark.parametrize('scale', [1e20, 1e100, 1e300])
@pytest.mark.parametrize('l1', [0., .05])
def test_negligible_weight_negative_outlier_preserves_penalty_standardization(con, solver, scale, l1):
    con.execute('CREATE TABLE weighted_center(x DOUBLE,y DOUBLE,w DOUBLE)')
    con.executemany('INSERT INTO weighted_center VALUES (?,?,?)',[(0.,0.,scale),(1.,1.,scale),(-scale,0.,1/scale)])
    coefficients = dict(con.execute(f"SELECT * FROM linreg_fit('weighted_center','y',weights_col:='w',solver:='{solver}',l2:=.2,l1:={l1})").fetchall())
    # Exact weighted moments approach mu_x=mu_y=.5, var_x=.75,
    # var_y=.25 and cov_xy=.25; all omitted terms are below 1e-19.
    slope = (.25-l1*np.sqrt(.75)*.5)/(.75*1.2)
    assert coefficients == pytest.approx({'(Intercept)':.5-.5*slope,'x':slope},abs=1e-8)


@pytest.mark.parametrize('solver', ['auto', 'irls', 'gd'])
@pytest.mark.parametrize('family,power', [('poisson',None),('gamma',None),('tweedie',1.5),('tweedie',3.),('nbinom',None)])
@pytest.mark.parametrize('slope', [0., .2])
def test_log_link_fit_preserves_overflowing_mean_scaled_response_contributions(con, solver, family, power, slope):
    con.execute('CREATE TABLE weighted_response(x DOUBLE,y DOUBLE,w DOUBLE)')
    rows = [(-1.,1e-300,1.),(1.,1e-300,1.)]
    if slope == 0:
        rows += [(0.,1e300,1e-320)]
        mean_unit = (1e-320*1e300)/2
    else:
        rows += [(x,1e300*np.exp(slope*x),1e-320) for x in [-1.,1.]]
        mean_unit = 1e-320*1e300
    con.executemany('INSERT INTO weighted_response VALUES (?,?,?)',rows)
    extra = '' if power is None else f',power:={power}'
    coefficients = dict(con.execute(f"SELECT * FROM {family}_fit('weighted_response','y',weights_col:='w',solver:='{solver}',max_iter:=1000{extra})").fetchall())
    # Within each x group, every family has its likelihood/quasi-likelihood
    # optimum at the weighted mean; symmetry fixes the intercept-only case.
    assert coefficients == pytest.approx({'(Intercept)':np.log(mean_unit),'x':slope},abs=1e-8)


@pytest.mark.parametrize('solver', ['auto','irls','gd'])
@pytest.mark.parametrize('scale', [1e-200,1e-310])
@pytest.mark.parametrize('offset', [1.,1e200])
def test_linear_constant_offsets_preserve_tiny_response_differences(con, solver, scale, offset):
    con.execute('CREATE TABLE tiny_response AS SELECT i::DOUBLE x,i*? y,?::DOUBLE o FROM range(-1,2)t(i)',[scale,offset])
    con.execute(f"CREATE TABLE tiny_model AS SELECT * FROM linreg_fit('tiny_response','y',offset_col:='o',solver:='{solver}',max_iter:=1000)")
    coefficients = dict(con.execute('SELECT * FROM tiny_model').fetchall())
    assert coefficients['(Intercept)'] == -offset
    assert coefficients['x']/scale == pytest.approx(1.,abs=1e-8)
    prediction = con.execute("SELECT prediction/? FROM linreg_predict('tiny_model','tiny_response',offset_col:='o') ORDER BY x",[scale]).fetchall()
    np.testing.assert_allclose(np.array(prediction).ravel(),[-1.,0.,1.],atol=1e-8,rtol=0.)


@pytest.mark.parametrize('solver', ['auto','irls','gd'])
@pytest.mark.parametrize('scale', [1e200,1e308])
def test_linear_varying_offsets_do_not_overflow_solver_sums(con, solver, scale):
    con.execute('CREATE TABLE large_offset AS SELECT i::DOUBLE x,1.0 y,i*? o FROM range(-1,2)t(i)',[scale])
    coefficients = dict(con.execute(f"SELECT * FROM linreg_fit('large_offset','y',offset_col:='o',solver:='{solver}',max_iter:=1000)").fetchall())
    assert coefficients['(Intercept)'] == pytest.approx(1.,abs=1e-8)
    assert coefficients['x']/scale == pytest.approx(-1.,abs=1e-8)


@pytest.mark.parametrize('solver', ['auto','irls','gd'])
@pytest.mark.parametrize('weighted', [False,True])
def test_linear_offset_normalization_preserves_outcome_based_l1_penalty(con, solver, weighted):
    x = np.arange(-2.,3.)
    y = 3+.2*x+np.array([.1,-.2,.3,-.1,.2])
    o = 5+30*x
    w = np.arange(1.,6.) if weighted else np.ones(5)
    con.execute('CREATE TABLE offset_penalty(x DOUBLE,y DOUBLE,o DOUBLE,w DOUBLE)')
    con.executemany('INSERT INTO offset_penalty VALUES (?,?,?,?)',list(zip(x,y,o,w)))
    mean_x,mean_y,mean_o = [np.average(v,weights=w) for v in [x,y,o]]
    var_x = np.average((x-mean_x)**2,weights=w)
    var_y = np.average((y-mean_y)**2,weights=w)
    covariance = np.average((x-mean_x)*(y-o-mean_y+mean_o),weights=w)
    slope = np.sign(covariance)*max(abs(covariance)-.05*np.sqrt(var_x*var_y),0)/(var_x*1.2)
    coefficients = dict(con.execute(f"SELECT * FROM linreg_fit('offset_penalty','y',offset_col:='o',weights_col:='w',solver:='{solver}',l1:=.05,l2:=.2)").fetchall())
    assert coefficients == pytest.approx({'(Intercept)':mean_y-mean_o-slope*mean_x,'x':slope},abs=1e-7)


@pytest.mark.parametrize('solver', ['auto', 'irls', 'gd'])
def test_linear_intercept_cancels_before_restoring_outcome_units(con, solver):
    con.execute('CREATE TABLE finite_intercept AS SELECT * FROM (VALUES (-2.5,-1e308),(-2.,-5e307),(-1.5,0.))t(x,y)')
    con.execute(f"CREATE TABLE finite_model AS SELECT * FROM linreg_fit('finite_intercept','y',solver:='{solver}')")
    coefficients = dict(con.execute('SELECT * FROM finite_model').fetchall())
    assert coefficients['(Intercept)']/1e308 == pytest.approx(1.5, abs=1e-8)
    assert coefficients['x']/1e308 == pytest.approx(1., abs=1e-8)
    predictions = np.array(con.execute("SELECT prediction FROM linreg_predict('finite_model','finite_intercept')").fetchall()).ravel()
    np.testing.assert_allclose(predictions/1e308, [-1., -.5, 0.], atol=1e-8)


@pytest.mark.parametrize('solver', ['auto', 'irls', 'gd'])
@pytest.mark.parametrize('shift', [-400., 0., 400.])
@pytest.mark.parametrize('offset_slope', [0., .1])
@pytest.mark.parametrize('weighted', [False, True])
def test_tweedie_fit_preserves_large_common_offset_shifts(con, solver, shift, offset_slope, weighted):
    con.execute('CREATE TABLE shifted_tweedie AS SELECT i::DOUBLE x,exp(.3*i) y,?+?*i expo FROM range(6)t(i)', [shift, offset_slope])
    if weighted:
        con.execute('ALTER TABLE shifted_tweedie ADD COLUMN w DOUBLE DEFAULT 1.')
        con.execute('UPDATE shifted_tweedie SET w = x+1')
    weights = ",weights_col:='w'" if weighted else ''
    coefficients = dict(con.execute(f"SELECT * FROM tweedie_fit('shifted_tweedie','y',power:=3,offset_col:='expo',solver:='{solver}'{weights})").fetchall())
    # Every observation is exactly fitted, independently of positive weights.
    assert coefficients['(Intercept)'] == pytest.approx(-shift, abs=1e-8)
    assert coefficients['x'] == pytest.approx(.3-offset_slope, abs=1e-8)
