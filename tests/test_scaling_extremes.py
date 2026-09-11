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
