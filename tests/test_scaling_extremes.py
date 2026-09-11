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


@pytest.mark.parametrize('scale', [1e-170,1e170])
def test_multinomial_fit_preserves_extreme_feature_units(con,scale):
    con.execute('CREATE TABLE base AS SELECT i/10.0 x,(i%3)::VARCHAR y FROM range(15)t(i)')
    con.execute(f'CREATE TABLE scaled AS SELECT * REPLACE(x*{scale} AS x) FROM base')
    models=[]
    for table in ['base','scaled']:
        rows=con.execute(f"SELECT * FROM multinom_fit('{table}','y',l2:=.1,max_iter:=1000)").fetchall()
        models.append({(label,feature):value for label,feature,value in rows})
    actual={key:value*scale if key[1]=='x' else value for key,value in models[1].items()}
    assert actual==pytest.approx(models[0],rel=1e-7,abs=1e-8)


@pytest.mark.parametrize('scale', [1e-170,1e170])
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


@pytest.mark.parametrize('scale', [1e-170,1e170])
def test_dispersion_profile_preserves_extreme_feature_units(con,scale):
    con.execute('CREATE TABLE base AS SELECT i/10.0 x,1.0+i%3 y FROM range(12)t(i)')
    con.execute(f'CREATE TABLE scaled AS SELECT * REPLACE(x*{scale} AS x) FROM base')
    profiles=[]
    for table in ['base','scaled']:
        profiles.append(con.execute(f"SELECT * FROM nbinom_dispersion('{table}','y',alpha_grid:=[.5,1.],max_iter:=300)").fetchall())
    np.testing.assert_allclose(np.array(profiles[1],dtype=float),np.array(profiles[0],dtype=float),rtol=1e-7,atol=1e-8)
