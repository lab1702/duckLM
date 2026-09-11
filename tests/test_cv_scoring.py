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


@pytest.mark.parametrize("sweep,grid", [("power", "[1.0,1.5,2.0]"), ("alpha", "[0.1,1.0,10.0]")])
def test_refinement_retains_the_same_scoring_scale(con, sweep, grid):
    coarse = con.execute(f"SELECT cv_deviance FROM cv_{sweep}('balanced','y',{grid})").fetchall()
    refined = con.execute(
        f"SELECT cv_deviance FROM cv_{sweep}_refine('balanced','y',{grid},n_refine:=4)"
    ).fetchall()
    assert len(refined) == 4
    assert [row[0] for row in refined] == pytest.approx([coarse[0][0]] * 4, rel=1e-9)


@pytest.mark.parametrize('call,outcome', [
    ("cv_l2('bad_domain','y','logistic',[0.,1.],k:=3,max_iter:=50)",'1+i%2'),
    ("cv_l1('bad_domain','y','poisson',[0.,1.],k:=3,max_iter:=50)",'-1+i%2'),
    ("cv_l2('bad_domain','y','gamma',[0.,1.],k:=3,max_iter:=50)",'i%2'),
    ("cv_alpha('bad_domain','y',[.5,1.],k:=3,max_iter:=50)",'-1+i%2'),
    ("cv_power('bad_domain','y',[1.5,2.],k:=3,max_iter:=50)",'i%2'),
    ("cv_power('bad_domain','y',[1.,1.5],k:=3,max_iter:=50)",'-1+i%2'),
])
def test_cv_rejects_invalid_family_outcomes(con,call,outcome):
    con.execute(f'CREATE TABLE bad_domain AS SELECT i::DOUBLE x,({outcome})::DOUBLE y FROM range(12)t(i)')
    with pytest.raises(duckdb.Error,match='cv: outcome must'):
        con.execute(f'SELECT * FROM {call}').fetchall()
