"""分布由来の輸送コスト(ステージ50)のテスト。"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from fsgen.models.likelihood_cost import (
    AccountModel,
    account_cost,
    conditional_probs,
    likelihood_cost_matrix,
    neg_log_density,
    scale_period,
)


def _model(family: str = "t", shape: float = 1.5) -> AccountModel:
    p = conditional_probs(70, 30, 10, 85, 5, 1e-4)
    return AccountModel(family, 0.1, 0.3, shape, **p)  # type: ignore[arg-type]


@pytest.mark.parametrize(("family", "shape", "ref"), [
    ("normal", 0.0, stats.norm(0.1, 0.3)),
    ("t", 1.5, stats.t(1.5, 0.1, 0.3)),
    ("subbotin", 0.6, stats.gennorm(0.6, 0.1, 0.3)),
])
def test_density_matches_scipy(family: str, shape: float, ref: object) -> None:
    g = np.linspace(-3, 3, 41)
    got = neg_log_density(_model(family, shape), g)
    np.testing.assert_allclose(got, -ref.logpdf(g), rtol=1e-10)  # type: ignore[attr-defined]


def test_conditional_probs_and_clip() -> None:
    p = conditional_probs(70, 30, 10, 85, 5, 1e-4)
    assert p["p_stay0"] == pytest.approx(0.7) and p["p_birth"] == pytest.approx(0.3)
    assert p["p_cont"] + p["p_death"] + p["p_flip"] == pytest.approx(1.0)
    q = conditional_probs(3, 0, 0, 100, 0, 1e-4)  # 新規計上・消滅・反転が 0 件
    assert q["p_birth"] == q["p_death"] == q["p_flip"] == 1e-4
    r = conditional_probs(0, 0, 0, 100, 1, 1e-4)  # 始点 0 の観測がない
    assert r["p_stay0"] == r["p_birth"] == 1e-4


def test_account_cost_categories() -> None:
    m = _model()
    x = np.array([0.0, 0.2, -0.2])
    y = np.array([0.0, 0.3, -0.1])
    c = account_cost(m, x, y)
    assert c[0, 0] == pytest.approx(-np.log(m.p_stay0))
    assert c[0, 1] == pytest.approx(-np.log(m.p_birth))
    assert c[1, 0] == pytest.approx(-np.log(m.p_death))
    assert c[1, 2] == pytest.approx(-np.log(m.p_flip))
    g = np.log(0.3 / 0.2)
    assert c[1, 1] == pytest.approx(-np.log(m.p_cont) + neg_log_density(m, np.array([g]))[0])
    g2 = np.log(0.1 / 0.2)  # 負の科目は絶対値の比
    assert c[2, 2] == pytest.approx(-np.log(m.p_cont) + neg_log_density(m, np.array([g2]))[0])


def test_matrix_skips_none_and_is_finite() -> None:
    m = _model()
    x = np.array([[0.2, 5.0], [0.0, -1.0]])
    y = np.array([[0.1, 0.0], [0.0, 3.0], [0.4, 7.0]])
    c = likelihood_cost_matrix(x, y, [m, None])
    np.testing.assert_allclose(c, account_cost(m, x[:, 0], y[:, 0]))
    assert np.isfinite(c).all()


def test_scale_period() -> None:
    m = scale_period(_model(), 5.0)
    assert m.loc == pytest.approx(0.5) and m.scale == pytest.approx(0.3 * np.sqrt(5))
    assert m.p_birth == _model().p_birth
