"""基準単価 θ と企業ごとのコスト ψ_i(ステージ82)のテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from fsgen.models.unit_costs import (
    THETA_MAX,
    emergence_rate,
    estimate_theta,
    inactive_mask,
    individual_theta,
    kde_density_at_zero,
    pairwise_cost,
    psi,
)


def _pool() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x0 = rng.uniform(0.0, 0.5, size=(400, 3))
    sign = np.where(rng.uniform(size=(400, 3)) < 0.3, 1.0, -1.0)
    d = sign * rng.exponential(0.1, size=(400, 3)) * (x0 + 0.01)
    d[:, 2] = 0.0  # 動かない科目
    return d, x0, np.full(3, 0.01)


def test_r1_is_ln2_over_median_and_probability() -> None:
    d, x0, sig = _pool()
    th = estimate_theta(d, x0, sig, "R1").theta
    z = np.abs(d[:, 0]) / (x0[:, 0] + 0.01)
    up = d[:, 0] > 0
    assert th[0, 0] == pytest.approx(np.log(2) / np.median(z[up]) / up.mean())
    assert (th[:, 2] == THETA_MAX).all()  # 動いた企業がない向きは上限


def test_kde0_close_to_r1_for_exponential_widths() -> None:
    d, x0, sig = _pool()
    r1 = estimate_theta(d, x0, sig, "R1").theta[:, :2]
    kde = estimate_theta(d, x0, sig, "KDE0")
    # 指数分布の f(0) = 1/θ = ln2/median なので、大標本では近い(鏡像の KDE は 0 で下向きの偏り)
    assert np.all(np.abs(np.log(kde.theta[:, :2] / r1)) < 0.35)
    assert kde.n_fallback == 0


def test_kde_density_matches_known_exponential() -> None:
    z = np.random.default_rng(3).exponential(2.0, size=20000)
    assert kde_density_at_zero(z) == pytest.approx(0.5, rel=0.1)


def test_modes_ipw_scale_a1() -> None:
    d, x0, sig = _pool()
    up = d[:, 0] > 0
    z = np.abs(d[:, 0]) / (x0[:, 0] + 0.01)
    m = np.median(z[up])
    assert estimate_theta(d, x0, sig, "IPW").theta[0, 0] == pytest.approx(1 / up.mean())
    assert estimate_theta(d, x0, sig, "scale").theta[0, 0] == pytest.approx(1 / m)
    om = np.minimum(z[up], m).mean()
    assert estimate_theta(d, x0, sig, "A1").theta[0, 0] == pytest.approx(1 / (up.mean() * om))


def test_individual_theta_scaling_and_inactive_cap() -> None:
    th = np.array([[2.0, 4.0], [3.0, 5.0]])
    x0 = np.array([0.0, 0.5])
    sig = np.array([0.01, 0.01])
    out = individual_theta(th, x0, sig, indiv_scale=True, inactive=np.array([True, False]))
    assert out[0, 0] == THETA_MAX
    assert out[1, 0] == pytest.approx(300.0)
    assert out[0, 1] == pytest.approx(4.0 / 0.51)
    same = individual_theta(th, x0, sig, indiv_scale=False)
    assert np.array_equal(same, th)


def test_pairwise_cost_uses_row_specific_theta() -> None:
    xs = np.array([[0.0, 0.0], [1.0, 1.0]])
    xt = np.array([[1.0, -1.0]])
    rows = np.array([[[1.0, 1.0], [2.0, 2.0]], [[10.0, 10.0], [20.0, 20.0]]])
    c = pairwise_cost(xs, xt, rows)
    assert c[0, 0] == pytest.approx(1.0 * 1 + 2.0 * 1)
    assert c[1, 0] == pytest.approx(20.0 * 2)
    assert psi(rows[0], xt[0] - xs[0]) == pytest.approx(c[0, 0])


def test_inactive_mask_and_emergence() -> None:
    names = ("inventories", "jppfs_cor_CashAndDeposits", "long_term_loans")
    x0 = np.array([[0.0, 0.0, 0.2], [0.0, 0.1, 0.0]])
    assert inactive_mask(x0[0], names).tolist() == [True, False, False]
    y = np.array([[2e-3, 0.5, 0.2], [0.0, 0.1, 5e-6]])
    # 組は (0, 棚卸) と (1, 棚卸)・(1, 長期借入金) の 3 つ
    assert emergence_rate(x0, y, names, 1e-6) == pytest.approx(2 / 3)
    assert emergence_rate(x0, y, names, 1e-3) == pytest.approx(1 / 3)


def test_unknown_mode_raises() -> None:
    d, x0, sig = _pool()
    with pytest.raises(ValueError, match="mode"):
        estimate_theta(d, x0, sig, "bogus")
