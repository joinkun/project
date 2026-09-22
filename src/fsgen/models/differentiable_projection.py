"""射影(二次計画)の微分可能版(ステージ10 §3.3、CAFM用)。

``cvxpylayers`` で ``fsgen.accounting.projection_qp.project_onto_feasible`` と
同じ白色化QP定式化を構築する。重み(コスト)は企業ごとに固定(学習対象では
ない)なので、cvxpyの``Parameter``にするのは射影対象の点 ``y`` だけでよい
(DPP制約下で重みをパラメータ化すると非アフィンになり扱いにくいため)。

正規化空間(x̃ = x/Assets)で使うことを想定しており、ステージ7で必要だった
「円単位のまま(〜1e11)だと不安定」というスケーリング問題は最初から
起きない(x̃は概ねO(1))。ただし重み比の悪化(条件数の問題)は変わらず
起こりうるため、白色化は引き続き適用する。
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer

from fsgen.accounting.constraints import (
    BasisSpec,
    CanonicalSpec,
    build_constraint_matrix,
    state_vector_names,
)
from fsgen.accounting.projection_qp import DirectionConstraintSpec, evaluate_condition
from fsgen.errors import SolverNotConvergedError

# SCSの既定(eps=1e-4)では、通常の入力(|y|max=17)でも制約違反が相対2.8e-3残る例が学習中に
# 見つかった(ステージ13)。eps=1e-6にすると精度が約100倍良くなり、速度は変わらない。
_SCS_SOLVER_ARGS = {"eps": 1e-6, "max_iters": 20000}
_FORWARD_FEASIBILITY_TOL = 1e-3  # SCSの精度(既定eps=1e-4)を踏まえた緩めの許容(入力のスケール比)


@dataclass(frozen=True)
class ProjectionLayer:
    """企業1社ぶんの、微分可能な射影(cvxpylayers)。"""

    layer: CvxpyLayer
    sqrt_w: torch.Tensor  # (D,) float64
    names: tuple[str, ...]
    c: np.ndarray | None = None  # (D, m) 恒等式行列(前進解の検証用)
    signs: tuple[str, ...] = ()  # names順のsign("nonneg"/"nonpos"/"free")

    def project(self, y: torch.Tensor) -> torch.Tensor:
        """y(正規化空間、(D,)または(B,D))を射影した点を返す。勾配が通る。

        前進解が有限で、恒等式・符号制約を許容内で満たすことを確認し、
        破れていれば ``SolverNotConvergedError``(§A.1)。**これは前進解の
        検証であり、勾配の正しさは保証しない**(ステージ10.5で、前進は
        正常でも逆伝播が完全に破綻した事例がある)。勾配は
        ``check_layer_gradient`` で別途確認する。
        """
        (u,) = self.layer(y * self.sqrt_w, solver_args=_SCS_SOLVER_ARGS)
        result: torch.Tensor = u / self.sqrt_w
        scale = max(float(y.detach().abs().max()), 1.0)
        self._verify_forward(result.detach().numpy(), scale)
        return result

    def _verify_forward(self, x: np.ndarray, scale: float) -> None:
        if not np.isfinite(x).all():
            raise SolverNotConvergedError("微分可能射影の前進解にNaN/infが含まれる")
        if self.c is None:
            return
        resid = float(np.max(np.abs(x @ self.c)))
        viol = 0.0
        for i, sg in enumerate(self.signs):
            v = x[..., i]
            if sg == "nonneg":
                viol = max(viol, float(np.max(np.maximum(0.0, -v))))
            elif sg == "nonpos":
                viol = max(viol, float(np.max(np.maximum(0.0, v))))
        if max(resid, viol) > _FORWARD_FEASIBILITY_TOL * scale:
            raise SolverNotConvergedError(
                "微分可能射影の前進解が制約を満たさない: "
                f"恒等式残差={resid:.2e}、符号違反={viol:.2e}"
            )


def check_layer_gradient(
    layer: ProjectionLayer,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    n_points: int = 3,
    rel_step: float = 1e-4,
    max_median_rel_error: float = 0.1,
) -> float:
    """解析勾配(cvxpylayers)と中心差分の相対誤差の中央値を返し、大きければ例外(§A.1)。

    ステージ10.5で、重み比が極端(≥1e8)だと前進は``optimal``なのに解析勾配の
    相対誤差が~100%になる事例があった。学習に使う前に、少数の点で確認する。
    """
    dim = len(y)
    errors = []
    for _ in range(n_points):
        y_pt = y + rng.normal(scale=0.3, size=dim) * np.abs(y).mean()
        direction = rng.normal(size=dim)
        direction /= np.linalg.norm(direction)
        d_t = torch.tensor(direction, dtype=torch.float64)
        y_t = torch.tensor(y_pt, dtype=torch.float64, requires_grad=True)
        (layer.project(y_t) * d_t).sum().backward()  # type: ignore[no-untyped-call]
        assert y_t.grad is not None
        analytic = y_t.grad.numpy()
        numeric = np.zeros(dim)
        for i in range(dim):
            step = rel_step * max(abs(y_pt[i]), 1.0)
            hi, lo = y_pt.copy(), y_pt.copy()
            hi[i] += step
            lo[i] -= step
            with torch.no_grad():
                f_hi = float((layer.project(torch.tensor(hi, dtype=torch.float64)) * d_t).sum())
                f_lo = float((layer.project(torch.tensor(lo, dtype=torch.float64)) * d_t).sum())
            numeric[i] = (f_hi - f_lo) / (2 * step)
        errors.append(
            float(np.linalg.norm(analytic - numeric) / max(np.linalg.norm(numeric), 1e-8))
        )
    median = float(np.median(errors))
    if not np.isfinite(median) or median > max_median_rel_error:
        raise SolverNotConvergedError(
            f"微分可能射影の解析勾配が数値勾配と合わない: 相対誤差の中央値={median:.3f}"
            f"(許容{max_median_rel_error})"
        )
    return median


def build_projection_layer(
    spec: CanonicalSpec,
    x_start: np.ndarray,
    weights: np.ndarray,
    direction_constraints: tuple[DirectionConstraintSpec, ...] = (),
    frozen_accounts: tuple[str, ...] = (),
) -> ProjectionLayer:
    """企業1社の x_start・weights(固定)に対する微分可能射影層を構築する。

    project_onto_feasible と同じ定式化: u=sqrt(w)*x に置き換えて
    min ||u - sqrt(w)*y||^2 s.t. (C/sqrt(w))^T u = 0, 符号制約, 方向制約,
    frozen_accounts の等式制約。y だけを cvxpy Parameter にする。
    """
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    dim = len(names)
    index = {n: i for i, n in enumerate(names)}

    sqrt_w = np.sqrt(weights)
    sqrt_w_y = cp.Parameter(dim)  # sqrt(w)*y をパラメータとして渡す(project()側で乗算)
    u = cp.Variable(dim)
    objective = cp.Minimize(cp.sum_squares(u - sqrt_w_y))
    c_whitened = c / sqrt_w[:, None]

    constraints = [c_whitened.T @ u == 0]
    for b in spec.basis:
        i = index[b.name]
        if b.sign == "nonneg":
            constraints.append(u[i] >= 0)
        elif b.sign == "nonpos":
            constraints.append(u[i] <= 0)

    for dc in direction_constraints:
        if not evaluate_condition(dc.condition, x_start, spec):
            continue
        if dc.rule != "monotone_from_start":
            raise ValueError(f"unknown direction constraint rule: {dc.rule!r}")
        i = index[dc.name]
        sign = next(bb.sign for bb in spec.basis if bb.name == dc.name)
        u_start_i = sqrt_w[i] * x_start[i]
        if sign == "nonpos":
            constraints.append(u[i] >= u_start_i)
        elif sign == "nonneg":
            constraints.append(u[i] <= u_start_i)
        else:
            raise ValueError(f"monotone_from_start requires nonneg/nonpos, got {sign!r}")

    for name in frozen_accounts:
        i = index[name]
        constraints.append(u[i] == sqrt_w[i] * x_start[i])

    prob = cp.Problem(objective, constraints)
    layer = CvxpyLayer(prob, parameters=[sqrt_w_y], variables=[u])
    return ProjectionLayer(
        layer=layer, sqrt_w=torch.tensor(sqrt_w, dtype=torch.float64), names=names,
        c=c, signs=tuple(b.sign for b in _basis_in_order(spec, names)),
    )


def _basis_in_order(spec: CanonicalSpec, names: tuple[str, ...]) -> list[BasisSpec]:
    by_name = {b.name: b for b in spec.basis}
    return [by_name[n] for n in names]


__all__ = ["ProjectionLayer", "build_projection_layer", "check_layer_gradient"]
