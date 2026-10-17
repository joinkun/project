"""ステージ66: 純資産を階層分離した第 1 項と、白色化した非対称 Elastic-Net の移動コストによる凸 QP。

    min_y  ½ Σ_{a∈A_BS} w_a (y_a − τ_a)² + ½ w_NA (NA(y) − NA(τ))²
           + λ [ Σ_a (ũ_a + ṽ_a) + (η/2) Σ_a (ũ_a² + ṽ_a²) ]
    s.t.   y_a − x_a = ũ_a / c⁺_a − ṽ_a / c⁻_a,  ũ, ṽ ≥ 0   (c± > 0 の科目)
           Cᵀy = 0、基礎科目の符号、凍結、方向制約、追加の線形制約 a·y ≥ rhs

ũ = c⁺u、ṽ = c⁻v と変数変換すると、移動コストの係数がすべて 1 になる(白色化)。
第 1 項は A_BS の科目を z_a = √w_a · y_a に置き換えて白色化する。
目標 τ は ``cvxpy.Parameter`` なので、同じ始点で目標だけを変えて何度も解ける
(1 段階統合の枝刈り用)。収束しない・最適でない・制約違反が大きい場合は例外にする。

``lower_bound`` は、NA の項と制約を外した、科目ごとに分離した 1 次元問題の和の閉形式で、
QP の最適値の下界になる(外した項は非負、制約を外すと最小値は小さくなる)。
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from fsgen.accounting.constraints import (
    CanonicalSpec,
    build_constraint_matrix,
    state_vector_names,
)
from fsgen.accounting.plausibility import LinearConstraint
from fsgen.accounting.projection_qp import DirectionConstraintSpec, evaluate_condition
from fsgen.errors import InfeasibleProjectionError, SolverNotConvergedError

_VIOL_TOL = 1e-7
_SOLVERS: tuple[tuple[str, dict[str, object]], ...] = (
    (cp.CLARABEL, {}),
    (cp.OSQP, {"eps_abs": 1e-10, "eps_rel": 1e-10, "max_iter": 400000, "polish": True}),
    (cp.ECOS, {}),
)


@dataclass(frozen=True)
class RecourseWeights:
    w_bs: np.ndarray  # (D,) A_BS の外は 0
    na_coef: np.ndarray  # (D,) NA(y) = na_coef · y
    w_na: float
    c_plus: np.ndarray  # (D,) 0 の科目は移動の項なし
    c_minus: np.ndarray
    eta: float


@dataclass(frozen=True)
class RecourseResult:
    y: np.ndarray
    objective: float
    status: str
    move_up: np.ndarray  # u(元の単位)
    move_down: np.ndarray
    max_violation: float


def recourse_objective(y: np.ndarray, x: np.ndarray, tau: np.ndarray, rw: RecourseWeights,
                       lam: float) -> float:
    """目的関数の値(元の単位)。u = max(y − x, 0)、v = max(x − y, 0) とする(最適解では一致)。"""
    d = y - x
    up = np.where(rw.c_plus > 0, np.maximum(d, 0.0) * rw.c_plus, 0.0)
    dn = np.where(rw.c_minus > 0, np.maximum(-d, 0.0) * rw.c_minus, 0.0)
    first = 0.5 * float(np.sum(rw.w_bs * (y - tau) ** 2))
    na = 0.5 * rw.w_na * float(rw.na_coef @ (y - tau)) ** 2
    move = float(np.sum(up + dn) + 0.5 * rw.eta * np.sum(up ** 2 + dn ** 2))
    return first + na + lam * move


def lower_bound(x: np.ndarray, targets: np.ndarray, rw: RecourseWeights,
                lam: float) -> np.ndarray:
    """各目標 y_j について、QP の最適値の下界(n_targets,)。

    NA の項と制約を外した、科目ごとに分離した 1 次元問題の閉形式の和。
    """
    w = rw.w_bs[None, :]
    cpl, cmi = rw.c_plus[None, :], rw.c_minus[None, :]
    delta = targets - x[None, :]
    up_d = (w * delta - lam * cpl) / (w + lam * rw.eta * cpl ** 2 + 1e-300)
    dn_d = (w * delta + lam * cmi) / (w + lam * rw.eta * cmi ** 2 + 1e-300)
    d = np.where(w * delta > lam * cpl, up_d, np.where(w * delta < -lam * cmi, dn_d, 0.0))
    u = np.maximum(d, 0.0) * cpl
    v = np.maximum(-d, 0.0) * cmi
    val = 0.5 * w * (d - delta) ** 2 + lam * (u + v + 0.5 * rw.eta * (u ** 2 + v ** 2))
    val = np.where(w > 0, val, 0.0)  # A_BS の外は動かさないのが最小(0)
    out: np.ndarray = val.sum(axis=1)
    return out


class RecourseQP:
    """1 社の始点 x について組み立てた QP。目標 τ を変えて ``solve`` を繰り返し呼べる。"""

    def __init__(self, spec: CanonicalSpec, x_start: np.ndarray, rw: RecourseWeights, lam: float,
                 *, extra_constraints: tuple[LinearConstraint, ...] = (),
                 direction_constraints: tuple[DirectionConstraintSpec, ...] = (),
                 frozen_accounts: tuple[str, ...] = ()) -> None:
        names = state_vector_names(spec, include_subtotals=False)
        index = {n: i for i, n in enumerate(names)}
        dim = len(names)
        self.x, self.rw, self.lam, self.spec = x_start, rw, lam, spec
        bs = rw.w_bs > 0
        scale = np.where(bs, 1.0 / np.sqrt(np.where(bs, rw.w_bs, 1.0)), 1.0)
        self.scale = scale
        self.z = cp.Variable(dim)
        y = cp.multiply(scale, self.z)
        self.tau = cp.Parameter(dim)
        self.tau_bs = cp.Parameter(int(bs.sum()))  # √w ⊙ τ(A_BS)
        self.na_tau = cp.Parameter()
        self.bs = bs
        mv = (rw.c_plus > 0) & (rw.c_minus > 0)
        self.mv = mv
        n_mv = int(mv.sum())
        self.ut = cp.Variable(n_mv, nonneg=True)
        self.vt = cp.Variable(n_mv, nonneg=True)
        obj = 0.5 * cp.sum_squares(self.z[np.where(bs)[0]] - self.tau_bs)
        obj += 0.5 * rw.w_na * cp.square(rw.na_coef @ y - self.na_tau)
        obj += lam * (cp.sum(self.ut) + cp.sum(self.vt)
                      + 0.5 * rw.eta * (cp.sum_squares(self.ut) + cp.sum_squares(self.vt)))
        idx_mv = np.where(mv)[0]
        cons = [build_constraint_matrix(spec, include_subtotals=False).T @ y == 0,
                y[idx_mv] - x_start[idx_mv] == cp.multiply(1.0 / rw.c_plus[mv], self.ut)
                - cp.multiply(1.0 / rw.c_minus[mv], self.vt)]
        for b in spec.basis:
            i = index[b.name]
            if b.sign == "nonneg":
                cons.append(y[i] >= 0)
            elif b.sign == "nonpos":
                cons.append(y[i] <= 0)
        for name in frozen_accounts:
            cons.append(y[index[name]] == x_start[index[name]])
        sign_of = {b.name: b.sign for b in spec.basis}
        for dc in direction_constraints:
            if not evaluate_condition(dc.condition, x_start, spec):
                continue
            i = index[dc.name]
            if sign_of[dc.name] == "nonpos":
                cons.append(y[i] >= x_start[i])
            elif sign_of[dc.name] == "nonneg":
                cons.append(y[i] <= x_start[i])
            else:
                raise ValueError(f"monotone_from_start には符号が要る: {dc.name}")
        for lc in extra_constraints:
            cons.append(lc.a @ y >= lc.rhs)
        self.extra = extra_constraints
        self.cmat = build_constraint_matrix(spec, include_subtotals=False)
        self.prob = cp.Problem(cp.Minimize(obj), cons)
        self.n_fallback = 0  # CLARABEL 以外で最適解を得た回数

    def _violation(self, y: np.ndarray) -> float:
        viol = float(np.abs(y @ self.cmat).max()) if self.cmat.size else 0.0
        for lc in self.extra:
            viol = max(viol, float(lc.rhs - lc.a @ y))
        return viol

    def solve(self, tau: np.ndarray) -> RecourseResult:
        rw = self.rw
        self.tau.value = tau
        self.tau_bs.value = np.sqrt(rw.w_bs[self.bs]) * tau[self.bs]
        self.na_tau.value = float(rw.na_coef @ tau)
        status, last_viol = "unsolved", float("nan")
        tol = _VIOL_TOL * max(1.0, float(np.abs(self.x).max()))
        y: np.ndarray | None = None
        # CLARABEL で最適解でない・制約違反が許容を超える場合は、OSQP(高精度)、ECOS の順に解き直す
        for name, kwargs in _SOLVERS:
            try:
                self.prob.solve(solver=name, **kwargs)
            except cp.error.SolverError:
                continue
            status = str(self.prob.status)
            if "infeasible" in status:
                raise InfeasibleProjectionError(f"QP が実行不可能: {status}")
            if status != "optimal" or self.z.value is None:
                continue
            cand = self.scale * np.asarray(self.z.value, dtype=float)
            viol = self._violation(cand)
            last_viol = viol
            if viol <= tol:
                y = cand
                if name != cp.CLARABEL:
                    self.n_fallback += 1
                break
        if y is None:
            raise SolverNotConvergedError(
                f"QP の解が得られない(状態 {status}、最後の制約違反 {last_viol:.3e}、"
                f"許容 {tol:.1e})")
        viol = self._violation(y)
        up = np.zeros(len(y))
        dn = np.zeros(len(y))
        up[self.mv] = np.asarray(self.ut.value) / rw.c_plus[self.mv]
        dn[self.mv] = np.asarray(self.vt.value) / rw.c_minus[self.mv]
        return RecourseResult(y=y, objective=recourse_objective(y, self.x, tau, rw, self.lam),
                              status=status, move_up=up, move_down=dn, max_violation=viol)


__all__ = ["RecourseQP", "RecourseResult", "RecourseWeights", "lower_bound",
           "recourse_objective"]
