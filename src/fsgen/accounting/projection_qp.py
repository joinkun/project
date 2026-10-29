"""符号制約・方向制約つきの実行可能集合への射影(二次計画、ステージ7)。

実行可能集合 C_i は次の3つの交わりとして定義する:

    {x : C^T x = 0}                     会計恒等式(貸借一致)
    ∩ {符号制約}                          core.yaml の sign(nonneg/nonpos/free)
    ∩ {方向制約_i}                        企業ごとに異なる(例: 自己株式)

微分可能性はこの段階では不要なので `cvxpy` で素直に二次計画として解く
(微分可能層は Flow Matching を導入する次段階の課題)。
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import numpy as np
import yaml

from fsgen.accounting.constraints import (
    BasisSpec,
    CanonicalSpec,
    build_constraint_matrix,
    state_vector_names,
)
from fsgen.accounting.plausibility import INACTIVE_ZERO, LinearConstraint, subtotal_members
from fsgen.errors import InfeasibleProjectionError, SolverNotConvergedError

_DEFAULT_WEIGHT_FLOOR = 1e-3  # 重みの床(ステージ8 §A.5)。理由は_default_weightsのdocstring参照。


@dataclass(frozen=True)
class DirectionConstraintSpec:
    """企業ごとに条件付きで有効になる方向制約(ステージ7 §2.3)。"""

    name: str  # 対象基礎科目名
    condition: str  # 例: "net_assets_negative"
    rule: str  # 例: "monotone_from_start" (x(t) >= x(0) を科目のsign方向に解釈)


def load_direction_constraints(path: Path) -> tuple[DirectionConstraintSpec, ...]:
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    raw = doc.get("direction_constraints", [])
    return tuple(
        DirectionConstraintSpec(name=d["name"], condition=d["condition"], rule=d["rule"])
        for d in raw
    )


def evaluate_condition(condition: str, x_start: np.ndarray, spec: CanonicalSpec) -> bool:
    """方向制約の発動条件を、企業ごとの基準点 x_start について判定する。"""
    if condition == "net_assets_negative":
        net_assets_idx = [i for i, b in enumerate(spec.basis) if b.side == "net_assets"]
        return bool(np.sum(x_start[net_assets_idx]) < 0)
    raise ValueError(f"unknown direction constraint condition: {condition!r}")


@dataclass(frozen=True)
class ProjectionInfo:
    status: str  # cvxpy の prob.status ("optimal" / "infeasible" / "optimal_inaccurate" 等)
    distance_before_after: float  # ‖x_projected - y‖(円単位、非加重)
    max_constraint_violation: float  # 解いた後の制約違反の最大値(数値誤差レベルのはず)
    solve_time_seconds: float
    active_constraints: tuple[str, ...] = ()  # 拘束(スラック≈0)している追加制約の名前
    # ステージ60: L1 の移動コストを足したときの増加分 u と減少分 v(元の単位)。なければ None
    move_up: np.ndarray | None = None
    move_down: np.ndarray | None = None


def _default_weights(x_start: np.ndarray, assets: float) -> np.ndarray:
    """W = diag(1 / (|x_start| + delta*Assets))(§3.1、ステージ8 §A.5で床を導入)。

    当初(ステージ7)は eps=assets*1e-6 (ゼロ割回避のみが目的の極小値)だった。
    この場合、科目間の重み比が実データで最大54万倍に達し(白色化で対処済み)、
    さらにステージ8 §A.5で68社全体を調べると最大3,035万倍に達する社が
    見つかった(オンコセラピー・サイエンス株式会社: 資本剰余金263億円が
    総資産8.67億円の30倍—欠損の累積で総資産が縮小した結果であり、
    データ誤りではない)。

    delta=1e-3(総資産の0.1%)を床として使うと、通常のケース(全科目が
    総資産以下)では重み比は高々1/delta=1,000倍程度に収まる(上記のような
    純資産側の科目が総資産を上回る企業では、床を入れてもなお比が大きく
    なりうる)。実測で比較した結果、床ありの重みで解いた結果は現行(床なし)
    とほぼ一致した(68社中、解の相対差は最大2.2e-4、中央値1.8e-5)ため、
    数値的な安全マージンとして床を採用する。ただし床だけでは白色化を
    省略できるとは限らない(床ありでも比が数万倍に達する社があり、
    白色化なしで解くと現行解から最大100%相当乖離することを確認済み)ため、
    白色化(``u = sqrt(w) ⊙ x``)は引き続き常に行う。
    """
    floor = assets * _DEFAULT_WEIGHT_FLOOR
    return 1.0 / (np.abs(x_start) + floor)


def _max_violation(
    x: np.ndarray, spec: CanonicalSpec, c: np.ndarray, index: dict[str, int]
) -> float:
    violations = [float(np.max(np.abs(c.T @ x)))]
    for b in spec.basis:
        v = x[index[b.name]]
        if b.sign == "nonneg":
            violations.append(max(0.0, -v))
        elif b.sign == "nonpos":
            violations.append(max(0.0, v))
    return max(violations)


def _start_assets(spec: CanonicalSpec, names: tuple[str, ...], x_start: np.ndarray) -> float:
    """始点の総資産の絶対値(資産合計の小計がない仕様では max|x_start|、最低 1)。"""
    members = subtotal_members(spec, names)
    a = members.get("jppfs_cor_Assets")
    total = abs(float(a @ x_start)) if a is not None else float(np.max(np.abs(x_start)))
    return max(total, 1.0) if a is None else total


def project_onto_feasible(
    y: np.ndarray,
    spec: CanonicalSpec,
    x_start: np.ndarray,
    weights: np.ndarray | None = None,
    enforce_inactive: bool = False,
    direction_constraints: tuple[DirectionConstraintSpec, ...] = (),
    frozen_accounts: tuple[str, ...] = (),
    whiten: bool = True,
    solver: str | None = None,
    strict: bool = True,
    extra_constraints: tuple[LinearConstraint, ...] = (),
    l1_move: tuple[np.ndarray, np.ndarray, float] | None = None,
) -> tuple[np.ndarray, ProjectionInfo]:
    """y を実行可能集合 C_i(x_start に基づく方向制約込み)へ射影する。

    min ‖W^{1/2}(x-y)‖^2  s.t.  x ∈ C_i

    weights=None なら W=I(重みなし)。``_default_weights`` で
    W = diag(1/(|x_start|+eps)) を作れる(§3.1、重み付け推奨)。

    **frozen_accounts(ステージ10 §0.1で追加)**: sign="free"(符号制約なし)
    の科目は、重み(コスト)をいくら大きくしても目的関数上の「動かす代償」が
    増えるだけで、値を拘束する制約が他に無い限りその科目の最適解は常に
    厳密に y の値に一致してしまう(他の変数との間でトレードオフする対象が
    無いため)。ステージ9で「非操作科目(利益剰余金等)をw=1e6の重みで
    実質固定したはず」が、実際には経路のRE変化が総資産比40%超のまま
    だったのはこれが原因だった。frozen_accountsに挙げた科目は
    ``u[i] == sqrt_w[i]*x_start_s[i]`` という等式制約を直接課し、
    重みの大小によらず厳密にx_startへ固定する。

    **内部スケーリング・白色化**: 円単位のまま(〜1e11オーダー)解くと、
    OSQPが既定の反復回数では収束しきらず"user_limit"で終わることが多い
    ことが実データで判明した。企業の規模(|y|, |x_start|の最大値)で問題を
    正規化してから解くと解決する。

    **さらに重要な問題**: W=diag(1/(|x_start|+eps)) を使う場合、大企業の
    科目間で重みが数十万倍のオーダーで異なる(実データで最大54万倍を
    確認)。``cp.sum_squares(cp.multiply(sqrt(w), x-y))`` のまま解くと、
    目的関数のヘッセ行列の条件数が悪化し、OSQP/ECOS/CLARABELのいずれでも
    "optimal"と報告しながら実際には全く最適でない解(実測で数百億円規模の
    誤った解)を返すことがあった。これを避けるため、変数を
    ``u = sqrt(w) ⊙ x`` に置き換える(白色化)。目的関数は
    ``sum_squares(u - sqrt(w)⊙y)``(ヘッセ行列が単位行列になり
    条件数の問題が消える)、制約は ``(C ⊘ sqrt(w))^T u = 0`` 等、符号は
    sqrt(w)>0 なので向きが保たれる。解いた後 ``x = u / sqrt(w)`` で戻す。

    ``whiten=False``(ステージ10.6 §1.4の比較専用): 上記の変数置換を行わず、
    ``x`` を直接変数として ``sum_squares(sqrt(w)⊙(x-y))`` を最小化する
    (数学的には同じ問題だが、白色化前の条件数のまま解くことになる)。
    白色化の効果そのものを検証するためのフラグで、通常はTrueのまま使う。

    **収束チェック(ステージ13 §A.1)**: ``prob.status`` が ``optimal`` 以外、
    解が得られない、または解の制約違反が許容(``1e-6 × scale``)を超える場合は
    ``SolverNotConvergedError`` を投げる(以前は解なしのとき黙って ``y`` を
    返していた)。``strict=False`` は、白色化なしの既知の不良版を意図的に
    動かすステージ10.6の比較実験専用。

    **追加の線形制約(ステージ14 §B)**: ``extra_constraints`` の各 ``a·x >= rhs``
    (小計の非負、純資産の非悪化、資本金の下限、総資産の変化幅)を、白色化後の
    変数では ``(a/sqrt(w))·u >= rhs/scale`` として課す。実行可能集合が空なら
    ``InfeasibleProjectionError``(``SolverNotConvergedError``の子)。
    拘束している制約の名前は ``ProjectionInfo.active_constraints`` に入る。

    **L1 の移動コスト(ステージ60)**: ``l1_move = (c_plus, c_minus, lam)`` なら、目的関数に
    ``lam · Σ_a (c_plus[a]·u_a + c_minus[a]·v_a)``、``x − x_start = u − v``、``u, v ≥ 0`` を足す
    (``min Σ w (x − y)² + lam Σ (c⁺u + c⁻v)``)。スケーリング後の問題では lam / scale を掛けるので、
    lam の意味は元の単位のまま。u・v は ``ProjectionInfo.move_up``・``move_down`` に入る。

    ``solver``(ステージ10.6 §1.5の独立ソルバ検算専用): Noneなら既定の
    OSQP(失敗時ECOSへフォールバック)。明示的に指定すると(例: "ECOS",
    "CLARABEL", "SCS")そのソルバのみで解く(フォールバックなし)。
    """
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    dim = len(names)
    index = {n: i for i, n in enumerate(names)}

    w = np.ones(dim) if weights is None else weights
    sqrt_w = np.sqrt(w)

    scale = max(float(np.max(np.abs(y))), float(np.max(np.abs(x_start))), 1.0)
    y_s = y / scale
    x_start_s = x_start / scale

    u = cp.Variable(dim)
    if whiten:
        objective = cp.Minimize(cp.sum_squares(u - cp.multiply(sqrt_w, y_s)))
        c_used = c / sqrt_w[:, None]
        u_start = sqrt_w * x_start_s  # frozen/direction制約での基準点(白色化空間)
    else:
        objective = cp.Minimize(cp.sum_squares(cp.multiply(sqrt_w, u - y_s)))
        c_used = c
        u_start = x_start_s  # uはxそのもの

    constraints = [c_used.T @ u == 0]
    for b in spec.basis:
        i = index[b.name]
        if b.sign == "nonneg":
            constraints.append(u[i] >= 0)
        elif b.sign == "nonpos":
            constraints.append(u[i] <= 0)

    if enforce_inactive:
        # ステージ74(A-4): ゼロの判定を plausibility.inactive_accounts と同じ
        # |x| < INACTIVE_ZERO × |総資産(始点)| にそろえ、始点の値に固定する
        # (厳密な 0 との比較をやめた)。
        # 対象は全基礎科目のまま(INACTIVE_ACCOUNTS の 4 科目ではない)なので、非推奨とする
        warnings.warn(
            "enforce_inactive は非推奨。frozen_accounts=inactive_accounts(x_start, names) を使う"
            "(plausibility の不活性の定義と同じになる)",
            DeprecationWarning, stacklevel=2)
        zero_tol = INACTIVE_ZERO * _start_assets(spec, names, x_start)
        for b in spec.basis:
            i = index[b.name]
            if abs(float(x_start[i])) < zero_tol:
                constraints.append(u[i] == u_start[i])

    for name in frozen_accounts:
        i = index[name]
        constraints.append(u[i] == u_start[i])

    for dc in direction_constraints:
        if not evaluate_condition(dc.condition, x_start, spec):
            continue
        if dc.rule != "monotone_from_start":
            raise ValueError(f"unknown direction constraint rule: {dc.rule!r}")
        i = index[dc.name]
        sign = _sign_of(spec, dc.name)
        u_start_i = u_start[i]
        if sign == "nonpos":
            # 負の値が増えない(絶対値が増えない)= x(t) >= x(0)
            constraints.append(u[i] >= u_start_i)
        elif sign == "nonneg":
            constraints.append(u[i] <= u_start_i)
        else:
            raise ValueError(
                f"monotone_from_start requires nonneg/nonpos sign for {dc.name!r}, got {sign!r}"
            )

    for lc in extra_constraints:
        a_u = lc.a / sqrt_w if whiten else lc.a
        constraints.append(a_u @ u >= lc.rhs / scale)

    up: cp.Variable | None = None
    down: cp.Variable | None = None
    if l1_move is not None:
        c_plus, c_minus, lam = l1_move
        if lam < 0 or np.any(c_plus < 0) or np.any(c_minus < 0):
            raise ValueError("l1_move の係数は非負")
        up, down = cp.Variable(dim, nonneg=True), cp.Variable(dim, nonneg=True)
        x_s_expr = cp.multiply(1.0 / sqrt_w, u) if whiten else u
        constraints.append(x_s_expr - x_start_s == up - down)
        objective = cp.Minimize(objective.args[0]
                                + (lam / scale) * (c_plus @ up + c_minus @ down))

    prob = cp.Problem(objective, constraints)
    start = time.perf_counter()
    if solver is not None:
        prob.solve(solver=solver)
    else:
        try:
            prob.solve(solver=cp.OSQP, max_iter=50000, eps_abs=1e-8, eps_rel=1e-8)
        except cp.error.SolverError:
            prob.solve(solver=cp.ECOS)
        # OSQPが optimal_inaccurate(重みの比が巨大なW3+追加制約で起きる)なら内点法で解き直す。
        # 解き直しが optimal に届かなければ、下の strict チェックが例外にする。
        if prob.status != "optimal" and "infeasible" not in str(prob.status):
            for fallback in (cp.CLARABEL, cp.ECOS):
                try:
                    prob.solve(solver=fallback)
                except cp.error.SolverError:
                    continue
                if prob.status == "optimal":
                    break
    elapsed = time.perf_counter() - start

    if u.value is None:
        if strict:
            if "infeasible" in str(prob.status):
                raise InfeasibleProjectionError(f"射影QPが実行不可能: status={prob.status}")
            raise SolverNotConvergedError(f"射影QPの解が得られない: status={prob.status}")
        x_result = y.copy()  # strict=False(比較実験専用)のみ、従来どおり元の点を返す
    else:
        u_val = np.asarray(u.value, dtype=float)
        x_s = u_val / sqrt_w if whiten else u_val  # 白色化を戻す(スケール後の単位)
        x_result = x_s * scale  # 円単位へ戻す
    max_violation = _max_violation(x_result, spec, c, index)
    active: list[str] = []
    for lc in extra_constraints:
        slack = float(lc.a @ x_result - lc.rhs)
        max_violation = max(max_violation, -slack)
        if slack <= 1e-6 * scale:
            active.append(lc.name)
    if strict:
        if prob.status != "optimal":
            raise SolverNotConvergedError(f"射影QPが最適解でない: status={prob.status}")
        if not np.isfinite(x_result).all() or max_violation > 1e-6 * scale:
            raise SolverNotConvergedError(
                f"射影解の制約違反が大きい: {max_violation:.3e} (scale={scale:.3e})"
            )

    move_up = (np.asarray(up.value, dtype=float) * scale
               if up is not None and up.value is not None else None)
    move_down = (np.asarray(down.value, dtype=float) * scale
                 if down is not None and down.value is not None else None)
    info = ProjectionInfo(
        status=str(prob.status),
        distance_before_after=float(np.linalg.norm(x_result - y)),
        max_constraint_violation=max_violation,
        solve_time_seconds=elapsed,
        active_constraints=tuple(active),
        move_up=move_up,
        move_down=move_down,
    )
    return x_result, info


def project_onto_union(
    y: np.ndarray,
    spec: CanonicalSpec,
    x_start: np.ndarray,
    branches: Sequence[tuple[LinearConstraint, ...]],
    weights: np.ndarray | None = None,
    extra_constraints: tuple[LinearConstraint, ...] = (),
    direction_constraints: tuple[DirectionConstraintSpec, ...] = (),
    frozen_accounts: tuple[str, ...] = (),
) -> tuple[np.ndarray, ProjectionInfo, int]:
    """凸集合の和集合への射影(ステージ29 R4)。分岐ごとに射影し、目的関数が最小の解を採る。

    目的関数は ``project_onto_feasible`` と同じ ‖W^{1/2}(x − y)‖²。実行不可能な分岐は飛ばし、
    すべて実行不可能なら ``InfeasibleProjectionError``。未収束(``SolverNotConvergedError``)は
    そのまま送出する。戻り値の最後は採った分岐の番号。
    """
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    best: tuple[float, np.ndarray, ProjectionInfo, int] | None = None
    for k, branch in enumerate(branches):
        try:
            x, info = project_onto_feasible(
                y, spec, x_start=x_start, weights=weights,
                direction_constraints=direction_constraints, frozen_accounts=frozen_accounts,
                extra_constraints=(*extra_constraints, *branch),
            )
        except InfeasibleProjectionError:
            continue
        obj = float(np.sum(w * (x - y) ** 2))
        if best is None or obj < best[0]:
            best = (obj, x, info, k)
    if best is None:
        raise InfeasibleProjectionError("和集合のどの分岐も実行不可能")
    return best[1], best[2], best[3]


def default_weights(x_start: np.ndarray, assets: float) -> np.ndarray:
    """W = diag(1 / (|x_start| + eps))(§3.1)。eps は総資産の1e-6程度。"""
    return _default_weights(x_start, assets)


def apply_projection_to_paths(
    x_paths: np.ndarray,
    x_anchor: np.ndarray,
    assets: np.ndarray,
    spec: CanonicalSpec,
    mode: str,
    enforce_inactive: bool = False,
    weighted: bool = True,
    direction_constraints: tuple[DirectionConstraintSpec, ...] = (),
    frozen_accounts: tuple[str, ...] = (),
) -> tuple[np.ndarray, list[ProjectionInfo]]:
    """経路 x_paths (n, T, D) に射影を適用する(ステージ7 §5.1)。

    mode="every_step": 各 t で x(t) = P(x_i + B(z(t)-z_i)) (t=0 も含む)。
    mode="endpoint_only": t=T-1(終端)のみ射影し、途中(t=0含む)は無補正の
    まま(線形補間、CAFM原著 §4 と同じ方針)。
    """
    if mode not in ("every_step", "endpoint_only"):
        raise ValueError(f"unknown mode: {mode!r}")

    n, t_len, _dim = x_paths.shape
    x_projected = x_paths.copy()
    infos: list[ProjectionInfo] = []
    t_indices = range(t_len) if mode == "every_step" else (t_len - 1,)

    for i in range(n):
        x_start = x_anchor[i]
        weights = default_weights(x_start, float(assets[i])) if weighted else None
        for t_i in t_indices:
            x_proj, info = project_onto_feasible(
                x_paths[i, t_i],
                spec,
                x_start=x_start,
                weights=weights,
                enforce_inactive=enforce_inactive,
                direction_constraints=direction_constraints,
                frozen_accounts=frozen_accounts,
            )
            x_projected[i, t_i] = x_proj
            infos.append(info)

    return x_projected, infos


def _sign_of(spec: CanonicalSpec, name: str) -> str:
    for b in spec.basis:
        if b.name == name:
            return b.sign
    raise KeyError(name)


def is_feasible(
    x: np.ndarray,
    spec: CanonicalSpec,
    tol: float = 1e-6,
) -> tuple[bool, list[tuple[str, str, float]]]:
    """x が(方向制約を除く){C^Tx=0} ∩ {符号制約} を満たすかを検証する。

    返り値: (満たすか, [(科目名, 違反の種類, 値), ...])
    """
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    index = {n: i for i, n in enumerate(names)}

    violations: list[tuple[str, str, float]] = []
    balance_residual = float(np.max(np.abs(c.T @ x)))
    if balance_residual > tol:
        violations.append(("(貸借一致)", "balance", balance_residual))

    for b in spec.basis:
        v = float(x[index[b.name]])
        if b.sign == "nonneg" and v < -tol:
            violations.append((b.name, "nonneg", v))
        elif b.sign == "nonpos" and v > tol:
            violations.append((b.name, "nonpos", v))

    return len(violations) == 0, violations


def basis_names_and_signs(spec: CanonicalSpec) -> list[tuple[str, str]]:
    return [(b.name, b.sign) for b in spec.basis]


__all__ = [
    "BasisSpec",
    "DirectionConstraintSpec",
    "ProjectionInfo",
    "basis_names_and_signs",
    "evaluate_condition",
    "is_feasible",
    "load_direction_constraints",
    "project_onto_feasible",
    "project_onto_union",
]
