"""最適輸送による経路生成(ステージ6)。

class 2(GC注記あり)の各社の低次元表現 z を、class 0(健全企業)の
低次元表現へエントロピー正則化OT(Sinkhorn、`POT` ライブラリ)で対応づけ、
その対応先への線形補間で経路を作る。

Flow Matching・CAFM・ニューラルネットは使わない。目的は
(1) データから経路までの配管が通ることの確認、
(2) 生成された貸借対照表を人間が見て会計的に読めるかの判断、の2つ。

**スケール復元**: 経路は低次元表現 z(t) の線形補間として作り、
始点企業からの「移動量」だけを低次元空間で計算して実測値に加算する
ことで元のスケールへ戻す(ステージ6.5 §B、``paths_to_state_vectors``)。
t=0 では実測値と厳密に一致するが、総資産合計は t とともに変化しうる
(C は貸借一致だけを制約し、規模を制約しないため)。実際の再建では
増資・資産売却で総資産自体が変わりうる点も含め、規模の制約をどう
扱うかは次段階の課題として残す。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ot

from fsgen.errors import SolverNotConvergedError

_SINKHORN_STOP_THR = 1e-9
_SINKHORN_MAX_ITER = 1000
_SINKHORN_LOG_MAX_ITER = 20000
DEFAULT_MARGINAL_TOL = 1e-6  # §A.2: 周辺分布の誤差の既定閾値


def cost_matrix(z0: np.ndarray, z1: np.ndarray) -> np.ndarray:
    """二乗ユークリッド距離のコスト行列 (n0, n1)。"""
    diff = z0[:, None, :] - z1[None, :, :]
    result: np.ndarray = np.sum(diff**2, axis=2)
    return result


@dataclass(frozen=True)
class OtResult:
    plan: np.ndarray  # (n0, n1), Σ_ij P[i,j] = 1
    converged: bool
    n_checkpoints: int
    final_error: float
    max_marginal_error: float  # 実際の周辺分布と目標(一様)分布とのずれの最大値
    n_zero_mass_rows: int  # 質量がほぼ0になった始点(数値アンダーフローの兆候)


def solve_ot(
    z0: np.ndarray, z1: np.ndarray, epsilon: float, *, log_domain: bool = True
) -> np.ndarray:
    """エントロピー正則化OT(Sinkhorn)を解き、輸送計画 P ∈ R^{n0×n1} を返す。

    コスト行列は二乗ユークリッド距離をその中央値で正規化したもの
    (``C_normalized = C / median(C)``)。始点・終点の重みは一様分布。
    ステージ13 §A.3: 既定は対数領域(標準Sinkhornは比較用)。未収束は例外。
    """
    return solve_ot_verbose(z0, z1, epsilon, log_domain=log_domain).plan


def solve_ot_verbose(
    z0: np.ndarray, z1: np.ndarray, epsilon: float, *, log_domain: bool = True
) -> OtResult:
    """solve_ot に加え、収束状況などの診断情報も返す(§3.2の報告用)。"""
    cost = cost_matrix(z0, z1)
    return solve_ot_from_cost_verbose(
        cost, len(z0), len(z1), epsilon, log_domain=log_domain
    )


def solve_ot_from_cost(
    cost: np.ndarray, epsilon: float, *, log_domain: bool = True
) -> np.ndarray:
    """コスト行列を直接指定してOTを解く(ステージ9: 科目ごとの重み付きコスト用)。"""
    n0, n1 = cost.shape
    return solve_ot_from_cost_verbose(cost, n0, n1, epsilon, log_domain=log_domain).plan


def solve_ot_from_cost_verbose(
    cost: np.ndarray,
    n0: int,
    n1: int,
    epsilon: float,
    *,
    log_domain: bool = True,
    require_converged: bool = True,
) -> OtResult:
    """solve_ot_from_cost に加え、収束状況などの診断情報も返す。

    ``ot.sinkhorn`` 自身が返す ``log['err']`` は、コスト行列に極端な外れ値
    (今回の実データでは正規化後で中央値の700倍超)があると、カーネル
    ``exp(-cost/epsilon)`` がその時点で丸ごとアンダーフローし、最初の
    反復にすら到達できず空リストのまま返ることがある。これは
    ``log['err']`` の有無だけでは検出できないため、実際に得られた輸送
    計画の周辺分布を一様分布(a, b)と直接比較し、質量がほぼ0まで潰れた
    始点(行)の数も別途数える。

    **ステージ12で判明**: 現行の実データ(68社×3951社、ステージ9の重み付き
    コスト)では、この標準(線形領域)Sinkhornが収束せず、68社中9社の行が
    質量0になる。その行の``argmax``は(全要素0のため)任意のindexになり、
    「対応先」として使えない。対応先の分析・学習ペアの生成には、対数領域の
    ``solve_ot_normalized(..., log_domain=True)`` を使うこと。
    """
    cost_normalized = cost / np.median(cost)
    a = np.full(n0, 1.0 / n0)
    b = np.full(n1, 1.0 / n1)
    return solve_ot_normalized(
        cost_normalized, a, b, epsilon, log_domain=log_domain,
        max_iter=_SINKHORN_LOG_MAX_ITER if log_domain else _SINKHORN_MAX_ITER,
        require_converged=require_converged,
    )


def solve_ot_normalized(
    cost_normalized: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
    *,
    log_domain: bool = True,
    max_iter: int = _SINKHORN_LOG_MAX_ITER,
    stop_thr: float = _SINKHORN_STOP_THR,
    require_converged: bool = True,
    marginal_tol: float = DEFAULT_MARGINAL_TOL,
) -> OtResult:
    """正規化済みコスト・任意の周辺分布(a, b)でSinkhornを解く(ステージ12)。

    質量0の目標(b=0)は先に取り除き、結果を元の列数へ0で戻す。
    ``log_domain=True`` は対数領域(``sinkhorn_log``)で、``exp(-C/ε)`` の
    アンダーフローに強い。診断値(周辺分布のずれ・質量0の行)は与えられた
    (a, b) に対して測る。
    """
    keep = b > 0
    cost_k = cost_normalized[:, keep]
    b_k = b[keep]
    plan_k, log = ot.sinkhorn(
        a,
        b_k,
        cost_k,
        epsilon,
        method="sinkhorn_log" if log_domain else "sinkhorn",
        numItermax=max_iter,
        stopThr=stop_thr,
        log=True,
        warn=False,
    )
    plan = np.zeros(cost_normalized.shape)
    plan[:, keep] = plan_k
    errors = log.get("err", [])
    final_error = float(errors[-1]) if errors else float("nan")
    row_error = float(np.max(np.abs(plan.sum(axis=1) - a)))
    col_error = float(np.max(np.abs(plan.sum(axis=0) - b)))
    max_marginal_error = max(row_error, col_error)
    n_zero_mass_rows = int(np.sum(plan.sum(axis=1) < 1e-12))
    finite = bool(np.isfinite(plan).all())
    converged = finite and max_marginal_error < marginal_tol and n_zero_mass_rows == 0
    if require_converged and not converged:
        raise SolverNotConvergedError(
            f"Sinkhornが収束しなかった(log_domain={log_domain}): "
            f"周辺分布の誤差={max_marginal_error:.3e}(許容{marginal_tol:.0e})、"
            f"質量0の行={n_zero_mass_rows}、有限={finite}、反復上限={max_iter}"
        )
    return OtResult(
        plan=plan,
        max_marginal_error=max_marginal_error,
        n_zero_mass_rows=n_zero_mass_rows,
        converged=converged,
        n_checkpoints=len(errors),
        final_error=final_error,
    )


@dataclass(frozen=True)
class PlanHealth:
    """輸送計画の健全性(§A.2)。"""

    min_row_mass: float
    max_marginal_error: float
    effective_support: np.ndarray  # 各始点の実効サポートサイズ(1/Σp²)。質量0の行はnan
    n_zero_mass_rows: int = 0  # allow_zero_mass_rows=True のときだけ非0になりうる


def check_plan_health(
    plan: np.ndarray,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    *,
    marginal_tol: float = DEFAULT_MARGINAL_TOL,
    check_marginals: bool = True,
    allow_zero_mass_rows: bool = False,
) -> PlanHealth:
    """輸送計画の健全性を検証し、違反があれば ``SolverNotConvergedError``(§A.2)。

    1. 各行の質量が0でない 2. 周辺分布の誤差が閾値以下(aを与えたとき、
    bも与えたときは列も) 3. NaN/infを含まない 4. 実効サポートサイズを返す。
    片側制約・不均衡OTのように周辺分布を意図的に守らない計画は
    ``check_marginals=False`` で、1・3のみを検証する。

    不均衡OTは外れ値の始点を意図的に切り捨てる(質量0の行になる)。これを
    **明示的に許す**場合のみ ``allow_zero_mass_rows=True``(件数は
    ``PlanHealth.n_zero_mass_rows`` に記録され、呼び出し側が報告・除外する)。
    """
    if not np.isfinite(plan).all():
        raise SolverNotConvergedError("輸送計画にNaN/infが含まれる")
    row_mass = plan.sum(axis=1)
    n_bad = int(np.sum(row_mass <= 1e-12))
    if n_bad and not allow_zero_mass_rows:
        raise SolverNotConvergedError(f"質量0の行が{n_bad}行ある(argmaxが任意になる)")
    err = 0.0
    if check_marginals and a is not None:
        err = float(np.max(np.abs(row_mass - a)))
        if b is not None:
            err = max(err, float(np.max(np.abs(plan.sum(axis=0) - b))))
        if err > marginal_tol:
            raise SolverNotConvergedError(
                f"周辺分布の誤差{err:.3e}が許容{marginal_tol:.0e}を超える"
            )
    support = effective_support_sizes(plan)
    support[row_mass <= 1e-12] = np.nan
    return PlanHealth(
        min_row_mass=float(row_mass.min()),
        max_marginal_error=err,
        effective_support=support,
        n_zero_mass_rows=n_bad,
    )


def solve_unbalanced_ot(z0: np.ndarray, z1: np.ndarray, epsilon: float, tau: float) -> np.ndarray:
    """質量保存を緩めた(unbalanced)エントロピー正則化OTを解く(ステージ6.5 §C.2)。

    通常のSinkhornは周辺分布(全始点・全終点の質量)を厳密に一様分布へ
    一致させようとするため、コスト行列に極端な外れ値がある始点は、
    その質量を無理やり(数値的にアンダーフローするほど)遠い終点へ
    押し込もうとして破綻することがある。``reg_m``(``tau``)で周辺分布の
    逸脱にペナルティを課す形に緩和すると、外れ値の始点は単に質量を
    あまり配らない(周辺の和が1/n0を下回る)ことを許容するようになり、
    数値的な破綻を避けられる可能性がある。``tau`` が小さいほど緩い
    (質量保存を諦めやすい)。
    """
    cost = cost_matrix(z0, z1)
    cost_normalized = cost / np.median(cost)
    a = np.full(len(z0), 1.0 / len(z0))
    b = np.full(len(z1), 1.0 / len(z1))
    return solve_unbalanced_normalized(cost_normalized, a, b, epsilon, tau)


def solve_unbalanced_normalized(
    cost_normalized: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
    tau: float,
    *,
    max_iter: int = _SINKHORN_LOG_MAX_ITER,
    tol: float = 1e-9,
    allow_dropped_sources: bool = False,
) -> np.ndarray:
    """不均衡OT(KL緩和、対数領域)を正規化済みコストで解く。収束チェック付き(§A.1)。

    ``ot.sinkhorn_unbalanced(a, b, M, reg=ε, reg_m=τ)`` と同じ目的関数
    ``<P,M> + ε·KL(P|a⊗b) + τ·KL(P1|a) + τ·KL(Pᵀ1|b)``(POTの現行の既定=
    reg_type='kl' と同じ、参照測度はa⊗b)を、対数領域の
    スケーリング反復 ``log u = φ(log a − LSE_j(−M/ε + log v))``(φ=τ/(τ+ε))で解く。
    POTのstabilized版は、この実データ(正規化後コストが中央値の約1400倍まで
    ある)で「Numerical errors」を出して質量0の行を返した(ステージ13)ため、
    自前で実装した。ポテンシャルの更新量が``tol``未満に達しなければ例外。
    """
    from scipy.special import logsumexp

    phi = tau / (tau + epsilon)
    log_a, log_b = np.log(a), np.log(b)
    neg_c = -cost_normalized / epsilon + log_a[:, None] + log_b[None, :]  # log K(参照測度a⊗b込み)
    log_u = np.zeros(len(a))
    log_v = np.zeros(len(b))
    delta = float("inf")
    for _ in range(max_iter):
        new_u = phi * (log_a - logsumexp(neg_c + log_v[None, :], axis=1))
        new_v = phi * (log_b - logsumexp(neg_c + new_u[:, None], axis=0))
        delta = float(max(np.max(np.abs(new_u - log_u)), np.max(np.abs(new_v - log_v))))
        log_u, log_v = new_u, new_v
        if delta < tol:
            break
    else:
        raise SolverNotConvergedError(
            f"不均衡Sinkhorn(対数領域)が{max_iter}反復で収束しない: 最終更新量={delta:.3e}"
        )
    plan: np.ndarray = np.exp(log_u[:, None] + neg_c + log_v[None, :])
    # 質量0の行(=切り捨てられた始点)は、allow_dropped_sources=Trueでない限り例外
    check_plan_health(plan, check_marginals=False, allow_zero_mass_rows=allow_dropped_sources)
    return plan


def solve_semi_unbalanced_normalized(
    cost_normalized: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
    tau1: float,
    *,
    max_iter: int = 200000,
    tol: float = 1e-9,
    marginal_tol: float = DEFAULT_MARGINAL_TOL,
) -> OtResult:
    """始点側を厳密に保存し(τ0 = ∞)、目標側だけ KL で緩める不均衡OT(対数領域、ステージ60)。

    ``<P,C> + ε·KL(P|a⊗b) + τ1·KL(Pᵀ1|b)``、``P1 = a``。反復は
    ``log u = log a − LSE_j(log K + log v)``、``log v = φ1 (log b − LSE_i(log K + log u))``
    (φ1 = τ1 / (τ1 + ε)、log K = −C/ε + log a + log b)。行の和は u の更新で毎回厳密に a になる。
    v に定数 c を足すと u が −c ずれて計画は変わらない。この方向は φ1 の速さでしか減らない
    (τ1 が大きいと非常に遅い)ので、収束は v の更新量の幅(最大 − 最小)で判定する。
    τ1 → 0 で目標側の制約が消え、各行が独立に exp(−C/ε) で重みを付ける形になる。
    未収束、非有限、行の周辺の誤差、質量 0 の行は例外。
    """
    from scipy.special import logsumexp

    if tau1 <= 0 or epsilon <= 0:
        raise ValueError("tau1 と epsilon は正")
    phi = tau1 / (tau1 + epsilon)
    log_a, log_b = np.log(a), np.log(b)
    log_k = -cost_normalized / epsilon + log_a[:, None] + log_b[None, :]
    log_u = np.zeros(len(a))
    log_v = np.zeros(len(b))
    delta = float("inf")
    n_iter = 0
    for _ in range(max_iter):
        n_iter += 1
        new_u = log_a - logsumexp(log_k + log_v[None, :], axis=1)
        new_v = phi * (log_b - logsumexp(log_k + new_u[:, None], axis=0))
        dv = new_v - log_v
        delta = float(dv.max() - dv.min())  # 定数のずれ(計画を変えない)を除いた更新量
        log_u, log_v = new_u, new_v
        if delta < tol:
            break
    else:
        raise SolverNotConvergedError(
            f"片側不均衡Sinkhorn(対数領域)が{max_iter}反復で収束しない: 最終更新量={delta:.3e}"
        )
    log_u = log_a - logsumexp(log_k + log_v[None, :], axis=1)  # 最後の v に合わせて行を厳密に
    plan: np.ndarray = np.exp(log_u[:, None] + log_k + log_v[None, :])
    row_error = float(np.max(np.abs(plan.sum(axis=1) - a)))
    n_zero = int(np.sum(plan.sum(axis=1) < 1e-12))
    finite = bool(np.isfinite(plan).all())
    if not finite or row_error >= marginal_tol or n_zero > 0:
        raise SolverNotConvergedError(
            f"片側不均衡OTの計画が不健全: 行の誤差={row_error:.3e}、質量0の行={n_zero}、"
            f"有限={finite}"
        )
    return OtResult(plan=plan, max_marginal_error=row_error, n_zero_mass_rows=0, converged=True,
                    n_checkpoints=n_iter, final_error=delta)


def plan_entropy(plan: np.ndarray) -> float:
    """輸送計画全体のエントロピー。0に近いほど決定的(1対1に近い)、
    大きいほど独立結合(周辺分布の積、つまり全始点が全終点に均等に
    質量を配る状態)に近い。"""
    p = plan[plan > 0]
    return float(-np.sum(p * np.log(p)))


def effective_support_sizes(plan: np.ndarray) -> np.ndarray:
    """各始点(行)が実効的に何点の終点に質量を配っているか(1/Σp²)。

    1に近ければほぼ1点に対応、終点数 n1 に近ければ均等分配
    (「健全企業の平均」へ向かうだけの状態)。
    """
    row_sums = plan.sum(axis=1, keepdims=True)
    safe = np.where(row_sums == 0, 1.0, row_sums)
    normalized = plan / safe
    result: np.ndarray = 1.0 / np.sum(normalized**2, axis=1)
    return result


def single_target_paths(
    z0: np.ndarray, z1: np.ndarray, plan: np.ndarray, ts: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """各始点 i について、輸送計画の行 P[i,:] から質量最大の終点 j* を取り、
    z(t) = (1-t) z_i + t z_j* の経路を作る(バリセンターではなく単一対応、
    §3.3・§7)。

    返り値: (paths (n0, len(ts), k), target_indices (n0,))
    """
    target_indices = np.argmax(plan, axis=1)
    n0, k = z0.shape
    ts_arr = np.asarray(ts)
    z_targets = z1[target_indices]  # (n0, k)
    # (n0, len(ts), k): (1-t)*z0 + t*z_target を全 t について一括計算
    paths = (1 - ts_arr)[None, :, None] * z0[:, None, :] + ts_arr[None, :, None] * z_targets[
        :, None, :
    ]
    assert paths.shape == (n0, len(ts), k)
    return paths, target_indices


def barycenter_paths(
    z0: np.ndarray, z1: np.ndarray, plan: np.ndarray, ts: tuple[float, ...]
) -> np.ndarray:
    """各始点 i について、輸送計画で重み付けた終点のバリセンターへの
    線形補間経路を作る(比較用、§3.3)。
    """
    row_sums = plan.sum(axis=1, keepdims=True)
    safe = np.where(row_sums == 0, 1.0, row_sums)
    weights = plan / safe  # (n0, n1)
    z_bary = weights @ z1  # (n0, k)
    ts_arr = np.asarray(ts)
    result: np.ndarray = (1 - ts_arr)[None, :, None] * z0[:, None, :] + ts_arr[
        None, :, None
    ] * z_bary[:, None, :]
    return result


def paths_to_state_vectors(
    z_paths: np.ndarray,
    z_anchor: np.ndarray,
    x_anchor: np.ndarray,
    b: np.ndarray,
    assets: np.ndarray,
) -> np.ndarray:
    """z(t) (n, T, k) を元の状態空間へ戻す(ステージ6.5 §B、移動量のみ低次元)。

    x(t) = x_anchor + Assets_i × B (z(t) - z_anchor)

    ステージ6では x(t) = B z(t) × Assets_i として直接再構成していたが、
    これは k=13 次元への圧縮復元誤差を t=0(現状)にも持ち込んでしまい、
    かつ z_i と z_target とで復元誤差の大きさが異なる(総資産側の和が
    それぞれ真の値からずれる比率が違う)ため、経路上で総資産合計が
    ドリフトする副作用があった(実測: ある社で t=0 の資産合計が真値の
    98.1%、t=1.0 で104.3%相当になっていた)。

    この式では「移動量」だけを低次元空間で計算し、実測値 x_anchor に
    加算する。t=0 で z(0)=z_anchor となるように呼べば x(0)=x_anchor が
    厳密に成立する(圧縮復元誤差が乗らない)。B の列は C^T B = 0 を
    満たすため、変位 Assets_i × B(z(t)-z_anchor) も C^T(...)=0 を満たし、
    x_anchor 自身が恒等式を満たす実測値であれば C^T x(t) = 0 が全 t で
    保たれる。ただし総資産合計そのものが一定に保たれるという保証は
    無い(C は貸借一致だけを制約し、規模を制約しないため)。z_target が
    別企業に由来する以上、移動量にも規模変化の成分が含まれうる。
    """
    displacement_z = z_paths - z_anchor[:, None, :]
    displacement_x_tilde = np.einsum("dk,ntk->ntd", b, displacement_z)
    displacement_x = displacement_x_tilde * assets[:, None, None]
    result: np.ndarray = x_anchor[:, None, :] + displacement_x
    return result
