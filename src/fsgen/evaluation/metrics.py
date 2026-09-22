"""経路(recourse path)の品質を測る指標群(ステージ8 §B.2)。

すべての指標は「経路 (T, D) と始点 x(0)」を受け取る形で統一する。手法
(baselines.py の各実装や提案手法)を差し替えても同じ関数で評価できる。

円単位の絶対値をそのまま比較すると企業規模の影響を強く受けるため、
proximity/sparsity は総資産(始点の Assets)で正規化する
(このプロジェクトを通じて繰り返し出てきた方針、ステージ4.6以降)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fsgen.accounting.constraints import CanonicalSpec

_SPARSITY_RELATIVE_THRESHOLD = 0.01  # 総資産比1%を超えた科目を「変化した」とみなす(§B.2)


def proximity_l2(path: np.ndarray, assets_start: float) -> float:
    """‖x(T-1) - x(0)‖ / Assets_0。"""
    return float(np.linalg.norm(path[-1] - path[0]) / assets_start)


def proximity_l1(path: np.ndarray, assets_start: float) -> float:
    """Σ|x_i(T-1) - x_i(0)| / Assets_0。"""
    return float(np.sum(np.abs(path[-1] - path[0])) / assets_start)


def proximity_l2_weighted(path: np.ndarray, weights: np.ndarray) -> float:
    """√(Σ w_i (x_i(T-1)-x_i(0))^2)(科目の規模差を吸収した重み付き版、§B.2)。"""
    diff = path[-1] - path[0]
    return float(np.sqrt(np.sum(weights * diff**2)))


def proximity_l1_weighted(path: np.ndarray, weights: np.ndarray) -> float:
    diff = np.abs(path[-1] - path[0])
    return float(np.sum(weights * diff))


def proximity_cost_weighted(
    path: np.ndarray, assets_start: float, w_plus: np.ndarray, w_minus: np.ndarray
) -> float:
    """proximity_weighted = Σ_a w_a |x_a(1)-x_a(0)| / Assets_0(ステージ10 §0.2)。

    ステージ9のコスト重み(cost_increase/cost_decrease)をそのまま使う
    (default_weightsのような1/|x|の逆スケール項は含まない——純粋にコストの
    大小だけで「動かしにくい科目をどれだけ動かしたか」を測る指標)。
    実現した方向(増加/減少)に応じて w_a^+ / w_a^- のどちらかを科目ごとに選ぶ。
    """
    diff = path[-1] - path[0]
    w = np.where(diff >= 0, w_plus, w_minus)
    return float(np.sum(w * np.abs(diff)) / assets_start)


def sparsity(
    path: np.ndarray, assets_start: float, threshold: float = _SPARSITY_RELATIVE_THRESHOLD
) -> int:
    """相対変化(総資産比)が threshold を超えた科目の数(§B.2)。

    科目自身の値(x_i(0))で正規化すると、x_i(0)=0 の科目で割り算が
    破綻する(多くの科目がt=0時点で0円になりうる)ため、他の指標と同様に
    総資産で正規化する。
    """
    diff = np.abs(path[-1] - path[0]) / assets_start
    return int(np.sum(diff > threshold))


def balance_residual(path: np.ndarray, c: np.ndarray, assets_start: float) -> float:
    """全tにおける恒等式残差 ‖C^T x(t)‖ の最大値 / Assets_0(§B.2 feasibility)。"""
    residuals = [float(np.max(np.abs(c.T @ path[t_i]))) for t_i in range(path.shape[0])]
    return max(residuals) / assets_start


def endpoint_balance_residual(path: np.ndarray, c: np.ndarray, assets_start: float) -> float:
    """終点(t=1)のみの恒等式残差 ‖C^T x(1)‖ / Assets_0(ステージ8.5 §C.2)。"""
    return float(np.max(np.abs(c.T @ path[-1]))) / assets_start


def sign_violation_count(
    path: np.ndarray,
    spec: CanonicalSpec,
    names: tuple[str, ...],
    assets_start: float,
    tol_ratio: float,
) -> int:
    """全tを通じた符号違反の合計件数(§B.2 feasibility)。tol_ratio*Assets_0 を許容誤差とする。"""
    index = {n: i for i, n in enumerate(names)}
    tol = assets_start * tol_ratio
    count = 0
    for t_i in range(path.shape[0]):
        for b in spec.basis:
            v = path[t_i, index[b.name]]
            if b.sign == "nonneg" and v < -tol:
                count += 1
            elif b.sign == "nonpos" and v > tol:
                count += 1
    return count


def direction_violation_count(
    path: np.ndarray,
    names: tuple[str, ...],
    direction_name: str,
    triggered: bool,
    assets_start: float,
    tol_ratio: float,
) -> int:
    """方向制約(monotone_from_start)の違反件数。triggered=Falseなら常に0。"""
    if not triggered:
        return 0
    idx = names.index(direction_name)
    tol = assets_start * tol_ratio
    x0 = path[0, idx]
    count = 0
    for t_i in range(1, path.shape[0]):
        if abs(path[t_i, idx]) > abs(x0) + tol:
            count += 1
    return count


def intermediate_feasibility_violations(
    path: np.ndarray,
    spec: CanonicalSpec,
    names: tuple[str, ...],
    assets_start: float,
    tol_ratio: float,
) -> int:
    """中間点(t=0.25,0.5,0.75、つまり先頭・末尾を除く全t)での符号違反件数(§B.2)。"""
    interior = path[1:-1]
    if interior.shape[0] == 0:
        return 0
    return sign_violation_count(interior, spec, names, assets_start, tol_ratio)


def monotonicity_reversals(path: np.ndarray) -> int:
    """各科目について、隣接差分の符号が反転した回数の合計(§B.2「経路の品質」)。

    単調に変化していれば0。増減を繰り返す科目が多いほど値が大きくなる。
    """
    diffs = np.diff(path, axis=0)  # (T-1, D)
    signs = np.sign(diffs)
    total = 0
    for d in range(path.shape[1]):
        s = signs[:, d]
        s_nonzero = s[s != 0]
        if s_nonzero.size < 2:
            continue
        total += int(np.sum(s_nonzero[1:] != s_nonzero[:-1]))
    return total


def nearest_neighbor_distance(x_target: np.ndarray, pool: np.ndarray) -> float:
    """x_target から pool(class0の実データ、正規化済み)への最短ユークリッド距離。"""
    diffs = pool - x_target[None, :]
    return float(np.min(np.linalg.norm(diffs, axis=1)))


def pairwise_nearest_neighbor_distances(pool: np.ndarray) -> np.ndarray:
    """pool内の各点について、pool内の他の点への最短距離(class0同士の距離分布、§B.2)。"""
    n = pool.shape[0]
    out = np.empty(n)
    for i in range(n):
        diffs = np.delete(pool, i, axis=0) - pool[i][None, :]
        out[i] = np.min(np.linalg.norm(diffs, axis=1))
    return out


def correlation_frobenius_distance(generated: np.ndarray, real: np.ndarray) -> float:
    """生成結果と実データ(class0)の科目間相関行列のフロベニウス距離(§B.2 plausibility)。

    分散が実質0の科目(常に0など)は相関が定義できないため、比較対象の両方で
    分散>0の科目だけに絞る。
    """
    std_g = generated.std(axis=0)
    std_r = real.std(axis=0)
    keep = (std_g > 0) & (std_r > 0)
    corr_g = np.corrcoef(generated[:, keep], rowvar=False)
    corr_r = np.corrcoef(real[:, keep], rowvar=False)
    return float(np.linalg.norm(corr_g - corr_r, ord="fro"))


def percentile_rank(value: float, reference: np.ndarray) -> float:
    """reference分布の中で value が何パーセンタイルに位置するか(0-100)。"""
    return float(np.mean(reference <= value) * 100.0)


def discriminative_auc(
    generated: np.ndarray, real: np.ndarray, seed: int, n_splits: int = 5
) -> float:
    """「生成されたn点」vs「class0から抽出したn点」を識別する分類器のAUC(§B.2)。

    0.5に近いほど見分けがつかない(=現実味がある)。サンプル数が少ない
    (68 vs 68程度)ため、単一のtrain/test分割ではなく交差検証AUCの平均を使う。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.concatenate([generated, real], axis=0)
    y = np.concatenate([np.ones(len(generated)), np.zeros(len(real))])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    from fsgen.evaluation.classifiers import strict_sklearn_convergence

    with strict_sklearn_convergence():  # 未収束は例外(ステージ13 §A.1)
        scores = cross_val_score(clf, x, y, cv=cv, scoring="roc_auc", error_score="raise")
    return float(np.mean(scores))


@dataclass(frozen=True)
class FeasibilitySummary:
    max_balance_residual_ratio: float
    total_sign_violations: int
    total_direction_violations: int
    total_intermediate_violations: int
