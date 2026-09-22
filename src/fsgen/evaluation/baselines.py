"""比較対象となる4つのベースライン + 提案手法の経路生成器(ステージ8 §B.0〜§B.1)。

各手法は ``RecoursePath`` に従い、始点 x(0) と経路上の時刻 ts を受け取って
(T, D) の経路を返す。「学習」(class 0 の統計量・OTの輸送計画・低次元基底の
構築)は、leave-one-out で企業ごとに異なりうるため、各クラスのコンストラクタ
で(既に学習済みの)参照データを受け取る形にし、企業ごとの再学習は呼び出し側
(loo.py)が担当する。

- ``PriorWorkBaseline``: 資産側3科目のみを class 0 中央値へ線形補間。射影なし。
- ``ConvexQpBaseline``: class 0 中央値への最近傍点を実行可能集合内で直接QPで
  求め、その点への線形補間。実行可能集合(恒等式 ∩ 符号制約 ∩ 方向制約)は
  凸(アフィン部分空間と半空間の交わり)なので、2つの実行可能点を結ぶ直線は
  常に実行可能——途中の各tで別途射影し直す必要がない。
- ``NaiveOtBaseline``: 低次元表現上のOT対応先への線形補間(ステージ6.5相当)。
  射影なし。
- ``GaussianBaseline``: class 0 を多変量正規で近似し、class 2 の多変量正規
  近似との間のBures-Wasserstein測地線(2つのガウス分布間の最適輸送写像は
  アフィン写像になる、閉形式)を用いて始点を写像する。
- ``ProposedMethod``: NaiveOtBaseline と同じOT対応先を使うが、各tで射影
  (二次計画、ステージ7)を適用する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from fsgen.accounting.constraints import CanonicalSpec
from fsgen.accounting.projection_qp import (
    DirectionConstraintSpec,
    default_weights,
    project_onto_feasible,
)


class RecoursePath(Protocol):
    def generate(
        self, x_start: np.ndarray, assets_start: float, ts: tuple[float, ...]
    ) -> np.ndarray:
        """始点 x_start(円単位、状態ベクトル)から時刻 ts の経路 (T, D) を返す。"""
        ...


@dataclass
class PriorWorkBaseline:
    """先行研究方式: 資産側3科目のみを class 0 中央値へ線形補間する(射影なし)。

    ``measure_prior_work_residual.py``(ステージ5)と同じ3科目・同じ定義を使う。
    """

    median: np.ndarray  # class 0 の中央値(D,)
    target_idx: tuple[int, ...]

    def generate(
        self, x_start: np.ndarray, assets_start: float, ts: tuple[float, ...]
    ) -> np.ndarray:
        path = np.tile(x_start, (len(ts), 1)).astype(float)
        for t_i, t in enumerate(ts):
            for idx in self.target_idx:
                path[t_i, idx] = x_start[idx] + t * (self.median[idx] - x_start[idx])
        return path


_PRIOR_WORK_TARGET_NAMES: tuple[str, ...] = (
    "inventories",
    "jppfs_cor_OtherCA",
    "__OTHER__jppfs_cor_NoncurrentAssets",
)


def build_prior_work_baseline(median: np.ndarray, names: tuple[str, ...]) -> PriorWorkBaseline:
    target_idx = tuple(names.index(n) for n in _PRIOR_WORK_TARGET_NAMES)
    return PriorWorkBaseline(median=median, target_idx=target_idx)


@dataclass
class ConvexQpBaseline:
    """生成モデルを使わず、実行可能集合内で class 0 中央値に最も近い点へ直接QP(§B.1 #2)。

    「最適輸送も低次元表現も要らず、QPを1回解けば十分ではないか」という
    最も重要な対抗馬。

    中央値は正規化空間 x̃=x/Assets で計算し(``median_tilde``)、
    ``generate`` 呼び出し時に企業自身の総資産(assets_start)倍して使う。
    円の絶対値そのままの中央値をQPの目標にすると、企業規模を無視した
    絶対円単位の目標になり(全社が同じ絶対金額を目指すことになる)、
    小規模企業では総資産の何倍もの移動量になってしまう(実測で確認)。
    他の全手法(naive_ot・proposed・gaussianはz空間、prior_workは絶対円だが
    3科目のみ)との公平な比較のため、この対抗馬だけ不利にならないよう
    正規化する。
    """

    median_tilde: np.ndarray
    spec: CanonicalSpec
    direction_constraints: tuple[DirectionConstraintSpec, ...]
    weighted: bool = True

    def generate(
        self, x_start: np.ndarray, assets_start: float, ts: tuple[float, ...]
    ) -> np.ndarray:
        y = self.median_tilde * assets_start
        weights = default_weights(x_start, assets_start) if self.weighted else None
        x_target, _info = project_onto_feasible(
            y,
            self.spec,
            x_start=x_start,
            weights=weights,
            direction_constraints=self.direction_constraints,
        )
        ts_arr = np.asarray(ts)
        path: np.ndarray = (1 - ts_arr)[:, None] * x_start[None, :] + ts_arr[:, None] * x_target[
            None, :
        ]
        return path


@dataclass
class _OtTarget:
    """低次元表現上でのOT対応先(naive_ot・proposedで共有する部分、§B.3)。"""

    b: np.ndarray  # (D, k) 低次元基底
    z_start: np.ndarray  # (k,) 始点の低次元表現
    z_target: np.ndarray  # (k,) 対応先の低次元表現


def ot_target_for_new_point(
    x_start: np.ndarray,
    assets_start: float,
    b: np.ndarray,
    z0: np.ndarray,
    z_train2: np.ndarray,
    targets_train2: np.ndarray,
) -> _OtTarget:
    """LOOで除外された企業iについて、他のclass2企業(train2)の輸送計画を
    「借りる」(ステージ8 §B.3)。

    μ0={i}のみでOTを解くと(周辺分布の制約で)輸送計画がコスト行列と無関係に
    一様分布へ退化するため、μ0=class2\\{i} で解いた計画(targets_train2、
    train2の各社が対応するclass0側のインデックス)を使い、iはlow-dim空間で
    最も近いtrain2企業の対応先を借りる。
    """
    z_start = (x_start / assets_start) @ b
    diffs = z_train2 - z_start[None, :]
    nn = int(np.argmin(np.sum(diffs**2, axis=1)))
    z_target = z0[targets_train2[nn]]
    return _OtTarget(b=b, z_start=z_start, z_target=z_target)


@dataclass
class NaiveOtBaseline:
    """OT + 低次元線形補間、射影なし(ステージ6.5相当、§B.1 #3)。"""

    b: np.ndarray
    z0: np.ndarray
    z_train2: np.ndarray
    targets_train2: np.ndarray

    def generate(
        self, x_start: np.ndarray, assets_start: float, ts: tuple[float, ...]
    ) -> np.ndarray:
        ot = ot_target_for_new_point(
            x_start, assets_start, self.b, self.z0, self.z_train2, self.targets_train2
        )
        ts_arr = np.asarray(ts)
        displacement_z = ts_arr[:, None] * (ot.z_target - ot.z_start)[None, :]
        displacement_x = np.einsum("dk,tk->td", self.b, displacement_z) * assets_start
        result: np.ndarray = x_start[None, :] + displacement_x
        return result


def build_naive_ot_baseline(
    x_tilde_pool: np.ndarray,
    c: np.ndarray,
    k: int,
    idx0_in_pool: np.ndarray,
    idx_train2_in_pool: np.ndarray,
    epsilon: float,
) -> NaiveOtBaseline:
    """基底Bと輸送計画をpool(iを除いた全社)から構築する(§B.3のLOO手順)。

    ``solve_ot``(fsgen.models.transport)はPOT(``ot``パッケージ)経由で
    sklearnを間接的にimportし、sklearnはtorchとは別のlibomp.dylibを
    同梱している(ステージ11-A §A.3で確認)。この関数を呼ばない限り
    baselines.py自体はtorch専用プロセスからimportしても安全なように、
    遅延import に留める。
    """
    from fsgen.accounting.subspace import build_reduced_basis
    from fsgen.models.transport import solve_ot

    b = build_reduced_basis(c, x_tilde_pool, k=k)
    z_all = x_tilde_pool @ b
    z0 = z_all[idx0_in_pool]
    z_train2 = z_all[idx_train2_in_pool]
    plan = solve_ot(z_train2, z0, epsilon)
    targets_train2 = np.argmax(plan, axis=1)
    return NaiveOtBaseline(b=b, z0=z0, z_train2=z_train2, targets_train2=targets_train2)


@dataclass
class ProposedMethod:
    """NaiveOtBaseline と同じOT対応先を使い、各tで射影を適用する(§B.0の提案手法)。"""

    naive: NaiveOtBaseline
    spec: CanonicalSpec
    direction_constraints: tuple[DirectionConstraintSpec, ...]
    weighted: bool = True

    def generate(
        self, x_start: np.ndarray, assets_start: float, ts: tuple[float, ...]
    ) -> np.ndarray:
        raw_path = self.naive.generate(x_start, assets_start, ts)
        weights = default_weights(x_start, assets_start) if self.weighted else None
        path = np.empty_like(raw_path)
        for t_i in range(raw_path.shape[0]):
            x_proj, _info = project_onto_feasible(
                raw_path[t_i],
                self.spec,
                x_start=x_start,
                weights=weights,
                direction_constraints=self.direction_constraints,
            )
            path[t_i] = x_proj
        return path


def _sqrtm_psd(m: np.ndarray) -> np.ndarray:
    """対称半正定値行列の主平方根(固有値分解、負の微小固有値は0に丸める)。"""
    vals, vecs = np.linalg.eigh(m)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


@dataclass
class GaussianBaseline:
    """class 0 と class 2 をそれぞれ多変量正規で近似し、2つのガウス分布間の
    Bures-Wasserstein測地線(=最適輸送写像、2次コストの下でアフィン写像に
    なる閉形式)で始点を写像する(§B.1 #4)。

    生の正規化空間 x̃=x/Assets(D=30次元)で共分散を推定すると、class2側は
    サンプル数67に対して次元がほぼ同程度で、共分散行列がほぼ特異になり
    (最小固有値がほぼ0)、その逆行列的な演算(S0^{-1/2})が爆発して
    始点から資産規模の1万倍以上も離れた完全に非現実的な点を出すことを
    実測で確認した。これはナイーブ実装の欠陥ではなく、次元67サンプルに
    対し次元30という設定そのものが共分散推定として無理があるため。

    そこで、OT系の手法(naive_ot・proposed)と同じ低次元基底B(k=13次元、
    class0が3951社あるためこちらは安定に推定できる)上で写像を作る
    (naive_ot・proposedとの比較も公平になる: 違いは「OTの離散対応付け」を
    「ガウス近似の閉形式アフィン写像」に置き換えた点だけになる)。
    z空間での写像後、x(t)=x(0)+Assets_0×B(z(t)-z(0)) で円単位へ戻す
    (paths_to_state_vectorsと同じ変位方式)。
    """

    b: np.ndarray  # (D, k) 低次元基底(naive_otと共有)
    mean0_z: np.ndarray  # class0(z空間)の平均
    cov0_z: np.ndarray
    mean2_z: np.ndarray  # class2\{i}(z空間)の平均
    cov2_z: np.ndarray

    def generate(
        self, x_start: np.ndarray, assets_start: float, ts: tuple[float, ...]
    ) -> np.ndarray:
        s0 = self.cov2_z
        s1 = self.cov0_z
        s0_half = _sqrtm_psd(s0)
        s0_half_inv = np.linalg.pinv(s0_half)
        inner = s0_half @ s1 @ s0_half
        inner_half = _sqrtm_psd(inner)
        a = s0_half_inv @ inner_half @ s0_half_inv
        a = (a + a.T) / 2.0

        z_start = (x_start / assets_start) @ self.b
        z_target = self.mean0_z + a @ (z_start - self.mean2_z)

        ts_arr = np.asarray(ts)
        displacement_z = ts_arr[:, None] * (z_target - z_start)[None, :]
        displacement_x = np.einsum("dk,tk->td", self.b, displacement_z) * assets_start
        result: np.ndarray = x_start[None, :] + displacement_x
        return result


def build_gaussian_baseline(
    b: np.ndarray, z0: np.ndarray, z_train2: np.ndarray
) -> GaussianBaseline:
    """z空間(k=13次元)で平均・共分散を推定する。

    class2側はLOOで67サンプルしかなく、生の標本共分散だとS0^{-1/2}(逆行列的
    な演算)がわずかな固有値の推定誤差を増幅し、始点から資産規模の数十〜
    百倍離れた非現実的な写像を生むことを実測で確認した(単純なリッジ加算
    (delta*trace(S)/k)では不十分だった)。Ledoit-Wolf縮小推定
    (データから最適な縮小強度を自動決定し、単位行列方向へ縮める、小標本の
    共分散推定における標準的な手法)に切り替えることで安定させる。
    """
    from sklearn.covariance import LedoitWolf

    mean0_z = z0.mean(axis=0)
    mean2_z = z_train2.mean(axis=0)
    cov0_z = LedoitWolf().fit(z0).covariance_
    cov2_z = LedoitWolf().fit(z_train2).covariance_
    return GaussianBaseline(b=b, mean0_z=mean0_z, cov0_z=cov0_z, mean2_z=mean2_z, cov2_z=cov2_z)


__all__ = [
    "ConvexQpBaseline",
    "GaussianBaseline",
    "NaiveOtBaseline",
    "PriorWorkBaseline",
    "ProposedMethod",
    "RecoursePath",
    "build_gaussian_baseline",
    "build_naive_ot_baseline",
    "build_prior_work_baseline",
    "ot_target_for_new_point",
]
