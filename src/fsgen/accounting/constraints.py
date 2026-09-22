"""canonical (core.yaml 等) から状態ベクトルの定義と会計制約行列を構築する。

状態ベクトル x は canonical の基礎科目(``basis:``)だけで構成する
(``include_subtotals=False``)か、基礎科目に加えて小計・合計科目
(``subtotals:``、``Assets``/``Liabilities``/``NetAssets`` 自身を含む)も
含めるか(``include_subtotals=True``)を選べる。どちらでも実効次元
(D - rank(C))は変わらない。

科目の順序は常に canonical の YAML 記載順を正とし、コードで並べ替えない
(``state_vector_names`` はそのまま返す)。

制約行列 C は (D, K) で、実データが ``C^T x = 0`` を満たすように書く。

- ``include_subtotals=False`` のとき、小計・合計は状態に無いので「小計 =
  配下の和」は制約にならない(x から計算される定義であって、xに対する
  制約ではない)。真の制約は貸借一致 1本だけ(K=1)。
- ``include_subtotals=True`` のとき、各小計・合計についても「(その値) -
  (直下の小計の値) - (直下の基礎科目の和) = 0」という定義制約が book(K =
  小計数 + 1)。特に Assets/Liabilities/NetAssets 自身の定義制約は
  ステージ4.6で導入した差額定義(__OTHER__ の値)と整合する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

# 合計・小計の階層(直下の小計)。ステージ4.6 inspect_values.py の
# _SUBTOTAL_CHILDREN と同じ定義。
SUBTOTAL_CHILDREN: dict[str, tuple[str, ...]] = {
    "jppfs_cor_Assets": ("jppfs_cor_CurrentAssets", "jppfs_cor_NoncurrentAssets"),
    "jppfs_cor_Liabilities": ("jppfs_cor_CurrentLiabilities", "jppfs_cor_NoncurrentLiabilities"),
    "jppfs_cor_NetAssets": (
        "jppfs_cor_ShareholdersEquity",
        "jppfs_cor_ValuationAndTranslationAdjustments",
    ),
}

ASSETS_ID = "jppfs_cor_Assets"
LIABILITIES_ID = "jppfs_cor_Liabilities"
NET_ASSETS_ID = "jppfs_cor_NetAssets"

_SVD_RANK_TOL = 1e-9


@dataclass(frozen=True)
class BasisSpec:
    name: str
    parent_subtotal: str | None
    side: str
    sign: str = "free"  # "nonneg" | "nonpos" | "free"(ステージ7 §2.1)
    sign_reason: str | None = None  # signの決定理由(ステージ7 §2.1)


@dataclass(frozen=True)
class SubtotalSpec:
    element_id: str
    side: str


@dataclass(frozen=True)
class CanonicalSpec:
    """constraints.py が必要とする canonical の最小限の情報。

    ``basis``/``subtotals`` の順序は YAML の記載順そのまま。
    """

    name: str
    basis: tuple[BasisSpec, ...]
    subtotals: tuple[SubtotalSpec, ...]


def load_canonical_spec(path: Path) -> CanonicalSpec:
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    basis = tuple(
        BasisSpec(
            name=b["name"],
            parent_subtotal=b["parent_subtotal"],
            side=b["side"],
            sign=b.get("sign", "free"),
            sign_reason=b.get("sign_reason"),
        )
        for b in doc["basis"]
    )
    subtotals = tuple(
        SubtotalSpec(element_id=s["element_id"], side=s["side"]) for s in doc["subtotals"]
    )
    return CanonicalSpec(name=doc["name"], basis=basis, subtotals=subtotals)


def state_vector_names(spec: CanonicalSpec, include_subtotals: bool) -> tuple[str, ...]:
    """状態ベクトルの各次元の名前。順序は YAML 記載順(基礎科目 → 小計)。"""
    names = tuple(b.name for b in spec.basis)
    if include_subtotals:
        names = names + tuple(s.element_id for s in spec.subtotals)
    return names


def build_constraint_matrix(spec: CanonicalSpec, include_subtotals: bool) -> np.ndarray:
    """C ∈ R^{D×K} を構成する。実データ x は C^T x = 0 を満たす。"""
    names = state_vector_names(spec, include_subtotals)
    index = {n: i for i, n in enumerate(names)}
    dim = len(names)

    if not include_subtotals:
        # 唯一の制約: 貸借一致(資産側の和 = 負債側+純資産側の和)。
        c = np.zeros((dim, 1))
        for b in spec.basis:
            c[index[b.name], 0] += 1.0 if b.side == "assets" else -1.0
        return c

    rows: list[np.ndarray] = []
    # 各小計・合計の定義制約: s - Σ(直下の小計) - Σ(直下の基礎科目) = 0。
    for s in spec.subtotals:
        row = np.zeros(dim)
        row[index[s.element_id]] = 1.0
        for child in SUBTOTAL_CHILDREN.get(s.element_id, ()):
            row[index[child]] -= 1.0
        for b in spec.basis:
            if b.parent_subtotal == s.element_id:
                row[index[b.name]] -= 1.0
        rows.append(row)

    # 貸借一致: Assets - Liabilities - NetAssets = 0。
    # (Assets/Liabilities/NetAssets 自身が状態に含まれるため、基礎科目を
    # 総なめする必要はない。)
    balance = np.zeros(dim)
    balance[index[ASSETS_ID]] = 1.0
    balance[index[LIABILITIES_ID]] = -1.0
    balance[index[NET_ASSETS_ID]] = -1.0
    rows.append(balance)

    return np.array(rows).T  # (D, K)


def _column_space_basis(matrix: np.ndarray, tol: float = _SVD_RANK_TOL) -> tuple[np.ndarray, int]:
    """SVD で matrix の列空間の正規直交基底と数値的な rank を求める。

    (C^T C)^{-1} や (C C^T)^{-1} を明示的に計算すると、C の列が
    ほぼ従属な場合に数値的に不安定になる(C は (D, K) で K < D なので
    C C^T は常に特異)。SVD ならこれを避けられる。
    """
    if matrix.size == 0:
        return np.zeros((matrix.shape[0], 0)), 0
    u, s, _vt = np.linalg.svd(matrix, full_matrices=True)
    threshold = tol * s[0] if s.size and s[0] > 0 else 0.0
    rank = int(np.sum(s > threshold))
    return u, rank


def compute_rank(matrix: np.ndarray, tol: float = _SVD_RANK_TOL) -> int:
    _u, rank = _column_space_basis(matrix, tol)
    return rank


def build_projection(matrix: np.ndarray) -> np.ndarray:
    """P = I - C(C^T C)^{-1} C^T を SVD 経由で数値的に安定して計算する。

    U の最初 rank(C) 列が col(C) の正規直交基底になるので、
    col(C) への射影は U_r @ U_r.T で書け、P はその補射影。
    """
    dim = matrix.shape[0]
    u, rank = _column_space_basis(matrix)
    u_r = u[:, :rank]
    result: np.ndarray = np.eye(dim) - u_r @ u_r.T
    return result


def orthonormal_basis(matrix: np.ndarray) -> np.ndarray:
    """null(C^T) (制約部分空間の直交補空間)の正規直交基底 (D, D-rank(C))。"""
    u, rank = _column_space_basis(matrix)
    return u[:, rank:]
