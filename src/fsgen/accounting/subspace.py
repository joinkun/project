"""制約部分空間 null(C^T) 内で低次元表現(基底)を構成する(ステージ5.5 §C)。

``build_reduced_basis`` は、データ X が既に会計恒等式 ``C^T x = 0`` を
満たしていることを利用し、null(C^T) への正規直交基底 Q を経由して
X を非中心化SVD(truncated SVD)で圧縮する。返り値の基底 B は

- 列が正規直交
- ``C^T @ B`` が(機械精度で)0、つまり ``B @ z`` は任意の z について
  自動的に恒等式を満たす

という2性質を持つ。``x ≈ B @ (B^T @ x)`` が低次元表現(圧縮率 D→k)になる。
"""

from __future__ import annotations

import numpy as np

from fsgen.accounting.constraints import orthonormal_basis


def build_reduced_basis(matrix: np.ndarray, x: np.ndarray, k: int) -> np.ndarray:
    """C ∈ R^{D×K} と実データ X ∈ R^{n×D}(各行が C^T row = 0 を満たす)から、
    上位 k 主成分方向でできた基底 B ∈ R^{D×k} を返す。

    非中心化(mean を引かない)SVDで、``X ≈ Z V_k^T`` の切断誤差を最小化する
    (Eckart-Young)。中心化しないのは、``x ≈ B @ z`` という純粋な線形
    (オフセット無し)再構成を成立させるため。
    """
    q = orthonormal_basis(matrix)  # (D, D-K), null(C^T) の正規直交基底
    z = x @ q  # (n, D-K): X は既に null(C^T) 上にあるので情報の欠落は無い
    _u, _s, vt = np.linalg.svd(z, full_matrices=False)
    v_k = vt[:k]  # (k, D-K)、上位k方向(分散降順、行が正規直交)
    b: np.ndarray = q @ v_k.T  # (D, k)
    return b


def reconstruction_error(b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """各行について ‖x - B B^T x‖ / ‖x‖ を返す(‖x‖=0の行は0とする)。"""
    x_hat = x @ b @ b.T
    diff_norm = np.linalg.norm(x - x_hat, axis=1)
    x_norm = np.linalg.norm(x, axis=1)
    safe_norm = np.where(x_norm == 0, 1.0, x_norm)
    result: np.ndarray = np.where(x_norm == 0, 0.0, diff_norm / safe_norm)
    return result
