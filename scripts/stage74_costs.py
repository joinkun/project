"""ステージ74: 始点ごとの移動コスト c̃_i^±(numpy だけ。POT・cvxpy の両方のプロセスから使う)。

規則は ``results/stage74/criteria.md`` §3・§4。
"""

from __future__ import annotations

import numpy as np

_RE, _VAL = "jppfs_cor_RetainedEarnings", "valuation_difference_securities"
# 設定 → (コストの種類, class 2 のプール, class 1 のプール)
SETTINGS = {"A1": ("abs", "57", "10"), "A2": ("abs", "57", "15"), "A3": ("abs", "57", "57"),
            "B1": ("rel", "57", "10"), "B2": ("hyb", "57", "10")}


def pool_of(setting: str, cls: str) -> str:
    _kind, p2, p1 = SETTINGS[setting]
    return p2 if cls == "c2" else p1


def entry(wj: dict[str, object], pool: str, code: str) -> dict[str, list[float]]:
    """プールの係数(始点と同じコードの企業がプールにあれば、それを除いた推定)。"""
    loo: dict[str, dict[str, list[float]]] = wj["loo"]  # type: ignore[assignment]
    pools: dict[str, dict[str, list[float]]] = wj["pools"]  # type: ignore[assignment]
    return loo.get(f"{pool}:{code}", pools[pool])


def firm_costs(kind: str, x0: np.ndarray, e: dict[str, list[float]], floor: np.ndarray,
               names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """始点 x0 の c̃⁺・c̃⁻。利益剰余金と評価差額金は 0(輸送でも QP の移動コストでも)。"""
    ap, am = np.array(e["abs_plus"], dtype=float), np.array(e["abs_minus"], dtype=float)
    if kind == "abs":
        cp, cm = ap, am
    else:
        cg, cs = np.array(e["rel_grow"], dtype=float), np.array(e["rel_shrink"], dtype=float)
        den = np.maximum(np.abs(x0), floor)
        pos = x0 >= 0
        rp = np.where(pos, cg, cs) / den
        rm = np.where(pos, cs, cg) / den
        if kind == "rel":
            cp, cm = rp, rm
        elif kind == "hyb":
            cp, cm = np.sqrt(ap * rp), np.sqrt(am * rm)
        else:
            raise ValueError(f"コストの種類が不明: {kind}")
    cp, cm = cp.copy(), cm.copy()
    for nm in (_RE, _VAL):
        cp[names.index(nm)] = cm[names.index(nm)] = 0.0
    return cp, cm
