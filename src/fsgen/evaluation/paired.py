"""企業単位に集約したペア検定・ブートストラップ信頼区間(ステージ13 §B.2)。

ステージ10以降のp値は 68社×3seed(fold は企業ごとに固定)=204行を独立標本として
検定していた疑いがある。同一企業がseedの数だけ繰り返し現れるため、有効標本数は
企業数(68)である。ここでは企業ごとにseed(・fold)をまたいで中央値を取り、
1社1観測にしてから検定する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairedResult:
    n: int
    median_diff: float
    ci_low: float
    ci_high: float
    p_value: float


def aggregate_by_company(
    df: pd.DataFrame, value_col: str, group_cols: tuple[str, ...] = ("method",)
) -> pd.DataFrame:
    """企業ごとにseed・foldをまたいで中央値を取り、行=企業(docID)・列=group_cols値の表にする。"""
    key = list(group_cols)
    agg = df.groupby(["docID", *key])[value_col].median().reset_index()
    if len(key) == 1:
        return agg.pivot(index="docID", columns=key[0], values=value_col)
    agg["_g"] = agg[key].astype(str).agg("|".join, axis=1)
    return agg.pivot(index="docID", columns="_g", values=value_col)


def paired_test(
    a: np.ndarray, b: np.ndarray, *, n_boot: int = 2000, seed: int = 0
) -> PairedResult:
    """差 a-b のWilcoxon符号付順位検定(両側)と、中央値のブートストラップ95%信頼区間。

    ブートストラップは**観測単位(企業)を復元抽出**する。a,bは同じ順序の企業ごとの値。
    """
    from scipy.stats import wilcoxon

    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    keep = np.isfinite(diff)
    diff = diff[keep]
    if len(diff) < 2 or np.all(diff == 0):
        return PairedResult(len(diff), float("nan"), float("nan"), float("nan"), float("nan"))
    p = float(wilcoxon(diff).pvalue)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    meds = np.median(diff[idx], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return PairedResult(len(diff), float(np.median(diff)), float(lo), float(hi), p)


def unaggregated_p_value(df: pd.DataFrame, value_col: str, method_a: str, method_b: str) -> float:
    """集約前(seed×企業の各行を独立とみなす)のWilcoxon p値。擬似反復の影響を示すための比較用。"""
    from scipy.stats import wilcoxon

    key = ["seed", "docID"]
    a = df[df["method"] == method_a].set_index(key)[value_col]
    b = df[df["method"] == method_b].set_index(key)[value_col]
    common = a.index.intersection(b.index)
    diff = (a.loc[common] - b.loc[common]).to_numpy()
    if np.all(diff == 0):
        return float("nan")
    return float(wilcoxon(diff).pvalue)


__all__ = ["PairedResult", "aggregate_by_company", "paired_test", "unaggregated_p_value"]
