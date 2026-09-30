"""会計構造が異なる業種の書類を、生成・評価の対象から除く(ステージ28 A)。

除外リストは ``scripts/special_industry_stage28.py`` が作る CSV(列 ``docID``)。
5-fold の分割(``data/interim/industry_12/folds.npz``)は作り直さず、除外した始点を
各 fold から取り除き、残った始点の位置を詰め直す(``subset_folds``)。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def load_excluded_docs(path: Path | None) -> frozenset[str]:
    """除外する docID の集合。``path`` が None なら空。"""
    if path is None:
        return frozenset()
    df = pd.read_csv(path, dtype={"docID": str})
    if "docID" not in df.columns:
        raise ValueError(f"{path} に docID 列がない")
    return frozenset(df["docID"].tolist())


def subset_folds(folds: Mapping[str, np.ndarray], keep: np.ndarray) -> dict[str, np.ndarray]:
    """``keep``(元の始点の並びでの真偽)で始点を絞り、各 fold の位置を詰め直す。

    元の位置 p(``keep[p]`` が真)は、残った始点の中での順位に写す。fold の割り当て
    (どの始点がどの fold のテストか)は変えない。
    """
    keep = np.asarray(keep, dtype=bool)
    new_pos = np.cumsum(keep) - 1
    out: dict[str, np.ndarray] = {}
    for key in folds:
        pos = np.asarray(folds[key])
        if pos.size and pos.max() >= len(keep):
            raise ValueError(f"{key} の位置 {int(pos.max())} が始点数 {len(keep)} を超える")
        out[key] = new_pos[pos[keep[pos]]]
    return out


def exclusion_masks(
    doc_ids: Sequence[str], labels: Sequence[int], excluded: frozenset[str],
    source_label: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """(全行で残す行, 始点(``source_label``、既定 class 2)の並びで残す始点) の真偽配列。"""
    row_keep = np.array([d not in excluded for d in doc_ids], dtype=bool)
    src_keep = np.array(
        [d not in excluded for d, lab in zip(doc_ids, labels, strict=True)
         if lab == source_label],
        dtype=bool,
    )
    return row_keep, src_keep


__all__ = ["exclusion_masks", "load_excluded_docs", "subset_folds"]
