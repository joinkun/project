"""ステージ60 B: 業種で層化した比例配分の抽出(numpy だけ。POT のプロセスと評価のプロセスで共有)。"""

from __future__ import annotations

import numpy as np


def stratified(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    """業種で層化した比例配分の無作為抽出(criteria §2)。返り値は列の添字。"""
    cats, counts = np.unique(labels, return_counts=True)
    share = counts / counts.sum()
    alloc = np.round(n * share).astype(int)
    order = np.argsort(-share)
    k = 0
    while alloc.sum() != n:
        j = order[k % len(order)]
        step = 1 if alloc.sum() < n else -1
        if 0 <= alloc[j] + step <= counts[j]:
            alloc[j] += step
        k += 1
    rng = np.random.default_rng(seed)
    idx = [rng.choice(np.where(labels == c)[0], size=m, replace=False)
           for c, m in zip(cats, alloc, strict=True) if m > 0]
    return np.sort(np.concatenate(idx))
