"""業種を考慮した輸送のための業種マスタとブロック割り当て(ステージ12)。

業種マスタ(``conf/accounts/industry.yaml``)はEDINETコードリストの「提出者業種」
(東証33業種と同一の体系)を edinetCode ごとに固定保存したもの。外部APIには
依存しない。

**ハード制約と輸送先の分割**: 各始点は同業種の class 0 企業にしか輸送できない。
class 0 の周辺分布を一様のままにすると、始点側と目標側で業種別の質量が
合わず(例: 始点の3%が業種Xだが目標の5%が業種X)輸送計画が存在しない。
そこで目標側の質量を「その業種(ブロック)に属する始点の質量」に合わせて
再配分する(ブロックごとに独立した釣り合いOTになる)。始点側は業種で
分割しない(正例68社をそのまま使う、§1.1)。

**フォールバック**(§2.2): 始点の業種に class 0 が ``min_targets`` 社未満なら
粗い分類の全業種を併合し、それでも足りなければ業種制約なしに落とす。
業種制約なしの始点は、ブロック分割(全始点が共有する分割)に混ぜると
他の始点の制約まで消えてしまう(ステージ12で実際に発生した)ため、ブロックOTから
外し(``src_block=-1``)、呼び出し側で始点ごとに独立に解く。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

IndustryLevel = Literal["33", "coarse"]
UNKNOWN_LABEL = "不明"
DEFAULT_MIN_TARGETS = 10


@dataclass(frozen=True)
class IndustryConfig:
    edinet_industry: dict[str, str]
    coarse_map: dict[str, str]

    def labels(self, edinet_codes: list[str], level: IndustryLevel = "33") -> list[str]:
        """edinetCode列を業種ラベル列に変換する(未登録は「不明」)。"""
        out = []
        for code in edinet_codes:
            ind = self.edinet_industry.get(code, UNKNOWN_LABEL)
            out.append(ind if level == "33" else self.coarse_map.get(ind, UNKNOWN_LABEL))
        return out


def load_industry_config(path: Path) -> IndustryConfig:
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return IndustryConfig(
        edinet_industry={str(k): str(v) for k, v in doc["edinet_industry"].items()},
        coarse_map={str(k): str(v) for k, v in doc["coarse_map"].items()},
    )


@dataclass(frozen=True)
class BlockAssignment:
    src_block: np.ndarray  # (n0,) 各始点のブロック番号。業種制約なしにフォールバックした始点は-1
    tgt_block: np.ndarray  # (n1,) 各目標企業のブロック番号
    src_level: list[str]  # 各始点が最終的に使った粒度: "33"/"coarse"/"all"(フォールバック記録)


class _UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {i: i for i in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def assign_blocks(
    src_labels: list[str],
    tgt_labels: list[str],
    coarse_map: dict[str, str],
    level: IndustryLevel,
    min_targets: int = DEFAULT_MIN_TARGETS,
) -> BlockAssignment:
    """業種ラベルから、輸送を制限するブロックを作る(フォールバック込み)。

    src_labels/tgt_labels は33業種のラベル。level="33" なら33業種単位、
    "coarse" なら粗い分類単位を出発点とする。
    """
    inds = sorted(set(src_labels) | set(tgt_labels))
    uf = _UnionFind(inds)
    by_coarse: dict[str, list[str]] = {}
    for ind in inds:
        by_coarse.setdefault(coarse_map.get(ind, UNKNOWN_LABEL), []).append(ind)

    def merge_coarse(ind: str) -> None:
        group = by_coarse[coarse_map.get(ind, UNKNOWN_LABEL)]
        for other in group[1:]:
            uf.union(group[0], other)

    if level == "coarse":
        for group in by_coarse.values():
            for other in group[1:]:
                uf.union(group[0], other)

    def n_targets(ind: str) -> int:
        root = uf.find(ind)
        return sum(1 for t in tgt_labels if uf.find(t) == root)

    start_level = "33" if level == "33" else "coarse"
    src_level = [start_level] * len(src_labels)
    # 第1段: 目標が足りない始点は粗い分類を併合する
    for i, ind in enumerate(src_labels):
        if n_targets(ind) < min_targets:
            merge_coarse(ind)
            src_level[i] = "coarse"
    # 第2段: それでも足りない始点は業種制約なし("all")。分割には混ぜない(src_block=-1)。
    unrestricted = [n_targets(ind) < min_targets for ind in src_labels]
    for i, flag in enumerate(unrestricted):
        if flag:
            src_level[i] = "all"

    roots = sorted({uf.find(i) for i in inds})
    block_id = {r: k for k, r in enumerate(roots)}
    src_block = np.array(
        [-1 if unrestricted[i] else block_id[uf.find(s)] for i, s in enumerate(src_labels)],
        dtype=int,
    )
    tgt_block = np.array([block_id[uf.find(t)] for t in tgt_labels], dtype=int)
    return BlockAssignment(src_block=src_block, tgt_block=tgt_block, src_level=src_level)


def block_marginals(src_block: np.ndarray, tgt_block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """始点は一様(1/n0)、目標は「そのブロックの始点質量を、ブロック内の目標に均等配分」。

    始点が1社もいないブロックの目標は質量0(輸送先にならない)。src_block=-1
    (業種制約なし)の始点は目標側の質量に含めない(呼び出し側で別に解く)ので、
    sum(b) = (制約つき始点の数)/n0 で、制約つき始点の行だけを取り出せばsum(a)と一致する。
    """
    n0, n1 = len(src_block), len(tgt_block)
    a = np.full(n0, 1.0 / n0)
    b = np.zeros(n1)
    for blk in np.unique(src_block[src_block >= 0]):
        mass = float(np.sum(src_block == blk)) / n0
        members = tgt_block == blk
        n_members = int(members.sum())
        if n_members == 0:
            raise ValueError(f"ブロック{blk}に目標企業がありません(フォールバック漏れ)")
        b[members] = mass / n_members
    return a, b


__all__ = [
    "DEFAULT_MIN_TARGETS",
    "UNKNOWN_LABEL",
    "BlockAssignment",
    "IndustryConfig",
    "IndustryLevel",
    "assign_blocks",
    "block_marginals",
    "load_industry_config",
]
