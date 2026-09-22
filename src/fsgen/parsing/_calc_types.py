"""calculation linkbase の最小限のデータ型。循環 import を避けるための内部モジュール。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalcArc:
    """calculationArc 1本。parent が合計、child が内訳。"""

    parent: str
    child: str
    weight: float
    order: float | None
