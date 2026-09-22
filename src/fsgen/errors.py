"""ソルバの収束失敗を表す例外(ステージ13 §A.1)。

本研究では、エラーを出さずに誤った値を返す故障が4件発生した(射影QPの誤解、
validityの符号反転、微分可能QP層の勾配破綻、Sinkhornの未収束)。うち2件は
ソルバの収束判定を見ていなかったことが原因だったため、最適化ソルバを呼ぶ
箇所は収束を確認し、未収束なら**警告ではなくこの例外**を投げる。
"""

from __future__ import annotations


class SolverNotConvergedError(RuntimeError):
    """最適化ソルバが収束しなかった(または結果の健全性チェックに失敗した)。"""


class InfeasibleProjectionError(SolverNotConvergedError):
    """射影QPが実行不可能(実行可能集合が空)。追加制約が強すぎる場合に起きる(ステージ14 §B.2)。"""


__all__ = ["InfeasibleProjectionError", "SolverNotConvergedError"]
