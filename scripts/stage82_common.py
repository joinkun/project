"""ステージ82 の共通の定義(重いライブラリを読み込まない。どのプロセスからも使える)。

規則は ``results/stage82/criteria.md``。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATA = Path("data/interim/stage82")
OUT = Path("results/stage82")
FOLDS = ("f0", "f1", "f2", "f3", "f4")
GAMMA, TAU1, EPS = 0.1, 0.05, 0.001
GAMMA_COL, GAMMA_COL_FALLBACK = 1e-6, 1e-5
ZERO_OT = ("jppfs_cor_RetainedEarnings", "valuation_difference_securities")


@dataclass(frozen=True)
class CostSpec:
    mode: str  # R1・KDE0・IPW・scale・A1
    indiv: bool  # 個体別の残高スケーリング
    penalty: bool  # A_inact(i) の θ⁺ = θ_max(輸送と QP)
    pool: str = "rec57"  # rec57(57 社、fold ごと)・alldist(苦境全体、full だけ)


COSTS: dict[str, CostSpec] = {
    "R1": CostSpec("R1", False, False),
    "KDE0": CostSpec("KDE0", False, False),
    "IPW": CostSpec("IPW", False, False),
    "scale": CostSpec("scale", False, False),
    "A1": CostSpec("A1", False, False),
    "R1i": CostSpec("R1", True, False),
    "R1p": CostSpec("R1", False, True),
    "R1ip": CostSpec("R1", True, True),
    # 実験 5: 苦境全体(回復 72 社 + 滞留群)から推定した R1(full だけ)
    "R1_all": CostSpec("R1", False, False, "alldist"),
    "R1i_all": CostSpec("R1", True, False, "alldist"),
    "R1p_all": CostSpec("R1", False, True, "alldist"),
    "R1ip_all": CostSpec("R1", True, True, "alldist"),
}
OT_MODES = ("prop", "pure_lp", "top5", "top8", "sinkhorn", "row", "col")


@dataclass(frozen=True)
class Setting:
    group: str  # 生成を置くディレクトリ(e1・e2・e3・full)
    cost: str
    ot: str = "prop"
    qp: str = "base"  # base・unified(A_inact(i) を QP で固定)
    tier: str = "t123"  # t1・t12・t123

    @property
    def tag(self) -> str:
        return f"{self.cost}_{self.ot}_{self.qp}_{self.tier}"


# D1 の設定(表示名 → 設定)。実験 1・2・3 の基準は、どれも R1(False、base、prop)
BASE = Setting("e1", "R1")
EXP1 = {"R1": BASE, "KDE0": Setting("e1", "KDE0"), "IPW": Setting("e1", "IPW"),
        "scale": Setting("e1", "scale"), "A1": Setting("e1", "A1")}
EXP2 = {"R1(False・base)": BASE, "R1(True・base)": Setting("e2", "R1i"),
        "R1(False・stage1_penalty)": Setting("e2", "R1p"),
        "R1(True・stage1_penalty)": Setting("e2", "R1ip"),
        "R1(False・unified_extended)": Setting("e2", "R1p", qp="unified"),
        "R1(True・unified_extended)": Setting("e2", "R1ip", qp="unified")}
EXP3 = {"proposed_quad_unbalanced": BASE, "pure_lp": Setting("e3", "R1", "pure_lp"),
        "topk_hard_5": Setting("e3", "R1", "top5"), "topk_hard_8": Setting("e3", "R1", "top8"),
        "sinkhorn": Setting("e3", "R1", "sinkhorn"), "quad_row_only": Setting("e3", "R1", "row"),
        "quad_col_only": Setting("e3", "R1", "col")}
D1_SETTINGS = {s.tag: s for grp in (EXP1, EXP2, EXP3) for s in grp.values()}
