"""生成された貸借対照表の会計的妥当性(ステージ14)。

ステージ13の指標(proximity・validity・恒等式残差・基礎科目の符号)がすべて良好でも、
小計・総合計は制約を受けないため、資産合計が負・純資産の悪化といった
「BSとして成立しない出力」が出ることが分かった。ここでは

1. 8項目の自動チェック(§A.2): 出力を機械的に判定する
2. 追加の線形制約(§B): 射影QPに課す(``project_onto_feasible``の``extra_constraints``)

を、基礎科目の状態ベクトル(総資産で正規化=tilde空間)に対して定義する。
小計は、基礎科目の``parent_subtotal``から配下の基礎科目の和として構成する
(Assets直下の``__OTHER__Assets``はAssetsのみに属し、流動/固定には属さない)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from fsgen.accounting.constraints import CanonicalSpec

_CASH = "jppfs_cor_CashAndDeposits"
_RECEIVABLES = "receivables_trade"
_INVENTORIES = "inventories"
_CAPITAL_STOCK = "jppfs_cor_CapitalStock"
_CAPITAL_SURPLUS = "jppfs_cor_CapitalSurplus"
# 事業継続の制約(ステージ16 §C)の対象: 事業の稼働に使う資産(売却=事業縮小)
# 控除科目を引いた純額(ステージ16で発見: 貸倒引当金が負に大きい生成点は、現金が総資産を上回る
# 出力を作れる。実データ4,156社で全て非負=違反0のものだけを採る)。
#   流動: 売上債権+その他流動資産+流動資産その他+貸倒引当金(流動) = 流動資産-現金-棚卸資産
#   固定: 投資その他の資産(総額)+貸倒引当金(固定) = 旧canonicalの投資その他の資産(純額)
# 売上債権+貸倒引当金(流動)だけでは実データの160社が負になる(引当金はその他債権にも掛かる)。
NET_OF_CONTRA: dict[str, tuple[str, ...]] = {
    "net:CA_ex_cash_inventories": (
        _RECEIVABLES, "jppfs_cor_OtherCA", "__OTHER__jppfs_cor_CurrentAssets",
        "allowance_doubtful_current",
    ),
    "net:IOA_after_allowance": (
        "jppfs_cor_InvestmentsAndOtherAssets", "allowance_doubtful_noncurrent",
    ),
}
RETENTION_ACCOUNTS = ("jppfs_cor_PropertyPlantAndEquipment", _INVENTORIES, _RECEIVABLES)

SUBTOTAL_NONNEG_IDS = (
    "jppfs_cor_Assets",
    "jppfs_cor_CurrentAssets",
    "jppfs_cor_NoncurrentAssets",
    "jppfs_cor_Liabilities",
    "jppfs_cor_CurrentLiabilities",
    "jppfs_cor_NoncurrentLiabilities",
)  # NetAssets・ShareholdersEquityは債務超過を許容するため含めない(§B.1)


# 下限を課す残余科目(8個)。ステージ14では、これらを自由符号にすると生成点が負の残余で小計の
# 非負制約をすり抜ける(例: 固定負債合計=0を、長期借入金+221と固定負債その他-221の相殺で
# 満たす)ため、下限を課した。ステージ16で、流動資産その他が負になる主因だった貸倒引当金を
# canonicalの基礎科目(allowance_doubtful_current)に追加した結果、これら8個は実データ4,156社で
# 総資産比-1e-3を下回らなくなった(最小-5e-4)。負の値は丸め(decimals=-6で最大±5e5円)の
# 産物なので、下限は「0から丸め許容(総資産比1e-3)を引いた値」とし、始点への緩和は不要になった
# (68社の始点は全て満たす。満たさない始点があれば例外にする)。評価換算差額その他は実データでも
# 負になる(最小-1.15、288社が-1e-3未満)ので対象外(自由のまま)。
RESIDUAL_FLOOR = -1e-3
RESIDUAL_NONNEG_ACCOUNTS = (
    "__OTHER__jppfs_cor_Assets",
    "__OTHER__jppfs_cor_CurrentAssets",
    "__OTHER__jppfs_cor_NoncurrentAssets",
    "__OTHER__jppfs_cor_CurrentLiabilities",
    "__OTHER__jppfs_cor_Liabilities",
    "__OTHER__jppfs_cor_NoncurrentLiabilities",
    "__OTHER__jppfs_cor_NetAssets",
    "__OTHER__jppfs_cor_ShareholdersEquity",
)
RESIDUAL_LOWER_BOUNDS: dict[str, float] = dict.fromkeys(RESIDUAL_NONNEG_ACCOUNTS, RESIDUAL_FLOOR)


def subtotal_members(spec: CanonicalSpec, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """小計・総合計 → 配下の基礎科目の指示ベクトル(D,)。"""
    idx = {n: i for i, n in enumerate(names)}
    members: dict[str, np.ndarray] = {}

    def vec(cond: object) -> np.ndarray:
        v = np.zeros(len(names))
        for b in spec.basis:
            if cond(b):  # type: ignore[operator]
                v[idx[b.name]] = 1.0
        return v

    for sub in spec.subtotals:
        if sub.element_id == "jppfs_cor_Assets":
            members[sub.element_id] = vec(lambda b: b.side == "assets")
        elif sub.element_id == "jppfs_cor_Liabilities":
            members[sub.element_id] = vec(lambda b: b.side == "liabilities")
        elif sub.element_id == "jppfs_cor_NetAssets":
            members[sub.element_id] = vec(lambda b: b.side == "net_assets")
        else:
            members[sub.element_id] = vec(lambda b, s=sub.element_id: b.parent_subtotal == s)
    return members


@dataclass(frozen=True)
class LinearConstraint:
    """a·x >= rhs(基礎科目の状態ベクトル x に対する線形不等式)。"""

    name: str
    a: np.ndarray
    rhs: float


@dataclass(frozen=True)
class PlausibilityConfig:
    """追加制約の設定(既定はすべて無効=S0=現行)。core.yamlの``plausibility_constraints``。"""

    subtotal_nonneg: bool = False
    net_assets_non_deterioration: bool = False
    capital_min_ratio: float | None = None  # α: 資本金・資本剰余金 >= α×始点(0で無効に近い)
    assets_ratio_range: tuple[float, float] | None = None  # (下限, 上限)×始点の総資産
    residual_nonneg: bool = False  # 実データで負が実質0の残余科目の下限0(RESIDUAL_NONNEG_ACCOUNTS)
    retention_beta: float | None = None  # β: 有形固定資産・棚卸資産・売上債権(t) >= β×始点
    contra_net_nonneg: bool = False  # 控除科目(貸倒引当金)を引いた後の純額が非負(ステージ16)
    label: str = field(default="S0", compare=False)


@dataclass(frozen=True)
class ConstraintLayers:
    """ステージ17 §C.1: 制約を2層に分ける。

    第1層(layer1): 会計恒等式(貸借一致)と各基礎科目の符号。project_onto_feasible が
    ``extra_constraints`` の有無によらず常に課す不変条件で、会計の定義そのものであり
    議論の余地がない。**無効化はできない**(layer1=falseはコード上の意味を持たず、
    設定ミスとして拒否する)。
    第2層(layer2): このモジュールの PlausibilityConfig が表す制約群(小計非負・純資産
    非悪化・残余下限・控除後純額・資本金下限α・事業継続β)。実務的妥当性のための
    設計判断で、無効にすると PlausibilityConfig() (=S0) と同じになる。
    """

    layer1: bool = True
    layer2: bool = True


def load_constraint_layers(path: Path) -> ConstraintLayers:
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    raw = doc.get("constraint_layers", {}) or {}
    layers = ConstraintLayers(
        layer1=bool(raw.get("layer1", True)), layer2=bool(raw.get("layer2", True))
    )
    if not layers.layer1:
        raise ValueError(
            "layer1(会計恒等式・符号)は無効化できない。project_onto_feasible の"
            "不変条件であり、設定ではない。"
        )
    return layers


def apply_layers(cfg: PlausibilityConfig, layers: ConstraintLayers) -> PlausibilityConfig:
    """layers.layer2 が False なら、cfg の中身によらず第2層の制約を全て外す(=S0相当)。"""
    if layers.layer2:
        return cfg
    return PlausibilityConfig(label=f"{cfg.label}(layer2 off)")


def load_plausibility_config(path: Path) -> PlausibilityConfig:
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    raw = doc.get("plausibility_constraints", {}) or {}
    rng = raw.get("assets_ratio_range")
    return PlausibilityConfig(
        subtotal_nonneg=bool(raw.get("subtotal_nonneg", False)),
        net_assets_non_deterioration=bool(raw.get("net_assets_non_deterioration", False)),
        capital_min_ratio=raw.get("capital_min_ratio"),
        assets_ratio_range=(float(rng[0]), float(rng[1])) if rng else None,
        residual_nonneg=bool(raw.get("residual_nonneg", False)),
        retention_beta=raw.get("retention_beta"),
        contra_net_nonneg=bool(raw.get("contra_net_nonneg", False)),
    )


def build_linear_constraints(
    spec: CanonicalSpec,
    names: tuple[str, ...],
    x_start: np.ndarray,
    cfg: PlausibilityConfig,
) -> tuple[LinearConstraint, ...]:
    """始点 x_start に応じた追加制約(a·x >= rhs)を作る。

    始点が既に制約を破っている場合(実データの小計が負など)、始点を実行可能に
    保つため、その制約の下限を始点の値まで緩める(「悪化させない」に読み替える)。
    """
    members = subtotal_members(spec, names)
    idx = {n: i for i, n in enumerate(names)}
    out: list[LinearConstraint] = []
    if cfg.subtotal_nonneg:
        for sid in SUBTOTAL_NONNEG_IDS:
            a = members[sid]
            out.append(LinearConstraint(f"{sid}>=0", a, min(0.0, float(a @ x_start))))
    if cfg.net_assets_non_deterioration:
        a = members["jppfs_cor_NetAssets"]
        out.append(LinearConstraint("NetAssets>=NetAssets(0)", a, float(a @ x_start)))
    if cfg.capital_min_ratio is not None:
        alpha = float(cfg.capital_min_ratio)
        for nm in (_CAPITAL_STOCK, _CAPITAL_SURPLUS):
            a = np.zeros(len(names))
            a[idx[nm]] = 1.0
            x0 = float(x_start[idx[nm]])
            out.append(LinearConstraint(f"{nm}>={alpha:g}*start", a, alpha * x0 if x0 >= 0 else x0))
    if cfg.residual_nonneg:
        for nm in RESIDUAL_NONNEG_ACCOUNTS:
            a = np.zeros(len(names))
            a[idx[nm]] = 1.0
            bound = RESIDUAL_FLOOR * float(members["jppfs_cor_Assets"] @ x_start)
            if float(x_start[idx[nm]]) < bound:
                raise ValueError(f"始点が残余科目の下限を破っている: {nm}(canonicalの見直しが必要)")
            out.append(LinearConstraint(f"residual:{nm[9:]}", a, bound))
    if cfg.contra_net_nonneg:
        for cname, members_c in NET_OF_CONTRA.items():
            a = np.zeros(len(names))
            for nm in members_c:
                a[idx[nm]] = 1.0
            out.append(LinearConstraint(f"{cname}>=0", a, min(0.0, float(a @ x_start))))
    if cfg.retention_beta is not None and cfg.retention_beta > 0:
        for nm in RETENTION_ACCOUNTS:
            a = np.zeros(len(names))
            a[idx[nm]] = 1.0
            x0 = float(x_start[idx[nm]])
            beta = cfg.retention_beta
            out.append(LinearConstraint(f"retention:{nm}>={beta:g}*start", a, beta * max(x0, 0.0)))
    if cfg.assets_ratio_range is not None:
        lo, hi = cfg.assets_ratio_range
        a = members["jppfs_cor_Assets"]
        a0 = float(a @ x_start)
        out.append(LinearConstraint(f"Assets>={lo:g}*Assets(0)", a, lo * a0))
        out.append(LinearConstraint(f"Assets<={hi:g}*Assets(0)", -a, -hi * a0))
    return tuple(out)


CHECK_NAMES = (
    "1_資産合計が負",
    "2_小計が負",
    "3_現金がほぼ0",
    "4_主要資産が全滅",
    "5_負債が2倍超",
    "6_純資産が悪化",
    "7_健全→債務超過",
    "8_資本金が半減以下",
)
_CASH_TOL = 1e-3  # 現金・主要資産の「ほぼ0」: 始点の総資産比
_NA_TOL = 1e-6  # 数値誤差の許容(tilde空間)
# 丸め・ソルバ許容(総資産比)。純資産の悪化判定に使う: 極端な高レバレッジ社ではQPの許容
# (1e-6×max|x|)が総資産比3e-4まで広がる
_ROUND_TOL = 1e-3
_LIAB_FLOOR = 0.01  # チェック5の対象: 終点の負債 > 総資産(0)の1%


def evaluate_checks(
    x0: np.ndarray, x1: np.ndarray, spec: CanonicalSpec, names: tuple[str, ...]
) -> np.ndarray:
    """(n, 8)のbool配列。行=企業、列=CHECK_NAMES。x0/x1は(n, D)のtilde空間。"""
    m = subtotal_members(spec, names)
    idx = {n: i for i, n in enumerate(names)}
    assets0 = x0 @ m["jppfs_cor_Assets"]
    assets1 = x1 @ m["jppfs_cor_Assets"]
    liab0 = x0 @ m["jppfs_cor_Liabilities"]
    liab1 = x1 @ m["jppfs_cor_Liabilities"]
    na0 = x0 @ m["jppfs_cor_NetAssets"]
    na1 = x1 @ m["jppfs_cor_NetAssets"]
    sub_neg = np.zeros(len(x1), dtype=bool)
    for sid in (
        "jppfs_cor_CurrentAssets", "jppfs_cor_NoncurrentAssets",
        "jppfs_cor_CurrentLiabilities", "jppfs_cor_NoncurrentLiabilities",
    ):
        sub_neg |= (x1 @ m[sid]) < -_NA_TOL
    cash1 = x1[:, idx[_CASH]] / assets0
    major_dead = (
        (np.abs(x1[:, idx[_CASH]]) / assets0 < _CASH_TOL)
        & (np.abs(x1[:, idx[_RECEIVABLES]]) / assets0 < _CASH_TOL)
        & (np.abs(x1[:, idx[_INVENTORIES]]) / assets0 < _CASH_TOL)
    )
    cs0, cs1 = x0[:, idx[_CAPITAL_STOCK]], x1[:, idx[_CAPITAL_STOCK]]
    flags = np.column_stack([
        assets1 < 0,
        sub_neg,
        cash1 < _CASH_TOL,
        major_dead,
        # 5: 始点の負債がごく小さい社で、わずかな増加が2倍超と判定されないよう、終点の負債が
        # 総資産(0)の1%を超える場合に限る(ステージ19 §A.2。資本金比と同じ趣旨の絶対額の下限)
        (liab0 > 0) & (liab1 / np.where(liab0 > 0, liab0, 1.0) > 2.0)
        & (liab1 / assets0 > _LIAB_FLOOR),
        na1 < na0 - _ROUND_TOL,
        # 7: 始点の純資産がごく小さい正値の社(68社中1社、0.95%)が丸め程度で負になっても
        # 数えないよう、6・8と同じ許容 _ROUND_TOL を使う(ステージ19 §A.3)
        (na0 > 0) & (na1 < -_ROUND_TOL),
        # 絶対許容(比だと資本金が極小の社で誤検出)。QPの受理判定は 1e-6×scale(円建て)の
        # 相対許容なので、tilde空間(÷総資産)に直すと大企業では 1e-6 を超えうる
        # (ステージ18で実測: 4.8e-6)。ソルバ精度の範囲として _ROUND_TOL を使う。
        (cs0 > 0) & (cs1 < 0.5 * cs0 - _ROUND_TOL),
    ])
    return flags


EXTRA_CHECK_NAMES = (
    "9_総負債が負", "10_残余科目が負", "11_現金>総資産", "12_基礎科目の符号違反",
    "13_控除後の純額が負",
)
_SIGN_TOL = 1e-6  # 基礎科目の符号違反の許容(tilde空間=総資産比)


def evaluate_extra_checks(
    x1: np.ndarray, x0: np.ndarray, spec: CanonicalSpec, names: tuple[str, ...]
) -> np.ndarray:
    """(n, 5)のbool配列。§A.2の8項目では見えない抜け穴(ステージ14〜16で発見)を捉える。

    9: 総負債が負。10: 残余科目(RESIDUAL_NONNEG_ACCOUNTS)のいずれかが下限(総資産比-1e-3)を
    下回る。11: 現金が総資産を上回る(ステージ15)。12: 基礎科目の符号違反(nonnegが負、
    nonposが正。許容1e-6×総資産、ステージ16)。13: 控除科目を引いた純額(NET_OF_CONTRA)が負
    (許容は丸め1e-3)。11の許容も丸め1e-3(総資産直下の残余の下限と同じ)。
    """
    m = subtotal_members(spec, names)
    idx = {n: i for i, n in enumerate(names)}
    assets0 = x0 @ m["jppfs_cor_Assets"]
    liab_neg = (x1 @ m["jppfs_cor_Liabilities"]) < -_NA_TOL
    below = np.column_stack([
        x1[:, idx[n]] / assets0 < RESIDUAL_FLOOR - _NA_TOL for n in RESIDUAL_NONNEG_ACCOUNTS
    ])
    sign_bad = np.zeros(len(x1), dtype=bool)
    for b in spec.basis:
        if b.sign == "nonneg":
            sign_bad |= x1[:, idx[b.name]] < -_SIGN_TOL
        elif b.sign == "nonpos":
            sign_bad |= x1[:, idx[b.name]] > _SIGN_TOL
    cash_over = x1[:, idx[_CASH]] > (x1 @ m["jppfs_cor_Assets"]) - RESIDUAL_FLOOR  # 許容=丸め(1e-3)
    net_bad = np.zeros(len(x1), dtype=bool)
    for members_c in NET_OF_CONTRA.values():
        net_bad |= x1[:, [idx[nm] for nm in members_c]].sum(axis=1) < RESIDUAL_FLOOR - _NA_TOL
    return np.column_stack([liab_neg, below.any(axis=1), cash_over, sign_bad, net_bad])


def check_summary(flags: np.ndarray, doc_ids: list[str]) -> list[dict[str, object]]:
    """チェックごとの違反企業数とdocID(§A.2の報告用)。"""
    rows: list[dict[str, object]] = []
    for j, name in enumerate(CHECK_NAMES):
        bad = [d for d, f in zip(doc_ids, flags[:, j], strict=True) if f]
        rows.append({"check": name, "n_violations": len(bad), "docIDs": ",".join(bad)})
    return rows


__all__ = [
    "ConstraintLayers",
    "apply_layers",
    "load_constraint_layers",
    "CHECK_NAMES",
    "EXTRA_CHECK_NAMES",
    "RESIDUAL_NONNEG_ACCOUNTS",
    "evaluate_extra_checks",
    "check_summary",
    "LinearConstraint",
    "PlausibilityConfig",
    "SUBTOTAL_NONNEG_IDS",
    "build_linear_constraints",
    "evaluate_checks",
    "load_plausibility_config",
    "subtotal_members",
]
