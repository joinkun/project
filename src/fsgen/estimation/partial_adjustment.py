"""部分調整モデルによる科目ごとの調整速度 λ_a の推定(ステージ22)。

総資産で正規化した科目 a の比率 y = x_a / A について

    y_t - y_{t-1} = λ (y*_t - y_{t-1}) + α_i + ε_t

を推定し、``adjustment_cost`` で c = (1 - λ) / λ に変換する。単年断面では調整コストと
ショックの変動性を分離できないため(Asker, Collard-Wexler & De Loecker 2014)、パネルで推定する。

入力は 1 行 = 1 社 × 1 決算年で、連続した 2 年以上を持つ。差分と前期の水準は同じ書類の
当期・前期から取る(ステージ21-D)。そのため t の ``y_prev`` と t-1 の ``y_cur`` は同じ年の
水準だが、書類が違うので一致するとは限らない。

目標 y* は 2 通り:
- ``industry``: 業種中央値 m_t(列 ``target``)。y_t = (1-λ) y_{t-1} + λ m_t + α_i + ε
- ``firm``: 企業特性への回帰(Flannery & Rangan 2006)。y_t = γ y_{t-1} + θ'X_{t-1} + α_i + ε、
  λ = 1 - γ、y* = (θ'X + α_i) / λ

推定方法は 2 通り:
- ``fe``: 固定効果(PanelOLS、企業でクラスタ)。T が短いと Nickell バイアスがある
- ``diff_gmm``: 1 階差分 + 操作変数(Arellano-Bond 型)。差分式を期ごとに縦に積み、t-2 以前の
  水準を期ごとの操作変数にする(ステージ23 で差分式 2 本以上に一般化)。AR(2) 検定には
  水準 5 期以上が必要で、水準 3〜4 期では計算できない

符号別(``direction``)は目標との乖離 y* - y_{t-1} の符号で標本を分ける。positive = 目標より下
(増やす方向)、negative = 目標より上(減らす方向)。GMM では、差分式 t を gap_t と gap_{t-1} が
同じ符号のときだけ使う(差分式 1 本なら「2 年とも同じ符号の社」と同じ)。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from linearmodels.iv import IVGMM
from linearmodels.panel import PanelOLS

Method = Literal["fe", "diff_gmm"]
Direction = Literal["both", "positive", "negative"]
Target = Literal["industry", "firm"]

FRAME_COLUMNS: tuple[str, ...] = ("edinetCode", "t", "y_prev", "y_cur", "target")

_METHODS: tuple[Method, ...] = ("fe", "diff_gmm")
_DIRECTIONS: tuple[Direction, ...] = ("both", "positive", "negative")
_TARGETS: tuple[Target, ...] = ("industry", "firm")
MIN_OBS = 30  # これ未満の標本では推定しない(status に理由を残す)
_Z = 1.959963984540054  # 標準正規の 97.5% 点


@dataclass(frozen=True)
class AdjustmentResult:
    """1 科目 × 目標 × 手法 × 方向の推定結果。推定できなかった場合は status に理由。"""

    account: str
    target: Target
    method: Method
    direction: Direction
    lam: float
    se: float
    n_firms: int
    n_obs: int
    status: str = "ok"
    j_stat: float | None = None  # Hansen の過剰識別検定(GMM、自由度 > 0 のとき)
    j_pval: float | None = None
    j_df: int | None = None
    first_stage_f: float | None = None  # 内生変数の第 1 段階の F(GMM)

    @property
    def ci_low(self) -> float:
        return self.lam - _Z * self.se

    @property
    def ci_high(self) -> float:
        return self.lam + _Z * self.se

    @property
    def in_unit_interval(self) -> bool:
        """λ ∈ (0, 1] か。外れると c = (1-λ)/λ への変換が破綻する。"""
        return math.isfinite(self.lam) and 0.0 < self.lam <= 1.0

    @property
    def cost(self) -> float | None:
        return adjustment_cost(self.lam) if self.in_unit_interval else None


def adjustment_cost(lam: float) -> float:
    """調整速度 λ から調整コストの重み c = (1 - λ) / λ に変換する。

    λ ∈ (0, 1] を前提とする(λ=1 は即時調整でコスト 0)。範囲外は推定の失敗か
    モデルの誤特定を意味するので、黙って丸めずに例外にする。
    """
    if not math.isfinite(lam) or not 0.0 < lam <= 1.0:
        raise ValueError(f"adjustment speed must be in (0, 1], got {lam!r}")
    return (1.0 - lam) / lam


def validate_frame(frame: pd.DataFrame, controls: Sequence[str] = ()) -> None:
    """必要な列があり、(edinetCode, t) が一意で、t が 2 年以上の連続した年であることを確かめる。"""
    missing = [c for c in (*FRAME_COLUMNS, *controls) if c not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing columns: {missing}")
    if frame.duplicated(subset=["edinetCode", "t"]).any():
        raise ValueError("frame has duplicated (edinetCode, t) rows")
    years = sorted(frame["t"].unique())
    if len(years) < 2 or any(b - a != 1 for a, b in zip(years, years[1:], strict=False)):
        raise ValueError(f"frame must have two or more consecutive years, got {years}")


def _failed(
    account: str, target: Target, method: Method, direction: Direction, status: str, n: int, f: int
) -> AdjustmentResult:
    return AdjustmentResult(account, target, method, direction, math.nan, math.nan, f, n, status)


def _fe_fit(dep: pd.Series, exog: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    res = PanelOLS(dep, exog, entity_effects=True).fit(cov_type="clustered", cluster_entity=True)
    return res.params, res.std_errors, res.estimated_effects.iloc[:, 0]


def _firm_gap(frame: pd.DataFrame, controls: Sequence[str]) -> pd.Series:
    """目標 #2 の乖離 y* - y_{t-1}(全標本の FE 推定から、企業効果込みの目標を作る)。"""
    panel = frame.set_index(["edinetCode", "t"])
    params, _, effects = _fe_fit(panel["y_cur"], panel[["y_prev", *controls]])
    lam = 1.0 - float(params["y_prev"])
    if not math.isfinite(lam) or lam == 0.0:
        raise ValueError(f"cannot form the firm target: lambda={lam!r}")
    fitted = panel[list(controls)].to_numpy() @ params[list(controls)].to_numpy() + effects
    target = fitted / lam
    return pd.Series(target.to_numpy() - panel["y_prev"].to_numpy(), index=panel.index)


def _gap(frame: pd.DataFrame, target: Target, controls: Sequence[str]) -> pd.Series:
    if target == "industry":
        return (frame["target"] - frame["y_prev"]).set_axis(
            pd.MultiIndex.from_frame(frame[["edinetCode", "t"]])
        )
    return _firm_gap(frame, controls)


def _select(gap: pd.Series, direction: Direction) -> pd.Series:
    if direction == "positive":
        return gap > 0
    if direction == "negative":
        return gap < 0
    return pd.Series(True, index=gap.index)


def _estimate_fe(
    frame: pd.DataFrame,
    account: str,
    target: Target,
    direction: Direction,
    controls: Sequence[str],
) -> AdjustmentResult:
    panel = frame.set_index(["edinetCode", "t"])
    keep = _select(_gap(frame, target, controls), direction).reindex(panel.index)
    sub = panel[keep.to_numpy()]
    # 企業効果を識別できるのは 2 観測以上が同じ群にいる社だけ(1 観測の社は寄与しない)
    counts = sub.groupby(level=0).size()
    sub = sub[sub.index.get_level_values(0).isin(counts[counts >= 2].index)]
    n_obs, n_firms = len(sub), sub.index.get_level_values(0).nunique()
    if n_obs < MIN_OBS:
        return _failed(account, target, "fe", direction, "insufficient_obs", n_obs, n_firms)
    if target == "industry":
        gap = (sub["target"] - sub["y_prev"]).rename("gap")
        dy = sub["y_cur"] - sub["y_prev"]
        params, se, _ = _fe_fit(dy, gap.to_frame())
        lam, lam_se = float(params["gap"]), float(se["gap"])
    else:
        params, se, _ = _fe_fit(sub["y_cur"], sub[["y_prev", *controls]])
        lam, lam_se = 1.0 - float(params["y_prev"]), float(se["y_prev"])
    return AdjustmentResult(account, target, "fe", direction, lam, lam_se, n_firms, n_obs)


def _estimate_gmm(
    frame: pd.DataFrame,
    account: str,
    target: Target,
    direction: Direction,
    controls: Sequence[str],
) -> AdjustmentResult:
    """1 階差分の式を期ごとに縦に積み、Arellano-Bond 型の操作変数で GMM 推定する。

    差分式 t(t = 2 年目以降)の操作変数は、t-2 以前の水準(各書類の前期の値)を期ごとの
    別の列に置いたもの(ブロック対角)。業種中央値の目標では、目標の変化 Δm_t も期ごとの列で
    操作変数に加える。企業特性の目標では、説明変数の変化 ΔX を外生変数、t-1 以前の書類の X を
    操作変数に加える。重み行列と標準誤差は企業でクラスタする。符号別では、差分式 t を
    gap_t と gap_{t-1} が同じ符号のときだけ使う。
    """
    years = sorted(frame["t"].unique())
    gap = _gap(frame, target, controls)
    wide = frame.set_index(["edinetCode", "t"]).assign(gap=gap).unstack("t")
    wide = wide.dropna(subset=[(c, t) for c in ("y_cur", "y_prev") for t in years])

    blocks = []
    for k in range(1, len(years)):
        t, tp = years[k], years[k - 1]
        rows = pd.DataFrame(index=wide.index)
        rows["_eq"] = t
        if target == "industry":
            dy_t = wide[("y_cur", t)] - wide[("y_prev", t)]
            dy_p = wide[("y_cur", tp)] - wide[("y_prev", tp)]
            rows["dep"] = dy_t - dy_p
            rows["endog"] = wide[("gap", t)] - wide[("gap", tp)]
            rows[f"z_dm_{t}"] = wide[("target", t)] - wide[("target", tp)]
        else:
            rows["dep"] = wide[("y_cur", t)] - wide[("y_cur", tp)]
            rows["endog"] = wide[("y_prev", t)] - wide[("y_prev", tp)]
            for c in controls:
                rows[f"d_{c}"] = wide[(c, t)] - wide[(c, tp)]
        # t-2 以前の水準: 書類 u(u <= t-1)の前期の値 = u-1 年の水準
        for u in years[:k]:
            rows[f"z_y{u - 1}_{t}"] = wide[("y_prev", u)]
            if target == "firm":
                for c in controls:
                    rows[f"z_{c}{u}_{t}"] = wide[(c, u)]
        if direction != "both":
            sign = 1.0 if direction == "positive" else -1.0
            same = (np.sign(wide[("gap", t)]) == sign) & (np.sign(wide[("gap", tp)]) == sign)
            rows = rows[same]
        blocks.append(rows)
    data = pd.concat(blocks).fillna(0.0)  # 他の期の操作変数の列は 0(ブロック対角)
    n_obs = len(data)
    n_firms = int(data.index.nunique())
    if n_obs < MIN_OBS:
        return _failed(account, target, "diff_gmm", direction, "insufficient_obs", n_obs, n_firms)

    eqs = sorted(data["_eq"].unique())
    exog = pd.DataFrame({f"eq_{t}": (data["_eq"] == t).astype(float) for t in eqs})
    if target == "firm":
        exog = pd.concat([exog, data[[f"d_{c}" for c in controls]]], axis=1)
    instruments = data[[c for c in data.columns if c.startswith("z_")]]
    # 分散 0 の操作変数(例: 業種中央値が変わらない、群に含まれない期)は落とす
    instruments = instruments.loc[:, instruments.std() > 0]
    exog = exog.loc[:, (exog.std() > 0) | exog.columns.str.startswith("eq_")]
    clusters = pd.factorize(data.index)[0]
    data = data.reset_index(drop=True)
    exog = exog.reset_index(drop=True)
    instruments = instruments.reset_index(drop=True)
    res: Any = IVGMM(
        data["dep"],
        exog,
        data[["endog"]],
        instruments,
        weight_type="clustered",
        clusters=clusters,
    ).fit(cov_type="clustered", clusters=clusters)
    coef, coef_se = float(res.params["endog"]), float(res.std_errors["endog"])
    lam = coef if target == "industry" else 1.0 - coef
    j_df = int(res.j_stat.df)
    diag = res.first_stage.diagnostics
    f_stat = float(diag.loc["endog", "f.stat"]) / instruments.shape[1]
    j_stat = float(res.j_stat.stat) if j_df > 0 else None
    j_pval = float(res.j_stat.pval) if j_df > 0 else None
    return AdjustmentResult(
        account,
        target,
        "diff_gmm",
        direction,
        lam,
        coef_se,
        n_firms,
        n_obs,
        j_stat=j_stat,
        j_pval=j_pval,
        j_df=j_df,
        first_stage_f=f_stat,
    )


def estimate_adjustment_speed(
    frame: pd.DataFrame,
    account: str,
    method: Method,
    target: Target = "industry",
    direction: Direction = "both",
    controls: Sequence[str] = (),
) -> AdjustmentResult:
    """科目 ``account`` の調整速度 λ を推定する。"""
    if method not in _METHODS:
        raise ValueError(f"unknown method: {method!r}")
    if direction not in _DIRECTIONS:
        raise ValueError(f"unknown direction: {direction!r}")
    if target not in _TARGETS:
        raise ValueError(f"unknown target: {target!r}")
    if target == "firm" and not controls:
        raise ValueError("target='firm' needs controls")
    validate_frame(frame, controls)
    cols = ["y_prev", "y_cur", "target", *controls]
    data = frame.dropna(subset=["y_prev", "y_cur", *controls]).copy()
    if target == "industry":
        data = data.dropna(subset=["target"])
    if not np.isfinite(data[cols].to_numpy(dtype=float)[:, [0, 1]]).all():
        raise ValueError("non-finite ratios in frame")
    try:
        if method == "fe":
            result = _estimate_fe(data, account, target, direction, controls)
        else:
            result = _estimate_gmm(data, account, target, direction, controls)
    except ValueError as exc:
        # 説明変数・操作変数が線形従属(例: ほぼ全社で 0 の科目、符号別の小さな群)。
        # 黙って捨てず、推定できなかった理由として status に残す。それ以外の例外は止める。
        if "full column rank" not in str(exc) and "rank" not in str(exc):
            raise
        n_firms = int(data["edinetCode"].nunique())
        return _failed(account, target, method, direction, "rank_deficient", len(data), n_firms)
    if result.status == "ok" and not (math.isfinite(result.lam) and math.isfinite(result.se)):
        raise ValueError(f"non-finite estimate for {account}: {result}")
    return result
