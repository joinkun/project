#!/usr/bin/env python
"""ステージ22 C: 科目ごとの調整速度 λ を推定する(lightgbm/sklearn/scipy 系のプロセス)。

入力は ``build_panel_stage22.py`` のパネル(同一書類の当期・前期、t = 2023, 2024)。
対象は __OTHER__ を除く基礎科目。比率 y = x / 総資産 は、科目ごとに全社・全期を合わせた
1〜99% 点で winsorize する(資本剰余金が総資産の数十倍になる社などの外れ値のため)。

目標 y*:
- industry: 33 業種 × 年の中央値(パネル内の全社)
- firm: 企業特性 X_{t-1} への回帰(Flannery & Rangan 2006)。X は同じ書類の前期から作る:
  規模 log(総資産)、有形固定資産比率、純資産比率、業種中央値(前期)。損益計算書を
  取得していないため、収益性・成長率は入れない。自分自身と恒等的に一致する変数は除く
  (有形固定資産の推定では有形固定資産比率を除く)

出力: ``results/stage22/estimates.csv``(パネル × 科目 × 目標 × 手法 × 方向)

    uv run python scripts/run_adjustment_stage22.py
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from fsgen.estimation.partial_adjustment import (
    AdjustmentResult,
    Direction,
    Method,
    Target,
    estimate_adjustment_speed,
)

_WINSOR = (0.01, 0.99)
_PPE = "jppfs_cor_PropertyPlantAndEquipment"
_METHODS: tuple[Method, ...] = ("fe", "diff_gmm")
_TARGETS: tuple[Target, ...] = ("industry", "firm")
_DIRECTIONS: tuple[Direction, ...] = ("both", "positive", "negative")


def account_frame(
    panel: pd.DataFrame, firms: pd.DataFrame, account: str, net_asset_names: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """1 科目の推定用フレーム(edinetCode, t, y_prev, y_cur, target, 説明変数)。"""
    prev_a = panel["prev_total_assets"].astype(float)
    cur_a = panel["cur_total_assets"].astype(float)
    df = pd.DataFrame(
        {
            "edinetCode": panel["edinetCode"],
            "t": panel["t"].astype(int),
            "industry33": panel["edinetCode"].map(firms["industry33"]),
            "y_prev": panel[f"prev_{account}"].fillna(0) / prev_a,
            "y_cur": panel[f"cur_{account}"].fillna(0) / cur_a,
            "size": np.log(prev_a),
            "tangibility": panel[f"prev_{_PPE}"].fillna(0) / prev_a,
            "equity_ratio": panel[[f"prev_{n}" for n in net_asset_names]].fillna(0).sum(axis=1)
            / prev_a,
        }
    )
    df = df[(prev_a > 0).to_numpy() & (cur_a > 0).to_numpy()]
    y = pd.concat([df["y_prev"], df["y_cur"]])
    lo, hi = y.quantile(_WINSOR[0]), y.quantile(_WINSOR[1])
    for c in ("y_prev", "y_cur"):
        df[c] = df[c].clip(lo, hi)
    for c in ("tangibility", "equity_ratio"):
        df[c] = df[c].clip(df[c].quantile(_WINSOR[0]), df[c].quantile(_WINSOR[1]))
    df["target"] = df.groupby(["industry33", "t"])["y_cur"].transform("median")
    df["ind_median_prev"] = df.groupby(["industry33", "t"])["y_prev"].transform("median")
    controls = ["size", "tangibility", "equity_ratio", "ind_median_prev"]
    if account == _PPE:
        controls.remove("tangibility")
    return df, controls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim", type=Path, default=Path("data/interim/stage22"))
    parser.add_argument("--canonical", type=Path, default=Path("conf/accounts/core.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results/stage22/estimates.csv"))
    args = parser.parse_args()

    basis = yaml.safe_load(args.canonical.read_text())["basis"]
    accounts = [b["name"] for b in basis if not b["name"].startswith("__OTHER__")]
    net_asset_names = [b["name"] for b in basis if b["side"] == "net_assets"]
    panel_all = pd.read_csv(args.interim / "panel" / "panel.csv", dtype={"edinetCode": str})
    firms = pd.read_csv(args.interim / "panel" / "firms.csv", dtype={"edinetCode": str})
    firms = firms.set_index("edinetCode")

    rows: list[dict[str, object]] = []
    for panel_name, flag in (("A", "panel_a"), ("B", "panel_b")):
        codes = set(firms.index[firms[flag]])
        panel = panel_all[panel_all["edinetCode"].isin(codes)]
        for account in tqdm(accounts, desc=f"panel {panel_name}"):
            frame, controls = account_frame(panel, firms, account, net_asset_names)
            for target in _TARGETS:
                for method in _METHODS:
                    for direction in _DIRECTIONS:
                        res: AdjustmentResult = estimate_adjustment_speed(
                            frame,
                            account,
                            method,
                            target,
                            direction,
                            controls if target == "firm" else (),
                        )
                        row: dict[str, object] = {"panel": panel_name}
                        row |= dataclasses.asdict(res)
                        row |= {
                            "ci_low": res.ci_low,
                            "ci_high": res.ci_high,
                            "in_unit_interval": res.in_unit_interval,
                            "cost": res.cost if res.cost is not None else math.nan,
                            "controls": ";".join(controls) if target == "firm" else "",
                        }
                        rows.append(row)
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.groupby(["panel", "status"]).size())


if __name__ == "__main__":
    main()
