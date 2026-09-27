#!/usr/bin/env python
"""ステージ23.5 B.3: 分母と推定手法を先行研究に揃えた再推定(lightgbm/sklearn/scipy 系のプロセス)。

対象は運転資本 4 科目(売上債権・仕入債務・棚卸資産・現金)。分母は総資産(本研究の既定)・
売上高(回転日数に相当。×365 は λ に影響しない)・売上原価(仕入債務・棚卸資産のみ、DPO・DIO
に相当)。目標は業種中央値、手法は固定効果・差分 GMM・システム GMM。

分母の違いだけを比べるため、どの分母も同じ部分標本(パネル A のうち、すべての年の書類で
当期・前期の売上高と売上原価が正の値で取れた社)で推定する。比率は科目・分母ごとに全社・全期を
合わせた 1〜99% 点で winsorize する。あわせて、パネル A の全社・総資産比でシステム GMM を
全 23 科目について推定する(手法の違いを見るため)。

    uv run python scripts/run_denominator_stage23_5.py
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run_adjustment_stage22 import account_frame  # noqa: E402

from fsgen.estimation.partial_adjustment import (  # noqa: E402
    Method,
    estimate_adjustment_speed,
)

_WINSOR = (0.01, 0.99)
_METHODS: tuple[Method, ...] = ("fe", "diff_gmm", "sys_gmm")
_P = "jppfs_cor_"
# (科目, 分母の一覧)
TARGETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("receivables_trade", ("assets", "sales")),
    ("payables_trade", ("assets", "sales", "cogs")),
    ("inventories", ("assets", "sales", "cogs")),
    (f"{_P}CashAndDeposits", ("assets", "sales")),
)
SETS = ("fy2024", "2023_2024", "2022_2023")


def load_sales(sales_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(sales_dir / f"sales_{s}.csv", dtype={"docID": str}) for s in SETS]
    return pd.concat(frames, ignore_index=True)


def frame_for(panel: pd.DataFrame, industry: pd.Series, account: str, denom: str) -> pd.DataFrame:
    col = {"assets": "total_assets", "sales": "sales", "cogs": "cogs"}[denom]
    prev_d = panel[f"prev_{col}"].astype(float)
    cur_d = panel[f"cur_{col}"].astype(float)
    df = pd.DataFrame(
        {
            "edinetCode": panel["edinetCode"],
            "t": panel["t"].astype(int),
            "industry33": panel["edinetCode"].map(industry),
            "y_prev": panel[f"prev_{account}"].fillna(0) / prev_d,
            "y_cur": panel[f"cur_{account}"].fillna(0) / cur_d,
        }
    )
    y = pd.concat([df["y_prev"], df["y_cur"]])
    lo, hi = y.quantile(_WINSOR[0]), y.quantile(_WINSOR[1])
    df["y_prev"] = df["y_prev"].clip(lo, hi)
    df["y_cur"] = df["y_cur"].clip(lo, hi)
    df["target"] = df.groupby(["industry33", "t"])["y_cur"].transform("median")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim", type=Path, default=Path("data/interim/stage23"))
    parser.add_argument("--sales-dir", type=Path, default=Path("data/interim/stage23_5"))
    parser.add_argument("--canonical", type=Path, default=Path("conf/accounts/core.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/stage23_5"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(args.interim / "panel" / "panel.csv", dtype={"edinetCode": str})
    firms = pd.read_csv(args.interim / "panel" / "firms.csv", dtype={"edinetCode": str})
    firms = firms.set_index("edinetCode")
    panel = panel[panel["edinetCode"].isin(firms.index[firms["panel_a"]])]
    sales = load_sales(args.sales_dir).rename(
        columns={
            "sales_cur": "cur_sales",
            "sales_prev": "prev_sales",
            "cogs_cur": "cur_cogs",
            "cogs_prev": "prev_cogs",
        }
    )
    merged = panel.merge(sales, on="docID", how="left")
    ok = (merged[["cur_sales", "prev_sales", "cur_cogs", "prev_cogs"]] > 0).all(axis=1)
    good = merged.assign(ok=ok).groupby("edinetCode")["ok"].all()
    n_years = merged.groupby("edinetCode")["t"].nunique()
    sub_codes = good.index[good & (n_years == merged["t"].nunique())]
    sub = merged[merged["edinetCode"].isin(sub_codes)]
    source_same = (sub["source_cur"] == sub["source_prev"]).mean()
    print(
        f"パネル A {panel['edinetCode'].nunique()} 社のうち、"
        f"売上高・売上原価がすべての年で取れた社: {len(sub_codes)}"
        f"(当期と前期で売上高の要素が同じ割合 {source_same:.3f})"
    )

    rows: list[dict[str, object]] = []
    industry = firms["industry33"]
    for account, denoms in tqdm(TARGETS, desc="denominators"):
        for denom in denoms:
            frame = frame_for(sub, industry, account, denom)
            for method in _METHODS:
                for direction in ("both", "positive", "negative"):
                    res = estimate_adjustment_speed(
                        frame,
                        account,
                        method,
                        "industry",
                        direction,
                    )
                    row: dict[str, object] = {"sample": "subsample", "denominator": denom}
                    row |= dataclasses.asdict(res)
                    row |= {"ci_low": res.ci_low, "ci_high": res.ci_high}
                    rows.append(row)

    # 手法の比較: パネル A 全社・総資産比で、全 23 科目のシステム GMM
    basis = yaml.safe_load(args.canonical.read_text())["basis"]
    accounts = [b["name"] for b in basis if not b["name"].startswith("__OTHER__")]
    net_asset_names = [b["name"] for b in basis if b["side"] == "net_assets"]
    for account in tqdm(accounts, desc="sys_gmm all accounts"):
        frame, _ = account_frame(panel, firms, account, net_asset_names)
        res = estimate_adjustment_speed(frame, account, "sys_gmm", "industry", "both")
        row = {"sample": "panel_a", "denominator": "assets"}
        row |= dataclasses.asdict(res)
        row |= {"ci_low": res.ci_low, "ci_high": res.ci_high}
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_dir / "estimates.csv", index=False)
    print(out.groupby(["sample", "status"]).size())
    meta = pd.DataFrame(
        [
            {
                "panel_a_firms": panel["edinetCode"].nunique(),
                "subsample_firms": len(sub_codes),
                "sales_source_same_share": source_same,
            }
        ]
    )
    meta.to_csv(args.out_dir / "sample.csv", index=False)


if __name__ == "__main__":
    main()
