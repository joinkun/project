#!/usr/bin/env python
"""ステージ38 B.4: 実データで 2 つの遷移(T1: class 2 → class 1、T2: class 1 → class 0)の変化を特徴づける。

生成の前に実行する。ラベルはステージ30 の年度別ラベル(``data/interim/stage30/firm_years.csv``)。
変化はステージ23 の推定用パネルの同一書類内の差分 / 前期総資産。

- 1 年の遷移(主): 年度 y のラベルと y+1 のラベルで遷移を判定し、y+1 年度の書類の前期末 → 当期末の
  変化を使う(y = 2022, 2023)。同じ企業が 2 回入りうる
- 2 年の遷移(参考): 2022 → 2024 のラベル。2023・2024 年度の書類の変化の和
- 対照: 同じ始点のまま留まった組(class 2 → 2、class 1 → 1)
- 特殊業種(ステージ28 の規則)を除く

出力: ``results/stage38/real_transitions.md``、``data/interim/stage38/real_deltas.parquet``
(組ごとの 32 科目の変化。生成との比較に使う)。

    uv run python scripts/real_transitions_stage38.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from evaluate_stage34 import special_reason

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import subtotal_members

GROUPS = (("T1: class 2 → 1", 2, 1), ("T2: class 1 → 0", 1, 0),
          ("対照: class 2 → 2", 2, 2), ("対照: class 1 → 1", 1, 1))
_THR = 1e-4


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel-dir", type=Path, default=Path("data/interim/stage23/panel"))
    ap.add_argument("--out", type=Path, default=Path("results/stage38"))
    args = ap.parse_args()
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    fy = pd.read_csv("data/interim/stage30/firm_years.csv", dtype={"edinetCode": str})
    lab = fy.set_index(["edinetCode", "year"])["label"]
    p = pd.read_csv(args.panel_dir / "panel.csv", dtype={"edinetCode": str})
    p = p[p["prev_total_assets"] > 0].copy()
    p["label_prev"] = [lab.get((c, t - 1), np.nan) for c, t in zip(p["edinetCode"], p["t"],
                                                                   strict=True)]
    p["label_cur"] = [lab.get((c, t), np.nan) for c, t in zip(p["edinetCode"], p["t"],
                                                             strict=True)]
    p = p[p["t"].isin([2023, 2024])]
    cand = p[p["label_prev"].isin([1, 2])]["edinetCode"].unique().tolist()
    reason = special_reason(fy, cand)
    special = set(reason.index[reason != ""])
    p = p[~p["edinetCode"].isin(special)]
    d = pd.DataFrame({n: (p[f"cur_{n}"].fillna(0) - p[f"prev_{n}"].fillna(0))
                      / p["prev_total_assets"] for n in names})
    xc = p[[f"cur_{n}" for n in names]].fillna(0).to_numpy()
    xp = p[[f"prev_{n}" for n in names]].fillna(0).to_numpy()
    na = members["jppfs_cor_NetAssets"]
    d["d_na"] = (xc @ na - xp @ na) / p["prev_total_assets"].to_numpy()
    d["edinetCode"] = p["edinetCode"].to_numpy()
    d["t"] = p["t"].to_numpy()
    d["label_prev"] = p["label_prev"].to_numpy()
    d["label_cur"] = p["label_cur"].to_numpy()
    Path("data/interim/stage38").mkdir(parents=True, exist_ok=True)
    d.to_parquet("data/interim/stage38/real_deltas.parquet", index=False)

    lines = ["# ステージ38 B.4: 実データでの 2 段階の特徴づけ(生成の前に確認)", "",
             f"1 年の遷移(年度 y → y+1、y = 2022・2023)。特殊業種 {len(special)} 社を除く。"
             "変化は総資産比(前期末)。", "",
             "| 遷移 | 組数(社数) | Δ純資産 中央値 | 純資産が増えた組 |", "|---|---:|---:|---:|"]
    sub: dict[str, pd.DataFrame] = {}
    for name, a, b in GROUPS:
        s = d[(d["label_prev"] == a) & (d["label_cur"] == b)]
        sub[name] = s
        lines.append(f"| {name} | {len(s)}({s['edinetCode'].nunique()}) "
                     f"| {s['d_na'].median():+.4f} | {(s['d_na'] > _THR).mean():.0%} |")
    for name, _a, _b in GROUPS[:2]:
        s = sub[name]
        med = s[list(names)].median()
        absmed = s[list(names)].abs().median().sort_values(ascending=False)
        dec = (s[list(names)] < -_THR).mean()
        inc = (s[list(names)] > _THR).mean()
        lines += ["", f"## {name}", "",
                  "動きの大きい科目 上位 5(|Δx|/A の中央値): " + "、".join(
                      f"{ja[k]} {absmed[k]:.4f}" for k in absmed.index[:5]), "",
                  "| 科目 | Δx/A 中央値 | 減った組 | 増えた組 |", "|---|---:|---:|---:|"]
        for k in names:
            if max(dec[k], inc[k]) >= 0.3:
                lines.append(f"| {ja[k]} | {med[k]:+.4f} | {dec[k]:.0%} | {inc[k]:.0%} |")
        lines.append("(減った組か増えた組が 30% 以上の科目のみ)")
    # 仮説の検証用の要約
    keys = [("jppfs_cor_PropertyPlantAndEquipment", "有形固定資産"),
            ("jppfs_cor_InvestmentsAndOtherAssets", "投資その他の資産"),
            ("jppfs_cor_CashAndDeposits", "現金預金"), ("short_term_loans", "短期借入金"),
            ("long_term_loans", "長期借入金"), ("jppfs_cor_RetainedEarnings", "利益剰余金"),
            ("jppfs_cor_CapitalStock", "資本金"), ("jppfs_cor_CapitalSurplus", "資本剰余金")]
    lines += ["", "## 仮説(T1 = 資産売却と資金繰り、T2 = 利益蓄積と設備回復)の確認", "",
              "| 科目 | " + " | ".join(f"{n} 中央値 / 減 / 増" for n, _a, _b in GROUPS) + " |",
              "|---|" + "---|" * len(GROUPS)]
    for k, lab_ in keys:
        cells = []
        for name, _a, _b in GROUPS:
            s = sub[name][k]
            cells.append(f"{s.median():+.4f} / {(s < -_THR).mean():.0%} / {(s > _THR).mean():.0%}")
        lines.append(f"| {lab_} | " + " | ".join(cells) + " |")
    (args.out / "real_transitions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
