#!/usr/bin/env python
"""ステージ21-D A.3: __OTHER__ 各科目の規模と、2023 年の重複の不一致を 3 段階で比べる。

段階: 統合前(data/interim/stage21)、21-C の統合後(integrated)、21-D の拡張要素マッピング後
(extmap)。いずれも ``verify_prior_stage21.py`` と ``build_panel_stage21.py`` の出力。

    uv run python scripts/compare_other_stages_stage21.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_STAGES: tuple[tuple[str, str], ...] = (
    ("統合前", "data/interim/stage21"),
    ("21-C", "data/interim/stage21/integrated"),
    ("21-D", "data/interim/stage21/extmap"),
)
_OTHER = "__OTHER__jppfs_cor_"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/stage21/other_three_stages.md"))
    args = parser.parse_args()

    lines = ["# __OTHER__ と 2023 年の重複の不一致(統合前 / 21-C / 21-D)", ""]
    lines += [
        "## __OTHER__ 各科目の |値| / 総資産 の中央値(fy2024 当期、非金融)",
        "",
        "| 科目 | " + " | ".join(s for s, _ in _STAGES) + " |",
        "|---|" + "---:|" * len(_STAGES),
    ]
    frames = {}
    for stage, root in _STAGES:
        df = pd.read_csv(Path(root) / "fy2024" / "values_current.csv", dtype={"docID": str})
        ok = (df["parse_status"] == "ok") & (df["is_financial"].astype(str) != "True")
        frames[stage] = df[ok & df["total_assets"].notna()].set_index("docID")
    others = [c for c in frames["統合前"].columns if c.startswith(_OTHER)]
    for col in others:
        cells = [
            f"{(f[col].fillna(0).abs() / f['total_assets'].abs()).median():.4f}"
            for f in frames.values()
        ]
        lines.append(f"| {col.removeprefix(_OTHER)} | " + " | ".join(cells) + " |")
    lines += ["", "平均(外れ値の影響を見るため):", ""]
    for col in ("__OTHER__jppfs_cor_CurrentAssets", "__OTHER__jppfs_cor_CurrentLiabilities"):
        cells = [
            f"{(f[col].fillna(0).abs() / f['total_assets'].abs()).mean():.4f}"
            for f in frames.values()
        ]
        lines.append(f"- {col.removeprefix(_OTHER)}: " + " → ".join(cells))

    mm = {
        stage: pd.read_csv(Path(root) / "overlap_mismatches.csv", dtype={"edinetCode": str})
        for stage, root in _STAGES
    }
    lines += [
        "",
        "## 2023 年の重複(fy2024 の前期 vs 2023-24 の当期)の不一致",
        "",
        "| 段階 | 不一致 | 総資産まで変わる |",
        "|---|---:|---:|",
    ]
    for stage, df in mm.items():
        lines.append(f"| {stage} | {len(df)} | {int(df['assets_changed'].sum())} |")
    pre, mid, post = (set(mm[s]["edinetCode"]) for s, _ in _STAGES)
    migrated = mid - pre  # 21-C で新たに不一致になった社(拡張要素 → 標準要素の付け替え)
    lines += [
        "",
        f"21-C で新たに不一致になった社: {len(migrated)}",
        f"  そのうち 21-D で解消: {len(migrated - post)}、残る: {len(migrated & post)}",
    ]
    rest = mm["21-D"][mm["21-D"]["edinetCode"].isin(migrated & post)]
    if not rest.empty:
        lines.append("  残る社の科目の組: " + "; ".join(rest["accounts"].value_counts().index[:5]))
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
