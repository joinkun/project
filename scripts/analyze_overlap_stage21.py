#!/usr/bin/env python
"""ステージ21-C C: 2023 年決算の重複(2 書類から取得)の不一致を分類し、採用方針を比べる。

``build_panel_stage21.py`` の ``overlap_mismatches.csv``(統合後)と、
``verify_prior_stage21.py`` の集約値を使う。

分類:
- 総資産まで変わる(遡及修正・修正再表示など)
- 総資産は不変で、古い書類に会社独自の拡張要素があり新しい書類で標準要素に替わった
  (タクソノミ改訂による付け替え)
- 総資産は不変のその他(表示の組替え)

採用方針の比較: 2024 年の差分 Δx = x_2024 - x_2023 を、
(a) 同じ書類(fy2024 の当期 - 前期)、(b) 書類をまたぐ(fy2024 の当期 - 2023-24 の当期)
で取ったときの差 |Δa - Δb| / 総資産 を、科目ごとに集計する。

    uv run python scripts/analyze_overlap_stage21.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_P = "jppfs_cor_"


def _values(interim: Path, set_name: str, period: str) -> pd.DataFrame:
    df = pd.read_csv(
        interim / set_name / f"values_{period}.csv", dtype={"docID": str, "edinetCode": str}
    )
    return df.set_index("docID")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim", type=Path, default=Path("data/interim/stage21/integrated"))
    parser.add_argument(
        "--old-leaves",
        type=Path,
        default=Path("data/interim/stage21/leaks/2023_2024_current/leaves.csv"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/stage21/integrated/overlap_analysis.md")
    )
    args = parser.parse_args()

    mm = pd.read_csv(
        args.interim / "overlap_mismatches.csv",
        dtype={"edinetCode": str, "doc_newer": str, "doc_older": str},
    )
    panel = pd.read_csv(args.interim / "panel_long.csv", dtype={"edinetCode": str})
    names = [c for c in panel.columns[7:]]

    old_leaves = pd.read_csv(args.old_leaves, dtype={"docID": str})
    ext_docs = set(
        old_leaves.loc[
            ~old_leaves["element"].str.startswith(_P)
            & old_leaves["bucket"].str.startswith("__OTHER__"),
            "docID",
        ]
    )
    changed = mm["assets_changed"]
    migrated = (
        ~changed
        & mm["doc_older"].isin(ext_docs)
        & mm["accounts"].str.contains("receivables_trade|payables_trade|inventories")
    )
    n_pairs = 3997  # build_panel_stage21.py の「重複する決算年の観測」
    lines = [
        "# ステージ21-C C: 2023 年決算の重複の不一致(統合後)",
        "",
        f"- 不一致 {len(mm)} 社(重複 {n_pairs} 社中)",
        f"  - 総資産まで変わる: {int(changed.sum())}",
        f"  - 総資産不変、拡張要素 → 標準要素の付け替えの可能性: {int(migrated.sum())}",
        f"  - 総資産不変、その他の表示の組替え: {int((~changed & ~migrated).sum())}",
        "",
        "総資産まで変わる社の、最大のずれ(総資産比)の分布:",
        f"  中央値 {mm.loc[changed, 'max_rel_diff'].median():.4f}、"
        f"p90 {mm.loc[changed, 'max_rel_diff'].quantile(0.9):.4f}、"
        f"最大 {mm.loc[changed, 'max_rel_diff'].max():.4f}",
    ]

    # 採用方針の比較: 2024 年の差分
    new_cur = _values(args.interim, "fy2024", "current")
    new_pri = _values(args.interim, "fy2024", "prior")
    old_cur = _values(args.interim, "2023_2024", "current")
    pairs = mm[["doc_newer", "doc_older"]]
    assets = new_cur.loc[pairs["doc_newer"], "total_assets"].abs().to_numpy()
    a = (
        new_cur.loc[pairs["doc_newer"], names].fillna(0).to_numpy()
        - new_pri.loc[pairs["doc_newer"], names].fillna(0).to_numpy()
    )
    b = (
        new_cur.loc[pairs["doc_newer"], names].fillna(0).to_numpy()
        - old_cur.loc[pairs["doc_older"], names].fillna(0).to_numpy()
    )
    gap = pd.DataFrame(abs(a - b) / assets[:, None], columns=names)
    delta_scale = pd.DataFrame(abs(a) / assets[:, None], columns=names)
    rows = []
    for n in names:
        nz = gap[n] > 0
        if not nz.any():
            continue
        rows.append(
            (
                n,
                int(nz.sum()),
                gap.loc[nz, n].median(),
                gap.loc[nz, n].max(),
                delta_scale.loc[nz, n].median(),
            )
        )
    rows.sort(key=lambda r: -r[1])
    lines += [
        "",
        "## 差分の取り方による違い(不一致の社について、2024 年の Δx)",
        "",
        "(a) 同じ書類: fy2024 の当期 − fy2024 の前期(前期は当期の表示に組み替え済み)",
        "(b) 書類をまたぐ: fy2024 の当期 − 2023-24 提出の当期",
        "",
        "| 科目 | 違いが出る社 | |Δa − Δb| / 総資産 中央値 | 最大 | 参考: |Δa| / 総資産 中央値 |",
        "|---|---:|---:|---:|---:|",
    ]
    for n, k, med, mx, ds in rows:
        lines.append(f"| {n} | {k} | {med:.4f} | {mx:.4f} | {ds:.4f} |")
    lines += [
        "",
        f"違いが出る社(いずれかの科目): {int((gap > 0).any(axis=1).sum())} / {len(mm)}",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
