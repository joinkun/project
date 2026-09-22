#!/usr/bin/env python
"""core.yaml/extended.yaml の各基礎科目の `sign` を実測に基づいて確定する
(ステージ7 §2.1)。

判定規則(全社(非金融)の実測値、`data/interim/values/aggregated_core.csv`):

    負が0社            -> nonneg
    正が0社            -> nonpos
    両方ある           -> free

ただし以下は上記の規則を上書きする(手動判断、理由を明記):

- `__OTHER__*`(残余科目、全て): 差額定義のため負になりうる -> free
- `jppfs_cor_RetainedEarnings`(利益剰余金): 欠損企業で負になるのは
  正当な会計事象 -> free
- `non_controlling_interests`(非支配株主持分): 実測で14社だけ負だが、
  子会社の欠損累積により非支配株主持分相当額が負になるのは実在する
  正当な会計事象(データ誤りではない) -> free
- `jppfs_cor_PropertyPlantAndEquipment`(有形固定資産): 実測で3社だけ
  負(いずれもちょうど-1,000円)。物理的な資産が負になることは会計上
  ありえず、端数処理由来の外れ値と判断 -> nonneg(3社は本来0円のはず)

決定理由は `sign_reason` フィールドとして各基礎科目に書き込む
(YAMLコメントは再ダンプ時に失われるため、実フィールドとして残す)。

使い方:
    uv run python scripts/finalize_signs.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

_DEFAULT_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_DEFAULT_TARGETS = (Path("conf/accounts/core.yaml"),)
# extended.yaml は core.yaml と異なる基礎科目集合を持ち、
# aggregated_core.csv (core.yaml で生成)には無い列を参照するため、
# 対応する aggregated 値データが別途必要。本ステージでは core.yaml のみ
# 対象とする(--targets で明示すれば extended.yaml も指定できる)。

_ALWAYS_FREE_EXACT = {"jppfs_cor_RetainedEarnings"}
_ALWAYS_FREE_PREFIX = "__OTHER__"

_MANUAL_OVERRIDES: dict[str, tuple[str, str]] = {
    "non_controlling_interests": (
        "free",
        "実測1751社中14社が負。子会社の欠損累積により非支配株主持分相当額が"
        "負になるのは正当な会計事象であり、データ誤りではないためfreeとする。",
    ),
    "jppfs_cor_PropertyPlantAndEquipment": (
        "nonneg",
        "実測4102社中3社のみ負(いずれも-1,000円ちょうど)。物理的な資産が"
        "負になることは会計上ありえず、端数処理由来の外れ値と判断しnonnegとする。",
    ),
}


def summarize(series: pd.Series) -> tuple[int, int, int]:
    n_pos = int((series > 0).sum())
    n_zero = int((series == 0).sum())
    n_neg = int((series < 0).sum())
    return n_pos, n_zero, n_neg


def decide_sign(name: str, n_pos: int, n_neg: int) -> tuple[str, str]:
    if name in _MANUAL_OVERRIDES:
        return _MANUAL_OVERRIDES[name]
    if name in _ALWAYS_FREE_EXACT:
        return "free", "欠損企業で負になるのは正当な会計事象のためfree(手動指定)。"
    if name.startswith(_ALWAYS_FREE_PREFIX):
        return "free", "残余科目(差額定義)のため負になりうる(手動指定)。"
    n_total = n_pos + n_neg
    if n_neg == 0:
        return "nonneg", f"実測{n_total}社(正のみ)中、負が0社。"
    if n_pos == 0:
        return "nonpos", f"実測{n_total}社(負のみ)中、正が0社。"
    return "free", f"実測{n_total}社中、正{n_pos}社・負{n_neg}社(両方あり)。"


def finalize(
    canonical_path: Path, aggregated_csv: Path
) -> list[tuple[str, str, int, int, int, str]]:
    with canonical_path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    agg = pd.read_csv(aggregated_csv, dtype={"docID": str})
    is_financial = agg["is_financial"].astype(str)
    agg = agg[(agg["parse_status"] == "ok") & (is_financial != "True")]

    report: list[tuple[str, str, int, int, int, str]] = []
    new_basis = []
    for entry in doc["basis"]:
        name = entry["name"]
        if name not in agg.columns:
            raise ValueError(f"{name} not found in {aggregated_csv} columns")
        series = pd.to_numeric(agg[name], errors="coerce").dropna()
        n_pos, _n_zero, n_neg = summarize(series)
        sign, reason = decide_sign(name, n_pos, n_neg)
        report.append((name, sign, n_pos, n_neg, len(series), reason))
        new_entry = {
            "name": entry["name"],
            "members": entry["members"],
            "parent_subtotal": entry["parent_subtotal"],
            "side": entry["side"],
            "sign": sign,
            "sign_reason": reason,
            "ratio": entry["ratio"],
            "always_include": entry.get("always_include", False),
            "include_reason": entry.get("include_reason"),
        }
        new_basis.append(new_entry)

    doc["basis"] = new_basis
    with canonical_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregated-csv", type=Path, default=_DEFAULT_AGGREGATED_CSV)
    parser.add_argument("--targets", type=Path, nargs="+", default=list(_DEFAULT_TARGETS))
    args = parser.parse_args()

    for target in args.targets:
        print(f"=== {target} ===")
        report = finalize(target, args.aggregated_csv)
        for name, sign, n_pos, n_neg, n_total, reason in report:
            print(f"  {name:<55} {sign:<8} (正{n_pos}/負{n_neg}/計{n_total}) {reason}")


if __name__ == "__main__":
    main()
