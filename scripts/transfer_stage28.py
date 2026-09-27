#!/usr/bin/env python
"""ステージ28 D.2: 振替パターン(ΔRE > 0 かつ Δ払込資本 < 0)の社数と割合を、除外の前後で比べる。

定義はステージ27(``analyze_re_stage27.py``)と同じ: 企業ごとの seed 中央値の年率(5 年で割る)で
ΔRE > 0.001 かつ Δ払込資本(資本金 + 資本剰余金)< −0.001。実データは推定用パネル
(ステージ23)の年率。除外後の実データは、除外リストの edinetCode の企業を除く。

    uv run python scripts/transfer_stage28.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from analyze_re_stage27 import empirical, generated

_THR = 0.001


def _row(name: str, s: pd.DataFrame) -> str:
    swap = (s["d_re"] > _THR) & (s["d_paid"] < -_THR)
    up = s["d_re"] > _THR
    # 振替パターンの社で、払込資本の減少が RE の増加の何割か(中央値)
    ratio = (-s.loc[swap, "d_paid"] / s.loc[swap, "d_re"]).median() if swap.any() else float("nan")
    return (
        f"| {name} | {len(s)} | {int(up.sum())} | {int(swap.sum())} | {swap.mean():.3f} "
        f"| {ratio:.2f} |"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exclude", type=Path, default=Path("data/interim/stage28/excluded.csv"))
    ap.add_argument("--panel-dir", type=Path, default=Path("data/interim/stage23/panel"))
    ap.add_argument("--out", type=Path, default=Path("results/stage28/transfer.md"))
    args = ap.parse_args()
    ex = pd.read_csv(args.exclude, dtype={"docID": str, "edinetCode": str})
    ex_docs, ex_codes = set(ex["docID"]), set(ex["edinetCode"])

    s27 = Path("data/interim/stage27")
    s28 = Path("data/interim/stage28")
    emp2 = empirical(args.panel_dir, 2)
    series: list[tuple[str, pd.DataFrame]] = [
        ("実データ class 2(除外前)", emp2),
        ("実データ class 2(除外後)", emp2[~emp2.index.isin(ex_codes)]),
    ]
    for v in ("r3", "r3_cap"):
        g27 = generated(s27, v)
        series += [
            (f"生成 {v}: 除外前(ステージ27、68 社)", g27),
            (f"生成 {v}: ステージ27 から除外対象の社を外しただけ", g27[~g27.index.isin(ex_docs)]),
            (f"生成 {v}: 除外後に生成し直し", generated(s28, v)),
        ]
    lines = [
        "# ステージ28 D.2: 振替パターン(ΔRE > 0 かつ Δ払込資本 < 0)", "",
        "年率(t=1 を 5 年)。閾値 ±0.001(ステージ27 と同じ)。最後の列は、振替パターンの社で"
        "払込資本の減少が RE の増加の何割か(中央値)。", "",
        "| 系列 | 社数 | ΔRE > 0 | 振替パターン | 割合 | 払込資本の減少 / RE の増加 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines += [_row(n, s) for n, s in series]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
