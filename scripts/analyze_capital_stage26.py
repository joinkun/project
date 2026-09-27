#!/usr/bin/env python
"""ステージ26 B.3: 生成された増資の規模を実データと照合し、目標選択の仕組みを確かめる。

- 生成: 版ごとに、企業ごとの (Δ資本金 + Δ資本剰余金) / A0 の seed 中央値。t=1 を 5 年と解釈し、
  年率 = 5 年の値 / 5
- 実データ: 推定用パネル(ステージ23、パネル A)の class 2 企業の 3 年間の累積 / 3(年率)
- 仕組み: 対応先の払込資本比率(資本金 + 資本剰余金)/ 総資産 − 始点の同比率

    uv run python scripts/analyze_capital_stage26.py
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from precompute_stage13_transport import load_data  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402

_CONFIG = "W1_S5_a05_b03"
_CS, _CSUR = "jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus"
_QS = (25, 50, 75, 90)


def generated_annual(directory: Path, version: str) -> pd.Series:
    files = sorted(glob.glob(str(directory / version / "nearest_na0__seed*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files])
    df = df[(df["method"] == "PCFM") & (df["config"] == _CONFIG)]
    x0 = df[f"x0_{_CS}"] + df[f"x0_{_CSUR}"]
    x1 = df[f"x1_{_CS}"] + df[f"x1_{_CSUR}"]
    # x は総資産で正規化済み(tilde)なので、差がそのまま A0 比
    per_row = pd.Series((x1 - x0).to_numpy(), index=df["docID"].to_numpy())
    result: pd.Series = per_row.groupby(level=0).median() / 5.0
    return result


def empirical_annual(panel_dir: Path) -> pd.Series:
    p = pd.read_csv(panel_dir / "panel.csv", dtype={"edinetCode": str})
    f = pd.read_csv(panel_dir / "firms.csv", dtype={"edinetCode": str}).set_index("edinetCode")
    p = p[p["edinetCode"].isin(f.index[f["panel_a"] & (f["label"] == 2)])]
    cur = p[f"cur_{_CS}"].fillna(0) + p[f"cur_{_CSUR}"].fillna(0)
    prev = p[f"prev_{_CS}"].fillna(0) + p[f"prev_{_CSUR}"].fillna(0)
    d = (cur - prev) / p["prev_total_assets"]
    n_years = p.groupby("edinetCode")["t"].nunique()
    result: pd.Series = d.groupby(p["edinetCode"]).sum() / n_years
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/interim/stage26"))
    ap.add_argument("--panel-dir", type=Path, default=Path("data/interim/stage23/panel"))
    ap.add_argument(
        "--versions",
        default="baseline,relative,median,median_estimated,median_qualitative,old_weights",
    )
    ap.add_argument("--out", type=Path, default=Path("results/stage26/capital_increase.md"))
    args = ap.parse_args()
    versions = args.versions.split(",")

    emp = empirical_annual(args.panel_dir)
    emp_p90 = float(emp.quantile(0.9))
    lines = [
        "# ステージ26 B.3: 増資の規模(年率、総資産比)",
        "",
        "生成: (Δ資本金 + Δ資本剰余金) / A0 を 5 年(t=1)で割った年率。企業ごとに seed の中央値。",
        "実データ: 推定用パネルの class 2 企業の 3 年累積 / 3。",
        "",
        "| 系列 | 社数 | p25 | p50 | p75 | p90 | 実データの p90 を超える社の割合 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def row(label: str, s: pd.Series) -> str:
        qs = " | ".join(f"{s.quantile(q / 100):.4f}" for q in _QS)
        return f"| {label} | {len(s)} | {qs} | {(s > emp_p90).mean():.3f} |"

    lines.append(row("実データ class 2", emp))
    for v in versions:
        lines.append(row(f"生成: {v}", generated_annual(args.dir, v)))

    # 仕組み: 対応先と始点の払込資本比率の差
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    x, _assets, _label, doc, _filer, _edinet = load_data(names)
    paid = pd.Series(x[:, names.index(_CS)] + x[:, names.index(_CSUR)], index=doc)
    lines += [
        "",
        "## 対応先の払込資本比率 − 始点の払込資本比率",
        "",
        "| 版 | 差の中央値 | 対応先の方が高い社の割合 |",
        "|---|---:|---:|",
    ]
    for v in versions:
        t = pd.read_csv(args.dir / v / "full_targets_nearest_na0.csv")
        gap = (
            paid.reindex(t["target_docID"]).to_numpy() - paid.reindex(t["source_docID"]).to_numpy()
        )
        lines.append(f"| {v} | {np.median(gap):+.3f} | {np.mean(gap > 0):.3f} |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
