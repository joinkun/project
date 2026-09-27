#!/usr/bin/env python
"""ステージ27 B.3: 利益剰余金の変化(ΔRE / A0)の規模を実データと照合する。

- 生成: 版ごとに、企業ごとの ΔRE / A0 の seed 中央値。t=1 を 5 年と解釈し、年率 = 5 年の値 / 5
- 実データ: 推定用パネル(ステージ23、パネル A、3 期の差分)の class 2・class 0 企業の、
  同一書類内の ΔRE / 前期総資産 を 3 年分足して 3 で割った年率
- 払込資本との振替: 実データでも生成でも、ΔRE と Δ払込資本(資本金 + 資本剰余金)が逆向きに
  動く社の割合(欠損填補の減資のような振替を含むかの目安)

    uv run python scripts/analyze_re_stage27.py
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

_CONFIG = "W1_S5_a05_b03"  # 既定の設定名。ディレクトリに PCFM の設定が 1 つだけならそれを使う


def select_pcfm(df: pd.DataFrame) -> pd.DataFrame:
    """PCFM の行を返す。設定が 1 つだけならそれを、複数あれば _CONFIG を使う(ステージ27)。"""
    p = df[df["method"] == "PCFM"]
    configs = sorted(p["config"].unique())
    if len(configs) > 1:
        p = p[p["config"] == _CONFIG]
    if p.empty:
        raise ValueError(f"PCFM の行がない(configs={configs})")
    return p
_RE = "jppfs_cor_RetainedEarnings"
_CS, _CSUR = "jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus"
_QS = (5, 25, 50, 75, 95)


def generated(directory: Path, version: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(directory / version / "nearest_na0__seed*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files])
    df = select_pcfm(df)
    out = pd.DataFrame(
        {
            "docID": df["docID"].to_numpy(),
            "d_re": (df[f"x1_{_RE}"] - df[f"x0_{_RE}"]).to_numpy(),
            "d_paid": (
                df[f"x1_{_CS}"] + df[f"x1_{_CSUR}"] - df[f"x0_{_CS}"] - df[f"x0_{_CSUR}"]
            ).to_numpy(),
        }
    )
    med = out.groupby("docID").median() / 5.0  # 年率(t=1 を 5 年)
    return med


def empirical(panel_dir: Path, label: int) -> pd.DataFrame:
    p = pd.read_csv(panel_dir / "panel.csv", dtype={"edinetCode": str})
    f = pd.read_csv(panel_dir / "firms.csv", dtype={"edinetCode": str}).set_index("edinetCode")
    p = p[p["edinetCode"].isin(f.index[f["panel_a"] & (f["label"] == label)])]
    a = p["prev_total_assets"]
    d_re = (p[f"cur_{_RE}"].fillna(0) - p[f"prev_{_RE}"].fillna(0)) / a
    paid = [f"{t}_{c}" for c in (_CS, _CSUR) for t in ("cur", "prev")]
    pp = p[paid].fillna(0)
    d_paid = (pp[f"cur_{_CS}"] + pp[f"cur_{_CSUR}"] - pp[f"prev_{_CS}"] - pp[f"prev_{_CSUR}"]) / a
    n = p.groupby("edinetCode")["t"].nunique()
    out = pd.DataFrame(
        {
            "d_re": d_re.groupby(p["edinetCode"]).sum() / n,
            "d_paid": d_paid.groupby(p["edinetCode"]).sum() / n,
        }
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/interim/stage27"))
    ap.add_argument("--panel-dir", type=Path, default=Path("data/interim/stage23/panel"))
    ap.add_argument("--versions", default="r1,r2,r3,r3_median")
    ap.add_argument("--out", type=Path, default=Path("results/stage27/re_change.md"))
    args = ap.parse_args()

    series = {
        "実データ class 2(GC 注記)": empirical(args.panel_dir, 2),
        "実データ class 0(健全)": empirical(args.panel_dir, 0),
    }
    for v in args.versions.split(","):
        series[f"生成: {v}"] = generated(args.dir, v)
    emp2 = series["実データ class 2(GC 注記)"]["d_re"]
    lines = [
        "# ステージ27 B.3: 利益剰余金の変化の規模(年率、総資産比)",
        "",
        "生成: ΔRE / A0 を 5 年(t=1)で割った年率(企業ごとに seed の中央値)。",
        "実データ: 推定用パネル(3 期の差分)の同一書類内の ΔRE / 前期総資産 を"
        " 3 年分足して 3 で割った年率。",
        "",
        "| 系列 | 社数 | "
        + " | ".join(f"p{q}" for q in _QS)
        + " | 年率 30% 超 | class 2 の p95 超 | ΔRE>0 かつ Δ払込資本<0 |",
        "|---|---:|" + "---:|" * (len(_QS) + 3),
    ]
    p95 = float(emp2.quantile(0.95))
    for name, s in series.items():
        d = s["d_re"]
        qs = " | ".join(f"{d.quantile(q / 100):+.4f}" for q in _QS)
        swap = ((s["d_re"] > 0.001) & (s["d_paid"] < -0.001)).mean()
        lines.append(
            f"| {name} | {len(d)} | {qs} | {(d > 0.30).mean():.3f} | {(d > p95).mean():.3f} "
            f"| {swap:.3f} |"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
