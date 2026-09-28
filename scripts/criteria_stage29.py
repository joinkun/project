#!/usr/bin/env python
"""ステージ29 A: 実データ(推定用パネル)の利益剰余金の変化を測り、採否の基準を機械的に決める。

生成の前に実行し、``results/stage29/criteria.md``・``criteria.json`` を書く(生成結果は読まない)。

実データ: ステージ23 の推定用パネル(パネル A、4 期 = 3 年分の同一書類内の差分)の class 2
(GC 注記)企業から、ステージ28 の除外リストの企業を除いたもの。企業ごとに 3 年分の
ΔX / 前期総資産 を足して 3 で割った年率を使う(ステージ27・28 と同じ)。

分解(R4 と同じ規則): 払込資本(資本金 + 資本剰余金)が減った社は、その減少分を振替
T = −Δ払込資本 とみなし、利益部分 P = ΔRE − T。払込資本が減っていない社は T = 0、P = ΔRE。

基準(閾値はすべて下の数値から機械的に決まる):
1. 13 項目の違反 0(外れ値 2 社を除く 55 社)
2. 利益剰余金が増える社の割合が、実データの割合 ± 10 ポイント以内
3. 利益部分の上限 γ = 5 × (実データの P の年率の p90)。負なら 0
4. 振替のある社の振替 T / A0(5 年)の中央値と p95 が、実データの振替のある社の
   T の年率 × 5 の p5〜p95 に入る

    uv run python scripts/criteria_stage29.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

_RE = "jppfs_cor_RetainedEarnings"
_CS, _CSUR = "jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus"
THR = 0.001  # 年率で「増えた」「減った」とみなす閾値(ステージ27・28 と同じ)
YEARS = 5.0  # 生成の t=1 を 5 年と解釈
_QS = (5, 25, 50, 75, 90, 95)


def decompose(d_re: pd.Series, d_paid: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(振替 T, 利益部分 P)。払込資本が減った分を振替とみなす。"""
    transfer = (-d_paid).clip(lower=0.0)
    return transfer, d_re - transfer


def firm_level(panel_dir: Path, label: int, excluded_codes: set[str]) -> pd.DataFrame:
    """企業ごとの年率(3 年分の同一書類内の差分 / 前期総資産 の和 / 年数)と、3 年の累計。"""
    p = pd.read_csv(panel_dir / "panel.csv", dtype={"edinetCode": str})
    f = pd.read_csv(panel_dir / "firms.csv", dtype={"edinetCode": str}).set_index("edinetCode")
    keep = f.index[f["panel_a"] & (f["label"] == label)]
    p = p[p["edinetCode"].isin(keep) & ~p["edinetCode"].isin(excluded_codes)]
    a = p["prev_total_assets"]
    d_re = (p[f"cur_{_RE}"].fillna(0) - p[f"prev_{_RE}"].fillna(0)) / a
    paid_cur = p[f"cur_{_CS}"].fillna(0) + p[f"cur_{_CSUR}"].fillna(0)
    paid_prev = p[f"prev_{_CS}"].fillna(0) + p[f"prev_{_CSUR}"].fillna(0)
    d_paid = (paid_cur - paid_prev) / a
    g = p["edinetCode"]
    n = p.groupby("edinetCode")["t"].nunique()
    out = pd.DataFrame(
        {
            "d_re_3y": d_re.groupby(g).sum(),
            "d_paid_3y": d_paid.groupby(g).sum(),
        }
    )
    out["n_years"] = n
    out["d_re"] = out["d_re_3y"] / n
    out["d_paid"] = out["d_paid_3y"] / n
    out["transfer"], out["profit"] = decompose(out["d_re"], out["d_paid"])
    return out


def _q(s: pd.Series) -> str:
    return " | ".join(f"{s.quantile(q / 100):+.4f}" for q in _QS)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel-dir", type=Path, default=Path("data/interim/stage23/panel"))
    ap.add_argument("--exclude", type=Path, default=Path("data/interim/stage28/excluded.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/stage29"))
    args = ap.parse_args()
    ex = pd.read_csv(args.exclude, dtype={"edinetCode": str})
    codes = set(ex["edinetCode"])
    gc = firm_level(args.panel_dir, 2, codes)
    healthy = firm_level(args.panel_dir, 0, codes)
    args.out.mkdir(parents=True, exist_ok=True)
    gc.to_csv(args.out / "empirical_gc_firms.csv")

    up = gc["d_re"] > THR
    trans = up & (gc["d_paid"] < -THR)
    pure = up & (gc["d_paid"].abs() <= THR)
    issue = up & (gc["d_paid"] > THR)
    share_up = float(up.mean())
    p90_profit = float(gc["profit"].quantile(0.90))
    gamma = max(0.0, YEARS * p90_profit)
    t_pos = gc.loc[gc["transfer"] > THR, "transfer"] * YEARS
    criteria = {
        "n_gc_firms": len(gc),
        "share_re_up": share_up,
        "share_re_up_band": [max(0.0, share_up - 0.10), share_up + 0.10],
        "profit_annual_p90": p90_profit,
        "gamma": gamma,
        "n_transfer_firms": len(t_pos),
        "transfer_5y_band": [float(t_pos.quantile(0.05)), float(t_pos.quantile(0.95))],
        "threshold_annual": THR,
        "years": YEARS,
    }
    (args.out / "criteria.json").write_text(json.dumps(criteria, indent=2), encoding="utf-8")

    head = " | ".join(f"p{q}" for q in _QS)
    lines = [
        "# ステージ29 A.1: 実データの利益剰余金の変化(GC 注記企業、除外後)", "",
        f"推定用パネル(ステージ23、パネル A)の class 2 のうち、ステージ28 の除外対象を除いた "
        f"{len(gc)} 社。4 期(3 年分)の同一書類内の差分 / 前期総資産。年率 = 3 年の和 / 3、"
        "5 年換算 = 年率 × 5。", "",
        "## 分布(総資産比)", "",
        f"| 系列 | 社数 | {head} |", "|---|---:|" + "---:|" * len(_QS),
        f"| ΔRE(3 年の和) | {len(gc)} | {_q(gc['d_re_3y'])} |",
        f"| ΔRE(年率) | {len(gc)} | {_q(gc['d_re'])} |",
        f"| ΔRE(5 年換算) | {len(gc)} | {_q(gc['d_re'] * YEARS)} |",
        f"| Δ払込資本(年率) | {len(gc)} | {_q(gc['d_paid'])} |",
        f"| 振替 T(年率) | {len(gc)} | {_q(gc['transfer'])} |",
        f"| 利益部分 P(年率) | {len(gc)} | {_q(gc['profit'])} |",
        f"| 参考: 健全企業 class 0 の ΔRE(年率) | {len(healthy)} | {_q(healthy['d_re'])} |",
        f"| 参考: 健全企業 class 0 の P(年率) | {len(healthy)} | {_q(healthy['profit'])} |",
        "",
        f"## 利益剰余金が増えた社(年率 > {THR})", "",
        "| 区分 | 社数 | 割合(全社に対して) |", "|---|---:|---:|",
        f"| 増えた社 | {int(up.sum())} | {share_up:.3f} |",
        f"| うち振替(Δ払込資本 < −{THR}) | {int(trans.sum())} | {trans.mean():.3f} |",
        f"| うち純粋な利益(\\|Δ払込資本\\| ≤ {THR}) | {int(pure.sum())} | {pure.mean():.3f} |",
        f"| うち増資を伴う(Δ払込資本 > {THR}) | {int(issue.sum())} | {issue.mean():.3f} |",
        f"| 利益部分 P > {THR} の社 | {int((gc['profit'] > THR).sum())} "
        f"| {(gc['profit'] > THR).mean():.3f} |",
        "",
        "## 増えた社の内訳(年率)", "",
        "| edinetCode | ΔRE | Δ払込資本 | 振替 T | 利益部分 P |", "|---|---:|---:|---:|---:|",
    ]
    for code, r in gc[up].sort_values("d_re", ascending=False).iterrows():
        lines.append(
            f"| {code} | {r['d_re']:+.4f} | {r['d_paid']:+.4f} | {r['transfer']:.4f} "
            f"| {r['profit']:+.4f} |"
        )
    (args.out / "empirical.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps(criteria, indent=2))


if __name__ == "__main__":
    main()
