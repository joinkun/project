#!/usr/bin/env python
"""ステージ36: class 1(131 社)と class 2(55 社)での重みの 5 版の比較、検定力、実データとの一致。

numpy/pandas/scipy のみ(分類器を使う指標は ``run_stage25.py`` の出力
``metrics_class{1,2}/`` を読む)。

- B.1: 同じ比較(各版 − 等重み)の差の中央値の 95% CI の幅を class 2 と class 1 で比べる
- C: ステージ32 で記録した「勝つ」の定義(主指標 3 つ = 純資産の改善 ↑、債務超過の解消率 ↑、
  |総資産変化| ↓ のうち 2 つ以上で有意に優れ、有意に劣るものがない)を両方の class に当てる
- D: class 1 の生成(各版)の科目別の変化を、2022 年度に class 1 だった企業の実際の 2 年の変化
  (ステージ34 と同じ 127 社)と比べる。年率(生成は / 5、実際は / 2)

    uv run python scripts/evaluate_stage36.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from evaluate_stage34 import OUTLIER_CLASS1, special_reason
from persistence_tables_stage33 import spearman
from recovery_stage30 import changes

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names

VERSIONS = ("equal", "estimated", "qualitative", "frequency", "persistence")
LABELS = {"equal": "1 等重み", "estimated": "2 推定ベース", "qualitative": "3 定性ベース",
          "frequency": "4 頻度ベース", "persistence": "5 持続性ベース"}
PRIMARY = (("純資産の改善", "大"), ("債務超過の解消率(社ごとの 3 seed 平均)", "大"),
           ("|総資産変化|", "小"))
OTHER = (("sparsity", "小"), ("validity LightGBM", "大"), ("等重み proximity", "小"))
_P = "jppfs_cor_"
STL, PPE = "short_term_loans", f"{_P}PropertyPlantAndEquipment"
GEN_YEARS, ACT_YEARS = 5.0, 2.0


def verdict(lo: float, hi: float, better: str) -> str:
    if pd.isna(lo) or (lo == 0 and hi == 0):
        return "差なし"
    if (lo > 0 and better == "大") or (hi < 0 and better == "小"):
        return "優"
    if (hi < 0 and better == "大") or (lo > 0 and better == "小"):
        return "劣"
    return "−"


def esc(s: str) -> str:
    return s.replace("|", "\\|")


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/interim/stage36"))
    ap.add_argument("--panel-dir", type=Path, default=Path("data/interim/stage23/panel"))
    ap.add_argument("--out", type=Path, default=Path("results/stage36"))
    args = ap.parse_args()
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    paired = {
        1: pd.read_csv(args.out / "metrics_class1" / "paired_excl_outliers131.csv"),
        2: pd.read_csv(args.out / "metrics_class2" / "paired_excl_outliers55.csv"),
    }
    lines = ["# ステージ36: class 1 での重みの再比較(判定と実データとの一致)", ""]

    # ---- 各版 − 等重み、両 class ----
    lines += ["## 各版 − 等重み(差の中央値 [95% CI]、判定: 優 / 劣 / − = 有意差なし)", "",
              "| 版 | 指標 | class 1(n = 131) | 判定 | class 2(n = 55) | 判定 "
              "| CI の幅の比(class 2 / class 1) |",
              "|---|---|---|---|---|---|---:|"]
    wins: dict[tuple[str, int], tuple[int, int]] = {}
    widths = []
    for v in VERSIONS[1:]:
        for metric, better in (*PRIMARY, *OTHER):
            cells = []
            w = {}
            for c in (1, 2):
                r = paired[c][(paired[c]["version"] == v) & (paired[c]["metric"] == metric)].iloc[0]
                lo, hi, med = r["ci_low"], r["ci_high"], r["median_diff(version-equal)"]
                j = verdict(lo, hi, better)
                if (metric, better) in PRIMARY:
                    g, b = wins.get((v, c), (0, 0))
                    wins[(v, c)] = (g + (j == "優"), b + (j == "劣"))
                txt = ("全社で差 0" if pd.isna(med) else
                       f"{med:+.4f} [{lo:+.4f}, {hi:+.4f}](n = {int(r['n'])})")
                cells += [txt, j]
                w[c] = hi - lo if pd.notna(lo) else np.nan
            ratio = w[2] / w[1] if pd.notna(w[1]) and pd.notna(w[2]) and w[1] > 0 else np.nan
            if metric in ("純資産の改善", "|総資産変化|"):
                widths.append(ratio)
            lines.append(f"| {LABELS[v]} | {esc(metric)} | {cells[0]} | {cells[1]} | {cells[2]} "
                         f"| {cells[3]} | {'—' if pd.isna(ratio) else f'{ratio:.2f}'} |")
    lines += ["", f"純資産の改善と |総資産変化| の CI の幅の比(class 2 / class 1)の中央値: "
              f"**{np.nanmedian(widths):.2f}**(標本の大きさだけなら √(131/55) = "
              f"{np.sqrt(131 / 55):.2f})", ""]
    lines += ["## 判定(ステージ32 の「勝つ」の定義)", "",
              "| 版 | class 1: 主指標で優 / 劣 | 勝つ | class 2: 優 / 劣 | 勝つ | 両方で勝つ |",
              "|---|---|---|---|---|---|"]
    for v in VERSIONS[1:]:
        g1, b1 = wins[(v, 1)]
        g2, b2 = wins[(v, 2)]
        w1 = g1 >= 2 and b1 == 0
        w2 = g2 >= 2 and b2 == 0
        lines.append(f"| {LABELS[v]} | {g1} / {b1} | {'○' if w1 else '×'} | {g2} / {b2} "
                     f"| {'○' if w2 else '×'} | {'○' if w1 and w2 else '×'} |")

    # ---- D: 実データとの一致 ----
    fy = pd.read_csv("data/interim/stage30/firm_years.csv", dtype={"edinetCode": str})
    wide = fy.pivot(index="edinetCode", columns="year", values="label")
    c1 = wide[wide[2022] == 1].copy()
    c1["reason"] = special_reason(fy, c1.index.tolist())
    ch = changes(args.panel_dir, c1.index.tolist(), names)
    c1 = c1.join(ch, how="left")
    act_codes = c1.index[(c1["reason"] == "") & c1["n_years"].notna()]
    p = pd.read_csv(args.panel_dir / "panel.csv", dtype={"edinetCode": str})
    pp = p[p["edinetCode"].isin(act_codes) & p["t"].isin([2023, 2024])]
    dacc = pd.DataFrame({n: (pp[f"cur_{n}"].fillna(0) - pp[f"prev_{n}"].fillna(0))
                         / pp["prev_total_assets"] for n in names})
    dacc["edinetCode"] = pp["edinetCode"].to_numpy()
    act = dacc.groupby("edinetCode").sum() / ACT_YEARS
    rec_codes = [c for c in act.index if wide.loc[c, 2024] == 0]
    act_rec = act.loc[rec_codes]

    gen: dict[str, pd.DataFrame] = {}
    for v in VERSIONS:
        files = sorted((args.dir / "class1" / v).glob("nearest_na0__seed*.parquet"))
        g = pd.concat([pd.read_parquet(f) for f in files])
        g = g[(g["method"] == "PCFM") & ~g["docID"].isin(OUTLIER_CLASS1)]
        d = pd.DataFrame({n: g[f"x1_{n}"].to_numpy() - g[f"x0_{n}"].to_numpy() for n in names})
        d["docID"] = g["docID"].to_numpy()
        gen[v] = d.groupby("docID").median() / GEN_YEARS

    lines += ["", "## D.1 短期借入金と有形固定資産の向き(年率、総資産比)", "",
              f"実際: 2022 年度に class 1 だった {len(act)} 社"
              f"(うち class 0 へ回復 {len(act_rec)} 社)の "
              "2 年の変化。生成: class 1 の 131 社。", "",
              "| 系列 | 短期借入金: 中央値 | 減った社 | 増えた社 | 有形固定資産: 中央値 "
              "| 減った社 | 増えた社 |",
              "|---|---:|---:|---:|---:|---:|---:|"]

    def row(label: str, df: pd.DataFrame) -> str:
        s, t = df[STL], df[PPE]
        thr = 1e-4
        return (f"| {label} | {s.median():+.4f} | {(s < -thr).mean():.0%} | {(s > thr).mean():.0%} "
                f"| {t.median():+.4f} | {(t < -thr).mean():.0%} | {(t > thr).mean():.0%} |")

    lines.append(row("実際(class 1 全体)", act))
    lines.append(row("実際(class 0 へ回復)", act_rec))
    for v in VERSIONS:
        lines.append(row(f"生成 {LABELS[v]}", gen[v]))
    lines += ["", "(変化が総資産の ±0.01% 以内は「減った」「増えた」に数えない。)", "",
              "## D.2 科目別の動きと実データの一致", "",
              "| 版 | 32 科目の \\|Δx\\| の中央値の順位相関(実際 全体) | 同(回復した社) "
              "| 符号付き Δx の中央値の順位相関(実際 全体) |",
              "|---|---:|---:|---:|"]
    a_abs, r_abs, a_sig = act.abs().median(), act_rec.abs().median(), act.median()
    for v in VERSIONS:
        g_abs, g_sig = gen[v].abs().median(), gen[v].median()
        lines.append(
            f"| {LABELS[v]} | {spearman(g_abs.to_numpy(), a_abs.to_numpy()):+.3f} "
            f"| {spearman(g_abs.to_numpy(), r_abs.to_numpy()):+.3f} "
            f"| {spearman(g_sig.to_numpy(), a_sig.to_numpy()):+.3f} |")
    lines += ["", "科目別の Δx / A の中央値(年率、符号付き):", "",
              "| 科目 | 実際(全体) | 実際(回復) | "
              + " | ".join(LABELS[v] for v in VERSIONS) + " |",
              "|---|---:|---:|" + "---:|" * len(VERSIONS)]
    for n in names:
        lines.append(f"| {ja[n]} | {act[n].median():+.4f} | {act_rec[n].median():+.4f} | "
                     + " | ".join(f"{gen[v][n].median():+.4f}" for v in VERSIONS) + " |")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
