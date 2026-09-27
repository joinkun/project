#!/usr/bin/env python
"""ステージ24 A: 科目ごとのコスト重みを 3 版で作る(既存の costs.yaml は記録として残す)。

- 版 1 ``costs_equal.yaml``: すべて w+ = w- = 1
- 版 2 ``costs_estimated.yaml``: 調整速度の推定値から c = (1 - λ) / λ。
  売上債権・仕入債務・棚卸資産は売上高比(ステージ23.5、差分 GMM、部分標本 3,250 社)、
  それ以外は総資産比(ステージ23、パネル A 3,657 社、差分 GMM)。
  いずれも業種中央値の目標・符号を分けない推定(符号別は不安定なため)。
  λ ∉ (0, 1]、c > 100、推定なし(残余科目)、推定不能の科目は等重み(1.0)で補完する(文献値では
  補完しない)。現金預金 = 1.0 で正規化する
- 版 3 ``costs_qualitative.yaml``: 制度・会計上の性質から順序が確実なものだけを
  3 段階(1.0 / 2.0 / 4.0)で与え、それ以外は 1.0。類推はしない

non_actionable(利益剰余金・評価差額金・非支配株主持分)は 3 版とも既存の costs.yaml と同じ。

    uv run python scripts/build_costs_stage24.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

_P = "jppfs_cor_"
_CASH = f"{_P}CashAndDeposits"
_MAX_COST = 100.0
SALES_DENOMINATOR = ("receivables_trade", "payables_trade", "inventories")
NO_BASIS = "根拠なし。等重み"


@dataclass(frozen=True)
class Qualitative:
    increase: float
    decrease: float
    source: str


# 版 3: 出典を確認できたものだけ。値の向きは「値の増加・減少」(cost.py の w+ / w-)。
QUALITATIVE: dict[str, Qualitative] = {
    f"{_P}TreasuryStock": Qualitative(
        1.0,
        4.0,
        "自己株式の取得(負の値がさらに減る方向)= 4.0。"
        "会社法 461 条 1 項: 自己株式の取得等により交付する金銭等の帳簿価額の総額は"
        "分配可能額を超えてはならない(e-Gov 法令 API で条文を確認)。"
        "分配可能額が 0 の企業(債務超過など)では取得できない。"
        "制約は条件付きだが、重みは全社に一律で課す",
    ),
    f"{_P}PropertyPlantAndEquipment": Qualitative(
        1.0,
        2.0,
        "減少(売却)= 2.0。Shleifer & Vishny (1992, JF 47(4)) 要旨: "
        "困窮企業が資産を売る際は同業者も困窮しており、"
        "最善の用途の価値を下回る価格での売却になる(要旨で確認)",
    ),
    "bonds_current": Qualitative(
        1.0,
        2.0,
        "減少(返済・再交渉)= 2.0、借入金は 1.0。Gilson, John & Lang (1990, JFE 27) 要旨・本文: "
        "銀行借入の割合が高く貸し手が少ない企業ほど私的な債務再編に成功しやすい(本文で確認)",
    ),
    "bonds_noncurrent": Qualitative(
        1.0,
        2.0,
        "減少(返済・再交渉)= 2.0、借入金は 1.0。"
        "Gilson, John & Lang (1990) による(bonds_current と同じ)",
    ),
    f"{_P}CapitalSurplus": Qualitative(
        1.0,
        1.0,
        "減少 = 1.0(最も低い段階のまま)。布袋 (2024): 無償減資は実行頻度が高い(ユーザー確認済み、"
        "本ステージでは原典未確認)。既定値と同じため数値は変わらない",
    ),
}


def load_lambdas(t4: Path, t23_5: Path) -> dict[str, tuple[float, str]]:
    """科目 -> (λ, 根拠)。業種中央値の目標・差分 GMM・符号を分けない推定。"""
    e4 = pd.read_csv(t4)
    e4 = e4[
        (e4["panel"] == "A")
        & (e4["target"] == "industry")
        & (e4["method"] == "diff_gmm")
        & (e4["direction"] == "both")
    ]
    e5 = pd.read_csv(t23_5)
    e5 = e5[
        (e5["sample"] == "subsample")
        & (e5["denominator"] == "sales")
        & (e5["method"] == "diff_gmm")
        & (e5["direction"] == "both")
    ]
    out: dict[str, tuple[float, str]] = {}
    for r in e4.itertuples():
        lam = float(r.lam) if r.status == "ok" else math.nan
        out[str(r.account)] = (lam, f"総資産比・差分 GMM(ステージ23)λ={r.lam:.3f}")
    for r in e5[e5["account"].isin(SALES_DENOMINATOR)].itertuples():
        lam = float(r.lam) if r.status == "ok" else math.nan
        out[str(r.account)] = (lam, f"売上高比・差分 GMM(ステージ23.5)λ={r.lam:.3f}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--costs", type=Path, default=Path("conf/accounts/costs.yaml"))
    parser.add_argument("--t4", type=Path, default=Path("results/stage23/estimates.csv"))
    parser.add_argument("--t23-5", type=Path, default=Path("results/stage23_5/estimates.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("conf/accounts"))
    parser.add_argument("--table", type=Path, default=Path("results/stage24/costs_comparison.md"))
    args = parser.parse_args()

    base = yaml.safe_load(args.costs.read_text(encoding="utf-8"))
    names = [a["name"] for a in base["basis_costs"]]
    lambdas = load_lambdas(args.t4, args.t23_5)

    # 版 2: c = (1-λ)/λ、現金で正規化、破綻・推定なしは 1.0
    raw: dict[str, float] = {}
    note: dict[str, str] = {}
    filled: list[tuple[str, str]] = []
    for name in names:
        lam, src = lambdas.get(name, (math.nan, "推定なし(残余科目)"))
        if not math.isfinite(lam):
            filled.append((name, src if name not in lambdas else f"推定不能({src})"))
            note[name] = src
            continue
        if not 0.0 < lam <= 1.0:
            filled.append((name, f"λ ∉ (0, 1]({src})"))
            note[name] = src
            continue
        c = (1.0 - lam) / lam
        if c > _MAX_COST:
            filled.append((name, f"c = {c:.1f} > {_MAX_COST:.0f}({src})"))
            note[name] = src
            continue
        raw[name] = c
        note[name] = f"{src}、c={c:.3f}"
    scale = raw[_CASH]
    estimated = {n: (raw[n] / scale if n in raw else 1.0) for n in names}

    def write(path: Path, rows: dict[str, tuple[float, float, str]], header: str) -> None:
        doc = {
            "name": path.stem,
            "basis_costs": [
                {"name": n, "cost_increase": ci, "cost_decrease": cd, "source": s}
                for n, (ci, cd, s) in rows.items()
            ],
            "non_actionable": base["non_actionable"],
        }
        text = header + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=1000)
        path.write_text(text, encoding="utf-8")

    header = "# ステージ24: scripts/build_costs_stage24.py が生成(手で編集しない)。\n"
    write(
        args.out_dir / "costs_equal.yaml",
        {n: (1.0, 1.0, "等重み(ベースライン)") for n in names},
        header,
    )
    est_rows: dict[str, tuple[float, float, str]] = {}
    for n in names:
        if n in raw:
            w = round(estimated[n], 4)
            est_rows[n] = (w, w, f"{note[n]}。現金預金の c={scale:.3f} で正規化")
        else:
            est_rows[n] = (1.0, 1.0, f"{NO_BASIS}(補完: {note[n]})")
    write(args.out_dir / "costs_estimated.yaml", est_rows, header)
    qual_rows = {
        n: (
            (QUALITATIVE[n].increase, QUALITATIVE[n].decrease, QUALITATIVE[n].source)
            if n in QUALITATIVE
            else (1.0, 1.0, NO_BASIS)
        )
        for n in names
    }
    write(args.out_dir / "costs_qualitative.yaml", qual_rows, header)

    lines = [
        "# ステージ24 A.4: コスト重みの 3 版(現金預金 = 1.0)",
        "",
        "w+ / w- は値の増加・減少のコスト。推定ベースは符号を分けない推定なので w+ = w-。",
        "",
        "| 科目 | 等重み | 推定ベース | 定性ベース (w+ / w-) | 推定の根拠 | 定性の根拠 |",
        "|---|---:|---:|---|---|---|",
    ]
    for n in names:
        q = qual_rows[n]
        est = f"{estimated[n]:.3f}" + ("" if n in raw else "(補完)")
        lines.append(
            f"| {n.removeprefix(_P)} | 1.0 | {est} | {q[0]:.1f} / {q[1]:.1f} | {note[n]} | {q[2]} |"
        )
    lines += [
        "",
        f"## 推定ベースで等重み(1.0)に補完した科目: {len(filled)} / {len(names)}",
        "",
    ]
    lines += [f"- {n.removeprefix(_P)}: {why}" for n, why in filled]
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
