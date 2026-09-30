#!/usr/bin/env python
"""ステージ41 A.2: 不活性制約を課す科目の選定(生成の前に実行し、結果を記録する)。

入力は ``inactive_rates_stage41.py`` の ``rates.csv``。実データの割合 r は
class 1+2(始点の年度のラベル)の
1 年の割合(生成の始点は class 1・2)。

- 規則 S(スペックの区分): r < 5% → 課す、r > 20% → 課さない、5〜20% → 規則 M で決める。
  始点 0 の組がない科目(r を推定できない)は課さない
- 規則 M(機械的): 5 年換算 r5 = 1 − (1 − r)^5(毎年独立とみなす。持続的な 0 の企業が
  多いほど実際は小さい)と、
  現行の生成(最終設定、class 1+2 の行)の割合 g を比べ、r5 < g / 2(0 に固定したほうが g のままより
  実データに近い)なら課す

    uv run python scripts/inactive_select_stage41.py --rates results/stage41/rates.csv \\
        --out results/stage41/selection.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LOW, HIGH, YEARS = 0.05, 0.20, 5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    t = pd.read_csv(args.rates)
    r = t["rate_c12"]
    r5 = 1 - (1 - r) ** YEARS
    n_gen = t["n_zero_gen_c2"] + t["n_zero_gen_c1"]
    g = (t["k_gen_c2"] + t["k_gen_c1"]) / n_gen.where(n_gen > 0)
    rule_m = (r5 < g / 2).fillna(False)  # g が推定できない(生成で始点 0 がない)科目は課さない
    rule_s = np.where(r.isna(), False, np.where(r < LOW, True, np.where(r > HIGH, False, rule_m)))
    out = t[["account", "ja", "n_zero_c12", "k_c12", "rate_c12", "c12_lo", "c12_hi"]].copy()
    out["r5"], out["g"] = r5, g
    out["band"] = np.where(r.isna(), "推定不可", np.where(r < LOW, "5% 未満",
                                                    np.where(r > HIGH, "20% 超", "5〜20%")))
    out["rule_S"], out["rule_M"] = rule_s, rule_m.to_numpy()
    out.to_csv(args.out.with_suffix(".csv"), index=False)

    def f(v: float) -> str:
        return "—" if pd.isna(v) else f"{v:.1%}"

    lines = ["# ステージ41 A.2: 不活性制約の選定", "",
             "r = 実データ class 1+2 の 1 年の割合、r5 = 1 − (1 − r)^5、"
             "g = 現行の生成の割合(5 年、class 1+2 の行)。",
             "規則 S はスペックの区分(5〜20% は規則 M で決める)、規則 M は r5 < g / 2。", "",
             "| 科目 | r [95%] | 区分 | r5 | g | 規則 S | 規則 M |",
             "|---|---|---|---:|---:|---|---|"]
    for x in out.to_dict("records"):
        lines.append(f"| {x['ja']} | {f(x['rate_c12'])}({x['k_c12']}/{x['n_zero_c12']}) "
                     f"[{f(x['c12_lo'])}, {f(x['c12_hi'])}] | {x['band']} | {f(x['r5'])} "
                     f"| {f(x['g'])} | {'課す' if x['rule_S'] else ''} "
                     f"| {'課す' if x['rule_M'] else ''} |")
    s = out[out["rule_S"]]["account"].tolist()
    m = out[out["rule_M"]]["account"].tolist()
    lines += ["", f"規則 S で課す科目: {len(s)} 個", "", *[f"- `{a}`" for a in s], "",
              f"規則 M で課す科目: {len(m)} 個", "", *[f"- `{a}`" for a in m]]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
