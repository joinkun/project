#!/usr/bin/env python
"""ステージ72: 論文用の表(第 2 版)。``stage72_evaluate.py`` の ``summary.json`` から作る。

出力: ``results/stage72/paper_tables_v2.md``。

    uv run python scripts/stage72_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stage72_evaluate import LABELS

OUT = Path("results/stage72")
ORDER = ("f1", "f1m1", "a2", "a2t05", "bal_l2", "f1bal", "l2nn", "pre05", "pre05m1")


def sym(s: list[int]) -> str:
    return " ".join({1: "+", -1: "−", 0: "0"}[int(v)] for v in s)


def ci(t: list[float]) -> str:
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def main() -> None:
    sm: dict[str, Any] = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    res = sm["res"]
    out = ["# 論文用の表(第 2 版、ステージ72)", "",
           "すべて class 1(128 社)だけで λ̃*_0 を較正し(違反 0 の中で L 最小)、同じ λ を"
           " class 2(55 社)"
           "に当てはめた。新設計 = 72 社のコスト c = 1/s・非支配株主持分を動かす・負債の上限"
           "(1 − 10⁻⁶ のマージン)。差は設定 4(ステージ69 の最終設定)との企業単位のペア比較"
           "(2,000 回のブートストラップ)。記号 = 純資産の改善・解消率・順位相関。", "",
           f"**採用(事前の判定)**: {LABELS[sm['adopted']] if sm['adopted'] else 'なし'}", ""]
    for cls, n in (("c2", "class 2(55 社)"), ("c1", "class 1(128 社)")):
        out += [f"## 表 A({n}): 提案 F1 と比較の設定", "",
                "| 設定 | 統合した企業数 | λ̃*_0 | 違反 | 純資産の改善 | 設定 4 との差 [95%] "
                "| 解消率 "
                "| 払込資本↓ / ↑ | 変更科目数 | 順位相関 | 順位相関の差 [95%] | 記号 |",
                "|---|---:|---:|---:|---:|---|---:|---|---:|---:|---|---|"]
        b = res[f"s1|{cls}"]["m"]
        out.append(f"| 4 従来(ステージ69) | 1 | 1 | {b['viol']} | {b['nai']:.3f} | — "
                   f"| {b['res']:.2f} "
                   f"| {b['paid_dec']:.0%} / {b['paid_inc']:.0%} | {b['n_changed']:.0f} "
                   f"| {b['rho']:+.3f} | — | — |")
        for st in ORDER:
            r = res.get(f"{st}|{cls}")
            if r is None:
                continue
            v = r["m"]
            bold = "**" if st == sm["adopted"] else ""
            out.append(f"| {bold}{LABELS[st]}{bold} | {r['multi']:.2f} | {v['lam']} "
                       f"| {v['viol']} "
                       f"| {v['nai']:.3f} | {ci(r['vs']['nai'])} | {v['res']:.2f} "
                       f"| {v['paid_dec']:.0%} / {v['paid_inc']:.0%} | {v['n_changed']:.0f} "
                       f"| {v['rho']:+.3f} | {ci(r['vs']['rho'])} | {sym(r['vs']['sign'])} |")
        out.append("")
    out += ["## 表 B: 事前に混ぜる 対 事後に混ぜる(同じ τ1 = 0.05 の S(i)・同じ p̃)", "",
            "目標の性質は QP の前(F1 は候補を集めた集合)、終点は QP の後(|y| < 10⁻⁶ を 0)。", "",
            "| class | 設定 | 目標: ちょうど 0 | 目標: 貸借の残差 | 目標: AUC | 終点: 0 の科目数 "
            "| 終点: 残差 | 純資産の改善 | 順位相関 | 記号 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for cls in ("c2", "c1"):
        for st in ("a2t05", "pre05", "pre05m1", "f1"):
            r = res[f"{st}|{cls}"]
            out.append(f"| {cls} | {LABELS[st]} | {r['t_zero']:.2f} | {r['t_resid']:.1e} "
                       f"| {r['t_auc']:.3f} | {r['e_zero']:.2f} | {r['e_resid']:.1e} "
                       f"| {r['m']['nai']:.3f} | {r['m']['rho']:+.3f} | {sym(r['vs']['sign'])} |")
    out += ["", "健全企業(class 0)のちょうど 0 の科目数は 6.70。設定 4 の終点の 0 の科目数は "
            f"{res['s1|c2']['e_zero']:.2f} / {res['s1|c1']['e_zero']:.2f}(class 2 / class 1)。", ""]
    (OUT / "paper_tables_v2.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:12]))


if __name__ == "__main__":
    main()
