#!/usr/bin/env python
"""ステージ69 C.2: ``results/stage65/slides_numbers.md`` の数値が、``paper_tables.json``・``ver"
  "ify.json``
(ステージ69 の集計)と一致するかを照合する。数値を書式化して、スライドの文中に現れるかを調べる。

出力: ``results/stage69/consistency.md``。一致しない項目があれば終了コード 1。

    uv run python scripts/check_stage69.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path("results/stage69")


def main() -> None:
    t = json.loads((OUT / "paper_tables.json").read_text(encoding="utf-8"))
    v = json.loads((OUT / "verify.json").read_text(encoding="utf-8"))
    slides = Path("results/stage65/slides_numbers.md").read_text(encoding="utf-8")
    m, vs, real, auc = t["metrics"], t["versus"], t["real"], t["auc"]

    def pct(x: float) -> str:
        return f"{x:.0%}"

    checks: list[tuple[str, str]] = [
        ("提案 class 2 払込資本↓", f"**{pct(m['prop']['c2']['paid_dec'])}**"),
        ("提案 class 2 純資産の改善", f"**{m['prop']['c2']['nai']:.3f}**"),
        ("提案 class 2 解消率", f"**{m['prop']['c2']['res']:.2f}**"),
        ("提案 class 2 順位相関", f"**{m['prop']['c2']['rho']:+.3f}**"),
        ("提案 class 2 変更科目数", f"**{m['prop']['c2']['n_changed']:.0f}**"),
        ("第 1 層のみ class 2 違反", f"**{m['l1only']['c2']['viol']} 社**"),
        ("第 1 層のみ class 1 違反", f"class 1 で {m['l1only']['c1']['viol']} 社"),
        ("重心 class 2 純資産の改善", f"{m['bary']['c2']['nai']:.3f}"),
        ("重心 class 2 差", f"{vs['bary']['c2']['nai'][0]:+.3f}".replace("-", "−")),
        ("重心 class 1 "
         "差", f"{vs['bary']['c1']['nai'][0]:+.3f}".replace("+", "").replace("-", "−")),
        ("バランス OT の argmax class 2", f"**{m['balarg']['c2']['nai']:.3f}**"),
        ("等重み L2 class 1 純資産の改善の差", f"+{-vs['l2']['c1']['nai'][0]:.3f}"),
        ("等重み L2 class 1 順位相関の差", f"+{-vs['l2']['c1']['rho'][0]:.3f}"),
        ("提案 AUC", f"{auc['c2']['prop'][0]:.2f} / {auc['c1']['prop'][0]:.2f}"),
        ("重心 AUC", f"{auc['c2']['bary'][0]:.3f} / {auc['c1']['bary'][0]:.3f}"),
        ("等重み L2 AUC", f"{auc['c2']['l2'][0]:.2f} / {auc['c1']['l2'][0]:.2f}"),
        ("重心 AUC(LogReg)", f"{auc['c2']['bary'][1]:.3f} / {auc['c1']['bary'][1]:.3f}"),
        ("厳密解の実効サポート class 2", f"**{v['c2']['eff_med']:.1f}**"),
        ("厳密解の実効サポート class 1", f"**{v['c1']['eff_med']:.1f}**"),
        ("下限 n1/n0", f"{v['c2']['ratio']:.1f} | {v['c1']['ratio']:.1f}"),
        ("ゼロの科目数", f"{v['健全企業(class 0、実在)']['zero_mean']:.2f} → 2 社の平均 "
                       f"{v['2 社の平均']['zero_mean']:.2f}"),
        ("未払金がゼロ", f"{pct(v['健全企業(class 0、実在)']['ap_zero'])}、2 社の平均で "
                      f"{pct(v['2 社の平均']['ap_zero'])}"),
        ("m = 1 の払込資本↓", f"class 2 で {pct(m['m1']['c2']['paid_dec'])}、class 1 で "
                           f"{pct(m['m1']['c1']['paid_dec'])}"),
        ("m = 1 の純資産の改善", f"{m['m1']['c2']['nai']:.3f} / {m['m1']['c1']['nai']:.3f}"),
        ("実データとの比較 class 2", f"| **{pct(m['prop']['c2']['paid_dec'])} / "
                                 f"{pct(real['c2']['paid_dec'])}** |"),
        ("実データとの比較 class 1", f"| {pct(m['prop']['c1']['paid_dec'])} / "
                                 f"{pct(real['c1']['paid_dec'])} |"),
        ("RE↑ class 2", f"| {pct(m['prop']['c2']['re_up'])} / {pct(real['c2']['re_up'])} |"),
        ("RE↑ class 1", f"{pct(m['prop']['c1']['re_up'])} / {pct(real['c1']['re_up'])} |"),
        ("純資産の改善(年率) class 2", f"+{m['prop']['c2']['nai'] / 5:.3f} / "
                                    f"+{real['c2']['nai_annual']:.3f}"),
        ("純資産の改善(年率) class 1", f"+{m['prop']['c1']['nai'] / 5:.3f} / "
                                    f"+{real['c1']['nai_annual']:.3f}"),
        ("変更科目数", f"**{m['prop']['c2']['n_changed']:.0f} / {real['c2']['n_changed']:.0f}** "
                     f"| **{m['prop']['c1']['n_changed']:.0f} / {real['c1']['n_changed']:.0f}**"),
    ]
    lines = ["# ステージ69 C.2: `slides_numbers.md` と論文の表の照合", "",
             "`paper_tables.json`・`verify.json` の値を書式化し、`results/stage65/slides_number"
             "s.md` の"
             "文中に同じ文字列があるかを調べた(`scripts/check_stage69.py`)。", "",
             "| 項目 | 論文の表の値(書式化) | スライドにある |", "|---|---|---|"]
    bad = 0
    for lab, s in checks:
        ok = s in slides
        bad += not ok
        lines.append(f"| {lab} | `{s}` | {'一致' if ok else '**不一致**'} |")
    lines += ["", f"不一致: {bad} 件 / {len(checks)} 件。", ""]
    (OUT / "consistency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
