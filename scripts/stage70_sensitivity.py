#!/usr/bin/env python
"""ステージ70 B の感度(結果を見てから追加した。判定には使わない。lightgbm・sklearn のプロセス)。

新設計では、class 1 の 1 社(S100UG53、項目 5「負債が 2 倍超」)が λ̃ ≤ 3 で違反し、ほぼ
全ての設定で λ̃*_0 = 10(QP がほとんど動かない)が選ばれた。この 1 社の違反を較正の条件から
外したときの λ̃ と、そのときの主指標(設定 1 との差)を示す。生成はやり直さない(既存の格子の
点を使う)。両 class の違反は表に出す。

出力: ``results/stage70/sensitivity.md``。

    uv run python scripts/stage70_sensitivity.py
"""

from __future__ import annotations

from pathlib import Path

import evaluate_stage51 as e51
import numpy as np
from evaluate_stage61 import CAL, gen_firms, loss, real_targets
from stage70_evaluate import LAMS, S70, lab_of
from stage70_qp import SETTINGS
from tables_stage69 import S67, Src, ctx, metrics, setup_excl, versus

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import subtotal_members

OUT = Path("results/stage70")
IGNORE = "S100UG53"


def main() -> None:
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    setup_excl(names)
    r1, se1, _ = real_targets("c1", names)
    w1 = {m: 1.0 / se1[m] for m in CAL}
    base = Src(*S67, "s1__l1", "1")
    lines = ["# ステージ70 B の感度: S100UG53 の違反を較正の条件から外した場合(結果を見てから"
             "追加、"
             "判定には使わない)", "",
             "λ̃ は、class 1 で S100UG53 以外の違反が 0 の中で L が最小の点。差は設定 1 との"
             "企業単位のペア比較。違反は両 class の企業数(S100UG53 を含む)。", "",
             "| 設定 | λ̃ | 違反 c2 / c1 | 純資産の改善 c2 / c1 | 差 c2 | 差 c1 | 記号 c2 / c1 |",
             "|---|---:|---|---|---|---|---|"]
    for st in SETTINGS:
        best: tuple[float | None, float] = (None, np.inf)
        for lt in LAMS:
            src = Src(*S70, f"{st}__l{lt:g}", f"{lt:g}")
            c = ctx(src, "c1", names)
            v = c.viol[(c.viol["setting"] == src.setting) & (c.viol["docID"] != IGNORE)]
            f = gen_firms("c1", src.setting, names, members)
            f = f[~f.index.isin(e51.EXCL["c1"])]
            lv = loss({"paid_dec": float(f["paid_dec"].mean()), "re_up": float(f["re_up"].mean()),
                       "d_assets": float(f["d_assets"].median())}, r1, w1)
            if v["docID"].nunique() == 0 and lv < best[1]:
                best = (lt, lv)
        if best[0] is None:
            lines.append(f"| {lab_of(st)} | なし | — | — | — | — | — |")
            continue
        src = Src(*S70, f"{st}__l{best[0]:g}", f"{best[0]:g}")
        mm = {cls: metrics(src, cls, names, members) for cls in ("c2", "c1")}
        vv = {cls: versus(src, base, cls, names, np.random.default_rng(0)) for cls in ("c2", "c1")}

        def sym(s: list[int]) -> str:
            return " ".join({1: "+", -1: "−", 0: "0"}[int(x)] for x in s)

        def ci(t: tuple[float, float, float]) -> str:
            return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
        lines.append(f"| {lab_of(st)} | {best[0]:g} | {mm['c2']['viol']} / {mm['c1']['viol']} "
                     f"| {mm['c2']['nai']:.3f} / {mm['c1']['nai']:.3f} | {ci(vv['c2']['nai'])} "
                     f"| {ci(vv['c1']['nai'])} | {sym(vv['c2']['sign'])} / "
                     f"{sym(vv['c1']['sign'])} |")
        print(lines[-1], flush=True)
    lines.append("")
    (OUT / "sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
