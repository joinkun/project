#!/usr/bin/env python
"""ステージ69.5 B(確認のみ): 規則どおりの c_RE で解き直した class 1 の設定 2 を比べる。

比較の相手はステージ67 の設定 2。

λ̃*_0 は class 1 で、違反 0 の中で L 最小(ステージ67 と同じ)。lightgbm・sklearn のプロセス。

出力: ``results/stage69/cre_check.md`` に追記。

    uv run python scripts/eval_cre_stage695.py
"""

from __future__ import annotations

from pathlib import Path

import evaluate_stage51 as e51
import numpy as np
from evaluate_stage61 import CAL, gen_firms, loss, real_targets
from tables_stage69 import INTERIM, S67, Src, ctx, metrics, setup_excl, versus

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import subtotal_members

OUT = Path("results/stage69")
S695 = (INTERIM / "stage695/metrics", INTERIM / "stage695/all")
LAMS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def main() -> None:
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    setup_excl(names)
    r1, se1, _ = real_targets("c1", names)
    w1 = {m: 1.0 / se1[m] for m in CAL}
    cells: list[str] = []
    best: tuple[float | None, float] = (None, np.inf)
    for lt in LAMS:
        src = Src(*S695, f"s2r__l{lt:g}", f"{lt:g}")
        c = ctx(src, "c1", names)
        f = gen_firms("c1", src.setting, names, members)
        f = f[~f.index.isin(e51.EXCL["c1"])]
        nv = c.viol[c.viol["setting"] == src.setting]["docID"].nunique()
        lv = loss({"paid_dec": float(f["paid_dec"].mean()), "re_up": float(f["re_up"].mean()),
                   "d_assets": float(f["d_assets"].median())}, r1, w1)
        cells.append(f"{lv:.2f}({nv})")
        if nv == 0 and lv < best[1]:
            best = (lt, lv)
    if best[0] is None:
        raise ValueError("違反 0 の λ̃ がない")
    lt_s = best[0]
    new = Src(*S695, f"s2r__l{lt_s:g}", f"{lt_s:g}")
    old = Src(*S67, "s2__l1", "1")
    base = Src(*S67, "s1__l1", "1")
    mn, mo, mb = (metrics(s, "c1", names, members) for s in (new, old, base))
    d_old = versus(new, old, "c1", names, np.random.default_rng(0))
    lines = ["", "## class 1 の設定 2(利益剰余金にコスト)を規則どおりの c_RE で解き直した結果("
                 "確認のみ)",
             "", "λ の尺度 s は設定ごとに class 1 で計算(c_RE⁻ が変わるので s も変わる)。λ̃*_0 "
                 "は class 1 "
             "の 128 社で、違反 0 の中で L 最小。", "",
             "| λ̃ | " + " | ".join(f"{v:g}" for v in LAMS) + " |", "|---|" + "---:|" * len(LAMS),
             "| L(違反した企業数) | " + " | ".join(cells) + " |", "",
             "| 設定(class 1) | λ̃*_0 | 違反 | ΔRE/ΔNA < 0.99 | 払込資本↑ | 払込資本↓ | RE↑ | "
             "変更科目数 "
             "| 純資産の改善 | 解消率 | 順位相関 "
             "|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for lab, v in (("1 現行(利益剰余金 c = 0)", mb), ("2 ステージ67(c_RE⁻ = 3.14)", mo),
                   ("2 規則どおり(c_RE⁻ = 137.79)", mn)):
        lines.append(f"| {lab} | {v['lam']} | {v['viol']} | {v['re_lt1']:.0%} |"
                     f" {v['paid_inc']:.0%} "
                     f"| {v['paid_dec']:.0%} | {v['re_up']:.0%} | {v['n_changed']:.0f} "
                     f"| {v['nai']:.3f} | {v['res']:.2f} | {v['rho']:+.3f} |")
    lines += ["", f"規則どおり − ステージ67: 純資産の改善 {d_old['nai'][0]:+.3f} "
              f"[{d_old['nai'][1]:+.3f}, {d_old['nai'][2]:+.3f}]、順位相関 {d_old['rho'][0]:+.3f} "
              f"[{d_old['rho'][1]:+.3f}, {d_old['rho'][2]:+.3f}]。", ""]
    path = OUT / "cre_check.md"
    path.write_text(path.read_text(encoding="utf-8") + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
