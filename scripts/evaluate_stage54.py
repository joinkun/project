#!/usr/bin/env python
"""ステージ54 C・D: 条件付き密度 + 距離の対応先の評価(``evaluate_stage51.py`` の評価を流用)。

規則は ``results/stage54/criteria.md``(生成の前に記録)。設定: ``equal``(現行)、``indep``
(ステージ52 の周辺密度)、``cond``(本ステージ)。加えて C.3 の検定力、検出力が不足する場合の
事前に決めた判断、自己株式を持たない企業を含む BS 表(C.4)を出す。

出力(``results/stage54/``): ``comparison.md``・``assets_extreme.md``・``summary_e.json``
(evaluate_stage51 の形式)、``power.md``、``judgment.md``、``bs.md``。

    uv run python scripts/evaluate_stage54.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import numpy as np
import pandas as pd
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.viz.balance_sheet import build_bs_lines, load_labels

e51._M = Path("data/interim/stage54/metrics")
e51._GEN = Path("data/interim/stage54/gen")
e51.SETTINGS = {"equal": "1 最近傍(現行)", "indep": "2 周辺密度 + 距離(ステージ52)",
                "cond": "3 条件付き密度 + 距離(本ステージ)"}
e51.BASE = "equal"
e51.BS_ORDER = ("cond",)
e51.PAIRS_F2 = (("cond", "indep", "条件付け の効果(3 対 2)"),)
e51.STAGE = "ステージ54"
e51.TITLE = "ステージ54 C.2: 条件付き密度 + 距離の対応先と現行の比較"
e51.OUT = {"comparison": "comparison.md", "losses": "losses.md", "assets": "assets_extreme.md",
           "bs": "bs_e51.md", "summary": "summary_e.json"}
OUT = Path("results/stage54")
PRIMARY = ("純資産の改善", "解消率", "順位相関")


def boot_power(diffs: np.ndarray, n: int, rng: np.random.Generator, outer: int = 2000,
               inner: int = 1000) -> float:
    """``diffs`` から n 社を復元抽出したとき、中央値の 95% ブートストラップ区間が 0 を含まず正"
      "の割合。"""
    hits = 0
    for _ in range(outer):
        s = rng.choice(diffs, size=n, replace=True)
        meds = np.median(s[rng.integers(0, n, size=(inner, n))], axis=1)
        hits += int(np.percentile(meds, 2.5) > 0)
    return hits / outer


def power_and_judgment(names: tuple[str, ...]) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    ctx = {c: e51.Ctx(c, names) for c in ("c2", "c1")}
    comp = {c: ctx[c].compare("cond", "equal", rng) for c in ("c2", "c1")}
    viol = {c: sum(ctx[c].viol_items("cond").values()) for c in ("c2", "c1")}
    ok1, sym1 = e51.wins(comp["c1"], viol["c1"])
    ok2, sym2 = e51.wins(comp["c2"], viol["c2"])
    # C.3 検定力: 純資産の改善
    def firm_diff(c: str) -> np.ndarray:
        g = ctx[c].comp.set_index(["version", "docID"])["net_asset_improvement"]
        a, b = g.loc["cond"], g.loc["equal"].reindex(g.loc["cond"].index)
        out: np.ndarray = (a - b).dropna().to_numpy()
        return out

    d1, d2 = firm_diff("c1"), firm_diff("c2")
    shifted = d2 - np.median(d2) + np.median(d1)
    pw_shift = boot_power(shifted, 55, rng)
    pw_c1 = boot_power(d1, 55, rng)
    half = (comp["c2"]["順位相関"][2] - comp["c2"]["順位相関"][1]) / 2
    corr1 = comp["c1"]["順位相関"][0]
    lines = ["# ステージ54 C.3: class 2(n = 55)の検定力", "",
             "設定 3(条件付き)− 設定 1(現行)。`criteria.md` §4 の定義。", "",
             "| 量 | 値 |", "|---|---:|",
             f"| class 1 の純資産の改善の差(中央値) | {np.median(d1):+.4f} |",
             f"| class 2 の純資産の改善の差(中央値 [95% 区間]) |"
             f" {comp['c2']['純資産の改善'][0]:+.4f} "
             f"[{comp['c2']['純資産の改善'][1]:+.4f}, {comp['c2']['純資産の改善'][2]:+.4f}] |",
             f"| **検定力**: class 2 の差の分布を class 1 の中央値へ平行移動、n = 55 | *"
             f"*{pw_shift:.2f}** |",
             f"| 参考: class 1 の差の分布から n = 55 を抜いた検定力 | {pw_c1:.2f} |",
             f"| class 1 の順位相関の差(点推定) | {corr1:+.3f} |",
             f"| class 2 の順位相関の差の 95% 区間の半幅 | {half:.3f} |",
             f"| 順位相関: class 1 の大きさを n = 55 で検出できるか | "
             f"{'できる(半幅 < 点推定)' if half < corr1 else 'できない(半幅 ≥ 点推定)'} |", ""]
    # 検出力が不足するか: class 1 で有意に優れた主指標のうち 1 つでも検出できない
    sig1 = [k for k in PRIMARY if comp["c1"][k][3] == 1]
    under = []
    for k in sig1:
        if k == "純資産の改善" and pw_shift < 0.8:
            under.append(k)
        if k == "順位相関" and half >= corr1:
            under.append(k)
        if k == "解消率":
            under.append(k)  # 債務超過 12 社では検定力を計算しない(ほぼ常に不足)
    underpowered = bool(under)
    # 事前に決めた判断(criteria.md §4)
    cond_ok = (ok1 and viol["c2"] == 0 and not any(comp["c2"][k][3] == -1 for k in PRIMARY)
               and all(comp["c2"][k][0] > 0 for k in sig1) and underpowered)
    if ok1 and ok2:
        verdict = "採用(両 class で基準を満たす)"
    elif cond_ok:
        verdict = "採用(class 2 では検出できない、と明記)"
    else:
        verdict = "採用しない"
    jl = ["# ステージ54 D: 判定", "",
          "| class | 違反 | 判定の記号(純資産の改善・解消率・順位相関) | 基準 1・2 |", "|---|---"
                                                              ":|---|---|",
          f"| class 2 | {viol['c2']} | {sym2} | {'満たす' if ok2 else '満たさない'} |",
          f"| class 1 | {viol['c1']} | {sym1} | {'満たす' if ok1 else '満たさない'} |", "",
          f"- class 1 で有意に優れた主指標: {', '.join(sig1) or 'なし'}",
          f"- class 2 で検出力が不足する主指標: {', '.join(under) or 'なし'}",
          "- 検出力が不足する場合の事前の条件: (1) class 1 で基準 1・2、(2) class 2 で違反 0、"
          "(3) class 2 で有意に劣る主指標なし、(4) class 1 で有意な各主指標の class 2 の点推定"
          "が正、"
          f"(5) 検出力が不足 → {'すべて満たす' if cond_ok else '満たさない'}",
          f"  - (1) {ok1}、(2) {viol['c2'] == 0}、(3)"
          f" {not any(comp['c2'][k][3] == -1 for k in PRIMARY)}、"
          f"(4) {all(comp['c2'][k][0] > 0 for k in sig1)}、(5) {underpowered}", "",
          f"**判定: {verdict}**", ""]
    (OUT / "power.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "judgment.md").write_text("\n".join(jl) + "\n", encoding="utf-8")
    print("\n".join(lines + jl))
    return {"verdict": verdict, "power_na": pw_shift, "power_na_c1dist": pw_c1, "corr_half": half,
            "corr_c1": corr1, "underpowered": under, "ok_c1": ok1, "ok_c2": ok2}


def bs(names: tuple[str, ...]) -> None:
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    labels = load_labels(Path("conf/accounts/labels_ja.yaml"))
    gens = {}
    for s in ("equal", "indep", "cond"):
        (f,) = sorted((e51._GEN / "c2" / s).glob("*__seed0.parquet"))
        g = pd.read_parquet(f)
        gens[s] = g[g["method"] == "PCFM"].set_index("docID")
    base = gens["equal"]
    pool = sorted(set(base.index) - set(e51.EXCL["c2"]))
    x, label, docs_all, _k = load_tilde(names)  # 始点の実データ(PCFM の x(0) は射影の解で 0 がない)
    its = names.index("jppfs_cor_TreasuryStock")
    zero_ts = {d for d, r in zip(docs_all, x, strict=True) if r[its] == 0}
    ts0 = [d for d in pool if d in zero_ts]
    rest = [d for d in pool if d not in ts0]
    rng = np.random.default_rng(0)
    docs = ([str(d) for d in rng.choice(ts0, 2, replace=False)]
            + [str(d) for d in rng.choice(rest, 3, replace=False)])
    x0c, x1c = [f"x0_{n}" for n in names], [f"x1_{n}" for n in names]
    out = ["# ステージ54 C.4: BS 表(class 2、seed 0、百万円、t = 1 は 5 年)", "",
           "企業は乱数(seed 0)で、自己株式を持たない企業(始点の実データでちょうど 0)から 2 社、"
           "その他から 3 社。列は始点、現行、ステージ52(周辺)、本ステージ(条件付き)。"
           "自己株式は不活性制約(規則 M)で始点の値に固定されるので、どの設定でも 0 のまま。", ""]
    for d in docs:
        r = base.loc[d]
        cols = [("始点", r[x0c].to_numpy(float)), ("現行", r[x1c].to_numpy(float)),
                ("周辺(52)", gens["indep"].loc[d][x1c].to_numpy(float)),
                ("条件付き(54)", gens["cond"].loc[d][x1c].to_numpy(float))]
        bl = build_bs_lines(spec, names, [v * float(r["assets_start"]) for _, v in cols], labels)
        out += [f"## {r['filerName']}({d}){'、自己株式なし' if d in ts0 else ''}", "",
                "| 科目 | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---:|" * len(cols)]
        for line in bl:
            if line.values is None:
                out.append(f"| **{line.label}** |" + " |" * len(cols))
                continue
            lab = "&nbsp;&nbsp;" * line.indent + line.label
            bold = "**" if line.kind in ("subtotal", "total", "diff") else ""
            out.append(f"| {bold}{lab}{bold} | "
                       + " | ".join(f"{v / 1e6:,.0f}" for v in line.values) + " |")
        out.append("")
    (OUT / "bs.md").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    sys.argv = [sys.argv[0], "--out", str(OUT)]
    e51.main()
    res = power_and_judgment(names)
    bs(names)
    (OUT / "summary_d.json").write_text(json.dumps(res, ensure_ascii=False, indent=2,
                                                   default=float), encoding="utf-8")
