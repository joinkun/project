#!/usr/bin/env python
"""ステージ55 C・D: 学習した重みの最近傍と現行の比較(学習に使った 72 社を評価から除く)。

規則は ``results/stage55/criteria.md``(学習の前に記録)。``evaluate_stage51.py`` の評価と
``evaluate_stage54.py`` の検定力の関数を流用する。除外は、学習に使った 72 社と同じ EDINET コードを
持つ始点(外れ値の除外に加える)。

出力(``results/stage55/``): ``comparison.md``、``assets_extreme.md``、``bs.md``、``power.md``、
``judgment.md``、``summary_c.json``。

    uv run python scripts/evaluate_stage55.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import numpy as np
import pandas as pd
from evaluate_stage54 import PRIMARY, boot_power

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names

OUT = Path("results/stage55")


def configure() -> dict[str, int]:
    """e51 の設定をステージ55 用に書き換え、除外した社数を返す。"""
    e51._M = Path("data/interim/stage55/metrics")
    e51._GEN = Path("data/interim/stage55/gen")
    e51.SETTINGS = {"equal": "1 等重み最近傍(現行)", "learned": "2 学習した重みの最近傍"}
    e51.BASE = "equal"
    e51.BS_ORDER = ("learned",)
    e51.PAIRS_F2 = ()
    e51.STAGE = "ステージ55"
    e51.TITLE = "ステージ55 C.2: 学習した重みの最近傍と現行の比較(学習に使った企業を除く)"
    e51.OUT = {"comparison": "comparison.md", "losses": "losses.md", "assets": "assets_extreme.md",
               "bs": "bs.md", "summary": "summary_c.json"}
    pairs = pd.read_csv(OUT / "pairs.csv", dtype={"edinetCode": str})
    codes = set(pairs["edinetCode"])
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    out: dict[str, int] = {}
    for cls in ("c2", "c1"):
        g = pd.read_parquet(sorted((e51._GEN / cls / "equal").glob("*__seed0.parquet"))[0])
        docs = g["docID"].unique()
        drop = [d for d in docs if code_of.get(d, "") in codes]
        base = tuple(e51.EXCL[cls])
        n_before = len(set(docs) - set(base))
        e51.EXCL[cls] = (*base, *drop)
        out[cls] = n_before
        out[f"{cls}_dropped"] = len(set(drop) - set(base))
        out[f"{cls}_after"] = len(set(docs) - set(e51.EXCL[cls]))
    return out


def judge(names: tuple[str, ...], counts: dict[str, int]) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    ctx = {c: e51.Ctx(c, names) for c in ("c2", "c1")}
    comp = {c: ctx[c].compare("learned", "equal", rng) for c in ("c2", "c1")}
    viol = {c: sum(ctx[c].viol_items("learned").values()) for c in ("c2", "c1")}
    ok1, sym1 = e51.wins(comp["c1"], viol["c1"])
    ok2, sym2 = e51.wins(comp["c2"], viol["c2"])

    def firm_diff(c: str) -> np.ndarray:
        g = ctx[c].comp.set_index(["version", "docID"])["net_asset_improvement"]
        a, b = g.loc["learned"], g.loc["equal"].reindex(g.loc["learned"].index)
        out: np.ndarray = (a - b).dropna().to_numpy()
        return out

    d1, d2 = firm_diff("c1"), firm_diff("c2")
    n2 = len(d2)
    pw = boot_power(d2 - np.median(d2) + np.median(d1), n2, rng)
    half = (comp["c2"]["順位相関"][2] - comp["c2"]["順位相関"][1]) / 2
    corr1 = comp["c1"]["順位相関"][0]
    sig1 = [k for k in PRIMARY if comp["c1"][k][3] == 1]
    under = [k for k in sig1 if (k == "純資産の改善" and pw < 0.8)
             or (k == "順位相関" and half >= corr1) or k == "解消率"]
    cond_ok = (ok1 and viol["c2"] == 0 and not any(comp["c2"][k][3] == -1 for k in PRIMARY)
               and all(comp["c2"][k][0] > 0 for k in sig1) and bool(under))
    verdict = ("採用(両 class で基準を満たす)" if ok1 and ok2 else
               "採用(class 2 では検出できない、と明記)" if cond_ok else "採用しない")
    pl = ["# ステージ55 C.3: 検定力(学習に使った企業を除いた後)", "",
          f"評価の社数: class 2 {counts['c2']} → **{counts['c2_after']}**(除外"
          f" {counts['c2_dropped']})、"
          f"class 1 {counts['c1']} → **{counts['c1_after']}**(除外 {counts['c1_dropped']})。", "",
          "| 量 | 値 |", "|---|---:|",
          f"| class 1 の純資産の改善の差(中央値) | {np.median(d1):+.4f} |",
          f"| class 2 の純資産の改善の差(中央値 [95% 区間]) | {comp['c2']['純資産の改善'][0]:+.4f} "
          f"[{comp['c2']['純資産の改善'][1]:+.4f}, {comp['c2']['純資産の改善'][2]:+.4f}] |",
          f"| 検定力(class 2 の差の分布を class 1 の中央値へ平行移動、n = {n2}) | **{pw:.2f}** |",
          f"| class 1 の順位相関の差 | {corr1:+.3f} |",
          f"| class 2 の順位相関の差の 95% 区間の半幅 | {half:.3f} |", ""]
    jl = ["# ステージ55 D: 判定", "", "基準 0(B.4)は `accuracy.md` で満たした。", "",
          "| class | 違反 | 記号(純資産の改善・解消率・順位相関) | 基準 1・2 |", "|---|---:|---|"
                                                           "---|",
          f"| class 2 | {viol['c2']} | {sym2} | {'満たす' if ok2 else '満たさない'} |",
          f"| class 1 | {viol['c1']} | {sym1} | {'満たす' if ok1 else '満たさない'} |", "",
          f"- class 1 で有意に優れた主指標: {', '.join(sig1) or 'なし'}",
          f"- class 2 で検出力が不足する主指標: {', '.join(under) or 'なし'}",
          f"- 検出力が不足する場合の事前の条件(ステージ54 と同じ):"
          f" {'すべて満たす' if cond_ok else '満たさない'}",
          "", f"**判定: {verdict}**", "",
          "注: 順位相関の実データの参照(再建群)は、学習に使った企業と重なる"
          "(`criteria.md` §4)。", ""]
    (OUT / "power.md").write_text("\n".join(pl) + "\n", encoding="utf-8")
    (OUT / "judgment.md").write_text("\n".join(jl) + "\n", encoding="utf-8")
    print("\n".join(pl + jl))
    return {"verdict": verdict, "power": pw, "half": half, "corr1": corr1, "ok1": ok1, "ok2": ok2,
            **counts}


if __name__ == "__main__":
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    counts = configure()
    sys.argv = [sys.argv[0], "--out", str(OUT)]
    e51.main()
    res = judge(names, counts)
    (OUT / "summary_d.json").write_text(json.dumps(res, ensure_ascii=False, indent=2,
                                                   default=float), encoding="utf-8")
