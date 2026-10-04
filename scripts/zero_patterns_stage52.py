#!/usr/bin/env python
"""ステージ52 A: 健全企業のゼロパターン(どの科目がちょうど 0 か)の集計。生成なし。

定義と判定規則は ``results/stage52/criteria_a.md``(集計の前に記録)。母集団の主は生成の目標の
候補(非金融・ステージ28 の除外後の class 0、3,768 書類)、参考は除外前の 3,951 書類。科目は
基礎科目 32(主)と、4 つの丸め差の残余を除く 28(参考)。

出力(``results/stage52/``): ``patterns.md``(A.1・A.4)、``zero_rates.md``(A.2)、
``co_zero.md``(A.3)、``patterns.csv``、``summary_a.json``。

    uv run python scripts/zero_patterns_stage52.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.datasets.exclusion import exclusion_masks, load_excluded_docs

_AGG = Path("data/interim/values/aggregated_core.csv")
_LABELS = Path("data/interim/labels/labels.csv")
_EXCLUDE = Path("data/interim/stage28/excluded.csv")
ROUNDING = ("__OTHER__jppfs_cor_Assets", "__OTHER__jppfs_cor_NoncurrentAssets",
            "__OTHER__jppfs_cor_Liabilities", "__OTHER__jppfs_cor_ShareholdersEquity")
TOP = 30
COVER_THR = 0.80


def load(names: tuple[str, ...], exclude: bool) -> tuple[np.ndarray, np.ndarray]:
    agg = pd.read_csv(_AGG, dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    lab = pd.read_csv(_LABELS, dtype={"docID": str})
    lab = lab[pd.to_numeric(lab["label"], errors="coerce").notna()]
    m = agg.merge(lab[["docID", "label"]], on="docID", how="inner")
    m = m[m["total_assets"].notna() & (m["total_assets"] != 0)]
    if exclude:
        keep, _ = exclusion_masks(m["docID"].tolist(), m["label"].astype(int).tolist(),
                                  load_excluded_docs(_EXCLUDE))
        m = m[keep]
    x = m[list(names)].fillna(0.0).to_numpy(dtype=float) / m["total_assets"].to_numpy()[:, None]
    return x, m["label"].astype(int).to_numpy()


def pattern_stats(z: np.ndarray, cols: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """z: (n, 科目) の bool(True = 0)。パターンの表と要約。"""
    keys = [tuple(c for c, f in zip(cols, row, strict=True) if f) for row in z]
    cnt = Counter(keys)
    n = len(keys)
    tab = pd.DataFrame([(k, v) for k, v in cnt.most_common()], columns=["pattern", "n"])
    tab["share"] = tab["n"] / n
    tab["cum"] = tab["share"].cumsum()
    m80 = int(np.searchsorted(tab["cum"].to_numpy(), COVER_THR) + 1)
    summ = {"n": n, "n_patterns": len(tab), "top_share": float(tab["share"].iloc[0]),
            "top10_cover": float(tab["cum"].iloc[min(9, len(tab) - 1)]),
            "zero_free_share": float(cnt.get((), 0) / n), "m_for_80": m80,
            "singletons": int((tab["n"] == 1).sum()),
            "mean_zeros": float(z.sum(axis=1).mean())}
    return tab, summ


def independent_cover(z: np.ndarray, rng: np.random.Generator,
                      reps: int = 20) -> tuple[float, float]:
    """科目ごとに独立なゼロ確率を仮定して同じ社数を生成したときの、上位 10 の被覆と異なるパター"
      "ン数。"""
    p = z.mean(axis=0)
    cover, npat = [], []
    for _ in range(reps):
        sim = rng.random(z.shape) < p
        cnt = Counter(map(tuple, sim))
        top = sorted(cnt.values(), reverse=True)
        cover.append(sum(top[:10]) / len(sim))
        npat.append(len(cnt))
    return float(np.mean(cover)), float(np.mean(npat))


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage52"))
    args = ap.parse_args()
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    x, label = load(names, exclude=True)
    x_all, label_all = load(names, exclude=False)
    z0 = x[label == 0] == 0
    rng = np.random.default_rng(0)
    summary: dict[str, Any] = {}

    # A.1・A.4
    lines = ["# ステージ52 A.1・A.4: 健全企業(class 0)のゼロパターン", "",
             "ゼロ = 総資産比の値がちょうど 0。パターン = 0 である科目の集合。定義は `criteria_a"
             ".md`。", ""]
    sets = {"32": list(names), "28": [n for n in names if n not in ROUNDING]}
    for key, cols in sets.items():
        idx = [names.index(c) for c in cols]
        tab, s = pattern_stats(z0[:, idx], cols)
        icov, inp = independent_cover(z0[:, idx], rng)
        s |= {"indep_top10_cover": icov, "indep_n_patterns": inp}
        _, s_all = pattern_stats((x_all[label_all == 0] == 0)[:, idx], cols)
        summary[f"accounts{key}"] = s | {"all3951": s_all}
        lab = "32 科目(主)" if key == "32" else "28 科目(丸め差の残余 4 つを除く、参考)"
        lines += [f"## {lab}", "",
                  "| 項目 | class 0(3,768、主) | 参考: 除外前(3,951) | 参考: 科目ごとに独立と仮"
                  "定(20 回の平均) |",
                  "|---|---:|---:|---:|",
                  f"| 書類数 | {s['n']:,} | {s_all['n']:,} | — |",
                  f"| 異なるパターンの総数 | {s['n_patterns']:,} | {s_all['n_patterns']:,} |"
                  f" {inp:,.0f} |",
                  f"| 1 社だけのパターン | {s['singletons']:,} | {s_all['singletons']:,} | — |",
                  f"| 最頻パターンの割合 | {s['top_share']:.1%} | {s_all['top_share']:.1%} | — |",
                  f"| **上位 10 パターンの被覆** | **{s['top10_cover']:.1%}** |"
                  f" {s_all['top10_cover']:.1%} "
                  f"| {icov:.1%} |",
                  f"| 累積 80% に必要なパターン数 | {s['m_for_80']:,} | {s_all['m_for_80']:,} | "
                  f"— |",
                  f"| ゼロを持たない企業(全科目が非 0) | {s['zero_free_share']:.1%} "
                  f"| {s_all['zero_free_share']:.1%} | — |",
                  f"| 1 社あたりのゼロの科目数(平均) | {s['mean_zeros']:.2f} |"
                  f" {s_all['mean_zeros']:.2f} | — |",
                  "", f"### 上位 {TOP} パターン", "",
                  "| 順位 | 0 の科目 | 社数 | 割合 | 累積 |", "|---:|---|---:|---:|---:|"]
        for r, row in enumerate(tab.head(TOP).itertuples(), 1):
            pat = "、".join(ja.get(c, c) for c in row.pattern) or "(なし: 全科目が非 0)"
            lines.append(f"| {r} | {pat} | {row.n} | {row.share:.1%} | {row.cum:.1%} |")
        lines.append("")
        if key == "32":
            tab.assign(pattern=["|".join(p) for p in tab["pattern"]]).to_csv(
                args.out / "patterns.csv", index=False)
    s32, s28 = summary["accounts32"], summary["accounts28"]
    ok32, ok28 = s32["top10_cover"] >= COVER_THR, s28["top10_cover"] >= COVER_THR
    lines += ["## A.4 判定(`criteria_a.md` §2)", "",
              f"- 32 科目: 上位 10 パターンの被覆 {s32['top10_cover']:.1%} → "
              "**" + ("パターン混合が実装可能" if ok32
                      else "パターンが分散している(科目ごとに独立なゼロ確率)") + "**",
              f"- 28 科目(参考): {s28['top10_cover']:.1%} → "
              f"{'パターン混合が実装可能' if ok28 else 'パターンが分散している'}"
              + ("" if ok32 == ok28 else "(**32 科目と判定が変わる**)"), ""]
    summary["judgment32"] = "mixture" if ok32 else "dispersed"
    summary["judgment28"] = "mixture" if ok28 else "dispersed"
    (args.out / "patterns.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # A.2
    rates = pd.DataFrame({"account": names, "ja": [ja.get(n, n) for n in names],
                          **{f"c{k}": (x[label == k] == 0).mean(axis=0) for k in (0, 1, 2)}})
    nk = {k: int((label == k).sum()) for k in (0, 1, 2)}
    lines = ["# ステージ52 A.2: 科目別のゼロ率(値がちょうど 0 の割合)", "",
             f"除外後の class 0 {nk[0]:,}、class 1 {nk[1]}、class 2 {nk[2]} 書類。区分(class 0 "
             f"で): "
             "多い ≥ 10%、中 1〜10%、少ない < 1%。ステージ49 の「名目額で不動」(変化しない)とは"
             "別の量。", "",
             "| 科目 | class 0 | class 1 | class 2 | 区分 |", "|---|---:|---:|---:|---|"]
    for r in rates.sort_values("c0", ascending=False).itertuples():
        cat = "多い" if r.c0 >= 0.10 else ("中" if r.c0 >= 0.01 else "少ない")
        rnd = "(丸め差の残余)" if r.account in ROUNDING else ""
        lines.append(f"| {r.ja}{rnd} | {r.c0:.1%} | {r.c1:.1%} | {r.c2:.1%} | {cat} |")
    (args.out / "zero_rates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["zero_rates"] = rates.set_index("account")[["c0", "c1", "c2"]].to_dict("index")

    # A.3
    p = z0.mean(axis=0)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if p[i] < 0.01 or p[j] < 0.01:
                continue
            both = float((z0[:, i] & z0[:, j]).mean())
            exp = float(p[i] * p[j])
            pairs.append({"a": ja.get(names[i], names[i]), "b": ja.get(names[j], names[j]),
                          "pa": p[i], "pb": p[j], "both": both, "expected": exp,
                          "ratio": both / exp,
                          "rounding": names[i] in ROUNDING or names[j] in ROUNDING})
    pr = pd.DataFrame(pairs)
    lines = ["# ステージ52 A.3: 科目間のゼロの連動(class 0)", "",
             "両方 0 の割合と、独立を仮定した期待値(各科目のゼロ率の積)。対象は class 0 のゼロ"
             "率が両方 1% "
             f"以上の科目の組({len(pr)} 組)。比 > 1 は一緒に 0 になりやすい、< 1 は排他的。", "",
             f"- 比が 2 倍以上の組: {int((pr['ratio'] >= 2).sum())} / {len(pr)}、"
             f"0.5 倍以下: {int((pr['ratio'] <= 0.5).sum())} / {len(pr)}、"
             f"0.8〜1.25 倍(ほぼ独立): {int(pr['ratio'].between(0.8, 1.25).sum())} / {len(pr)}",
             f"- 独立を仮定して同じ社数を生成すると、上位 10 パターンの被覆は "
             f"{summary['accounts32']['indep_top10_cover']:.1%}、異なるパターンは "
             f"{summary['accounts32']['indep_n_patterns']:,.0f}(実際は {s32['top10_cover']:.1%}、"
             f"{s32['n_patterns']:,})", ""]
    strong = pr.sort_values("ratio", ascending=False).head(20)
    for title, df in (("連動が強い組(比の大きい順、上位 20)", strong),
                      ("排他的な組(比の小さい順、上位 10)", pr.sort_values("ratio").head(10))):
        lines += [f"## {title}", "", "| 科目 A | 科目 B | A の 0 率 | B の 0 率 | 両方 0 | 期待"
                                     "値 | 比 |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for r in df.itertuples():
            mark = "(丸め差を含む)" if r.rounding else ""
            lines.append(f"| {r.a} | {r.b}{mark} | {r.pa:.1%} | {r.pb:.1%} | {r.both:.1%} "
                         f"| {r.expected:.1%} | {r.ratio:.2f} |")
        lines.append("")
    bond = pr[((pr["a"] == "1年内償還予定社債") & (pr["b"] == "社債"))
              | ((pr["a"] == "社債") & (pr["b"] == "1年内償還予定社債"))]
    if len(bond):
        r = bond.iloc[0]
        i, j = names.index("bonds_current"), names.index("bonds_noncurrent")
        zi, zj = z0[:, i], z0[:, j]
        phi = float(np.corrcoef(zi.astype(float), zj.astype(float))[0, 1])
        lines += [f"**社債と 1 年内償還予定社債**: 両方 0 {r['both']:.1%}、期待値"
                  f" {r['expected']:.1%}、比 {r['ratio']:.2f}。ゼロ率が高いので比は 1 に近くなる。"
                  f"P(社債 = 0 | 1 年内 = 0) = {zj[zi].mean():.1%}、"
                  f"P(社債 = 0 | 1 年内 ≠ 0) = {zj[~zi].mean():.1%}、"
                  f"ゼロの指示変数の相関(φ)= {phi:.2f}。", ""]
    (args.out / "co_zero.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pr.to_csv(args.out / "co_zero.csv", index=False)
    (args.out / "summary_a.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                        default=float), encoding="utf-8")
    print(open(args.out / "patterns.md", encoding="utf-8").read()[:6000])


if __name__ == "__main__":
    main()
