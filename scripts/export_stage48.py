#!/usr/bin/env python
"""ステージ48: 最終設定(ステージ47、seed 0)の生成結果を全件出力し、経路を類型化する。

新しい生成は行わない(ステージ47 の出力を整形する)。生成は Euler 法 1 ステップ(``_TS = (0, 1)``)で、
モデルの出力は始点(t = 0)と終点(t = 1)だけ。t = 0.25・0.5・0.75 の値は**始点と終点の線形補間**
(1 ステップの Euler では経路は直線)。補間した点も 15 項目のチェックを満たすことを確かめる。

出力(``results/stage48/``): ``bs_class2/<docID>.md``、``bs_class1/<docID>.md``、``index.md``、
``all_paths.csv``、``typology.md``、``summary.json``。類型の閾値は ``criteria.md``(分類の前に記録)。

    uv run python scripts/export_stage48.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_stage34 import special_reason

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import (
    CHECK_NAMES,
    EXTRA_CHECK_NAMES,
    decompose_re_change,
    evaluate_checks,
    evaluate_extra_checks,
    subtotal_members,
)
from fsgen.models.industry import load_industry_config
from fsgen.viz.balance_sheet import build_bs_lines, load_labels

_GEN = Path("data/interim/stage47/gen")
_TARGETS = Path("data/interim/stage28/plans_equal/full_targets_nearest_na0.csv")
_PANEL = Path("data/interim/stage23/panel/panel.csv")
OUTLIERS = {"c2": ("S100TSS9", "S100TWFQ"), "c1": ("S100VGDJ",)}
TS = (0.0, 0.25, 0.5, 0.75, 1.0)
_RE, _CS, _CSUR = ("jppfs_cor_RetainedEarnings", "jppfs_cor_CapitalStock",
                   "jppfs_cor_CapitalSurplus")
# 類型の閾値(criteria.md、分類の前に記録)
NA_THR, AL_THR, RE_THR = 0.01, 0.05, 0.01


def load(cls: str, seed: int) -> pd.DataFrame:
    g = pd.read_parquet(_GEN / cls / "final" / f"nearest_na0__seed{seed}.parquet")
    return g[(g["method"] == "PCFM") & ~g["docID"].isin(OUTLIERS[cls])].set_index("docID")


def classify(x0: np.ndarray, x1: np.ndarray, names: tuple[str, ...],
             m: dict[str, np.ndarray]) -> dict[str, str]:
    """4 軸の区分(x0, x1 は同じ尺度。始点の総資産で割って比べる)。"""
    a0 = float(x0 @ m["jppfs_cor_Assets"])
    na0, na1 = float(x0 @ m["jppfs_cor_NetAssets"]), float(x1 @ m["jppfs_cor_NetAssets"])
    dna = (na1 - na0) / a0
    if na0 < 0 < na1:
        na = "債務超過を解消"
    elif dna > NA_THR:
        na = "改善(解消せず)"
    elif dna < -NA_THR:
        na = "悪化"
    else:
        na = "ほぼ変わらず"

    def three(d: float) -> str:
        return "増える" if d > AL_THR else ("減る" if d < -AL_THR else "ほぼ変わらず")

    da = (float(x1 @ m["jppfs_cor_Assets"]) - a0) / a0
    dl = float((x1 - x0) @ m["jppfs_cor_Liabilities"]) / a0
    transfer, _profit = decompose_re_change(x0, x1, names)
    dre = float(x1[names.index(_RE)] - x0[names.index(_RE)]) / a0
    tr = float(transfer) / a0
    if dre <= RE_THR:
        re_ = "増加なし"
    elif tr / dre >= 2 / 3:
        re_ = "振替が主"
    elif tr / dre <= 1 / 3:
        re_ = "利益が主"
    else:
        re_ = "両方"
    return {"純資産": na, "資産": three(da), "負債": three(dl), "利益剰余金": re_}


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage48"))
    args = ap.parse_args()
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    m = subtotal_members(spec, names)
    labels = load_labels(Path("conf/accounts/labels_ja.yaml"))
    ind = load_industry_config(Path("conf/accounts/industry.yaml"))
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    edinet = dict(zip(agg["docID"], agg["edinetCode"].astype(str), strict=True))

    def industry(doc: str) -> str:
        return ind.labels([edinet.get(doc, "")])[0]

    tg = pd.read_csv(_TARGETS, dtype={"source_docID": str, "target_docID": str}).set_index(
        "source_docID")
    x0c = [f"x0_{n}" for n in names]
    x1c = [f"x1_{n}" for n in names]
    gens = {cls: {s: load(cls, s) for s in (0, 1, 2)} for cls in ("c2", "c1")}
    # 債務超過の解消(3 seed の 2/3 以上)
    resolved: dict[str, set[str]] = {}
    for cls, gs in gens.items():
        cnt: dict[str, int] = {}
        for g in gs.values():
            na0 = g[x0c].to_numpy() @ m["jppfs_cor_NetAssets"]
            na1 = g[x1c].to_numpy() @ m["jppfs_cor_NetAssets"]
            for d, a, b in zip(g.index, na0, na1, strict=True):
                cnt[d] = cnt.get(d, 0) + int(a < 0 < b)
        resolved[cls] = {d for d, k in cnt.items() if k >= 2}

    csv_rows: list[dict[str, object]] = []
    index_lines = ["# ステージ48: 生成結果の索引(最終設定、seed 0)", "",
                   "**t = 0.25・0.5・0.75 の値は始点と終点の線形補間**(生成は Euler 法 1 ステッ"
                   "プで、モデルの出力は t = 0 と "
                   "t = 1 だけ)。★ = 債務超過を解消(3 seed の 2/3 以上。class 2 で 8 社)。純資産"
                   " / 総資産は始点の総資産で割った値。", ""]
    n_files = {}
    check_viol = 0
    for cls, sub in (("c2", "bs_class2"), ("c1", "bs_class1")):
        g = gens[cls][0]
        out_dir = args.out / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        index_lines += [f"## {'class 2' if cls == 'c2' else 'class 1'}({len(g)} 社)", "",
                        "| docID | 企業名 | 業種 | 始点の純資産 / 総資産 | 終点の純資産 / 総資産 "
                        "| 債務超過を解消 | seed 0 で解消 | 対応先 |",
                        "|---|---|---|---:|---:|---|---|---|"]
        for doc in sorted(g.index):
            r = g.loc[doc]
            a0 = float(r["assets_start"])
            x0 = r[x0c].to_numpy(dtype=float)
            x1 = r[x1c].to_numpy(dtype=float)
            xs = [x0 + t * (x1 - x0) for t in TS]
            # 補間した点のチェック(始点との比較)
            x0s, xts = np.stack([x0] * len(TS)), np.stack(xs)
            fl = np.column_stack([evaluate_checks(x0s, xts, spec, names),
                                  evaluate_extra_checks(xts, x0s, spec, names)])
            check_viol += int(fl.any(axis=1).sum())
            na = [float(x @ m["jppfs_cor_NetAssets"]) for x in xs]
            if cls == "c2" and doc in tg.index:
                t_ = tg.loc[doc]
                target = (f"{t_['target_name']}(docID {t_['target_docID']}、業種 "
                          f"{industry(str(t_['target_docID']))}、対応先の総資産 / 始点の総資産 "
                          f"{float(t_['asset_ratio']):.2f})")
                target_short = str(t_["target_name"])
            else:
                target = ("なし(class 1 は輸送計画を持たない。class 2 → class 0 で学習した速度"
                          "場を当てる)"
                          if cls == "c1" else "—")
                target_short = "—"
            star = "★" if doc in resolved[cls] else ""
            seed0 = "○" if na[0] < 0 < na[-1] else ""
            index_lines.append(
                f"| [{doc}]({sub}/{doc}.md) | {r['filerName']} | {industry(doc)} "
                f"| {na[0]:+.3f} | {na[-1]:+.3f} | {star} | {seed0} | {target_short} |")
            # BS 表
            bl = build_bs_lines(spec, names, [x * a0 for x in xs], labels)
            cls_name = "class 2" if cls == "c2" else "class 1"
            md = [f"# {r['filerName']}(docID: {doc}、{cls_name})", "",
                  f"- edinetCode: {edinet.get(doc, '')}、業種: {industry(doc)}",
                  f"- 対応先: {target}",
                  "- 最終設定(ステージ47、`s47_final`)、seed 0。単位 百万円。t = 1 は 5 年。",
                  "- **t = 0.25・0.5・0.75 は始点と終点の線形補間**"
                  "(生成は Euler 法 1 ステップ)。", "",
                  "| 科目 | " + " | ".join(f"t = {t:g}" for t in TS) + " |",
                  "|---|" + "---:|" * len(TS)]
            for line in bl:
                if line.values is None:
                    md.append(f"| **{line.label}** |" + " |" * len(TS))
                    continue
                lab = "&nbsp;&nbsp;" * line.indent + line.label
                bold = "**" if line.kind in ("subtotal", "total", "diff") else ""
                md.append(f"| {bold}{lab}{bold} | "
                          + " | ".join(f"{v / 1e6:,.0f}" for v in line.values) + " |")
            tr = [float(decompose_re_change(x0, x, names)[0]) * a0 for x in xs]
            pf = [float(decompose_re_change(x0, x, names)[1]) * a0 for x in xs]
            dpaid = [float((x - x0)[names.index(_CS)] + (x - x0)[names.index(_CSUR)]) * a0
                     for x in xs]
            dna = [(v - na[0]) * a0 for v in na]
            md += ["", "**内訳(始点からの累積、百万円)**", "",
                   "| 項目 | " + " | ".join(f"t = {t:g}" for t in TS) + " |",
                   "|---|" + "---:|" * len(TS)]
            for lab, vals in (("ΔRE のうち振替(払込資本の減少)", tr), ("ΔRE のうち利益部分", pf),
                              ("Δ払込資本(資本金 + 資本剰余金)", dpaid), ("Δ純資産", dna)):
                md.append(f"| {lab} | " + " | ".join(f"{v / 1e6:,.0f}" for v in vals) + " |")
            viol = [c for c, f in zip([*CHECK_NAMES, *EXTRA_CHECK_NAMES], fl.any(axis=0),
                                      strict=True) if f]
            md += ["", f"15 項目のチェック(5 時点): {'、'.join(viol) or '違反なし'}。", ""]
            (out_dir / f"{doc}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
            for t, x, tr_, pf_ in zip(TS, xs, tr, pf, strict=True):
                row: dict[str, object] = {"docID": doc, "企業名": r["filerName"],
                                          "class": 2 if cls == "c2" else 1, "t": t}
                for j, n in enumerate(names):
                    row[n] = x[j] * a0 / 1e6
                ta = float(x @ m["jppfs_cor_Assets"]) * a0 / 1e6
                tl = float(x @ m["jppfs_cor_Liabilities"]) * a0 / 1e6
                tn = float(x @ m["jppfs_cor_NetAssets"]) * a0 / 1e6
                row.update({"資産合計": ta, "負債合計": tl, "純資産合計": tn,
                            "貸借差額": ta - tl - tn, "振替": tr_ / 1e6, "利益部分": pf_ / 1e6})
                csv_rows.append(row)
        n_files[sub] = len(list(out_dir.glob("*.md")))
        index_lines.append("")
    pd.DataFrame(csv_rows).to_csv(args.out / "all_paths.csv", index=False)
    (args.out / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    # ---------------- 類型化 ----------------
    axes = ("純資産", "資産", "負債", "利益剰余金")
    cats = {"純資産": ("債務超過を解消", "改善(解消せず)", "ほぼ変わらず", "悪化"),
            "資産": ("増える", "ほぼ変わらず", "減る"), "負債": ("増える", "ほぼ変わらず", "減る"),
            "利益剰余金": ("振替が主", "両方", "利益が主", "増加なし")}

    def classify_gen(g: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({d: classify(g.loc[d, x0c].to_numpy(dtype=float),
                                         g.loc[d, x1c].to_numpy(dtype=float), names, m)
                             for d in g.index}).T

    typ = {cls: {s: classify_gen(gens[cls][s]) for s in (0, 1, 2)} for cls in ("c2", "c1")}
    # 実データ
    fy = pd.read_csv("data/interim/stage30/firm_years.csv", dtype={"edinetCode": str})
    wl = fy.pivot(index="edinetCode", columns="year", values="label")
    rf = pd.read_csv("results/stage30/recovery_firms.csv", dtype={"edinetCode": str})
    rec2 = rf[(rf["reason"].fillna("") == "") & rf["in_panel"].astype(bool) & (rf["group"] == "Y")]
    c1 = wl[wl[2022] == 1]
    reason = special_reason(fy, c1.index.tolist())
    pan = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    a23 = pan[pan["t"] == 2023].set_index("edinetCode")
    b24 = pan[pan["t"] == 2024].set_index("edinetCode")
    rec1_codes = [c for c in c1.index if reason.get(c, "") == "" and c in a23.index
                  and c in b24.index and wl.loc[c, 2024] == 0]

    def classify_real(codes: list[str]) -> pd.DataFrame:
        out = {}
        for c in codes:
            if c not in a23.index or c not in b24.index:
                continue
            x0 = a23.loc[c, [f"prev_{n}" for n in names]].fillna(0).to_numpy(dtype=float)
            x1 = b24.loc[c, [f"cur_{n}" for n in names]].fillna(0).to_numpy(dtype=float)
            out[c] = classify(x0, x1, names, m)
        return pd.DataFrame(out).T

    real2 = classify_real(list(rec2["edinetCode"]))
    real1 = classify_real(rec1_codes)
    t_lines = ["# ステージ48 C: 経路の類型化", "",
               "閾値は `criteria.md`(分類の前に記録)。生成は最終設定の seed 0(5 年)、実データは"
               "実際に再建した企業の 2 年"
               "(2022 → 2024 年度末)。期間が違う(5 年と 2 年)ので、生成のほうが大きく動く方向に"
               "偏りうる。", "",
               "## C.1・C.3 軸ごとの分布", "",
               f"| 軸 | 区分 | 生成 class 2({len(typ['c2'][0])} 社) | 生成 class 1"
               f"({len(typ['c1'][0])} 社) "
               f"| 実データ class 2 → 注記なし({len(real2)} 社) | 実データ class 1 → class 0"
               f"({len(real1)} 社) |",
               "|---|---|---:|---:|---:|---:|"]
    for ax in axes:
        for c in cats[ax]:
            cells = []
            for df in (typ["c2"][0], typ["c1"][0], real2, real1):
                k = int((df[ax] == c).sum())
                cells.append(f"{k}({k / len(df):.0%})")
            t_lines.append(f"| {ax} | {c} | " + " | ".join(cells) + " |")
    # 組み合わせ
    def combo(df: pd.DataFrame) -> pd.Series:
        out: pd.Series = df[list(axes)].agg(" / ".join, axis=1)
        return out

    cb2 = combo(typ["c2"][0])
    vc = cb2.value_counts()
    t_lines += ["", "## 類型(4 軸の組み合わせ、生成 class 2)", "",
                "純資産 / 資産 / 負債 / 利益剰余金 の順。", "",
                "| 類型 | 生成 class 2 | 生成 class 1 | 実データ 15 社 | 実データ 57 社 |",
                "|---|---:|---:|---:|---:|"]
    cb1, cr2, cr1 = combo(typ["c1"][0]), combo(real2), combo(real1)
    for k in sorted(set(vc.index) | set(cr2) | set(cr1),
                    key=lambda x: (-int((cb2 == x).sum()), -int((cr2 == x).sum()), x)):
        t_lines.append(f"| {k} | {int((cb2 == k).sum())} | {int((cb1 == k).sum())} "
                       f"| {int((cr2 == k).sum())} | {int((cr1 == k).sum())} |")
    # 頑健性
    same = {cls: float(np.mean([(combo(typ[cls][0]) == combo(typ[cls][s]).reindex(
        typ[cls][0].index)).mean() for s in (1, 2)])) for cls in ("c2", "c1")}
    t_lines += ["", f"頑健性: seed 0 と同じ類型になる企業の割合(seed 1・2 の平均): class 2"
                    f" {same['c2']:.0%}、"
                f"class 1 {same['c1']:.0%}。", ""]
    # 代表例
    rng = np.random.default_rng(0)
    reps: list[tuple[str, str]] = []
    for k in vc.index[:4]:
        docs = sorted(cb2[cb2 == k].index)
        reps.append((k, str(rng.choice(docs))))
    reps.append((str(cb2.get("S100VGW5", "")), "S100VGW5"))
    g0 = gens["c2"][0]
    t_lines += ["## C.2 代表例(社数の多い 4 類型から乱数 seed 0 で 1 社、とジェイホールディング"
                "ス)", "",
                "| 類型 | 企業 | Δ現金 | Δ有利子負債 | Δ払込資本 | ΔRE(振替 / 利益) | Δ純資産 | "
                "Δ総資産 |",
                "|---|---|---:|---:|---:|---|---:|---:|"]
    debt = [n for n in ("short_term_loans", "long_term_loans", "bonds_current", "bonds_noncurrent")
            if n in names]
    for typ_k, doc in reps:
        r = g0.loc[doc]
        a0 = float(r["assets_start"]) / 1e6
        x0 = r[x0c].to_numpy(dtype=float)
        x1 = r[x1c].to_numpy(dtype=float)
        d = x1 - x0
        trf, prf = decompose_re_change(x0, x1, names)
        t_lines.append(
            f"| {typ_k} | [{r['filerName']}](bs_class2/{doc}.md) "
            f"| {d[names.index('jppfs_cor_CashAndDeposits')] * a0:+,.0f} "
            f"| {sum(d[names.index(n)] for n in debt) * a0:+,.0f} "
            f"| {(d[names.index(_CS)] + d[names.index(_CSUR)]) * a0:+,.0f} "
            f"| {d[names.index(_RE)] * a0:+,.0f}({float(trf) * a0:,.0f} / {float(prf) * a0:+,.0f}) "
            f"| {float(d @ m['jppfs_cor_NetAssets']) * a0:+,.0f} "
            f"| {float(d @ m['jppfs_cor_Assets']) * a0:+,.0f} |")
    t_lines += ["", "(百万円、t = 0 → 1)", ""]
    (args.out / "typology.md").write_text("\n".join(t_lines) + "\n", encoding="utf-8")
    summary = {"files": n_files, "csv_rows": len(csv_rows),
               "interpolated_check_violations": check_viol,
               "resolved_c2": sorted(resolved["c2"]), "robustness": same,
               "representatives": reps}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n".join(t_lines))


if __name__ == "__main__":
    main()
