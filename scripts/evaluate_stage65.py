#!/usr/bin/env python
"""ステージ65 B・C: 最終設定と比較手法の全指標、実データとの比較、BS 表。

生成は ``data/interim/stage65/all/<cls>/<手法>__l<λ̃>``(各ステージの生成へのリンク)。事前に
``metrics_stage39.py --gen-dir data/interim/stage65/all --out data/interim/stage65/metrics`` と
``auc_stage65.py`` を実行する。class 1 の評価対象は 128 社(生成の始点 132 社から、外れ値 1 社と
再建群 72 社と重なる 3 社を除く)。

    uv run python scripts/evaluate_stage65.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import numpy as np
import pandas as pd
from compare_stage39 import corr
from evaluate_stage61 import CAL, gen_firms, loss, real_targets

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.plausibility import CHECK_NAMES, EXTRA_CHECK_NAMES, subtotal_members
from fsgen.viz.balance_sheet import build_bs_lines, load_labels

OUT = Path("results/stage65")
DATA = Path("data/interim/stage65")
LAMS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
METHODS = {"m2_l1only": "2 第 1 層のみ(恒等式・符号)", "m3_l2": "3 等重み L2 + 最近傍",
           "m4_l1s": "4 対称 L1 + 最近傍", "m5_prop": "**5 提案**",
           "m6_balarg": "6 対照: バランス OT の argmax", "m7_bary": "7 対照: バランス OT の重心"}
CHECKS = (*CHECK_NAMES, *EXTRA_CHECK_NAMES)
REAL_GROUP = {"c2": "class 2 → 回復", "c1": "class 1 → class 0"}
REAL_FIRMS = {"E33858": "和心", "E04355": "農協観光"}
_PANEL = Path("data/interim/stage23/panel/panel.csv")


def excl() -> None:
    codes = set(pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})["edinetCode"])
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    for cls in ("c2", "c1"):
        g = pd.read_parquet(sorted((e51._GEN / cls / "m5_prop__l1").glob("*__seed0.parquet"))[0])
        drop = [d for d in g["docID"].unique() if code_of.get(d, "") in codes]
        e51.EXCL[cls] = tuple(dict.fromkeys((*e51.EXCL[cls], *drop)))


def real_firm_values(cls: str, names: tuple[str, ...], members: dict[str, np.ndarray]
                     ) -> pd.DataFrame:
    """実データの企業ごとの年率の値(純資産の改善は (NA1 − NA0) / A0 / 2)。"""
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    codes = pairs[pairs["group"] == REAL_GROUP[cls]]["edinetCode"].tolist()
    a = p[p["t"] == 2022].set_index("edinetCode").loc[codes]
    b = p[p["t"] == 2024].set_index("edinetCode").loc[codes]
    x0 = a[[f"cur_{n}" for n in names]].fillna(0).to_numpy()
    x1 = b[[f"cur_{n}" for n in names]].fillna(0).to_numpy()
    a0 = a["cur_total_assets"].to_numpy()
    na0, na1 = x0 @ members["jppfs_cor_NetAssets"], x1 @ members["jppfs_cor_NetAssets"]
    paid = pd.read_csv("results/stage56/paidin.csv", dtype={"edinetCode": str})
    pg = "再建群(class 2 → 回復)" if cls == "c2" else "再建群(class 1 → class 0)"
    pd_ = paid[paid["group"] == pg].set_index("edinetCode")["annual_d_paid"].reindex(codes)
    return pd.DataFrame({
        "nai_annual": (na1 - na0) / a0 / 2.0,
        "paid_dec": (pd_ < -0.001).astype(float).where(pd_.notna()).to_numpy(),
        "re_up": ((x1 - x0)[:, names.index("jppfs_cor_RetainedEarnings")] / a0 / 2 > 0.001
                  ).astype(float),
        "n_changed": (np.abs(x1 - x0) / a0[:, None] / 2.0 > 0.002).sum(axis=1).astype(float),
        "d_assets": (b["cur_total_assets"].to_numpy() / a0 - 1.0) / 2.0}, index=codes)


def bs_table(spec: Any, names: tuple[str, ...], labels: Any, title: str,
             cols: list[tuple[str, np.ndarray]]) -> list[str]:
    bl = build_bs_lines(spec, names, [v for _, v in cols], labels)
    out = [f"## {title}", "", "| 科目 | " + " | ".join(c for c, _ in cols) + " |",
           "|---|" + "---:|" * len(cols)]
    for line in bl:
        if line.values is None:
            out.append(f"| **{line.label}** |" + " |" * len(cols))
            continue
        lab = "&nbsp;&nbsp;" * line.indent + line.label
        bold = "**" if line.kind in ("subtotal", "total", "diff") else ""
        out.append(f"| {bold}{lab}{bold} | " + " | ".join(f"{v / 1e6:,.0f}" for v in line.values)
                   + " |")
    return [*out, ""]


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    cmat = build_constraint_matrix(spec, include_subtotals=False)
    e51._M = DATA / "metrics"
    e51._GEN = DATA / "all"
    excl()
    ctx = {c: e51.Ctx(c, names) for c in ("c2", "c1")}
    rng = np.random.default_rng(0)
    star: dict[str, dict[str, float | None]] = {}
    lines = ["# ステージ65 B.2: 全指標(最終設定と比較手法)", "",
             "λ̃ は手法ごとに、違反 0 の中で較正の乖離 L(ステージ61)が最小のもの(λ̃*_0)。手法 2"
             " は提案の "
             "λ̃*_0 を使う(較正しない)。手法 1(先行研究方式)は定義と出力がこの環境にないので実"
             "施していない。"
             "違反は企業数(15 項目のどれか)。恒等式の残差 = max |Cᵀy|(総資産(0)比)。総資産は年"
             "率の中央値。"
             "評価対象: class 2 55 社、class 1 128 社。", ""]
    item_lines: list[str] = []
    for cls in ("c2", "c1"):
        r, se, n_real = real_targets(cls, names)
        w = {m: 1.0 / se[m] for m in CAL}
        auc = pd.read_csv(OUT / f"auc_{cls}.csv").set_index("method")
        star[cls] = {}
        firms: dict[str, pd.DataFrame] = {}
        for mkey in METHODS:
            if mkey == "m2_l1only":
                continue
            best: tuple[float | None, float] = (None, float("inf"))
            for lt in LAMS:
                s = f"{mkey}__l{lt:g}"
                if not (e51._GEN / cls / s).exists():
                    continue
                f = gen_firms(cls, s, names, members)
                f = f[~f.index.isin(e51.EXCL[cls])]
                nv = ctx[cls].viol[ctx[cls].viol["setting"] == s]["docID"].nunique()
                g = {"paid_dec": float(f["paid_dec"].mean()), "re_up": float(f["re_up"].mean()),
                     "d_assets": float(f["d_assets"].median())}
                lv = loss(g, r, w)
                if nv == 0 and lv < best[1]:
                    best = (lt, lv)
            star[cls][mkey] = best[0]
        star[cls]["m2_l1only"] = star[cls]["m5_prop"]
        n_eval = 0
        lines += [f"## {cls}", "",
                  "| 手法 | λ̃ | 違反(社) | 恒等式の残差 | 純資産の改善 | 解消率 | 払込資本↓ | R"
                  "E↑ | 変更科目数 "
                  "| 順位相関 | 目標の AUC | 総資産(年率) | proximity | L |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        item_lines += [f"## {cls}: 項目別の違反(企業数)", "",
                       "| 手法 | " + " | ".join(c.split("_")[0] for c in CHECKS) + " |",
                       "|---|" + "---:|" * len(CHECKS)]
        for mkey, lab in METHODS.items():
            lt_s = star[cls][mkey]
            if lt_s is None:
                lines.append(f"| {lab} | なし |" + " — |" * 12)
                continue
            s = f"{mkey}__l{lt_s:g}"
            g = e51.load_gen(cls, s)
            g = g[g["seed"] == 0]
            x1 = g[[f"x1_{n}" for n in names]].to_numpy()
            resid = float(np.abs(x1 @ cmat).max())
            f = gen_firms(cls, s, names, members)
            f = f[~f.index.isin(e51.EXCL[cls])]
            firms[mkey] = f
            n_eval = len(f)
            cr = ctx[cls].comp[ctx[cls].comp["version"] == s]
            v = ctx[cls].viol[ctx[cls].viol["setting"] == s]
            gv = {"paid_dec": float(f["paid_dec"].mean()), "re_up": float(f["re_up"].mean()),
                  "d_assets": float(f["d_assets"].median())}
            res = cr.loc[cr["insolvent_start"].astype(bool), "insolvency_resolved"].mean()
            akey = "m5_prop" if mkey == "m2_l1only" else mkey
            lines.append(f"| {lab} | {lt_s:g} | {v['docID'].nunique()} | {resid:.1e} "
                         f"| {cr['net_asset_improvement'].median():.3f} | {res:.2f} "
                         f"| {gv['paid_dec']:.0%} | {gv['re_up']:.0%} |"
                         f" {f['n_changed'].median():.0f} "
                         f"| {corr(ctx[cls].wide[s], ctx[cls].ref):+.3f} "
                         f"| {auc.loc[akey, 'auc_lgbm']:.3f} | {gv['d_assets']:+.4f} "
                         f"| {cr['proximity_l2'].median():.3f} | {loss(gv, r, w):.2f} |")
            vv = v.drop_duplicates(["docID", "check"])
            item_lines.append(f"| {lab} | " + " | ".join(
                str(int((vv["check"] == c).sum())) or "0" for c in CHECKS) + " |")
        item_lines.append("")
        lines += ["", f"(実データ: 払込資本↓ {r['paid_dec']:.0%}、RE↑ {r['re_up']:.0%}、変更科目数 "
                  f"{n_real:.0f}、総資産(年率) {r['d_assets']:+.3f}。手法 2 の目標の AUC は提案"
                  f"と同じ目標)",
                  "", "**提案(5)との企業単位のペア比較**(差 = 他の手法 − 提案。記号 = 純資産の"
                      "改善・"
                  "解消率・順位相関)", "",
                  "| 手法 | 払込資本↓ の差 [95%] | 純資産の改善の差 [95%] | 順位相関の差 [95%] |"
                  " 記号 |",
                  "|---|---|---|---|---|"]
        sp = f"m5_prop__l{star[cls]['m5_prop']:g}"
        fp = firms["m5_prop"]
        for mkey, lab in METHODS.items():
            if mkey == "m5_prop" or mkey not in firms:
                continue
            s = f"{mkey}__l{star[cls][mkey]:g}"
            fa = firms[mkey]
            dd = (fa["paid_dec"] - fp["paid_dec"].reindex(fa.index)).to_numpy()
            bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            c = ctx[cls].compare(s, sp, rng)
            sym = " ".join({1: "+", -1: "−", 0: "0"}[int(c[k][3])]
                           for k in ("純資産の改善", "解消率", "順位相関"))
            lines.append(f"| {lab} | {dd.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] "
                         f"| {e51.fc(c['純資産の改善'])} | {e51.fc(c['順位相関'])} | {sym} |")
        lines.append("")
        if cls == "c1" and n_eval != 128:
            raise ValueError(f"class 1 の評価対象が 128 社でない: {n_eval}")
        # B.3 実データとの比較
        rf = real_firm_values(cls, names, members)
        prop = firms["m5_prop"].copy()
        cp = ctx[cls].comp[ctx[cls].comp["version"] == sp].set_index("docID")
        prop["nai_annual"] = cp["net_asset_improvement"].reindex(prop.index) / 5.0
        spec_b3 = (("純資産の改善(年率、中央値)", "nai_annual", "median"),
                   ("払込資本を減らした割合", "paid_dec", "mean"),
                   ("利益剰余金が増えた割合", "re_up", "mean"),
                   ("変更された科目数(中央値)", "n_changed", "median"),
                   ("総資産変化(年率、中央値)", "d_assets", "median"))
        b3 = [f"## {cls}: 提案 対 実データ({REAL_GROUP[cls]}、{len(rf)} 社)", "",
              "差 = 提案 − 実データ。95% 区間は、生成と実データの企業を別々に復元抽出"
              "(2,000 回)。", "",
              "| 項目 | 提案 | 実データ | 差 [95%] |", "|---|---:|---:|---|"]
        for lab, col, how in spec_b3:
            a_ = prop[col].dropna().to_numpy()
            b_ = rf[col].dropna().to_numpy()
            fn = np.median if how == "median" else np.mean
            ds = [fn(a_[rng.integers(0, len(a_), len(a_))])
                  - fn(b_[rng.integers(0, len(b_), len(b_))]) for _ in range(2000)]
            lo, hi = np.percentile(ds, [2.5, 97.5])
            fmt = "{:.0%}" if how == "mean" else ("{:.0f}" if col == "n_changed" else "{:+.4f}")
            b3.append(f"| {lab} | {fmt.format(fn(a_))} | {fmt.format(fn(b_))} "
                      f"| {fn(a_) - fn(b_):+.4f} [{lo:+.4f}, {hi:+.4f}] |")
        (OUT / f"realdata_{cls}.md").write_text("\n".join([*b3, ""]) + "\n", encoding="utf-8")
    (OUT / "metrics.md").write_text("\n".join([*lines, "# 項目別の違反", "", *item_lines]) + "\n",
                                    encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({"star": star}, indent=1), encoding="utf-8")
    # C: BS 表
    labels = load_labels(Path("conf/accounts/labels_ja.yaml"))
    s3 = f"m3_l2__l{star['c2']['m3_l2']:g}"
    s5 = f"m5_prop__l{star['c2']['m5_prop']:g}"
    gens = {k: (lambda g: g[g["seed"] == 0].set_index("docID"))(e51.load_gen("c2", k))
            for k in (s3, s5)}
    g5 = gens[s5]
    x0c, x1c = [f"x0_{n}" for n in names], [f"x1_{n}" for n in names]
    na0 = g5[x0c].to_numpy() @ members["jppfs_cor_NetAssets"]
    docs = np.array(g5.index)
    r_ = np.random.default_rng(0)
    ins = docs[na0 < 0]
    pick_ins = list(r_.choice(ins, size=min(2, len(ins)), replace=False))
    rest = [d for d in docs if d not in pick_ins]
    picks = pick_ins + list(r_.choice(rest, size=3, replace=False))
    bs = ["# ステージ65 C: BS 表(class 2、百万円、生成の終点は 5 年後とみなす)", "",
          "企業は class 2(55 社)から乱数(seed 0)で選んだ。始点で債務超過の企業から 2 社、残りか"
          "ら 3 社。"
          f"列は始点、3 等重み L2 + 最近傍(λ̃ = {star['c2']['m3_l2']:g})、5 提案(λ̃ = "
          f"{star['c2']['m5_prop']:g})の終点。", ""]
    for d in picks:
        r0 = g5.loc[d]
        a0 = float(r0["assets_start"])
        cols = [("始点", r0[x0c].to_numpy(float) * a0),
                ("3 等重み L2", gens[s3].loc[d][x1c].to_numpy(float) * a0),
                ("5 提案", r0[x1c].to_numpy(float) * a0)]
        tag = "(始点で債務超過)" if d in pick_ins else ""
        bs += bs_table(spec, names, labels, f"{r0['filerName']}({d}){tag}", cols)
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    bs += ["# 実際に再建した企業(実データ、2022 → 2024 年度末)", "",
           "この 2 社の 2022 年度の書類は生成の始点に含まれないので、生成の BS はない(実データ"
           "だけを示す)。", ""]
    for code, nm in REAL_FIRMS.items():
        a = p[(p["t"] == 2022) & (p["edinetCode"] == code)].iloc[0]
        b = p[(p["t"] == 2024) & (p["edinetCode"] == code)].iloc[0]
        cols = [("2022 年度末", a[[f"cur_{n}" for n in names]].fillna(0).to_numpy(float)),
                ("2024 年度末", b[[f"cur_{n}" for n in names]].fillna(0).to_numpy(float))]
        bs += bs_table(spec, names, labels, f"{nm}({code}、class 2 → 回復)", cols)
    (OUT / "bs.md").write_text("\n".join(bs) + "\n", encoding="utf-8")
    print((OUT / "metrics.md").read_text(encoding="utf-8"))
    for cl in ("c2", "c1"):
        print((OUT / f"realdata_{cl}.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
