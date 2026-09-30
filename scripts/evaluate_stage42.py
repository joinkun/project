#!/usr/bin/env python
"""ステージ42: 確定した最終設定(不活性制約 規則 M を含む)の全指標と実データとの比較。

新しい探索ではなく、最終的な数値の取得。入力は
``metrics_stage39.py --gen-dir data/interim/stage42/gen``
の出力(``data/interim/stage42/metrics``)と生成の parquet。``final`` が最終設定、``prev`` が
不活性制約なしの従来の最終設定(ステージ31・34 と同一の出力)。

出力: ``results/stage42/final.md``(違反・全指標・出現・実データとの比較)、``summary.json``。

    uv run python scripts/evaluate_stage42.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from compare_stage39 import agree, boot, comp_level, corr, fmt_ci
from criteria_stage31 import wilson
from evaluate_stage34 import special_reason
from persistence_tables_stage33 import spearman
from recovery_stage30 import changes
from run_stage25 import CONTINUOUS, summarize

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import CHECK_NAMES, EXTRA_CHECK_NAMES, INACTIVE_ACCOUNTS

_M = Path("data/interim/stage42/metrics")
_GEN = Path("data/interim/stage42/gen")
_PANEL = Path("data/interim/stage23/panel/panel.csv")
OUTLIERS = {"c2": ("S100TSS9", "S100TWFQ"), "c1": ("S100VGDJ",)}
CLS = {"c2": "class 2", "c1": "class 1"}
_RE, _CS, _CSUR = ("jppfs_cor_RetainedEarnings", "jppfs_cor_CapitalStock",
                   "jppfs_cor_CapitalSurplus")
GEN_YEARS, THR, ZERO, NONZERO = 5.0, 0.001, 1e-6, 1.001e-3
LABEL = {
    "net_asset_improvement": "純資産の改善", "abs_asset_change": "\\|総資産変化\\|",
    "sparsity": "sparsity", "validity_lightgbm": "validity LightGBM",
    "validity_random_forest": "validity RF", "validity_logreg": "validity LogReg",
    "proximity_l2": "proximity(等重み L2)", "debt_reduction": "有利子負債の圧縮",
    "equity_ratio_change": "自己資本比率の改善",
}


def load_gen(cls: str, setting: str) -> pd.DataFrame:
    g = pd.concat([pd.read_parquet(f) for f in
                   sorted((_GEN / cls / setting).glob("*__seed*.parquet"))])
    return g[g["method"] == "PCFM"].reset_index(drop=True)


def firm_signed(g: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
    d = pd.DataFrame({n: g[f"x1_{n}"].to_numpy() - g[f"x0_{n}"].to_numpy() for n in names})
    d["docID"] = g["docID"].to_numpy()
    return d.groupby("docID").median()


def real_account_changes(codes: list[str], names: tuple[str, ...]) -> pd.DataFrame:
    """2022 → 2024 年度末の科目の変化(年率、総資産比)。企業 × 科目。"""
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    pp = p[p["edinetCode"].isin(codes) & p["t"].isin([2023, 2024])]
    d = pd.DataFrame({n: (pp[f"cur_{n}"].fillna(0) - pp[f"prev_{n}"].fillna(0))
                      / pp["prev_total_assets"] for n in names})
    d["edinetCode"] = pp["edinetCode"].to_numpy()
    return d.groupby("edinetCode").sum() / 2.0


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage42"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    rates = pd.read_csv("results/stage41/rates.csv").set_index("account")
    checks = [*CHECK_NAMES, *EXTRA_CHECK_NAMES]
    summary: dict[str, object] = {}
    lines = ["# ステージ42: 最終設定の全指標(中間報告 第 4 版用)", "",
             "最終設定: 等重み最近傍 + W1 + PCFM + R3 + 上限 γ = 0.757 + 第 2 層の制約 + α = 0.5 "
             "+ β = 0.3"
             " + **不活性制約(規則 M、4 科目)** + 目標 baseline。seed 0/1/2。",
             f"不活性制約の対象: {'、'.join(ja[a] for a in INACTIVE_ACCOUNTS)}。", "",
             "「従来」は不活性制約なし(ステージ31・34 の最終設定と同一の出力、差 0 を確認)。"
             "評価は外れ値を除く class 2 55 社・class 1 131 社。", ""]

    # ---------------- 1. 15 項目の違反 ----------------
    lines += ["## 1. 15 項目の違反(行)", "",
              "| チェック | 最終 class 2(55 社) | 最終 class 2(57 社) | 最終 class 1(131 社) "
              "| 最終 class 1(132 社) | 従来 class 2(55) | 従来 class 1(131) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    vt: dict[tuple[str, str, bool], pd.Series] = {}
    for cls in ("c2", "c1"):
        v = pd.read_csv(_M / f"viol_{cls}.csv", dtype={"docID": str})
        for setting in ("final", "prev"):
            for excl in (True, False):
                w = v[v["setting"] == setting]
                if excl:
                    w = w[~w["docID"].isin(OUTLIERS[cls])]
                vt[(cls, setting, excl)] = w.groupby("check")["rows"].sum().reindex(
                    checks, fill_value=0)
    for c in checks:
        lines.append(f"| {c} | {vt[('c2', 'final', True)][c]} | {vt[('c2', 'final', False)][c]} "
                     f"| {vt[('c1', 'final', True)][c]} | {vt[('c1', 'final', False)][c]} "
                     f"| {vt[('c2', 'prev', True)][c]} | {vt[('c1', 'prev', True)][c]} |")
    lines.append(f"| **合計(行の延べ)** | **{vt[('c2', 'final', True)].sum()}** "
                 f"| {vt[('c2', 'final', False)].sum()} | **{vt[('c1', 'final', True)].sum()}** "
                 f"| {vt[('c1', 'final', False)].sum()} | {vt[('c2', 'prev', True)].sum()} "
                 f"| {vt[('c1', 'prev', True)].sum()} |")
    lines += ["", "- 行数は 3 seed × 企業(class 2 55 社 = 165 行、class 1 131 社 = 393 行)。"
              "57 社の違反は外れ値 2 社"
              "(オンコセラピー・サイエンス、桃太郎源)のもの(既知、HANDOFF §8)。",
              "- 従来の設定は、チェック 15(規則 M の 4 科目の出現)に class 2 で "
              f"{vt[('c2', 'prev', True)]['15_不活性科目が動いた']} / 165 行、class 1 で "
              f"{vt[('c1', 'prev', True)]['15_不活性科目が動いた']} / 393 行が掛かる"
              "(ステージ41 の「162 / 165」は、チェック 15 が規則 S の 17 科目を"
              "見ていたときの数)。", ""]
    summary["violations"] = {f"{c}_{s}_{'eval' if e else 'all'}": int(x.sum())
                             for (c, s, e), x in vt.items()}

    # ---------------- 2. 全指標 ----------------
    for cls in ("c2", "c1"):
        rows = pd.read_parquet(_M / f"rows_{cls}.parquet")
        rows = rows[~rows["docID"].isin(OUTLIERS[cls])]
        comp = comp_level(rows)
        summ, pr = summarize(comp, (), ("prev", "final"), "prev")
        s = summ.set_index("version")
        pr = pr.set_index("metric")
        dels = pd.read_parquet(_M / f"deltas_{cls}.parquet")
        dels = dels[(dels["space"] == "native") & ~dels["docID"].isin(OUTLIERS[cls])]
        refs = pd.read_parquet(_M / "refs.parquet")
        ref = refs[(refs["cls"] == cls) & (refs["space"] == "native")].set_index(
            "account")["median"]
        wv = {st: dels[dels["setting"] == st].pivot(index="docID", columns="account",
                                                     values="delta")[list(names)]
              for st in ("final", "prev")}
        cf, cp = corr(wv["final"], ref), corr(wv["prev"], ref)
        lo, hi, dlo, dhi = boot(wv["final"], ref, wv["prev"].loc[wv["final"].index], ref, rng)
        plo, phi, _, _ = boot(wv["prev"], ref, None, None, rng)
        n = int(s.loc["final", "n"])
        ref_note = ("回復 15 社の 2 年(2022 → 2024 年度末)" if cls == "c2"
                    else "class 1 → class 0 の 1 年の遷移 68 組")
        lines += [f"## 2. 全指標({CLS[cls]}、{n} 社)", "",
                  "| 区分 | 指標 | 最終設定 | 従来 | 差(最終 − 従来)[95% CI] |",
                  "|---|---|---:|---:|---|",
                  f"| 必須 | 15 項目の違反(行) | **{vt[(cls, 'final', True)].sum()}** "
                  f"| {vt[(cls, 'prev', True)].sum()} | — |"]
        res_q = pr.loc["債務超過の解消率(社ごとの 3 seed 平均)"]
        order = [("主(改善)", "net_asset_improvement"), ("主(実行可能性)", "abs_asset_change"),
                 ("主(実行可能性)", "sparsity"), ("副", "validity_lightgbm"),
                 ("副", "validity_random_forest"), ("副", "validity_logreg"),
                 ("副", "proximity_l2"), ("参考", "debt_reduction"),
                 ("参考", "equity_ratio_change")]
        labels = {m: lab for m, lab, _ in CONTINUOUS}
        for i, (kind, m) in enumerate(order):
            q = pr.loc[labels[m]]
            dg = 0 if m == "sparsity" else 3
            lines.append(f"| {kind} | {LABEL[m]} | **{s.loc['final', m]:.{dg}f}** "
                         f"| {s.loc['prev', m]:.{dg}f} "
                         f"| {fmt_ci(q['median_diff(version-equal)'], q['ci_low'], q['ci_high'])} "
                         "|")
            if i == 0:
                lines.append(
                    f"| 主(改善) | 債務超過の解消(3 seed の 2/3 以上) "
                    f"| **{int(s.loc['final', 'resolved_firms(>=2/3 seeds)'])} / "
                    f"{int(s.loc['final', 'insolvent_start'])}** "
                    f"| {int(s.loc['prev', 'resolved_firms(>=2/3 seeds)'])} / "
                    f"{int(s.loc['prev', 'insolvent_start'])} "
                    "| 解消率 " + fmt_ci(res_q["median_diff(version-equal)"], res_q["ci_low"],
                                         res_q["ci_high"]) + " |")
        lines += [f"| 副 | 符号付き順位相関(実データ: {ref_note}) "
                  f"| **{cf:+.3f}** [{lo:+.3f}, {hi:+.3f}] "
                  f"| {cp:+.3f} [{plo:+.3f}, {phi:+.3f}] | {fmt_ci(cf - cp, dlo, dhi, 3)} |",
                  f"| 副 | 符号が一致した科目 | {agree(wv['final'], ref)} / 32 "
                  f"| {agree(wv['prev'], ref)} / 32 | — |", ""]
        summary[f"metrics_{cls}"] = {
            **{m: float(s.loc["final", m]) for _, m in order},
            "resolved": int(s.loc["final", "resolved_firms(>=2/3 seeds)"]),
            "insolvent_start": int(s.loc["final", "insolvent_start"]),
            "signed_corr": cf, "signed_corr_ci": [lo, hi], "n": n,
            "violations": int(vt[(cls, "final", True)].sum()),
        }

        # ---------------- 3. 出現 ----------------
        gf, gp = load_gen(cls, "final"), load_gen(cls, "prev")
        gf = gf[~gf["docID"].isin(OUTLIERS[cls])]
        gp = gp[~gp["docID"].isin(OUTLIERS[cls])]
        lines += [f"### 始点で 0 の科目の出現({CLS[cls]}、5 年、seed × 企業の行)", "",
                  "| 科目 | 実データ 1 年 class 1+2 | 2 年 | 最終設定 | 従来 |",
                  "|---|---:|---:|---:|---:|"]
        for a in names:
            cells = []
            for g in (gf, gp):
                z = np.abs(g[f"x0_{a}"].to_numpy()) < ZERO
                e = z & (np.abs(g[f"x1_{a}"].to_numpy()) > NONZERO)
                cells.append(f"{e.sum() / z.sum():.0%}({int(e.sum())}/{int(z.sum())})"
                             if z.sum() else "—")
            if cells[0] == "—":
                continue
            rr = rates.loc[a]
            mark = "(対象)" if a in INACTIVE_ACCOUNTS else ""
            lines.append(f"| {ja[a]}{mark} | {rr['rate_c12']:.1%} | {rr['rate_2y_c12']:.1%} "
                         f"| {cells[0]} | {cells[1]} |")
        lines.append("")

    # ---------------- 4. 実データとの比較(年率) ----------------
    fy = pd.read_csv("data/interim/stage30/firm_years.csv", dtype={"edinetCode": str})
    wide = fy.pivot(index="edinetCode", columns="year", values="label")
    rf = pd.read_csv("results/stage30/recovery_firms.csv", dtype={"edinetCode": str})
    rf = rf[(rf["reason"].fillna("") == "") & rf["in_panel"].astype(bool)]
    groups: dict[str, dict[str, pd.DataFrame]] = {
        "c2": {"再建群(class 2 → 注記なし)": rf[rf["group"] == "Y"].set_index("edinetCode"),
               "非再建群(注記が残る)": rf[rf["group"] == "Z"].set_index("edinetCode")}}
    c1_22 = wide[wide[2022] == 1].copy()
    c1_22["reason"] = special_reason(fy, c1_22.index.tolist())
    c1_22 = c1_22.join(changes(_PANEL.parent, c1_22.index.tolist(), names), how="left")
    act = c1_22[(c1_22["reason"] == "") & c1_22["n_years"].notna()]
    groups["c1"] = {"class 1(2022 年度)全体": act,
                    "うち class 0 へ回復": act[wide.loc[act.index, 2024] == 0]}
    lines += ["## 3. 実データとの比較(年率、総資産比)", "",
              "生成は t = 1(5 年)を 5 で割り、実データは 2022 → 2024 年度末(2 年)を 2 で割った。"
              f"閾値 {THR}(年率)。区間は Wilson 95%。", ""]
    for cls in ("c2", "c1"):
        g = load_gen(cls, "final")
        g = g[~g["docID"].isin(OUTLIERS[cls])]
        sig = firm_signed(g, names) / GEN_YEARS
        rows = pd.read_parquet(_M / f"rows_{cls}.parquet")
        rows = rows[(rows["version"] == "final") & ~rows["docID"].isin(OUTLIERS[cls])]
        cm = rows.groupby("docID").agg(na=("net_asset_improvement", "median"),
                                       ins=("insolvent_start", "first"),
                                       res=("insolvency_resolved", "mean"))
        gna = cm["na"] / GEN_YEARS
        paid = sig[_CS] + sig[_CSUR]
        k_re = int((sig[_RE] > THR).sum())
        lo_g, hi_g = wilson(k_re, len(sig))
        heads = list(groups[cls])
        lines += [f"### {CLS[cls]}", "",
                  f"| 項目 | 生成(最終設定、{len(sig)} 社) | "
                  + " | ".join(f"実際: {h}" for h in heads) + " |",
                  "|---|---:|" + "---:|" * len(heads)]

        def cellf(d: pd.DataFrame, col: str, fn: str) -> str:
            x = d[col].dropna()
            if fn == "med":
                return f"{x.median():+.4f}"
            k = int((x > THR).sum())
            lo, hi = wilson(k, len(x))
            return f"{k}/{len(x)}({k / len(x):.0%}、[{lo:.0%}, {hi:.0%}])"

        ins_g = cm["ins"].astype(bool)
        items = [
            ("社数", str(len(sig)), [str(len(groups[cls][h])) for h in heads]),
            ("純資産の改善(中央値)", f"{gna.median():+.4f}",
             [cellf(groups[cls][h], "d_na", "med") for h in heads]),
            ("純資産が改善した社", f"{int((gna > THR).sum())}/{len(gna)}({(gna > THR).mean():.0%})",
             [cellf(groups[cls][h], "d_na", "share") for h in heads]),
            ("利益剰余金が増えた社", f"{k_re}/{len(sig)}({k_re / len(sig):.0%}、"
             f"[{lo_g:.0%}, {hi_g:.0%}])", [cellf(groups[cls][h], "d_re", "share") for h in heads]),
            ("ΔRE(中央値)", f"{sig[_RE].median():+.4f}",
             [cellf(groups[cls][h], "d_re", "med") for h in heads]),
            ("Δ払込資本(中央値)", f"{paid.median():+.4f}",
             [cellf(groups[cls][h], "d_paid", "med") for h in heads]),
            ("始点で債務超過 → 解消", f"{int((cm.loc[ins_g, 'res'] >= 2 / 3).sum())} / "
             f"{int(ins_g.sum())}",
             [f"{int(groups[cls][h]['resolved'].sum())} / "
              f"{int(groups[cls][h]['insolvent_2022'].sum())}" for h in heads]),
        ]
        for lab, gcell, acells in items:
            lines.append(f"| {lab} | {gcell} | " + " | ".join(acells) + " |")
        real = real_account_changes(list(groups[cls][heads[0]].index), names)
        mag = spearman((sig.abs().median()).to_numpy(), real.abs().median().to_numpy())
        sgn = spearman(sig.median().to_numpy(), real.median().to_numpy())
        top_g = sig.abs().median().sort_values(ascending=False)
        top_a = real.abs().median().sort_values(ascending=False)
        lines += ["", f"科目別(32 科目、{heads[0]} と比較): "
                  "変化の大きさ(\\|Δx\\| の中央値)の順位相関 "
                  f"{mag:+.3f}、符号付きの中央値の順位相関 {sgn:+.3f}。", "",
                  "| 順位 | 生成(\\|Δx\\| 年率) | | 実際 | |", "|---:|---|---:|---|---:|"]
        for k in range(5):
            lines.append(f"| {k + 1} | {ja[top_g.index[k]]} | {top_g.iloc[k]:.4f} "
                         f"| {ja[top_a.index[k]]} | {top_a.iloc[k]:.4f} |")
        lines += ["", "| 科目 | 生成(年率、中央値) | 実際 | 向き |", "|---|---:|---:|---|"]
        for a in names:
            gm, am = sig[a].median(), real[a].median()

            def sg(v: float) -> int:
                return 0 if abs(v) < 1e-4 else int(np.sign(v))
            lines.append(f"| {ja[a]} | {gm:+.4f} | {am:+.4f} "
                         f"| {'一致' if sg(gm) == sg(am) else '不一致'} |")
        lines.append("")
        summary[f"real_{cls}"] = {"magnitude_rank_corr": mag, "signed_median_rank_corr": sgn,
                                  "re_increase_share": k_re / len(sig)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "final.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
