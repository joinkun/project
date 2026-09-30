#!/usr/bin/env python
"""ステージ38: 遷移を分けた生成の評価(指標、実データとの向きの一致、数値条件、BS 表)。

lightgbm/sklearn/scipy のプロセス(torch なし)。判定の基準は ``results/stage38/criteria.md``
(結果を見る前に記録)。validity は class 1 を学習に含めない分類器(class 0 対 class 2)。

    uv run python scripts/evaluate_stage38.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from evaluate_stage34 import load_real_no_class1  # noqa: E402
from persistence_tables_stage33 import spearman  # noqa: E402
from run_stage25 import company_level, rows_for_version, summarize  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.accounting.plausibility import (  # noqa: E402
    CHECK_NAMES,
    EXTRA_CHECK_NAMES,
    RESIDUAL_FLOOR,
    RESIDUAL_NONNEG_ACCOUNTS,
    evaluate_checks,
    evaluate_extra_checks,
    subtotal_members,
)
from fsgen.evaluation.classifiers import train_classifiers  # noqa: E402
from fsgen.viz.balance_sheet import build_bs_lines, load_labels  # noqa: E402

SETTINGS = ("s1", "s2", "s3", "s4", "s5")
SET_LABEL = {"s1": "1 輸送等重み・射影一様・速度場一様", "s2": "2 輸送重み", "s3": "3 輸送+射影",
             "s4": "4 輸送+射影+速度場", "s5": "5 射影+速度場(輸送等重み)"}
OUT_C2 = ("S100TSS9", "S100TWFQ")
_B = 1000
_ZERO = 1e-4
BASE_T1 = Path("data/interim/stage31"), "r3_cap076"  # 単一遷移(class 2 → class 0)
BASE_T2 = Path("data/interim/stage34"), "class1"  # ステージ36 の class 1 始点・等重み


def firm_deltas(directory: Path, version: str, names: tuple[str, ...],
                exclude: tuple[str, ...]) -> pd.DataFrame:
    files = sorted((directory / version).glob("nearest_na0__seed*.parquet"))
    g = pd.concat([pd.read_parquet(f) for f in files])
    g = g[(g["method"] == "PCFM") & ~g["docID"].isin(exclude)]
    d = pd.DataFrame({n: g[f"x1_{n}"].to_numpy() - g[f"x0_{n}"].to_numpy() for n in names})
    d["docID"] = g["docID"].to_numpy()
    return d.groupby("docID").median()


def signed_corr(gen: pd.DataFrame, real_med: pd.Series) -> float:
    return spearman(gen.median().to_numpy(), real_med.to_numpy())


def sign_agree(gen_med: pd.Series, real_med: pd.Series) -> int:
    def sg(s: pd.Series) -> np.ndarray:
        v = s.to_numpy()
        return np.where(np.abs(v) < _ZERO, 0, np.sign(v))
    return int((sg(gen_med) == sg(real_med)).sum())


def boot_corr(gen: pd.DataFrame, real_med: pd.Series, base: pd.DataFrame | None,
              rng: np.random.Generator) -> tuple[float, float, float, float]:
    """(相関の 95% 区間, 基準との差の 95% 区間)。企業を同時に復元抽出する。"""
    ids = gen.index.to_numpy()
    vals, diffs = [], []
    for _ in range(_B):
        pick = rng.choice(ids, size=len(ids), replace=True)
        c = signed_corr(gen.loc[pick], real_med)
        vals.append(c)
        if base is not None:
            diffs.append(c - signed_corr(base.loc[pick], real_med))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    dlo, dhi = (np.nanpercentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan))
    return float(lo), float(hi), float(dlo), float(dhi)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/interim/stage38"))
    ap.add_argument("--out", type=Path, default=Path("results/stage38"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    excl = set(pd.read_csv(args.dir / "excluded.csv", dtype={"docID": str})["docID"])
    x_real, is2 = load_real_no_class1(names, excl)
    clfs = train_classifiers(x_real, is2, seed=42)
    lines = ["# ステージ38: 遷移を分けた生成の評価", ""]

    # ---- 実データの参照 ----
    real = pd.read_parquet(args.dir / "real_deltas.parquet")
    ref = {
        "T1": real[(real["label_prev"] == 2) & (real["label_cur"] == 1)][list(names)].median(),
        "T2": real[(real["label_prev"] == 1) & (real["label_cur"] == 0)][list(names)].median(),
    }
    y30 = pd.read_csv("results/stage30/recovery_firms.csv", dtype={"edinetCode": str})
    y30 = y30[(y30["group"] == "Y") & (y30["reason"].fillna("") == "") & y30["in_panel"]]
    pan = pd.read_csv("data/interim/stage23/panel/panel.csv", dtype={"edinetCode": str})
    pp = pan[pan["edinetCode"].isin(y30["edinetCode"]) & pan["t"].isin([2023, 2024])]
    dd = pd.DataFrame({n: (pp[f"cur_{n}"].fillna(0) - pp[f"prev_{n}"].fillna(0))
                       / pp["prev_total_assets"] for n in names})
    dd["edinetCode"] = pp["edinetCode"].to_numpy()
    ref["chain"] = dd.groupby("edinetCode").sum().median()

    # ---- 指標 ----
    groups = {"T1": OUT_C2, "T2": (), "chain": OUT_C2}
    summaries: dict[str, pd.DataFrame] = {}
    paired: dict[str, pd.DataFrame] = {}
    viol_detail: dict[str, str] = {}
    for grp, out in groups.items():
        rows = []
        for s in SETTINGS:
            r, _ = rows_for_version(args.dir / grp, s, names, members, clfs)
            rows.append(r)
            g = pd.concat([pd.read_parquet(f) for f in
                           sorted((args.dir / grp / s).glob("nearest_na0__seed*.parquet"))])
            g = g[(g["method"] == "PCFM") & ~g["docID"].isin(out)].reset_index(drop=True)
            x0 = g[[f"x0_{n}" for n in names]].to_numpy()
            x1 = g[[f"x1_{n}" for n in names]].to_numpy()
            fl = np.column_stack([evaluate_checks(x0, x1, spec, names),
                                  evaluate_extra_checks(x1, x0, spec, names)])
            cnt = pd.Series(fl.sum(axis=0), index=[*CHECK_NAMES, *EXTRA_CHECK_NAMES])
            viol_detail[f"{grp}/{s}"] = "、".join(f"{k} {int(v)}" for k, v in cnt.items() if v)
        comp = company_level(pd.concat(rows, ignore_index=True))
        summ, pr = summarize(comp, out, SETTINGS, "s1")
        summaries[grp], paired[grp] = summ, pr
        summ.to_csv(args.out / f"summary_{grp}.csv", index=False)
        pr.to_csv(args.out / f"paired_{grp}.csv", index=False)

    # 連結 vs 単一遷移(D.1 の 3)
    r_single, _ = rows_for_version(BASE_T1[0], BASE_T1[1], names, members, clfs)
    r_chain, _ = rows_for_version(args.dir / "chain", "s1", names, members, clfs)
    comp_cs = company_level(pd.concat([r_single.assign(version="single"),
                                       r_chain.assign(version="chain_s1")], ignore_index=True))
    summ_cs, pr_cs = summarize(comp_cs, OUT_C2, ("single", "chain_s1"), "single")

    # ---- 向きの一致 ----
    deltas: dict[str, dict[str, pd.DataFrame]] = {}
    for grp, out in groups.items():
        deltas[grp] = {s: firm_deltas(args.dir / grp, s, names, out) for s in SETTINGS}
    base_t1 = firm_deltas(BASE_T1[0], BASE_T1[1], names, OUT_C2)
    base_t2 = firm_deltas(BASE_T2[0], BASE_T2[1], names, ("S100VGDJ",))
    base_t2 = base_t2.loc[deltas["T2"]["s1"].index]
    base_of = {"T1": base_t1, "T2": base_t2, "chain": base_t1}
    corr_rows = []
    for grp in groups:
        b = base_of[grp].loc[deltas[grp]["s1"].index]
        bc = signed_corr(b, ref[grp])
        blo, bhi, _, _ = boot_corr(b, ref[grp], None, rng)
        corr_rows.append({"group": grp, "setting": "基準(単一遷移)", "corr": bc, "lo": blo,
                          "hi": bhi, "dlo": np.nan, "dhi": np.nan, "dslo": np.nan,
                          "dshi": np.nan, "agree": sign_agree(b.median(), ref[grp])})
        for s in SETTINGS:
            gd = deltas[grp][s]
            c = signed_corr(gd, ref[grp])
            lo, hi, dlo, dhi = boot_corr(gd, ref[grp], b.loc[gd.index], rng)
            _, _, dslo, dshi = boot_corr(gd, ref[grp], deltas[grp]["s1"].loc[gd.index], rng) \
                if s != "s1" else (0.0, 0.0, np.nan, np.nan)
            corr_rows.append({"group": grp, "setting": s, "corr": c, "lo": lo, "hi": hi,
                              "dlo": dlo, "dhi": dhi, "dslo": dslo, "dshi": dshi,
                              "agree": sign_agree(gd.median(), ref[grp])})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(args.out / "signed_corr.csv", index=False)

    # ---- 出力: 指標 ----
    n_real = {"T1": int(((real["label_prev"] == 2) & (real["label_cur"] == 1)).sum()),
              "T2": int(((real["label_prev"] == 1) & (real["label_cur"] == 0)).sum()),
              "chain": len(y30)}
    for grp, title in (("T1", "T1: class 2 → class 1(55 社)"),
                       ("T2", "T2: class 1 → class 0(131 社)"),
                       ("chain", "連結 T1 → T2(55 社、元の class 2 の始点に対して評価)")):
        s = summaries[grp].set_index("version")
        lines += [f"## {title}", "",
                  "| 設定 | 違反(行) | 純資産の改善 | 解消 | \\|総資産変化\\| | sparsity "
                  "| validity LGBM "
                  "| proximity | 符号付き順位相関 [95%] | 符号が一致した科目 |",
                  "|---|---:|---:|---|---:|---:|---:|---:|---|---:|"]
        c = corr[corr["group"] == grp].set_index("setting")
        br = c.loc["基準(単一遷移)"]
        for v in SETTINGS:
            r = s.loc[v]
            cr = c.loc[v]
            lines.append(
                f"| {SET_LABEL[v]} | {int(r['violations(rows)'])} "
                f"| {r['net_asset_improvement']:.3f} "
                f"| {int(r['resolved_firms(>=2/3 seeds)'])} / {int(r['insolvent_start'])} "
                f"| {r['abs_asset_change']:.3f} | {r['sparsity']:.0f} "
                f"| {r['validity_lightgbm']:.3f} "
                f"| {r['proximity_l2']:.3f} | {cr['corr']:+.3f} [{cr['lo']:+.3f}, {cr['hi']:+.3f}] "
                f"| {int(cr['agree'])} / 32 |")
        lines += [f"| 基準(単一遷移) | | | | | | | | {br['corr']:+.3f} [{br['lo']:+.3f}, "
                  f"{br['hi']:+.3f}] | {int(br['agree'])} / 32 |", "",
                  f"実データの参照: {n_real[grp]} 組 / 社。違反の内訳: " + "; ".join(
                      f"{k.split('/')[1]}: {v}" for k, v in viol_detail.items()
                      if k.startswith(grp + "/") and v) or "なし", ""]
        lines += ["各設定 − 設定 1(差の中央値 [95% CI])と、符号付き順位相関の差:", "",
                  "| 設定 | 純資産の改善 | \\|総資産変化\\| | 解消率 | 順位相関の差 [95%] |",
                  "|---|---|---|---|---|"]
        p = paired[grp]
        for v in SETTINGS[1:]:
            def cell(metric: str, v: str = v, p: pd.DataFrame = p) -> str:
                q = p[(p["version"] == v) & (p["metric"] == metric)].iloc[0]
                if pd.isna(q["ci_low"]):
                    return "差 0"
                return (f"{q['median_diff(version-equal)']:+.4f} "
                        f"[{q['ci_low']:+.4f}, {q['ci_high']:+.4f}]")
            cr = c.loc[v]
            lines.append(f"| {SET_LABEL[v]} | {cell('純資産の改善')} | {cell('|総資産変化|')} "
                         f"| {cell('債務超過の解消率(社ごとの 3 seed 平均)')} "
                         f"| {cr['corr'] - c.loc['s1', 'corr']:+.3f} "
                         f"[{cr['dslo']:+.3f}, {cr['dshi']:+.3f}] |")
        s1 = c.loc["s1"]
        lines += ["", f"設定 1 − 基準(単一遷移)の符号付き順位相関の差: "
                  f"{s1['corr'] - br['corr']:+.3f} [{s1['dlo']:+.3f}, {s1['dhi']:+.3f}]", ""]
    pc = pr_cs.set_index("metric")
    lines += ["## 連結 − 単一遷移(55 社、設定 1)", "",
              "| 指標 | 単一遷移 | 連結 | 差 [95% CI] |", "|---|---:|---:|---|"]
    sc = summ_cs.set_index("version")
    for metric, col in (("純資産の改善", "net_asset_improvement"),
                        ("|総資産変化|", "abs_asset_change"), ("sparsity", "sparsity"),
                        ("等重み proximity", "proximity_l2")):
        q = pc.loc[metric]
        lines.append(f"| {metric.replace('|', chr(92) + '|')} | {sc.loc['single', col]:.3f} "
                     f"| {sc.loc['chain_s1', col]:.3f} "
                     f"| {q['median_diff(version-equal)']:+.4f} [{q['ci_low']:+.4f}, "
                     f"{q['ci_high']:+.4f}] |")
    lines.append(f"| 債務超過の解消 | {int(sc.loc['single', 'resolved_firms(>=2/3 seeds)'])} / "
                 f"{int(sc.loc['single', 'insolvent_start'])} "
                 f"| {int(sc.loc['chain_s1', 'resolved_firms(>=2/3 seeds)'])} / "
                 f"{int(sc.loc['chain_s1', 'insolvent_start'])} | |")

    # ---- B.3・B.4: 科目別(設定 1)----
    lines += ["", "## B.3・B.4 科目別の変化(設定 1、中央値。生成は 5 年、"
              "実データは T1・T2 が 1 年、"
              "連結の実データは 2 年)", "",
              "| 科目 | T1 生成 | T1 実際 | T2 生成 | T2 実際 | 連結 生成 | 単一遷移 生成 "
              "| 回復 15 社 実際 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        lines.append(f"| {ja[n]} | {deltas['T1']['s1'][n].median():+.4f} | {ref['T1'][n]:+.4f} "
                     f"| {deltas['T2']['s1'][n].median():+.4f} | {ref['T2'][n]:+.4f} "
                     f"| {deltas['chain']['s1'][n].median():+.4f} "
                     f"| {base_t1[n].median():+.4f} | {ref['chain'][n]:+.4f} |")
    for grp in ("T1", "T2"):
        gd = deltas[grp]["s1"]
        top = gd.abs().median().sort_values(ascending=False).index[:5]
        dec = [ja[n] for n in names if (gd[n] < -_ZERO).mean() > 0.5]
        inc = [ja[n] for n in names if (gd[n] > _ZERO).mean() > 0.5]
        lines += ["", f"{grp} 生成(設定 1): 動きの大きい科目 上位 5 = "
                  + "、".join(ja[n] for n in top),
                  f"過半の社で減少: {'、'.join(dec) or 'なし'}",
                  f"過半の社で増加: {'、'.join(inc) or 'なし'}"]

    # ---- D.3 数値条件 ----
    lines += ["", "## D.3 数値条件", "",
              "| 遷移 | 設定 | 射影の重み比(最大) | 同(中央値) "
              "| 最適性の検定(違反 / 検定数) | 最大の超過 "
              "| 勾配の相対誤差(最大) | 実行不可能 |",
              "|---|---|---:|---:|---|---:|---:|---:|"]
    for grp in ("T1", "T2", "chain"):
        for s in SETTINGS:
            ds = [json.loads(f.read_text()) for f in
                  sorted((args.dir / grp / s).glob("diag_seed*.json"))]
            ge = [d["grad_check_max_rel_err"] for d in ds
                  if d["grad_check_max_rel_err"] is not None]
            lines.append(
                f"| {grp} | {s} | {max(d['proj_weight_ratio_max'] for d in ds):,.0f} "
                f"| {np.median([d['proj_weight_ratio_median'] for d in ds]):,.0f} "
                f"| {sum(d['optimality_violations'] for d in ds)} / "
                f"{sum(d['optimality_checked'] for d in ds)} "
                f"| {max(d['optimality_worst_rel_excess'] for d in ds):.2e} "
                f"| {max(ge):.1e} | {sum(d['infeasible'] for d in ds)} |" if ge else
                f"| {grp} | {s} | {max(d['proj_weight_ratio_max'] for d in ds):,.0f} "
                f"| {np.median([d['proj_weight_ratio_median'] for d in ds]):,.0f} "
                f"| {sum(d['optimality_violations'] for d in ds)} / "
                f"{sum(d['optimality_checked'] for d in ds)} "
                f"| {max(d['optimality_worst_rel_excess'] for d in ds):.2e} | — "
                f"| {sum(d['infeasible'] for d in ds)} |")
    ds1 = json.loads((args.dir / "T1" / "s4" / "diag_seed0.json").read_text())
    lines.append(f"\nコスト重み(持続性)の比: {ds1['weight_ratio_cost']:.1f}(最大 / 最小)。")
    # 連結で下限を緩めた社
    relaxed = 0
    total = 0
    for s in SETTINGS:
        for f in sorted((args.dir / "chain" / s).glob("nearest_na0__seed*.parquet")):
            g = pd.read_parquet(f)
            xm = g[[f"xm_{n}" for n in names]].to_numpy()
            am = xm @ members["jppfs_cor_Assets"]
            bad = np.zeros(len(g), dtype=bool)
            for nm in RESIDUAL_NONNEG_ACCOUNTS:
                bad |= xm[:, names.index(nm)] < RESIDUAL_FLOOR * am
            relaxed += int(bad.sum())
            total += len(g)
    lines.append(f"連結で T2 の始点(T1 の終点)が残余科目の下限を下回り、始点の値まで緩めた行: "
                 f"{relaxed} / {total}。")
    (args.out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---- BS 表(設定 1、seed 0)----
    labels = load_labels(Path("conf/accounts/labels_ja.yaml"))
    bs_dir = args.out / "bs"
    bs_dir.mkdir(parents=True, exist_ok=True)

    def load_s0(path: Path) -> pd.DataFrame:
        g = pd.read_parquet(path / "nearest_na0__seed0.parquet")
        return g[g["method"] == "PCFM"].set_index("docID")

    t1, t2, ch = (load_s0(args.dir / g / "s1") for g in ("T1", "T2", "chain"))
    single = load_s0(BASE_T1[0] / BASE_T1[1])
    pick_rng = np.random.default_rng(0)
    pool1 = sorted(set(t1.index) - set(OUT_C2) - {"S100TBIV"})
    t1_docs = ["S100TBIV", *(str(d) for d in pick_rng.choice(pool1, 4, replace=False))]
    t2_docs = [str(d) for d in pick_rng.choice(sorted(t2.index), 5, replace=False)]
    ch_docs = ["S100TBIV", *(str(d) for d in pick_rng.choice(pool1, 2, replace=False))]

    def table(title: str, cols: list[tuple[str, np.ndarray]], assets: float) -> list[str]:
        bl = build_bs_lines(spec, names, [v * assets for _, v in cols], labels)
        out = [f"## {title}", "", "| 科目 | " + " | ".join(c for c, _ in cols) + " |",
               "|---|" + "---:|" * len(cols)]
        for line in bl:
            lab = "&nbsp;&nbsp;" * line.indent + line.label
            if line.values is None:
                out.append(f"| **{line.label}** |" + " |" * len(cols))
                continue
            bold = "**" if line.kind in ("subtotal", "total", "diff") else ""
            out.append(f"| {bold}{lab}{bold} | "
                       + " | ".join(f"{v / 1e6:,.0f}" for v in line.values) + " |")
        return [*out, ""]

    x0c = [f"x0_{n}" for n in names]
    x1c = [f"x1_{n}" for n in names]
    xmc = [f"xm_{n}" for n in names]
    doc_lines = ["# ステージ38 C.3: BS 表(設定 1、seed 0、単位 百万円、5 年 = t=1)", ""]
    for d in t1_docs:
        r = t1.loc[d]
        doc_lines += table(f"T1 {r['filerName']}({d})", [
            ("始点(class 2)", r[x0c].to_numpy(float)), ("T1 の終点", r[x1c].to_numpy(float)),
            ("単一遷移の終点", single.loc[d][x1c].to_numpy(float))], float(r["assets_start"]))
    for d in t2_docs:
        r = t2.loc[d]
        doc_lines += table(f"T2 {r['filerName']}({d})", [
            ("始点(class 1)", r[x0c].to_numpy(float)), ("T2 の終点", r[x1c].to_numpy(float))],
            float(r["assets_start"]))
    for d in ch_docs:
        r = ch.loc[d]
        doc_lines += table(f"連結 {r['filerName']}({d})", [
            ("始点(class 2)", r[x0c].to_numpy(float)), ("T1 の終点", r[xmc].to_numpy(float)),
            ("T2 の終点(連結)", r[x1c].to_numpy(float)),
            ("単一遷移の終点", single.loc[d][x1c].to_numpy(float))], float(r["assets_start"]))
    (bs_dir / "bs.md").write_text("\n".join(doc_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
