#!/usr/bin/env python
"""ステージ34: class 1 を始点とした生成の検証(A)、3 群の変化と持続性(B)、class 1 の遷移と γ(C)。

lightgbm/sklearn/scipy のプロセス(torch なし)。新しい生成は ``run_generation_stage34.py`` の
出力だけ。

- A: 生成(``data/interim/stage34/class1``)の 13 項目チェック・回復指標・RE の増減・動いた科目を、
  2022 年度に class 1 だった企業の実際の 2 年間の変化(推定用パネル)と並べる。年率に直して比べる
  (生成は t=1 を 5 年、実際は 2 年)。validity は **class 1 を学習に含めない**分類器(class 0 対
  class 2、seed 42)で測る(これまでの分類器は class 1 を「class 2 でない」側に含めていた)。
- B: 推定用パネルの書類内の 1 年の組を、各書類の年度のラベル(ステージ30)で 3 群に分け、
  |Δx| / 前期総資産 の p50・p90・p99 と ρ1 を比べる。潜在的可動性
  m = p90(class 1 と 2)/ p90(class 0)。
- C: 2022 → 2024 の class 1 の遷移。class 1 → class 0 の企業の財務変化(ステージ30 と同じ計算)と、
  ステージ30 の再建群 15 社を合わせた γ(Tukey の外れ値を除く p90 × 5)。

    uv run python scripts/evaluate_stage34.py
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
from extract_business_stage28 import business_text  # noqa: E402
from persistence_stage32 import load_panel  # noqa: E402
from persistence_tables_stage33 import boot_ci, spearman  # noqa: E402
from recovery_stage30 import changes  # noqa: E402
from run_stage25 import rows_for_version  # noqa: E402
from special_industry_stage28 import classify  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.accounting.plausibility import (  # noqa: E402
    CHECK_NAMES,
    EXTRA_CHECK_NAMES,
    evaluate_checks,
    evaluate_extra_checks,
    subtotal_members,
)
from fsgen.evaluation.classifiers import train_classifiers  # noqa: E402
from fsgen.models.industry import load_industry_config  # noqa: E402

_P = "jppfs_cor_"
_RE = f"{_P}RetainedEarnings"
THR = 0.001
GEN_YEARS, ACT_YEARS = 5.0, 2.0
# シンバイオ製薬: RE / A が class 0 の最大を超える(ステージ19 と同じ基準)
OUTLIER_CLASS1 = ("S100VGDJ",)


def load_real_no_class1(
    names: tuple[str, ...], excluded: set[str]
) -> tuple[np.ndarray, np.ndarray]:
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    lab = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str})
    lab = lab[pd.to_numeric(lab["label"], errors="coerce").isin([0, 2])]
    m = agg.merge(lab[["docID", "label"]], on="docID", how="inner")
    m = m[m["total_assets"].notna() & (m["total_assets"] != 0) & ~m["docID"].isin(excluded)]
    x = m[list(names)].fillna(0.0).to_numpy(dtype=float) / m["total_assets"].to_numpy()[:, None]
    return x, m["label"].astype(int).to_numpy() == 2


def special_reason(fy: pd.DataFrame, codes: list[str]) -> pd.Series:
    """各社の最新の書類にステージ28 の規則を当てる。"""
    latest = fy.sort_values("year").drop_duplicates("edinetCode", keep="last").set_index(
        "edinetCode").loc[codes]
    ind = load_industry_config(Path("conf/accounts/industry.yaml"))
    df = pd.DataFrame({
        "filerName": latest["filerName"],
        "industry33": ind.labels(list(codes)),
        "business": [business_text(d) or "" for d in latest["docID"]],
    }, index=codes)
    return classify(df)


def fmt_or_dash(v: float, d: int) -> str:
    return "—" if pd.isna(v) or not np.isfinite(v) else f"{v:.{d}f}"


def q(s: pd.Series | np.ndarray, p: float) -> float:
    a = np.asarray(s, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.quantile(a, p)) if len(a) else float("nan")


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/interim/stage34"))
    ap.add_argument("--exclude", type=Path, default=Path("data/interim/stage28/excluded.csv"))
    ap.add_argument("--panel-dir", type=Path, default=Path("data/interim/stage23/panel"))
    ap.add_argument("--out", type=Path, default=Path("results/stage34"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    ex = pd.read_csv(args.exclude, dtype={"docID": str, "edinetCode": str})
    lines: list[str] = ["# ステージ34: class 1 の活用(検証と試算)", ""]

    # ---------------- A: 生成 ----------------
    x_real, is2 = load_real_no_class1(names, set(ex["docID"]))
    clfs = train_classifiers(x_real, is2, seed=42)
    rows, _dec = rows_for_version(args.dir, "class1", names, members, clfs)
    rows_eval = rows[~rows["docID"].isin(OUTLIER_CLASS1)]
    gen = pd.concat([pd.read_parquet(f) for f in sorted((args.dir / "class1").glob("*.parquet"))])
    gen = gen[gen["method"] == "PCFM"].reset_index(drop=True)
    x0 = gen[[f"x0_{n}" for n in names]].to_numpy()
    x1 = gen[[f"x1_{n}" for n in names]].to_numpy()
    flags = np.column_stack([evaluate_checks(x0, x1, spec, names),
                             evaluate_extra_checks(x1, x0, spec, names)])
    check_names = [*CHECK_NAMES, *EXTRA_CHECK_NAMES]
    viol = pd.DataFrame(flags, columns=check_names).assign(docID=gen["docID"], seed=gen["seed"],
                                                          filerName=gen["filerName"])
    n_firms = gen["docID"].nunique()
    comp = rows_eval.groupby("docID").agg(
        na=("net_asset_improvement", "median"), ins=("insolvent_start", "first"),
        res=("insolvency_resolved", "mean"), aac=("abs_asset_change", "median"),
        sp=("sparsity", "median"), vl=("validity_lightgbm", "median"),
        prox=("proximity_l2", "median"))
    d_re = pd.Series((x1 - x0)[:, names.index(_RE)], index=gen.index).groupby(gen["docID"]).median()
    d_re = d_re.drop(index=[d for d in OUTLIER_CLASS1 if d in d_re.index])
    gen_abs = pd.DataFrame(np.abs(x1 - x0), columns=names).assign(docID=gen["docID"])
    gen_abs = gen_abs[~gen_abs["docID"].isin(OUTLIER_CLASS1)].groupby("docID").median()
    gen_sig = pd.DataFrame(x1 - x0, columns=names).assign(docID=gen["docID"])
    gen_sig = gen_sig[~gen_sig["docID"].isin(OUTLIER_CLASS1)].groupby("docID").median()

    # 比較用: 最終設定(class 2)の validity を、class 1 を含まない分類器で測り直す
    rows2, _ = rows_for_version(Path("data/interim/stage31"), "r3_cap076", names, members, clfs)
    rows2 = rows2[~rows2["docID"].isin(("S100TSS9", "S100TWFQ"))]
    v2_new = rows2.groupby("docID")["validity_lightgbm"].median().median()

    # ---------------- 実際(2022 年度に class 1)----------------
    fy = pd.read_csv("data/interim/stage30/firm_years.csv", dtype={"edinetCode": str})
    wide = fy.pivot(index="edinetCode", columns="year", values="label")
    c1_22 = wide[wide[2022] == 1].copy()
    c1_22["reason"] = special_reason(fy, c1_22.index.tolist())
    ch = changes(args.panel_dir, c1_22.index.tolist(), names)
    c1_22 = c1_22.join(ch, how="left")
    c1_22["in_panel"] = c1_22["n_years"].notna()
    act = c1_22[(c1_22["reason"] == "") & c1_22["in_panel"]]
    # 科目別の実際の変化(2 年分の書類内の差分 / 前期総資産 の和 / 2)
    p = pd.read_csv(args.panel_dir / "panel.csv", dtype={"edinetCode": str})
    pp = p[p["edinetCode"].isin(act.index) & p["t"].isin([2023, 2024])]
    dacc = pd.DataFrame({n: (pp[f"cur_{n}"].fillna(0) - pp[f"prev_{n}"].fillna(0))
                         / pp["prev_total_assets"] for n in names})
    dacc["edinetCode"] = pp["edinetCode"].to_numpy()
    act_sig = dacc.groupby("edinetCode").sum() / ACT_YEARS
    act_abs = act_sig.abs()

    g_na = comp["na"] / GEN_YEARS
    gen_paid = float(((gen_sig[f"{_P}CapitalStock"] + gen_sig[f"{_P}CapitalSurplus"])
                      / GEN_YEARS).median())
    lines += [
        "## A.2・A.4 class 1 を始点とした生成", "",
        f"- 始点: class 1 は 137 社。ステージ28 の規則で金融 2 社・ゴルフ場 3 社を除いた "
        f"**{n_firms} 社**を生成し、外れ値 1 社(シンバイオ製薬、RE / 総資産 6.58 が class 0 の"
        "最大 5.57 を超える)を除いた "
        f"**{n_firms - len(OUTLIER_CLASS1)} 社**で評価した。",
        "- 速度場は class 2 の 57 社 → class 0 で学習(class 1 は学習にも目標にも使わない)。"
        "3 seed とも実行不可能 0。",
        f"- **13 項目の違反: {int(flags.any(axis=1).sum())} 行 / {len(gen)} 行"
        f"({viol.loc[flags.any(axis=1), 'docID'].nunique()} 社)**",
    ]
    if flags.any():
        cnt = viol[check_names].sum()
        lines += ["  - 違反の内訳(行): " + "、".join(f"{k} {int(v)}" for k, v in cnt.items() if v),
                  "  - 違反した社: " + "、".join(
                      f"{r['filerName']}(seed {r['seed']})"
                      for _, r in viol[flags.any(axis=1)].iterrows())]
    lines += ["", "## A.3 生成と実際の比較(年率、総資産比)", "",
              f"実際: 2022 年度に class 1 だった {len(c1_22)} 社のうち、特殊業種を除きパネルにある "
              f"{len(act)} 社の 2022 → 2024 年度末(2 年)。生成: 上の {len(comp)} 社の t = 1(5 年)。"
              "どちらも年率(生成は / 5、実際は / 2)。", "",
              "| 項目 | 生成(class 1 始点) | 実際(class 1 の 2 年) "
              "| 実際のうち class 0 へ回復した社 |",
              "|---|---:|---:|---:|"]
    rec = act[wide.loc[act.index, 2024] == 0]
    ins_g = comp["ins"].astype(bool)
    lines += [
        f"| 社数 | {len(comp)} | {len(act)} | {len(rec)} |",
        f"| 純資産の改善(年率、中央値) | {g_na.median():+.4f} | {act['d_na'].median():+.4f} "
        f"| {rec['d_na'].median():+.4f} |",
        f"| 純資産が改善した社 | {(g_na > THR).mean():.1%} | {(act['d_na'] > THR).mean():.1%} "
        f"| {(rec['d_na'] > THR).mean():.1%} |",
        f"| 始点で債務超過の社 | {int(ins_g.sum())} | {int(act['insolvent_2022'].sum())} "
        f"| {int(rec['insolvent_2022'].sum())} |",
        f"| うち解消(生成は 3 seed の 2/3 以上) | {int((comp.loc[ins_g, 'res'] >= 2 / 3).sum())} "
        f"| {int(act['resolved'].sum())} | {int(rec['resolved'].sum())} |",
        f"| 利益剰余金が増えた社(年率 > {THR}) | {(d_re / GEN_YEARS > THR).mean():.1%} "
        f"| {(act['d_re'] > THR).mean():.1%} | {(rec['d_re'] > THR).mean():.1%} |",
        f"| ΔRE(年率、中央値) | {(d_re / GEN_YEARS).median():+.4f} | {act['d_re'].median():+.4f} "
        f"| {rec['d_re'].median():+.4f} |",
        f"| Δ払込資本(年率、中央値) | "
        f"{gen_paid:+.4f}"
        f" | {act['d_paid'].median():+.4f} | {rec['d_paid'].median():+.4f} |",
        f"| \\|総資産変化\\|(5 年、中央値) | {comp['aac'].median():.3f} | — | — |",
        f"| validity LightGBM(class 1 を含まない分類器) | {comp['vl'].median():.3f} | — | — |",
        "",
    ]
    top_g = (gen_abs.median() / GEN_YEARS).sort_values(ascending=False)
    top_a = act_abs.median().sort_values(ascending=False)
    lines += ["動いた科目(|Δx| / A の年率の中央値)の上位 5:", "",
              "| 順位 | 生成 | 年率 | 実際 | 年率 |", "|---:|---|---:|---|---:|"]
    for k in range(5):
        lines.append(f"| {k + 1} | {ja[top_g.index[k]]} | {top_g.iloc[k]:.4f} "
                     f"| {ja[top_a.index[k]]} | {top_a.iloc[k]:.4f} |")
    both = pd.DataFrame({"gen": gen_abs.median() / GEN_YEARS, "act": act_abs.median()})
    lines += ["", f"全 32 科目での順位相関(生成 対 実際): "
              f"{spearman(both['gen'].to_numpy(), both['act'].to_numpy()):+.3f}", "",
              "科目別の変化(年率、中央値。符号付き):", "",
              "| 科目 | 生成 | 実際 | 実際(回復した社) |", "|---|---:|---:|---:|"]
    rec_sig = act_sig.loc[act_sig.index.isin(rec.index)]
    for n in names:
        lines.append(f"| {ja[n]} | {gen_sig[n].median() / GEN_YEARS:+.4f} "
                     f"| {act_sig[n].median():+.4f} | {rec_sig[n].median():+.4f} |")
    lines += ["", "validity の分類器について: これまでの分類器(`load_real`)は class 1 を"
              "「class 2 でない」側に含めて学習していた。class 1 を除いた分類器で最終設定"
              "(class 2 始点、55 社)の validity を"
              f"測り直すと {v2_new:.3f}(ステージ31 の値 0.998)。", ""]

    # ---------------- B ----------------
    codes = set(ex["edinetCode"])
    pan = load_panel(args.panel_dir, codes)
    pan = pan[(pan["cur_total_assets"] > 0) & (pan["prev_total_assets"] > 0)].reset_index(drop=True)
    lab = fy.set_index(["edinetCode", "year"])["label"]
    pan["label"] = [lab.get((c, t), np.nan) for c, t in zip(pan["edinetCode"], pan["t"],
                                                            strict=True)]
    brow = []
    for n in names:
        xc = pan[f"cur_{n}"].fillna(0.0).to_numpy()
        xp = pan[f"prev_{n}"].fillna(0.0).to_numpy()
        rc, rp = xc / pan["cur_total_assets"].to_numpy(), xp / pan["prev_total_assets"].to_numpy()
        d = np.abs(xc - xp) / pan["prev_total_assets"].to_numpy()
        br: dict[str, float | str] = {"account": n, "ja": ja[n]}
        for c in (0, 1, 2):
            m = (pan["label"] == c).to_numpy()
            br[f"p50_{c}"], br[f"p90_{c}"], br[f"p99_{c}"] = q(d[m], .5), q(d[m], .9), q(d[m], .99)
            br[f"rho_{c}"] = spearman(rc[m], rp[m])
            if c in (1, 2):
                lo, hi = boot_ci(rc[m], rp[m], pan.loc[m, "edinetCode"].to_numpy(), rng)
                br[f"rho_{c}_lo"], br[f"rho_{c}_hi"] = lo, hi
            br[f"n_{c}"] = int(m.sum())
        m12 = pan["label"].isin([1, 2]).to_numpy()
        br["p90_12"] = q(d[m12], .9)
        brow.append(br)
    bt = pd.DataFrame(brow)
    bt["m"] = np.where(bt["p90_0"] > 0, bt["p90_12"] / bt["p90_0"], np.nan)
    bt["c"] = 1 / bt["m"]
    cash_c = float(bt.loc[bt["account"] == f"{_P}CashAndDeposits", "c"].iloc[0])
    bt["c_norm"] = bt["c"] / cash_c
    bt.to_csv(args.out / "class_compare.csv", index=False)

    def mono(a: float, b: float, c: float) -> str:
        if any(pd.isna(v) for v in (a, b, c)):
            return "—"
        if a <= b <= c:
            return "単調"
        return "単調でない"

    n_obs = {c: int((pan["label"] == c).sum()) for c in (0, 1, 2)}
    n_firm = {c: pan.loc[pan["label"] == c, "edinetCode"].nunique() for c in (0, 1, 2)}
    lines += ["## B.1 3 群の変化の大きさ(|Δx| / 前期総資産、1 年)", "",
              f"組数: class 0 {n_obs[0]:,}({n_firm[0]:,} 社)、class 1 {n_obs[1]}({n_firm[1]} 社)、"
              f"class 2 {n_obs[2]}({n_firm[2]} 社)。ラベルは各書類の年度のもの。", "",
              "| 科目 | p50 c0 | c1 | c2 | p90 c0 | c1 | c2 | p99 c0 | c1 | c2 | p90 の順序 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for _, r in bt.iterrows():
        lines.append(
            f"| {r['ja']} | {r['p50_0']:.4f} | {r['p50_1']:.4f} | {r['p50_2']:.4f} "
            f"| {r['p90_0']:.4f} | {r['p90_1']:.4f} | {r['p90_2']:.4f} | {r['p99_0']:.3f} "
            f"| {r['p99_1']:.3f} | {r['p99_2']:.3f} | {mono(r['p90_0'], r['p90_1'], r['p90_2'])} |")
    n_mono = sum(mono(r["p90_0"], r["p90_1"], r["p90_2"]) == "単調" for _, r in bt.iterrows())
    lines += ["", f"p90 が class 0 ≤ class 1 ≤ class 2 の順の科目: {n_mono} / 32", ""]
    lines += ["## B.2 ρ1 の 3 群比較", "",
              "class 1・2 の区間は企業単位のブートストラップ(1,000 回)。"
              "「単調」は ρ0 ≥ ρ1 ≥ ρ2。", "",
              "| 科目 | ρ class 0 | ρ class 1 [95%] | ρ class 2 [95%] | 単調か "
              "| class 1 が class 0 より有意に低い |",
              "|---|---:|---|---|---|---|"]
    n_mono_rho = n_low1 = 0
    for _, r in bt.iterrows():
        mr = "単調" if r["rho_0"] >= r["rho_1"] >= r["rho_2"] else "単調でない"
        low1 = r["rho_1_hi"] < r["rho_0"]
        n_mono_rho += mr == "単調"
        n_low1 += bool(low1)
        lines.append(f"| {r['ja']} | {r['rho_0']:.3f} | {r['rho_1']:.3f} "
                     f"[{r['rho_1_lo']:.3f}, {r['rho_1_hi']:.3f}] | {r['rho_2']:.3f} "
                     f"[{r['rho_2_lo']:.3f}, {r['rho_2_hi']:.3f}] | {mr} | {'○' if low1 else ''} |")
    lines += ["", f"ρ が単調(ρ0 ≥ ρ1 ≥ ρ2)な科目: {n_mono_rho} / 32。class 1 が class 0 より有意に"
              f"低い科目: {n_low1} / 32。", ""]
    lines += ["## B.3 潜在的可動性の試算(生成には使わない)", "",
              "m = p90(|Δx|/A | class 1 と 2 をあわせた組)/ p90(|Δx|/A | class 0)、"
              "c = 1/m を現金預金 = 1 に正規化。"
              "class 0 の p90 が 0 の科目は定義できない。", "",
              "| 科目 | p90 class 0 | p90 class 1・2 | m | c = 1/m(現金 = 1) |",
              "|---|---:|---:|---:|---:|"]
    for _, r in bt.sort_values("m", ascending=False).iterrows():
        lines.append(f"| {r['ja']} | {r['p90_0']:.4f} | {r['p90_12']:.4f} "
                     f"| {fmt_or_dash(r['m'], 1)} | {fmt_or_dash(r['c_norm'], 3)} |")

    # ---------------- C ----------------
    st = {0: "class 0", 1: "class 1", 2: "class 2"}
    to = c1_22[2024].map(lambda v: "欠測" if pd.isna(v) else st[int(v)]).value_counts()
    gc = wide[wide[2022] == 2][2024]
    to2 = gc.map(lambda v: "欠測" if pd.isna(v) else st[int(v)]).value_counts()
    lines += ["", "## C.1 2022 → 2024 の遷移", "",
              "| 2022 | 社数 | 2024: class 0 | class 1 | class 2 | 欠測 |",
              "|---|---:|---:|---:|---:|---:|"]
    for lab_, t_ in (("class 1", to), ("class 2(ステージ30)", to2)):
        tot = int(t_.sum())
        n0 = int(t_.get("class 0", 0))
        lines.append(f"| {lab_} | {tot} | {n0}({n0 / tot:.0%})"
                     f" | {int(t_.get('class 1', 0))} | {int(t_.get('class 2', 0))} "
                     f"| {int(t_.get('欠測', 0))} |")
    # C.2
    y30 = pd.read_csv("results/stage30/recovery_firms.csv", dtype={"edinetCode": str})
    y30 = y30[(y30["group"] == "Y") & (y30["reason"].fillna("") == "") & y30["in_panel"]]

    def stats(s: pd.DataFrame) -> dict[str, float]:
        up = s["d_re"] > THR
        return {"n": len(s), "share_up": float(up.mean()),
                "n_transfer_up": int((up & (s["d_paid"] < -THR)).sum()),
                "n_profit_pos": int((s["profit"] > THR).sum()),
                "profit_med": float(s["profit"].median()),
                "n_issue": int((s["d_paid"] > THR).sum()),
                "n_ins": int(s["insolvent_2022"].sum()), "n_res": int(s["resolved"].sum()),
                "na_med": float(s["d_na"].median())}

    s30, s1 = stats(y30), stats(rec)
    lines += ["", "## C.2 回復群の財務変化(2022 → 2024、年率)", "",
              "| 項目 | class 2 → 注記なし(ステージ30、15 社) | class 1 → class 0 |",
              "|---|---:|---:|",
              f"| 社数 | {s30['n']} | {s1['n']} |",
              f"| 利益剰余金が増えた社 | {s30['share_up']:.1%} | {s1['share_up']:.1%} |",
              f"| うち振替を伴う | {s30['n_transfer_up']} | {s1['n_transfer_up']} |",
              f"| 利益部分 P > 0 の社 | {s30['n_profit_pos']} | {s1['n_profit_pos']} |",
              f"| 利益部分の年率(中央値) | {s30['profit_med']:+.4f} | {s1['profit_med']:+.4f} |",
              f"| 増資のある社 | {s30['n_issue']} | {s1['n_issue']} |",
              f"| Δ純資産の年率(中央値) | {s30['na_med']:+.4f} | {s1['na_med']:+.4f} |",
              f"| 2022 年度末に債務超過 → 解消 | {s30['n_res']} / {s30['n_ins']} "
              f"| {s1['n_res']} / {s1['n_ins']} |"]

    # C.3 γ
    def gamma(prof: np.ndarray) -> tuple[float, float, list[int]]:
        q1, q3 = np.quantile(prof, [0.25, 0.75])
        fence = q3 + 1.5 * (q3 - q1)
        keep = prof <= fence
        return 5 * float(np.quantile(prof[keep], 0.9)), float(fence), [int((~keep).sum())]

    g30 = gamma(y30["profit"].to_numpy())
    g1 = gamma(rec["profit"].to_numpy())
    pooled = np.concatenate([y30["profit"].to_numpy(), rec["profit"].to_numpy()])
    gp = gamma(pooled)
    lines += ["", "## C.3 γ の再計算(採用しない)", "",
              "γ = 5 × p90(利益部分の年率、Tukey の外れ値 Q3 + 1.5 IQR を除く)。", "",
              "| 標本 | 社数 | 外れ値 | γ |", "|---|---:|---:|---:|",
              f"| class 2 → 注記なし(ステージ31 の値) | {len(y30)} | {g30[2][0]} | {g30[0]:.3f} |",
              f"| class 1 → class 0 | {len(rec)} | {g1[2][0]} | {g1[0]:.3f} |",
              f"| 両者を合わせる | {len(pooled)} | {gp[2][0]} | {gp[0]:.3f} |"]
    summary = {"n_gen": int(n_firms), "violations_rows": int(flags.any(axis=1).sum()),
               "gamma_class2": g30[0], "gamma_class1": g1[0], "gamma_pooled": gp[0],
               "validity_final_without_class1": float(v2_new)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    (args.out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
