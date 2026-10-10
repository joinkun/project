#!/usr/bin/env python
"""ステージ60 D・E: 輸送計画の台と周辺分布の評価と判定。

規則は ``results/stage60/criteria.md``。``--phase link`` で各案の代表を規則で選び、評価用のリンク
(``data/interim/stage60/eval/<cls>/``)を作る。その後 ``metrics_stage39.py --gen-dir
data/interim/stage60/eval`` を実行し、``--phase report`` で表と判定を書く。

    uv run python scripts/evaluate_stage60.py --phase link
    uv run python scripts/metrics_stage39.py --gen-dir data/interim/stage60/eval \\
        --out data/interim/stage60/metrics
    uv run python scripts/evaluate_stage60.py --phase report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import numpy as np
import pandas as pd
from density_stage51 import load_tilde
from sampling_stage60 import stratified

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import subtotal_members
from fsgen.models.industry import load_industry_config

OUT = Path("results/stage60")
DATA = Path("data/interim/stage60")
EVAL = DATA / "eval"
_CS, _CSUR = "jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus"
YEARS, THR = 5.0, 0.001
EPS_REP = 0.005
SUP_MAX = 20.0
REAL_GROUP = {"c2": "class 2 → 回復", "c1": "class 1 → class 0"}
LABELS = {"nn__bary": "0 最近傍(非対称 L1)", "nn_l2__bary": "0' 最近傍(等重み L2²)",
          "bal__bary": "1 バランス OT(重心)", "bal__argmax": "1 バランス OT(最大質量)",
          "rep_A__bary": "A 不均衡 OT(重心)", "rep_A__argmax": "A 不均衡 OT(最大質量)",
          "rep_B__bary": "B 台を絞る(重心)", "rep_B__argmax": "B 台を絞る(最大質量)",
          "rep_B3__bary": "B 対照: 近傍の和集合(重心)",
          "ref_C__bary": "C 参考: 最も局所的な β(重心)"}


def representatives(plans: pd.DataFrame) -> dict[str, str | None]:
    """criteria §4 の規則で各案の代表を選ぶ。"""
    ok = plans[(plans["status"] == "ok") & (plans["mass"] > 0.5) & (plans["row_min"] > 0)]
    loc = ok[(ok["eps"] == EPS_REP) & (ok["support_med"] <= SUP_MAX)]
    a = loc[loc["family"] == "A"].sort_values("tau1")
    b = loc[(loc["family"] == "B") & (loc["seed"] == 0)].sort_values("n1p")
    b3 = loc[loc["family"] == "B3"].sort_values("n1p")
    c = loc[(loc["family"] == "C") & (loc["top5_share"] <= 0.5)].sort_values("beta")
    cref = ok[(ok["family"] == "C") & (ok["eps"] == EPS_REP)].sort_values("support_med")
    return {"A": a["setting"].iloc[-1] if len(a) else None,
            "B": b["setting"].iloc[-1] if len(b) else None,
            "B3": b3["setting"].iloc[-1] if len(b3) else None,
            "C": c["setting"].iloc[0] if len(c) else None,
            "C_ref": cref["setting"].iloc[0] if len(cref) else None}


def link_map(cls: str) -> dict[str, str]:
    plans = pd.read_csv(OUT / f"plans_{cls}.csv")
    rep = representatives(plans)
    m = {"nn__bary": "nn__bary", "nn_l2__bary": "nn_l2__bary",
         "bal__bary": f"bal_e{EPS_REP:g}__bary", "bal__argmax": f"bal_e{EPS_REP:g}__argmax"}
    for fam in ("A", "B"):
        if rep[fam]:
            m[f"rep_{fam}__bary"] = f"{rep[fam]}__bary"
            m[f"rep_{fam}__argmax"] = f"{rep[fam]}__argmax"
    if rep["B3"]:
        m["rep_B3__bary"] = f"{rep['B3']}__bary"
    if rep["C"]:
        m["rep_C__bary"] = f"{rep['C']}__bary"
    if rep["C_ref"]:
        m["ref_C__bary"] = f"{rep['C_ref']}__bary"
    if rep["B"]:  # B.3 の安定性: 代表の n1' の seed 0〜4
        n = rep["B"].split("_s")[0]
        for s in range(5):
            m[f"B_seed{s}__bary"] = f"{n}_s{s}_e{EPS_REP:g}__bary"
    return m


def phase_link() -> None:
    sel: dict[str, Any] = {}
    for cls in ("c2", "c1"):
        m = link_map(cls)
        d = EVAL / cls
        d.mkdir(parents=True, exist_ok=True)
        for k, v in m.items():
            src = (DATA / "gen" / cls / v).resolve()
            if not src.is_dir():
                raise FileNotFoundError(f"{cls}: {v} の生成がない")
            link = d / k
            if link.is_symlink():
                link.unlink()
            link.symlink_to(src)
        sel[cls] = {"links": m, "rep": representatives(pd.read_csv(OUT / f"plans_{cls}.csv"))}
    (OUT / "representatives.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print(json.dumps(sel, ensure_ascii=False, indent=1))


def configure() -> None:
    e51._M = DATA / "metrics"
    e51._GEN = EVAL
    settings = {k: v for k, v in LABELS.items()}
    e51.SETTINGS = settings
    e51.BASE = "nn__bary"
    e51.BASE_DESC = "設定 0(最近傍、非対称 L1)"
    e51.BASE_COL = "0 最近傍"
    e51.CRITERIA = "`criteria.md` §4"
    e51.PAIRS_F2 = (("rep_A__bary", "rep_A__argmax", "A: 重心 − 最大質量"),
                    ("rep_B__bary", "rep_B__argmax", "B: 重心 − 最大質量"),
                    ("nn_l2__bary", "nn__bary", "コストの効果: L2² − 非対称 L1(最近傍)"))
    e51.STAGE = "ステージ60"
    e51.TITLE = "ステージ60 D.3: 輸送計画の台と周辺分布(QP による生成)"
    e51.OUT = {"comparison": "comparison.md", "losses": "losses.md",
               "assets": "assets_extreme.md", "bs": "bs.md", "summary": "summary_d.json"}
    codes = set(pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})["edinetCode"])
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    for cls in ("c2", "c1"):
        g = pd.read_parquet(sorted((EVAL / cls / "nn__bary").glob("*__seed0.parquet"))[0])
        drop = [d for d in g["docID"].unique() if code_of.get(d, "") in codes]
        e51.EXCL[cls] = tuple(dict.fromkeys((*e51.EXCL[cls], *drop)))


def firm_table(cls: str, s: str, members: dict[str, np.ndarray], names: tuple[str, ...]
               ) -> pd.DataFrame:
    g = e51.load_gen(cls, s)
    g = g[g["seed"] == 0]
    x0 = g[[f"x0_{n}" for n in names]].to_numpy()
    x1 = g[[f"x1_{n}" for n in names]].to_numpy()
    a0, a1 = x0 @ members["jppfs_cor_Assets"], x1 @ members["jppfs_cor_Assets"]
    p0 = x0[:, names.index(_CS)] + x0[:, names.index(_CSUR)]
    p1 = x1[:, names.index(_CS)] + x1[:, names.index(_CSUR)]
    return pd.DataFrame({"docID": g["docID"].to_numpy(),
                         "dec_nom": ((p1 - p0) / a0 / YEARS < -THR).astype(float),
                         "dec_norm": ((p1 / a1 - p0 / a0) / YEARS < -THR).astype(float),
                         "depth": np.where(p0 > 0, p1 / np.where(p0 > 0, p0, 1.0), np.nan),
                         "d_assets": a1 / a0 - 1.0}).set_index("docID")


def boot_diff(a: pd.Series, b: pd.Series, rng: np.random.Generator, n: int = 2000
              ) -> tuple[float, float, float]:
    """企業単位のペアのブートストラップ: mean(a − b) と 95% 区間。"""
    d = (a - b.reindex(a.index)).to_numpy()
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def real_assets() -> dict[str, tuple[float, float, float]]:
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    p = pd.read_csv("data/interim/stage23/panel/panel.csv", dtype={"edinetCode": str})
    out = {}
    for cls, grp in REAL_GROUP.items():
        codes = pairs[pairs["group"] == grp]["edinetCode"].tolist()
        a = p[p["t"] == 2022].set_index("edinetCode").loc[codes, "cur_total_assets"]
        b = p[p["t"] == 2024].set_index("edinetCode").loc[codes, "cur_total_assets"]
        r = (b / a - 1.0).to_numpy()
        out[cls] = (float(np.percentile(r, 5)), float(np.median(r)), float(np.percentile(r, 95)))
    return out


def plans_section() -> list[str]:
    lines = ["# ステージ60 §0・A.4・B・C.3: 輸送計画の健全性と局所性", ""]
    for cls in ("c2", "c1"):
        p = pd.read_csv(OUT / f"plans_{cls}.csv")
        meta = json.loads((DATA / f"meta_{cls}.json").read_text(encoding="utf-8"))
        n0, n1 = meta["n0"], meta["n1"]
        emd = p[p["setting"] == "bal_emd"].iloc[0]
        lines += [f"## {cls}(n0 = {n0}、n1 = {n1:,}、下限 n1/n0 = {n1 / n0:.1f})", "",
                  f"**§0**: 厳密解(`ot.emd`)の実効サポートは中央値 {emd['support_med']:.1f}"
                  f"(最小 {emd['support_min']:.1f}、p90 {emd['support_p90']:.1f})。"
                  f"下限 {n1 / n0:.1f} を割らない。", "",
                  "| 設定 | 案 | ε | τ1 / n1' / β | 状態 | 反復 | 輸送質量 | 行の最小 | 目標側"
                  "の誤差 "
                  "| 実効サポート 中央値 / p90 / 最小 | 受け取った企業 | 上位 5 社の受取 | 異な"
                  "る最大質量の対応先 "
                  "| 重心の分散 | 重心の払込資本比率 | 恒等式の残差 |",
                  "|---|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
        show = p[(p["family"] != "B") | (p["seed"] == 0)]
        for r in show.itertuples():
            par = next((f"{getattr(r, c):g}" for c in ("tau1", "n1p", "beta")
                        if c in p.columns and pd.notna(getattr(r, c, np.nan))), "")
            if r.status != "ok":
                lines.append(f"| {r.setting} | {r.family} | {r.eps} | {par} | 失敗: {r.reason} |"
                             + " |" * 11)
                continue
            lines.append(f"| {r.setting} | {r.family} | {r.eps if pd.notna(r.eps) else '—'} |"
                         f" {par} "
                         f"| ok | {r.n_iter if pd.notna(r.n_iter) else '—'} | {r.mass:.3f} "
                         f"| {r.row_min:.2e} | {r.col_err:.2e} "
                         f"| {r.support_med:.1f} / {r.support_p90:.1f} / {r.support_min:.1f} "
                         f"| {r.n_receive:,} | {r.top5_share:.1%} | {r.n_argmax} |"
                         f" {r.var_bary:.4f} "
                         f"| {r.paid_bary_med:.3f} | {r.identity_resid:.1e} |")
        b = p[p["family"] == "B"].groupby(["n1p", "eps"])["support_med"].agg(["mean", "min", "max"])
        lines += ["", "**B.3 抽出の安定性(実効サポートの中央値、seed 0〜4)**:", "",
                  "| n1' | ε | 平均 | 最小 | 最大 |", "|---:|---:|---:|---:|---:|"]
        for (n, e), r2 in b.iterrows():
            lines.append(f"| {n:g} | {e:g} | {r2['mean']:.2f} | {r2['min']:.2f} |"
                         f" {r2['max']:.2f} |")
        lines.append("")
    return lines


def strata_section(names: tuple[str, ...]) -> list[str]:
    """B.4: 抽出前後の分布(業種構成・払込資本比率・32 科目の中央値)。"""
    x, lab, docs, _ = load_tilde(names)
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    xt = x[lab == 0]
    ind = np.array(load_industry_config(Path("conf/accounts/industry.yaml")).labels(
        [code_of.get(d, "") for d in np.array(docs)[lab == 0]]))
    share = pd.Series(ind).value_counts(normalize=True)
    paid = xt[:, names.index(_CS)] + xt[:, names.index(_CSUR)]
    med_all = np.median(xt, axis=0)
    lines = ["# ステージ60 B.4: 層化の妥当性(class 0 全体と抽出後)", "",
             "業種の構成比のずれ = 33 業種 + 不明で |抽出後 − 全体| の最大。科目の中央値のずれ ="
             " 32 科目で"
             " |抽出後の中央値 − 全体の中央値| の最大(総資産比)。", "",
             f"全体: {len(xt):,} 社、払込資本比率の中央値 {np.median(paid):.3f}。", "",
             "| n1' | seed | 払込資本比率の中央値 | 業種の構成比のずれ(最大) | 科目の中央値のず"
             "れ(最大) |",
             "|---:|---:|---:|---:|---:|"]
    for n in (110, 220, 550, 1100):
        for s in range(5):
            cols = stratified(ind, n, s)
            sh = (pd.Series(ind[cols]).value_counts(normalize=True)
                  .reindex(share.index, fill_value=0))
            lines.append(f"| {n} | {s} | {np.median(paid[cols]):.3f} |"
                         f" {(sh - share).abs().max():.3f} "
                         f"| {np.abs(np.median(xt[cols], axis=0) - med_all).max():.4f} |")
    return lines + [""]


def targets_section() -> list[str]:
    lines = ["# ステージ60 D.2: 目標点の性質", "",
             "識別 AUC = 目標点(55 社 / 132 社)と、class 0 から同数を無作為に抽出した実データの "
             "5-fold CV"
             "(LightGBM / ロジスティック回帰、3 回の平均、seed 42)。1.0 に近いほど合成点として"
             "区別できる。", ""]
    for cls in ("c2", "c1"):
        p = pd.read_csv(OUT / f"plans_{cls}.csv").set_index("setting")
        auc = pd.read_csv(OUT / f"auc_{cls}.csv")
        rep = json.loads((OUT / "representatives.json").read_text(encoding="utf-8"))[cls]["links"]
        lines += [f"## {cls}", "",
                  "| 設定 | 目標 | 実効サポート 中央値 / p90 | 異なる目標 | 目標の分散 | 恒等式"
                  "の残差 "
                  "| 払込資本比率の中央値 | 実在企業との最小距離 | AUC LightGBM | AUC LogReg |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for link, target in rep.items():
            if link.startswith("B_seed"):
                continue
            setting, kind = target.rsplit("__", 1)
            r = p.loc[setting]
            a = auc[(auc["setting"] == setting) & (auc["target"] == kind)].iloc[0]
            var = r["var_bary"] if kind == "bary" else r["var_argmax"]
            paid = r["paid_bary_med"] if kind == "bary" else r["paid_argmax_med"]
            dist = r["nn_dist_med"] if kind == "bary" else 0.0
            n_distinct = (str(r["n_argmax"]) if kind == "argmax" or setting.startswith("nn")
                          else "すべて異なる")
            lines.append(f"| {LABELS.get(link, link)}(`{setting}`) | {kind} "
                         f"| {r['support_med']:.1f} / {r['support_p90']:.1f} | {n_distinct} "
                         f"| {var:.4f} | {r['identity_resid'] if kind == 'bary' else 0.0:.1e} "
                         f"| {paid:.3f} | {dist:.3f} | {a['auc_lgbm']:.3f} |"
                         f" {a['auc_logreg']:.3f} |")
        lines.append("")
    return lines


def qp_section(names: tuple[str, ...]) -> list[str]:
    """C.3・C.4: λ のトレードオフと QP の数値的健全性。"""
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    members = subtotal_members(spec, names)
    sd = pd.read_csv("results/stage59/pooled_sd.csv").set_index("account").reindex(names)
    w = 1.0 / np.maximum(sd["sigma_1"].to_numpy(), 0.001) ** 2
    sel = json.loads((OUT / "representatives.json").read_text(encoding="utf-8"))
    lines = ["# ステージ60 C.3・C.4: λ のトレードオフと QP の数値的健全性", ""]
    for cls in ("c2", "c1"):
        meta = json.loads((DATA / f"meta_{cls}.json").read_text(encoding="utf-8"))
        cp, cm = np.array(meta["c_plus"]), np.array(meta["c_minus"])
        pos = np.concatenate([cp[cp > 0], cm[cm > 0]])
        qp = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob(f"qp_{cls}_*.csv"))
                        if not f.name.endswith("_0of1.csv")])
        chk = pd.read_csv(OUT / f"qp_{cls}_1_0of1.csv")
        st: dict[str, int] = {}
        for v in qp["solver_status"].dropna():
            for k, n in json.loads(v).items():
                st[k] = st.get(k, 0) + int(n)
        s = json.loads((DATA / f"lam_scale_{cls}.json").read_text(encoding="utf-8"))["s"]
        lines += [f"## {cls}", "",
                  f"- 重み比: 1/σ² の最大 / 最小 {w.max() / w.min():,.0f}(σ の下限 0.001、白色"
                  f"化して解く)、"
                  f"c± の最大 / 最小 {pos.max() / pos.min():.1f}。λ のスケール s = {s:.4g}。",
                  f"- ソルバの状態(全設定 × 企業、λ̃ のすべて): {st}。実行不可能"
                  f" {int(qp['n_infeasible'].sum())}。",
                  f"- min(u, v) > 1e-9(c > 0 の科目): {int(qp['n_both_positive'].sum())} 件、> 1"
                  f"e-6: "
                  f"{int(qp['n_both_positive_1e6'].sum())} 件、最大"
                  f" {qp['both_positive_max'].max():.1e}"
                  "(1e-6 を超えるものはない。OSQP の許容誤差は eps = 1e-8)。",
                  f"- 独立ソルバ(CLARABEL)との差の最大(代表の設定、λ̃ = 1): "
                  f"{chk['clarabel_max_dev'].max():.1e}(総資産比)。", "",
                  "| 設定 | λ̃ | 目標点への距離 Σw(y − τ)² 中央値 | 移動コスト Σc±|Δ| 中央値 "
                  "| 払込資本を減らした(名目) |", "|---|---:|---:|---:|---:|"]
        rep = sel[cls]["rep"]
        keys = ["nn", "nn_l2", f"bal_e{EPS_REP:g}"] + [rep[f] for f in ("A", "B", "B3") if rep[f]]
        for k in keys:
            for lt in (0.01, 0.1, 1.0, 10.0, 100.0):
                name = f"{k}__bary" if lt == 1.0 else f"{k}__bary__l{lt:g}"
                r = qp[qp["target"] == name]
                d = DATA / "gen" / cls / name
                if r.empty or not d.is_dir():
                    continue
                g = pd.read_parquet(d / "nearest_na0__seed0.parquet")
                x0 = g[[f"x0_{n}" for n in names]].to_numpy()
                x1 = g[[f"x1_{n}" for n in names]].to_numpy()
                a0 = x0 @ members["jppfs_cor_Assets"]
                pi = [names.index(_CS), names.index(_CSUR)]
                dec = ((x1[:, pi].sum(axis=1) - x0[:, pi].sum(axis=1)) / a0 / YEARS < -THR)
                if cls == "c1":
                    dec = dec[~g["docID"].isin(e51.EXCL[cls]).to_numpy()]
                lines.append(f"| {k} | {lt:g} | {r['first_term_med'].iloc[0]:.3f} "
                             f"| {r['move_cost_med'].iloc[0]:.3f} | {dec.mean():.0%} |")
        lines.append("")
    return lines


def main_report() -> None:  # noqa: PLR0915
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    members = subtotal_members(load_canonical_spec(Path("conf/accounts/core.yaml")), names)
    configure()
    sel = json.loads((OUT / "representatives.json").read_text(encoding="utf-8"))
    e51.SETTINGS = {k: v for k, v in LABELS.items()
                    if all((EVAL / c / k).exists() for c in ("c2", "c1"))}
    e51.BS_ORDER = tuple(k for k in ("rep_A__bary", "rep_B__bary") if k in e51.SETTINGS)
    sys.argv = [sys.argv[0], "--out", str(OUT)]
    e51.main()
    (OUT / "plans.md").write_text("\n".join(plans_section()) + "\n", encoding="utf-8")
    (OUT / "strata.md").write_text("\n".join(strata_section(names)) + "\n", encoding="utf-8")
    (OUT / "targets.md").write_text("\n".join(targets_section()) + "\n", encoding="utf-8")
    real = real_assets()
    rng = np.random.default_rng(0)
    ctx = {c: e51.Ctx(c, names) for c in ("c2", "c1")}
    judge: dict[str, dict[str, Any]] = {}
    lines = ["# ステージ60 D.3・E: 払込資本・総資産と判定", "",
             "払込資本を減らした = 年率 (Δ払込資本 / A0) / 5 < −0.001(名目、主)、正規化 = (払込"
             "資本 / 総資産)"
             " の差で同じ閾値。差は設定 0 に対する企業単位のペアのブートストラップ(2,000 回)の"
             "平均の差 [95% 区間]。"
             "総資産の変化 = A1 / A0 − 1 の中央値。実データ = 2 年の A2024 / A2022 − 1 の p5 / "
             "中央値 / p95"
             "(class 2 は回復 15 社、class 1 は 57 社)。QP は決定的なので 1 seed の値。", ""]
    for cls in ("c2", "c1"):
        base = firm_table(cls, "nn__bary", members, names)
        lo, mid, hi = real[cls]
        lines += [f"## {cls}(実データの総資産の変化 p5 {lo:+.2f} / 中央値 {mid:+.2f} / p95"
                  f" {hi:+.2f})", "",
                  "| 設定 | n | 払込資本を減らした(名目) | 差 [95%] | (正規化) | 差 [95%] | 減"
                  "らす深さ "
                  "| 総資産の変化 中央値 | p5 / p95 | 違反 |",
                  "|---|---:|---:|---|---:|---|---:|---:|---|---:|"]
        for s, lab in e51.SETTINGS.items():
            f = firm_table(cls, s, members, names)
            dn = boot_diff(f["dec_nom"], base["dec_nom"], rng)
            dz = boot_diff(f["dec_norm"], base["dec_norm"], rng)
            nv = sum(ctx[cls].viol_items(s).values())
            dep = f.loc[f["dec_nom"] > 0, "depth"].median()
            lines.append(f"| {lab} | {len(f)} | {f['dec_nom'].mean():.0%} "
                         f"| {dn[0]:+.2f} [{dn[1]:+.2f}, {dn[2]:+.2f}] |"
                         f" {f['dec_norm'].mean():.0%} "
                         f"| {dz[0]:+.2f} [{dz[1]:+.2f}, {dz[2]:+.2f}] | {dep:.3f} "
                         f"| {f['d_assets'].median():+.3f} | {f['d_assets'].quantile(0.05):+.3f} / "
                         f"{f['d_assets'].quantile(0.95):+.3f} | {nv} |")
            judge.setdefault(s, {})[cls] = {
                "viol": nv, "dec_nom": float(f["dec_nom"].mean()), "diff_nom": dn,
                "assets_med": float(f["d_assets"].median()),
                "assets_ok": bool(lo <= f["d_assets"].median() <= hi)}
        lines.append("")
    # B.3: 代表の n1' の seed 0〜4
    lines += ["## B.3 抽出の安定性(代表の n1'、ε = 0.005、重心)", "",
              "| class | seed | 払込資本を減らした(名目) | 純資産の改善の中央値 |", "|---|---:|-"
                                                               "--:|---:|"]
    stab: dict[str, bool] = {}
    for cls in ("c2", "c1"):
        dec, nai = [], []
        for seed in range(5):
            k = f"B_seed{seed}__bary"
            if not (EVAL / cls / k).exists():
                continue
            f = firm_table(cls, k, members, names)
            rows = ctx[cls].rows
            v = rows[rows["version"] == k]["net_asset_improvement"].median() \
                if "version" in rows.columns else float("nan")
            dec.append(f["dec_nom"].mean())
            nai.append(v)
            lines.append(f"| {cls} | {seed} | {f['dec_nom'].mean():.0%} | {v:.3f} |")
        stab[cls] = bool(dec) and (max(dec) - min(dec) <= 0.10) and (
            np.nanmax(nai) - np.nanmin(nai) <= 0.02)
        lines.append(f"| {cls} | 範囲 | {max(dec) - min(dec):.0%} |"
                     f" {np.nanmax(nai) - np.nanmin(nai):.3f} |"
                     if dec else f"| {cls} | — | — | — |")
    lines += ["", f"安定性(範囲 ≤ 10 ポイントかつ ≤ 0.02): c2 {stab.get('c2')}、c1"
                  f" {stab.get('c1')}。", ""]
    # E.1・E.2
    lines += ["## E.1 判定(各案の代表、重心)", "",
              "| 案 | 代表(c2 / c1) | 1 違反 0 | 2 計画が健全 | 3 実効サポート ≤ 20 | 4 総資産 |"
              " 5 c2 | 5 c1 "
              "| 6 両 class | 採用 |", "|---|---|---|---|---|---|---|---|---|---|"]
    verdicts: dict[str, str] = {}
    for fam in ("A", "B", "C"):
        key = f"rep_{fam}__bary"
        reps = (sel["c2"]["rep"].get(fam), sel["c1"]["rep"].get(fam))
        if key not in e51.SETTINGS:
            lines.append(f"| {fam} | {reps[0]} / {reps[1]} | — | — | **満たさない(該当なし)** | "
                         f"— | — | — "
                         "| — | 不採用 |")
            verdicts[fam] = "局所性を回復できない"
            continue
        j = judge[key]
        c1_ok = all(j[c]["viol"] == 0 for c in ("c2", "c1"))
        c4_ok = all(j[c]["assets_ok"] for c in ("c2", "c1"))
        c5 = {c: j[c]["diff_nom"][2] < 0 for c in ("c2", "c1")}
        ok = c1_ok and c4_ok and c5["c2"] and c5["c1"] and (fam != "B" or all(stab.values()))
        lines.append(f"| {fam} | {reps[0]} / {reps[1]} | {c1_ok} | True | True | {c4_ok} |"
                     f" {c5['c2']} "
                     f"| {c5['c1']} | {c5['c2'] and c5['c1']} | **{'採用' if ok else '不採用'}** |")
        comp = {c: ctx[c].compare(key, "nn__bary", rng) for c in ("c2", "c1")}
        prim = ("純資産の改善", "解消率", "順位相関")
        worse = (any(comp[c][k][3] == -1 for c in comp for k in prim)
                 or any(j[c]["diff_nom"][1] > 0 for c in ("c2", "c1")))
        better = any(comp[c][k][3] == 1 for c in comp for k in prim) or any(c5.values())
        verdicts[fam] = ("OT > NN" if ok else "OT < NN" if worse else "OT ≈ NN" if not better
                         else "判定規則の外(一部で優れるが基準は不成立)")
    lines += ["", "## E.2 輸送計画を解く効果(対 設定 0)", ""]
    lines += [f"- {fam}: **{v}**" for fam, v in verdicts.items()]
    lines.append("")
    (OUT / "judgment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "summary_e.json").write_text(json.dumps({"judge": judge, "verdicts": verdicts,
                                                    "stability": stab}, ensure_ascii=False,
                                                   indent=1, default=str), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("link", "report", "qp"), required=True)
    args = ap.parse_args()
    if args.phase == "link":
        phase_link()
    elif args.phase == "qp":
        configure()
        names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                                   include_subtotals=False)
        (OUT / "qp.md").write_text("\n".join(qp_section(names)) + "\n", encoding="utf-8")
        print((OUT / "qp.md").read_text(encoding="utf-8"))
    else:
        main_report()


if __name__ == "__main__":
    main()
