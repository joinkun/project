#!/usr/bin/env python
"""ステージ69 A: 論文・発表用のマスター表(既存の生成と指標から集計するだけ。新しい生成はしない)。

すべて class 1 で較正した λ(リークなし)の生成を使う。class 2 には class 1 の λ の絶対値を当てはめた
生成だけを使い、存在しないもの(1-NN と等重み L2 の class 2)は空欄にして、近い格子点の値を
参考に示す。
lightgbm・sklearn のプロセス(``metrics_stage39`` の出力を読む)。

出力: ``results/stage69/paper_tables.md``・``paper_tables.json``。

    uv run python scripts/tables_stage69.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import numpy as np
import pandas as pd
from compare_stage39 import boot, corr, verdict
from diag_stage68 import flags
from evaluate_stage61 import gen_firms, real_targets
from evaluate_stage65 import real_firm_values
from evaluate_stage67 import extra_cols, real_ratio
from targets_stage68 import pool, props, sources
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import CHECK_NAMES, EXTRA_CHECK_NAMES, subtotal_members
from fsgen.evaluation.paired import paired_test

OUT = Path("results/stage69")
INTERIM = Path("data/interim")
CHECKS = (*CHECK_NAMES, *EXTRA_CHECK_NAMES)


@dataclass(frozen=True)
class Src:
    metrics: Path
    gen: Path
    setting: str
    lam: str  # 表示用の λ̃(class 1 の較正値)
    note: str = ""


S65 = (INTERIM / "stage65/metrics", INTERIM / "stage65/all")
S67 = (INTERIM / "stage67/metrics", INTERIM / "stage67/all")
S68A = (INTERIM / "stage68/metrics_a", INTERIM / "stage68/all_a")
S68 = (INTERIM / "stage68/metrics", INTERIM / "stage68/gen")
S64 = (INTERIM / "stage64/metrics", INTERIM / "stage64/all")
# 手法 → class → 生成(None は class 1 の λ を当てはめた生成がない)
CELLS: dict[str, dict[str, Src | None]] = {
    "prop": {"c2": Src(*S67, "s1__l1", "1"), "c1": Src(*S67, "s1__l1", "1")},
    "l1only": {"c2": Src(*S68A, "m2_l1only__l1", "1"), "c1": Src(*S65, "m2_l1only__l1", "1")},
    "bary": {"c2": Src(*S68A, "m7_bary__l3", "3"), "c1": Src(*S65, "m7_bary__l3", "3")},
    "balarg": {"c2": Src(*S68A, "m6_balarg__l30", "30"), "c1": Src(*S65, "m6_balarg__l30", "30")},
    "nn": {"c2": None, "c1": Src(*S64, "t1_nn__l1a_l1", "1")},
    "l2": {"c2": None, "c1": Src(*S65, "m3_l2__l10", "10")},
    "m1": {"c2": Src(*S68, "t0m1__l3", "3"), "c1": Src(*S68, "t0m1__l3", "3")},
    "rep": {"c2": Src(*S68, "t0m10__l1", "1"), "c1": Src(*S68, "t0m10__l1", "1")},
    "re": {"c2": Src(*S67, "s2__l1", "1"), "c1": Src(*S67, "s2__l1", "1")},
    "t1": {"c2": Src(*S68, "t1m1__l10", "10"), "c1": Src(*S68, "t1m1__l10", "10")},
    "t2": {"c2": Src(*S68, "t2m1__l30", "30"), "c1": Src(*S68, "t2m1__l30", "30")},
    "t2ref": {"c2": Src(*S68, "t2m1__l0.01", "0.01(参考、違反あり)"),
              "c1": Src(*S68, "t2m1__l0.01", "0.01(参考、違反あり)")},
}
# class 2 に class 1 の λ を当てはめた生成がないものの参考(近い格子点。λ の絶対値が違う)
REF_C2 = {"nn": Src(*S64, "t1_nn__l1a_l1", "1(class 2 の格子、λ は class 1 の値の 0.92 倍)"),
          "l2": Src(*S65, "m3_l2__l10", "10(class 2 の格子、λ は class 1 の値の 0.81 倍)")}
_CTX: dict[tuple[Path, str], e51.Ctx] = {}


def ctx(src: Src, cls: str, names: tuple[str, ...]) -> e51.Ctx:
    e51._M, e51._GEN = src.metrics, src.gen
    key = (src.metrics, cls)
    if key not in _CTX:
        _CTX[key] = e51.Ctx(cls, names)
    return _CTX[key]


def setup_excl(names: tuple[str, ...]) -> None:
    codes = set(pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})["edinetCode"])
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    _x, _lab, docs, _ = load_tilde(names)
    drop = [d for d in docs if code_of.get(d, "") in codes]
    for cls in ("c2", "c1"):
        e51.EXCL[cls] = tuple(dict.fromkeys((*e51.EXCL[cls], *drop)))


def metrics(src: Src, cls: str, names: tuple[str, ...], members: dict[str, np.ndarray]
            ) -> dict[str, Any]:
    c = ctx(src, cls, names)
    f = gen_firms(cls, src.setting, names, members).join(
        extra_cols(cls, src.setting, names, members))
    f = f[~f.index.isin(e51.EXCL[cls])]
    cr = c.comp[c.comp["version"] == src.setting]
    v = c.viol[c.viol["setting"] == src.setting].drop_duplicates(["docID", "check"])
    return {"lam": src.lam, "n": int(len(f)), "viol": int(v["docID"].nunique()),
            "items": {k.split("_")[0]: int((v["check"] == k).sum()) for k in CHECKS
                      if int((v["check"] == k).sum())},
            "nai": float(cr["net_asset_improvement"].median()),
            "res": float(cr.loc[cr["insolvent_start"].astype(bool), "insolvency_resolved"].mean()),
            "paid_dec": float(f["paid_dec"].mean()), "paid_inc": float(f["paid_inc"].mean()),
            "re_up": float(f["re_up"].mean()), "n_changed": float(f["n_changed"].median()),
            "re_lt1": float(f["re_lt1"].mean()), "d_assets": float(f["d_assets"].median()),
            "rho": float(corr(c.wide[src.setting], c.ref))}


def versus(a: Src, b: Src, cls: str, names: tuple[str, ...], rng: np.random.Generator
           ) -> dict[str, Any]:
    """a − b の企業単位のペア比較(純資産の改善・解消率・順位相関)。"""
    ca, cb = ctx(a, cls, names), ctx(b, cls, names)
    ga = ca.comp[ca.comp["version"] == a.setting].set_index("docID")
    gb = cb.comp[cb.comp["version"] == b.setting].set_index("docID").reindex(ga.index)
    r = paired_test(ga["net_asset_improvement"].to_numpy(float),
                    gb["net_asset_improvement"].to_numpy(float))
    sel = gb["insolvent_start"].astype(bool)
    rr = paired_test(ga.loc[sel, "insolvency_resolved"].to_numpy(float),
                     gb.loc[sel, "insolvency_resolved"].to_numpy(float))
    wa = ca.wide[a.setting]
    wb = cb.wide[b.setting].loc[wa.index]
    d = corr(wa, ca.ref) - corr(wb, cb.ref)
    _, _, lo, hi = boot(wa, ca.ref, wb, cb.ref, rng)
    return {"nai": (r.median_diff, r.ci_low, r.ci_high),
            "sign": [verdict("大", r.median_diff, r.ci_low, r.ci_high),
                     verdict("大", rr.median_diff, rr.ci_low, rr.ci_high),
                     verdict("大", d, lo, hi)],
            "rho": (d, lo, hi)}


def real_row(cls: str, names: tuple[str, ...], members: dict[str, np.ndarray]) -> dict[str, Any]:
    r, _se, n_real = real_targets(cls, names)
    rf = real_firm_values(cls, names, members)
    paid = pd.read_csv("results/stage56/paidin.csv", dtype={"edinetCode": str})
    pg = "再建群(class 2 → 回復)" if cls == "c2" else "再建群(class 1 → class 0)"
    pp = paid[paid["group"] == pg]["annual_d_paid"]
    rr = real_ratio(cls, names, members)
    return {"n": int(len(rf)), "paid_de"
                               "c": float(r["paid_dec"]), "paid_inc": float((pp > 0.001).mean()),
            "re_up": float(r["re_up"]), "n_change"
                                        "d": float(n_real), "re_lt1": float((rr < 0.99).mean()),
            "nai_annual": float(rf["nai_annual"].median()), "d_assets": float(r["d_assets"]),
            "re_share": float(np.median(rr))}


def pct(v: float) -> str:
    return f"{v:.0%}"


def sym(s: list[int]) -> str:
    return " ".join({1: "+", -1: "−", 0: "0"}[int(v)] for v in s)


def ci(t: tuple[float, float, float]) -> str:
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    OUT.mkdir(parents=True, exist_ok=True)
    setup_excl(names)
    rng = np.random.default_rng(0)
    m: dict[str, dict[str, Any]] = {k: {} for k in (*CELLS, "ref_nn", "ref_l2")}
    for k, by in CELLS.items():
        for cls, src in by.items():
            m[k][cls] = None if src is None else metrics(src, cls, names, members)
    for k, src in REF_C2.items():
        m[f"ref_{k}"]["c2"] = metrics(src, "c2", names, members)
    vs: dict[str, dict[str, Any]] = {}
    for k in ("l1only", "bary", "balarg", "nn", "l2", "m1", "rep", "re", "t1", "t2"):
        vs[k] = {}
        for cls in ("c2", "c1"):
            sa, sb = CELLS[k][cls], CELLS["prop"][cls]
            vs[k][cls] = None if sa is None or sb is None else versus(sa, sb, cls, names, rng)
    real = {cls: real_row(cls, names, members) for cls in ("c2", "c1")}
    auc: dict[str, dict[str, tuple[float, float]]] = {}
    for cls in ("c2", "c1"):
        a65 = pd.read_csv(f"results/stage65/auc_{cls}.csv").set_index("method")
        a64 = pd.read_csv(f"results/stage64/auc_{cls}.csv").set_index("key")
        auc[cls] = {"prop": tuple(a65.loc["m5_prop", ["auc_lgbm", "auc_logreg"]]),
                    "bary": tuple(a65.loc["m7_bary", ["auc_lgbm", "auc_logreg"]]),
                    "balarg": tuple(a65.loc["m6_balarg", ["auc_lgbm", "auc_logreg"]]),
                    "nn": tuple(a64.loc["nn", ["auc_lgbm", "auc_logreg"]]),
                    "l2": tuple(a65.loc["m3_l2", ["auc_lgbm", "auc_logreg"]])}
    plans = {cls: pd.read_csv(f"results/stage64/plans_{cls}.csv").set_index("setting")
             for cls in ("c2", "c1")}

    def g(k: str, cls: str, key: str, fmt: str = "{:.3f}") -> str:
        v = m[k][cls]
        if v is None:
            return "—"
        val = v[key]
        return pct(val) if fmt == "%" else fmt.format(val)

    out: list[str] = ["# ステージ69 論文・発表用マスター表", "",
                    "**すべて class 1 で較正した λ(リークなし)の確定値。** class 2 には class 1"
                    " の λ の"
                    "絶対値(λ̃*_0 × s_c1)を当てはめた生成だけを使う。評価対象は class 2 55 社、"
                    "class 1 "
                    "128 社(外れ値と、再建群と重なる 3 社を除く)。純資産の改善は 5 年後とみなし"
                    "た終点の"
                    "中央値(総資産比)。払込資本↑↓・RE↑ は年率 ±0.001、変更科目数は年率 0.002 超。"
                    "差は提案との企業単位のペア比較(2,000 回のブートストラップ)、記号 = 純資産"
                    "の改善・"
                    "解消率・順位相関(+ 有意に優れる、− 有意に劣る)。集計は "
                    "`scripts/tables_stage69.py`(新しい生成はしていない)。", ""]
    # 表 1
    out += ["## 表 1: 第 1 層(恒等式・符号)のみ 対 第 1 層 + 第 2 層(提案)", "",
          "| class | 手法 | λ̃ | 違反した企業 | 違反の内訳(項目: 企業数) | 純資産の改善 | 解消率 |",
          "|---|---|---:|---:|---|---:|---:|"]
    for cls, n in (("c2", "class 2(55 社)"), ("c1", "class 1(128 社)")):
        for k, lab in (("l1only", "第 1 層のみ"), ("prop", "**提案(第 1 層 + 第 2 層)**")):
            v = m[k][cls]
            it = "、".join(f"{a}: {b}" for a, b in v["items"].items()) or "—"
            out.append(f"| {n} | {lab} | {v['lam']} | {v['viol']} | {it} | {v['nai']:.3f} "
                     f"| {v['res']:.2f} |")
    out += ["", "項目 6 = 純資産が悪化、13 = 控除後の純額が負、2 = 小計が負、15 = 不活性科目が"
                "動いた"
          "(全 15 項目の名前は表 5)。第 1 層のみは、提案と同じ目標・λ で第 2 層・不活性・現金の"
          "下限を外したもの(非操作科目の凍結は残す)。", ""]
    # 表 2
    out += ["## 表 2: 目標(対応先)の選び方の比較(QP は提案と同じ、λ は手法ごとに class 1 で較正"
            ")", "",
          "| class | 目標 | λ̃ | 違反 | 純資産の改善 | 提案との差 [95%] | 解消率 | 払込資本↓ "
          "| 順位相関 | 順位相関の差 [95%] | 記号 | 識別 AUC(LightGBM / LogReg) "
          "| 実効サポート(中央値) |",
          "|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---|---:|"]
    sup = {"prop": "semi_t0.01_e0.001", "bary": "bal_e0.001", "balarg": "bal_e0.001", "nn": "nn"}
    for cls, n in (("c2", "class 2"), ("c1", "class 1")):
        for k, lab in (("prop", "**1 提案: 不均衡 OT(τ1 = 0.01)の argmax**"),
                       ("bary", "2 バランス OT の重心"), ("balarg", "3 バランス OT の argmax"),
                       ("nn", "4 最近傍(非対称 L1)")):
            v = m[k][cls]
            au = auc[cls][k]
            sp = float(plans[cls].loc[sup[k], "support_med"])
            if v is None:
                rv = m["ref_nn"]["c2"]
                out.append(f"| {n} | {lab} | — | — | —(参考 {rv['nai']:.3f}) | — | —(参考 "
                         f"{rv['res']:.2f}) | —(参考 {pct(rv['paid_dec'])}) "
                         f"| —(参考 {rv['rho']:+.3f}) | — | — | {au[0]:.3f} / {au[1]:.3f} "
                         f"| {sp:.2f} |")
                continue
            d = "—" if k == "prop" else ci(vs[k][cls]["nai"])
            dr = "—" if k == "prop" else ci(vs[k][cls]["rho"])
            s_ = "—" if k == "prop" else sym(vs[k][cls]["sign"])
            out.append(f"| {n} | {lab} | {v['lam']} | {v['viol']} | {v['nai']:.3f} | {d} "
                       f"| {v['res']:.2f} | {pct(v['paid_dec'])} | {v['rho']:+.3f} | {dr} | {s_} "
                       f"| {au[0]:.3f} / {au[1]:.3f} | {sp:.2f} |")
    nn_match = {cls: int(plans[cls].loc["semi_t0.01_e0.001", "nn_match"]) for cls in ("c2", "c1")}
    emd = {cls: float(plans[cls].loc["bal_emd", "support_med"]) for cls in ("c2", "c1")}
    out += ["", f"- 提案の argmax が最近傍と一致: class 2 で {nn_match['c2']} / 55、class 1 で "
          f"{nn_match['c1']} / 132。",
          f"- バランス OT の実効サポート(1/Σp²、行で正規化)は、ε = 0.001 で上の値、厳密解で "
          f"{emd['c2']:.2f} / {emd['c1']:.2f}。下界 n1/n0 = 68.51 / 28.55(命題 1)。",
          "- class 1 の重心の順位相関の差は、区間の上端が 0 に接する境界の値(ステージ65 の同じ"
          "比較では"
          "[−0.187, +0.001] で有意でなかった)。ブートストラップの乱数で判定が変わりうる。",
          "- 重心の AUC 0.995 / 0.999 は LightGBM の値で、ロジスティック回帰では 0.589 / 0.599。"
          "木のモデルが「ちょうど 0」の構造で見分ける(命題 2)。",
          "- 4 の class 2: class 1 の λ(λ̃ = 1 × s_c1)を当てはめた生成がない(本ステージは新しい"
          "生成を"
          "しない)。「参考」は class 2 の格子の λ̃ = 1(λ は class 1 の値の 0.92 倍)の値。", ""]
    # 表 3
    out += ["## 表 3: QP の設計と事後選択 m = 10 の感度分析", "",
            "| class | 設定 | λ̃ | 違反 | 純資産の改善 | 提案との差 [95%] | 解消率 | 払込資本↓ "
            "| 払込資本↑ | RE↑ | 変更科目数 | ΔRE/ΔNA < 0.99 | 順位相関 | 順位相関の差 [95%] "
            "| 記号 |",
            "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for cls, n in (("c2", "class 2"), ("c1", "class 1")):
        for k, lab in (("prop", "**1 提案(非対称 L1 + QP、払込資本 m = 10)**"),
                       ("rep", "1r 提案の再現(`RecourseQP`)"),
                       ("m1", "2 事後選択なし(m = 1)"), ("l2", "3 等重み L2 + QP(最近傍)"),
                       ("re", "4 利益剰余金にコスト(c_RE > 0)")):
            v = m[k][cls]
            if v is None:
                rv = m["ref_l2"]["c2"]
                out.append(f"| {n} | {lab} | — | — | —(参考 {rv['nai']:.3f}) | — "
                           f"| —(参考 {rv['res']:.2f}) | —(参考 {pct(rv['paid_dec'])}) | — | — "
                           f"| —(参考 {rv['n_changed']:.0f}) | — | —(参考 {rv['rho']:+.3f}) "
                           "| — | — |")
                continue
            d = "—" if k == "prop" else ci(vs[k][cls]["nai"])
            dr = "—" if k == "prop" else ci(vs[k][cls]["rho"])
            s_ = "—" if k == "prop" else sym(vs[k][cls]["sign"])
            out.append(f"| {n} | {lab} | {v['lam']} | {v['viol']} | {v['nai']:.3f} | {d} "
                       f"| {v['res']:.2f} | {pct(v['paid_dec'])} | {pct(v['paid_inc'])} "
                       f"| {pct(v['re_up'])} | {v['n_changed']:.0f} | {pct(v['re_lt1'])} "
                       f"| {v['rho']:+.3f} | {dr} | {s_} |")
        r = real[cls]
        out.append(f"| {n} | 実データ(再建群 {r['n']} 社) | — | — | {r['nai_annual']:.3f}(年率) "
                   f"| — | — | {pct(r['paid_dec'])} | {pct(r['paid_inc'])} | {pct(r['re_up'])} "
                   f"| {r['n_changed']:.0f} | {pct(r['re_lt1'])} | — | — | — |")
    out += ["", "- 1 は最終設定(`project_onto_feasible`)。2・4 は `RecourseQP` で解いたので、同じ"
          "ソルバでの再現 1r と並べる(1r と 1 の差は表の「提案との差」)。",
          "- 2 と 1 の比較: m = 10 を外すと、class 1 の較正で λ̃ = 3 が選ばれる。払込資本↓ は過剰に"
          "ならない(較正した λ が抑える)が、純資産の改善は下がる。",
          "- 3 の class 2: class 1 の λ(λ̃ = 10 × s_c1)を当てはめた生成がない。「参考」は class 2 の"
          "格子の λ̃ = 10(λ は class 1 の値の 0.81 倍)の値。",
          "- 実データの純資産の改善は 2 年の年率。生成は 5 年分の終点(年率にすると ÷ 5)。", ""]
    # 表 4
    out += ["## 表 4: 目標の定義の比較(T0 水準のマッチング・T1 再建終点・T2 遷移加算)", "",
          "**(a) QP の前の目標そのものの向き**(差 = τ − x。払込資本↑ は差 > 0.005、実データは年"
          "率 > "
          "0.001)", "",
          "| class | 目標 | τ の払込資本 < 始点 | τ での ΔRE/ΔNA 中央値 | ΔRE/ΔNA < 0.99 "
          "| τ での払込資本↑ |", "|---|---|---:|---:|---:|---:|"]
    x, labels_a, docs, _ = load_tilde(names)
    docs_a = np.array(docs)
    na = members["jppfs_cor_NetAssets"]
    for cls in ("c2", "c1"):
        xs = x[sources(cls, labels_a, docs_a)]
        z = np.load(INTERIM / f"stage68/targets_{cls}.npz", allow_pickle=True)
        for key, lab_t in (("t0", "T0(健全企業、現行)"), ("t1", "T1(再建終点)"),
                           ("t2", "T2(遷移加算)")):
            p_ = props(xs, np.asarray(z[key]), names, na)
            out.append(f"| {cls} | {lab_t} | {pct(p_['paid_lower'])} | {p_['re_share']:.2f} "
                     f"| {pct(p_['re_lt1'])} | {pct(p_['paid_up'])} |")
        r = real[cls]
        out.append(f"| {cls} | 実データ({r['n']} 社) | — | {r['re_share']:.2f} |"
                   f" {pct(r['re_lt1'])} "
                 f"| {pct(r['paid_inc'])} |")
    out += ["", "**(b) QP の後(λ は設定ごとに class 1 で較正。T2 の λ̃ = 0.01 は違反があり、必"
                "須条件を"
          "満たさない参考値)**", "",
          "| class | 設定 | λ̃ | 違反 | ΔRE/ΔNA < 0.99 | 払込資本↑ | 払込資本↓ | 変更科目数 "
          "| 総資産(年率) | 純資産の改善 | 順位相関 "
          "|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cls in ("c2", "c1"):
        for k, lab_t in (("rep", "T0・m = 10(現行の再現"
                                 ")"), ("m1", "T0・m = 1"), ("t1", "T1・m = 1"),
                         ("t2", "T2・m = 1"), ("t2ref", "T2・m = 1(参考)")):
            v = m[k][cls]
            out.append(f"| {cls} | {lab_t} | {v['lam']} | {v['viol']} | {pct(v['re_lt1'])} "
                     f"| {pct(v['paid_inc'])} | {pct(v['paid_dec'])} | {v['n_changed']:.0f} "
                     f"| {v['d_assets']:+.4f} | {v['nai']:.3f} | {v['rho']:+.3f} |")
        r = real[cls]
        out.append(f"| {cls} | 実データ | — | — | {pct(r['re_lt1'])} | {pct(r['paid_inc'])} "
                 f"| {pct(r['paid_dec'])} | {r['n_changed']:.0f} | {r['d_assets']:+.4f} "
                 f"| {r['nai_annual']:.3f}(年率) | — |")
    out += ["", "T2 の参考(λ̃ = 0.01)の違反の内訳: "
          + "、".join(f"{cls} {m['t2ref'][cls]['items']}" for cls in ("c2", "c1"))
          + "(項目 5 = 負債が 2 倍超)。事前の判定(`results/stage68/criteria.md` §4)では、T1・T2"
            " とも"
          "不採用。", ""]
    # 表 5
    out += ["## 表 5: 実際の再建企業と T0 の目標が、15 項目のチェックに違反する件数", "",
          "15 項目のチェックを、生成の終点の代わりに (a) 実際の再建企業の 2022 → 2024 の変化、"
          "(b) 始点から T0 の目標への変化、に当てた(企業数)。", "",
          "| 対象 | 違反した企業 | " + " | ".join(c.split("_")[0] for c in CHECKS) + " |",
          "|---|---:|" + "---:|" * len(CHECKS)]
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    codes, comp22, _c24, dnom = pool(names)
    grp = pairs.set_index("edinetCode")["group"]
    t5: dict[str, list[int]] = {}
    for gname in ("class 2 → 回復", "class 1 → class 0"):
        ix = [i for i, c in enumerate(codes) if grp[c] == gname]
        fl = flags(comp22[ix], comp22[ix] + dnom[ix], spec, names)
        t5[gname] = [int(fl.any(axis=1).sum()), len(ix), *[int(v) for v in fl.sum(axis=0)]]
        out.append(f"| 実データ {gname}({len(ix)} 社) | {t5[gname][0]} / {len(ix)} | "
                 + " | ".join(str(v) for v in t5[gname][2:]) + " |")
    for cls in ("c2", "c1"):
        xs = x[sources(cls, labels_a, docs_a)]
        z = np.load(INTERIM / f"stage68/targets_{cls}.npz", allow_pickle=True)
        fl = flags(xs, np.asarray(z["t0"]), spec, names)
        t5[f"T0 {cls}"] = [int(fl.any(axis=1).sum()), len(xs), *[int(v) for v in fl.sum(axis=0)]]
        out.append(f"| T0 の目標 {cls}({len(xs)} 社) | {t5[f'T0 {cls}'][0]} / {len(xs)} | "
                 + " | ".join(str(v) for v in t5[f"T0 {cls}"][2:]) + " |")
    out += ["", "項目: " + "、".join(CHECKS) + "。", "",
          "T0 の目標の行は、生成の始点 132 社(class 1)を含む(評価の 128 社ではない)。", ""]
    (OUT / "paper_tables.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    (OUT / "paper_tables.json").write_text(json.dumps(
        {"metrics": m, "versus": vs, "real": real, "table5": t5,
         "auc": {c: {k: list(v) for k, v in a.items()} for c, a in auc.items()}},
        ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
