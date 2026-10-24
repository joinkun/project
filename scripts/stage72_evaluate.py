#!/usr/bin/env python
"""ステージ72: 較正・比較・目標と終点の性質・F1 の判定・BS 表(lightgbm・sklearn のプロセス)。

規則は ``results/stage72/criteria.md``。事前に ``metrics_stage39.py --gen-dir
data/interim/stage72/gen --out data/interim/stage72/metrics`` を実行する。
基準は設定 4(ステージ69 の最終設定、ステージ67 の ``s1__l1``)。

    uv run python scripts/stage72_evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import numpy as np
import pandas as pd
from evaluate_stage61 import CAL, gen_firms, loss, real_targets
from evaluate_stage65 import bs_table
from stage71_evaluate import endpoints
from tables_stage69 import INTERIM, S67, Src, ctx, metrics, setup_excl, versus
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.plausibility import subtotal_members
from fsgen.evaluation.identification import identification_auc_repeated
from fsgen.viz.balance_sheet import load_labels

OUT = Path("results/stage72")
S72 = (INTERIM / "stage72/metrics", INTERIM / "stage72/gen")
LAMS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
LABELS = {"f1": "1 F1 事後アンサンブル(τ1 = 0.05、m = 10)",
          "f1m1": "2 F1(m = 1、事後選択なし)",
          "a2": "3a argmax(τ1 = 0.01)", "a2t05": "3b argmax(τ1 = 0.05)",
          "bal_l2": "5 バランス OT の事前 L2 重心", "f1bal": "6 バランス OT の F1",
          "l2nn": "7 等重み L2 + QP", "pre05": "8a 事前 L2 重心(同じ S(i)、τ1 = 0.05)",
          "pre05m1": "8b 事前 M1(同じ S(i)、τ1 = 0.05)"}
ITEM5 = "5_負債が2倍超"
ZERO = 1e-6
AUC_MAX = {"c2": 0.762, "c1": 0.736}
ZERO_MIN = 6.70


def sym(s: list[int]) -> str:
    return " ".join({1: "+", -1: "−", 0: "0"}[int(v)] for v in s)


def ci(t: tuple[float, float, float]) -> str:
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    cmat = build_constraint_matrix(spec, include_subtotals=False)[:, 0]
    setup_excl(names)
    x, lab, _docs, _ = load_tilde(names)
    xt = x[lab == 0]
    r1, se1, _ = real_targets("c1", names)
    w1 = {m: 1.0 / se1[m] for m in CAL}
    cal = ["# ステージ72: λ の較正(class 1 の 128 社だけ)", "",
           "| 設定 | " + " | ".join(f"{v:g}" for v in LAMS) + " | λ̃*_0 |",
           "|---|" + "---:|" * (len(LAMS) + 1)]
    star: dict[str, float | None] = {}
    lval: dict[str, float] = {}
    n5 = {"c2": 0, "c1": 0}
    for st, lab_s in LABELS.items():
        cells: list[str] = []
        best: tuple[float | None, float] = (None, np.inf)
        for lt in LAMS:
            src = Src(*S72, f"{st}__l{lt:g}", f"{lt:g}")
            for cls in ("c2", "c1"):
                c_ = ctx(src, cls, names)
                v_ = c_.viol[(c_.viol["setting"] == src.setting) & (c_.viol["check"] == ITEM5)]
                n5[cls] += int(v_["docID"].nunique())
            c = ctx(src, "c1", names)
            nv = c.viol[c.viol["setting"] == src.setting]["docID"].nunique()
            f = gen_firms("c1", src.setting, names, members)
            f = f[~f.index.isin(e51.EXCL["c1"])]
            lv = loss({"paid_dec": float(f["paid_dec"].mean()), "re_up": float(f["re_up"].mean()),
                       "d_assets": float(f["d_assets"].median())}, r1, w1)
            cells.append(f"{lv:.2f}({nv})")
            if nv == 0 and lv < best[1]:
                best = (lt, lv)
        star[st], lval[st] = best[0], best[1]
        cal.append(f"| {lab_s} | " + " | ".join(cells)
                   + f" | **{'なし' if best[0] is None else f'{best[0]:g}'}** |")
    cal += ["", "セルは L(違反した企業数)。class 2 には同じ λ を当てはめる。", "",
            f"**項目 5 の違反**(全 {len(LABELS)} 設定 × {len(LAMS)} λ̃、企業数の合計): "
            f"class 2 で {n5['c2']}、class 1 で {n5['c1']}。ステージ71(マージンなし)は "
            "class 1 で 38。", ""]
    base = Src(*S67, "s1__l1", "1")
    p71 = {c: np.load(INTERIM / f"stage71/plan_{c}.npz", allow_pickle=True) for c in ("c2", "c1")}
    p72 = {c: np.load(INTERIM / f"stage72/plan_{c}.npz", allow_pickle=True) for c in ("c2", "c1")}
    z70 = {c: np.load(INTERIM / f"stage70/targets_{c}_main.npz", allow_pickle=True)
           for c in ("c2", "c1")}
    ot70 = {c: pd.read_csv(f"results/stage70/ot_{c}_main.csv").set_index("key")
            for c in ("c2", "c1")}

    def target_info(st: str, cls: str) -> dict[str, float]:
        """QP が参照する目標の性質(F1 は候補を集めた集合)。"""
        if st in ("f1", "f1m1", "f1bal"):
            z = p72[cls] if st == "f1bal" else p71[cls]
            key = "bal" if st == "f1bal" else "t0.05"
            idx, pt = np.asarray(z[f"{key}__idx"]), np.asarray(z[f"{key}__pt"])
            pool = xt[idx[pt > 0]]
            multi = float(np.median(1.0 / (pt ** 2).sum(axis=1)))
        else:
            src_key = {"a2": (z70, "t0.01__argmax"), "a2t05": (z70, "t0.05__argmax"),
                       "bal_l2": (z70, "bal__l2"), "l2nn": (p72, "l2nn"),
                       "pre05": (p72, "pre05_l2"), "pre05m1": (p72, "pre05_m1")}[st]
            pool = np.asarray(src_key[0][cls][src_key[1]], dtype=float)
            multi = {"bal_l2": float(ot70[cls].loc["bal", "support_med"]),
                     "pre05": float(np.median(1.0 / (np.asarray(p71[cls]["t0.05__pt"]) ** 2)
                                              .sum(axis=1))),
                     "pre05m1": float(np.median(1.0 / (np.asarray(p71[cls]["t0.05__pt"]) ** 2)
                                                .sum(axis=1)))}.get(st, 1.0)
        r = identification_auc_repeated(pool, xt, 3)
        return {"t_auc": float(r["lightgbm"].mean()),
                "t_zero": float((pool == 0).sum(axis=1).mean()),
                "t_resid": float(np.median(np.abs(pool @ cmat))), "multi": multi}

    res: dict[str, dict[str, Any]] = {}
    for cls in ("c2", "c1"):
        yb = endpoints(base, cls, names)
        res[f"s1|{cls}"] = {"m": metrics(base, cls, names, members),
                            "e_zero": float((np.abs(yb) < ZERO).sum(axis=1).mean()),
                            "e_resid": float(np.median(np.abs(yb @ cmat)))}
        for st in LABELS:
            if star[st] is None:
                continue
            src = Src(*S72, f"{st}__l{star[st]:g}", f"{star[st]:g}")
            y = endpoints(src, cls, names)
            res[f"{st}|{cls}"] = {"m": metrics(src, cls, names, members),
                                  "vs": versus(src, base, cls, names, np.random.default_rng(0)),
                                  "e_zero": float((np.abs(y) < ZERO).sum(axis=1).mean()),
                                  "e_resid": float(np.median(np.abs(y @ cmat))),
                                  **target_info(st, cls)}
            print(f"  {cls} {st}: {res[f'{st}|{cls}']['m']['nai']:.3f}", flush=True)
    comp = ["# ステージ72: 生成の比較", "",
            "差は設定 4(ステージ69 の最終設定)との企業単位のペア比較(2,000 回のブートストラップ)。"
            "記号 = 純資産の改善・解消率・順位相関。統合した企業数は p̃ の実効サポート(または計画の"
            "実効サポート)の中央値。", ""]
    for cls in ("c2", "c1"):
        comp += [f"## {cls}", "",
                 "| 設定 | 統合した企業数 | λ̃*_0 | 違反 | 純資産の改善 | 設定 4 との差 [95%] "
                 "| 解消率 | 払込資本↓ / ↑ | 変更科目数 | 順位相関 | 順位相関の差 [95%] | 記号 |",
                 "|---|---:|---:|---:|---:|---|---:|---|---:|---:|---|---|"]
        b = res[f"s1|{cls}"]["m"]
        comp.append(f"| 4 従来(ステージ69) | 1 | 1 | {b['viol']} | {b['nai']:.3f} | — "
                    f"| {b['res']:.2f} | {b['paid_dec']:.0%} / {b['paid_inc']:.0%} "
                    f"| {b['n_changed']:.0f} | {b['rho']:+.3f} | — | — |")
        for st, lab_s in LABELS.items():
            if f"{st}|{cls}" not in res:
                comp.append(f"| {lab_s} | — | なし | 較正できない |" + " — |" * 8)
                continue
            r = res[f"{st}|{cls}"]
            v = r["m"]
            comp.append(f"| {lab_s} | {r['multi']:.2f} | {v['lam']} | {v['viol']} | {v['nai']:.3f} "
                        f"| {ci(r['vs']['nai'])} | {v['res']:.2f} "
                        f"| {v['paid_dec']:.0%} / {v['paid_inc']:.0%} | {v['n_changed']:.0f} "
                        f"| {v['rho']:+.3f} | {ci(r['vs']['rho'])} | {sym(r['vs']['sign'])} |")
        comp.append("")
    real_zero = float((xt == 0).sum(axis=1).mean())
    prop = ["# ステージ72: 目標と終点の性質", "",
            f"健全企業のちょうど 0 の科目数は {real_zero:.2f}。目標の AUC は LightGBM(F1 は全始点の"
            "候補を集めた集合)。目標のゼロ・残差は、F1 では候補の平均。終点のゼロは"
            " |y| < 10⁻⁶。", "",
            "| class | 設定 | 目標: AUC | 目標: ちょうど 0 | 目標: 貸借の残差 "
            "| 終点: 0 とみなす科目数 "
            "| 終点: 貸借の残差 |", "|---|---|---:|---:|---:|---:|---:|"]
    for cls in ("c2", "c1"):
        s1r = res[f"s1|{cls}"]
        prop.append(f"| {cls} | 4 従来(ステージ69) | 0.712 / 0.686(参考) | — | — "
                    f"| {s1r['e_zero']:.2f} | {s1r['e_resid']:.1e} |")
        for st, lab_s in LABELS.items():
            if f"{st}|{cls}" not in res:
                continue
            r = res[f"{st}|{cls}"]
            prop.append(f"| {cls} | {lab_s} | {r['t_auc']:.3f} | {r['t_zero']:.2f} "
                        f"| {r['t_resid']:.1e} | {r['e_zero']:.2f} | {r['e_resid']:.1e} |")
    prop.append("")
    jl = ["# ステージ72: F1 の判定(criteria.md §4)", "",
          "| 設定 | 1 違反 0 | 2 統合 ≥ 3 | 3 目標の AUC・終点のゼロ | 4 主指標 | 採用 |",
          "|---|---|---|---|---|---|"]
    ok_set: list[str] = []
    for st in ("f1", "f1m1"):
        if f"{st}|c2" not in res or f"{st}|c1" not in res:
            jl.append(f"| {LABELS[st]} | 較正できない | — | — | — | 否 |")
            continue
        rr = {c: res[f"{st}|{c}"] for c in ("c2", "c1")}
        c1_ = all(rr[c]["m"]["viol"] == 0 for c in rr)
        c2_ = all(rr[c]["multi"] >= 3 for c in rr)
        c3_ = all(rr[c]["t_auc"] <= AUC_MAX[c] and rr[c]["e_zero"] >= ZERO_MIN for c in rr)
        c4_ = all(-1 not in [int(v) for v in rr[c]["vs"]["sign"]]
                  and 1 in [int(v) for v in rr[c]["vs"]["sign"]] for c in rr)
        ok = c1_ and c2_ and c3_ and c4_

        def yn(b_: bool) -> str:
            return "成立" if b_ else "**不成立**"
        jl.append(f"| {LABELS[st]} | {yn(c1_)}({rr['c2']['m']['viol']} / {rr['c1']['m']['viol']}) "
                  f"| {yn(c2_)}({rr['c2']['multi']:.1f} / {rr['c1']['multi']:.1f}) "
                  f"| {yn(c3_)}(AUC {rr['c2']['t_auc']:.3f} / {rr['c1']['t_auc']:.3f}、ゼロ "
                  f"{rr['c2']['e_zero']:.2f} / {rr['c1']['e_zero']:.2f}) "
                  f"| {yn(c4_)}({sym(rr['c2']['vs']['sign'])} / {sym(rr['c1']['vs']['sign'])}) "
                  f"| {'採用候補' if ok else '否'} |")
        if ok:
            ok_set.append(st)
    adopted = "f1m1" if "f1m1" in ok_set else ("f1" if "f1" in ok_set else None)
    verdict = LABELS[adopted] if adopted else "なし(ステージ69 の最終設定のまま)"
    jl += ["", f"**採用: {verdict}**", ""]
    # 事前に混ぜる 対 事後に混ぜる(同じ S(i))
    con = ["# ステージ72: 事前に混ぜる(QP(Σ p̃ y_j))対 事後に混ぜる(Σ p̃ QP(y_j))", "",
           "同じ τ1 = 0.05 の S(i)・同じ p̃。λ はそれぞれ class 1 で較正。", "",
           "| class | 設定 | λ̃*_0 | 目標: ちょうど 0 | 目標: 貸借の残差 | 目標: AUC "
           "| 終点: 0 とみなす "
           "| 終点: 残差 | 違反 | 純資産の改善 | 順位相関 | 記号 |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for cls in ("c2", "c1"):
        for st in ("a2t05", "pre05", "pre05m1", "f1"):
            if f"{st}|{cls}" not in res:
                continue
            r = res[f"{st}|{cls}"]
            con.append(f"| {cls} | {LABELS[st]} | {r['m']['lam']} | {r['t_zero']:.2f} "
                       f"| {r['t_resid']:.1e} | {r['t_auc']:.3f} | {r['e_zero']:.2f} "
                       f"| {r['e_resid']:.1e} | {r['m']['viol']} | {r['m']['nai']:.3f} "
                       f"| {r['m']['rho']:+.3f} | {sym(r['vs']['sign'])} |")
    con.append("")
    for name_, body in (("calibration.md", cal), ("comparison.md", comp),
                        ("properties.md", prop), ("judgement.md", jl), ("contrast.md", con)):
        (OUT / name_).write_text("\n".join(body) + "\n", encoding="utf-8")
        print("\n".join(body))
    (OUT / "summary.json").write_text(json.dumps(
        {"star": star, "L": lval, "adopted": adopted, "item5": n5, "res": res},
        ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (OUT / "bs.md").write_text("\n".join(bs_lines(names, members, star)) + "\n", encoding="utf-8")


def bs_lines(names: tuple[str, ...], members: dict[str, np.ndarray],
             star: dict[str, float | None]) -> list[str]:
    """BS 表(class 2、ステージ65〜71 と同じ 5 社)。始点・設定 4・3b・8a・1(F1)。"""
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    e51._GEN = INTERIM / "stage67/all"
    g1 = (lambda g: g[g["seed"] == 0].set_index("docID"))(e51.load_gen("c2", "s1__l1"))
    e51._GEN = INTERIM / "stage72/gen"
    show = [st for st in ("a2t05", "pre05", "f1") if star.get(st) is not None]
    gens = {st: (lambda g: g[g["seed"] == 0].set_index("docID"))(
        e51.load_gen("c2", f"{st}__l{star[st]:g}")) for st in show}
    x0c, x1c = [f"x0_{n}" for n in names], [f"x1_{n}" for n in names]
    na0 = g1[x0c].to_numpy() @ members["jppfs_cor_NetAssets"]
    docs = np.array(g1.index)
    r_ = np.random.default_rng(0)
    ins = docs[na0 < 0]
    pick_ins = list(r_.choice(ins, size=min(2, len(ins)), replace=False))
    rest = [d for d in docs if d not in pick_ins]
    picks = pick_ins + list(r_.choice(rest, size=3, replace=False))
    labels = load_labels(Path("conf/accounts/labels_ja.yaml"))
    short = {"a2t05": "argmax", "pre05": "事前 L2 重心", "f1": "F1"}
    out = ["# ステージ72: BS 表(class 2、百万円、生成の終点は 5 年後とみなす)", "",
           "企業はステージ65〜71 と同じ規則(乱数 seed 0、始点で債務超過から 2 社、残りから 3 社。"
           "ANAP と極楽湯ホールディングスを含む)。列は始点、設定 4(ステージ69)、新設計の argmax"
           "(τ1 = 0.05)・事前 L2 重心(同じ S(i))・F1 の終点(λ はそれぞれ class 1 で較正)。", ""]
    for d in picks:
        r0 = g1.loc[d]
        a0 = float(r0["assets_start"])
        cols = [("始点", r0[x0c].to_numpy(float) * a0), ("従来", r0[x1c].to_numpy(float) * a0),
                *((f"{short[st]} λ̃={star[st]:g}", gens[st].loc[d][x1c].to_numpy(float) * a0)
                  for st in show)]
        tag = "(始点で債務超過)" if d in pick_ins else ""
        out += bs_table(spec, names, labels, f"{r0['filerName']}({d}){tag}", cols)
    return out


if __name__ == "__main__":
    main()
