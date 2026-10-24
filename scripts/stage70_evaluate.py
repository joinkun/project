#!/usr/bin/env python
"""ステージ70 B: 較正・比較・判定(lightgbm・sklearn のプロセス)。

規則は ``results/stage70/criteria.md`` §3・§4。事前に ``metrics_stage39.py --gen-dir
data/interim/stage70/gen --out data/interim/stage70/metrics`` と ``stage70_auc.py`` を実行する。
比較の基準は設定 1(ステージ69 の最終設定、ステージ67 の ``s1__l1``)。

    uv run python scripts/stage70_evaluate.py
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
from stage70_qp import SETTINGS
from tables_stage69 import INTERIM, S67, Src, ctx, metrics, setup_excl, versus
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import subtotal_members
from fsgen.viz.balance_sheet import load_labels

OUT = Path("results/stage70")
S70 = (INTERIM / "stage70/metrics", INTERIM / "stage70/gen")
LAMS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
LABELS = {"s2": "2 新設計・argmax(τ1 = 0.01)", "s3a": "3a 新設計・M1(τ1 = 0.05)",
          "s3b": "3b 新設計・M1(τ1 = 0.1)", "s4": "4 新設計・M1(バランス OT)",
          "s5a": "5a 新設計・M2(τ1 = 0.05)", "s5b": "5b 新設計・M2(τ1 = 0.1)",
          "s6a": "6a 新設計 + c_RE・M1(τ1 = 0.05)", "s6b": "6b 新設計 + c_RE・M1(τ1 = 0.1)"}
TAU_OF = {"t0.01": "0.01", "t0.05": "0.05", "t0.1": "0.1", "t0.5": "0.5", "bal": "∞(バランス)"}
CANDIDATES = ("s3a", "s3b", "s4", "s5a", "s5b", "s6a", "s6b")


def lab_of(st: str) -> str:
    base, m = st.rsplit("_", 1)
    return f"{LABELS[base]}・m = {m[1:]}"


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    setup_excl(names)
    a2 = pd.read_csv(OUT / "precheck_a2.csv")
    r1, se1, _ = real_targets("c1", names)
    w1 = {m: 1.0 / se1[m] for m in CAL}
    cal = ["# ステージ70 B: λ の較正(class 1 の 128 社だけ)", "",
           "| 設定 | " + " | ".join(f"{v:g}" for v in LAMS) + " | λ̃*_0 |",
           "|---|" + "---:|" * (len(LAMS) + 1)]
    star: dict[str, float | None] = {}
    lval: dict[str, float] = {}
    for st in SETTINGS:
        cells: list[str] = []
        best: tuple[float | None, float] = (None, np.inf)
        for lt in LAMS:
            src = Src(*S70, f"{st}__l{lt:g}", f"{lt:g}")
            c = ctx(src, "c1", names)
            f = gen_firms("c1", src.setting, names, members)
            f = f[~f.index.isin(e51.EXCL["c1"])]
            nv = c.viol[c.viol["setting"] == src.setting]["docID"].nunique()
            lv = loss({"paid_dec": float(f["paid_dec"].mean()), "re_up": float(f["re_up"].mean()),
                       "d_assets": float(f["d_assets"].median())}, r1, w1)
            cells.append(f"{lv:.2f}({nv})")
            if nv == 0 and lv < best[1]:
                best = (lt, lv)
        star[st], lval[st] = best[0], best[1]
        cal.append(f"| {lab_of(st)} | " + " | ".join(cells)
                   + f" | **{'なし' if best[0] is None else f'{best[0]:g}'}** |")
    cal += ["", "セルは L(違反した企業数)。class 2 には同じ λ(class 1 の s × λ̃*_0)を"
            "当てはめる。", ""]
    base = Src(*S67, "s1__l1", "1")
    res: dict[str, dict[str, Any]] = {}
    for cls in ("c2", "c1"):
        res[f"s1|{cls}"] = {"m": metrics(base, cls, names, members)}
        for st, (key, _re, _m) in SETTINGS.items():
            if star[st] is None:
                continue
            src = Src(*S70, f"{st}__l{star[st]:g}", f"{star[st]:g}")
            tk, meth = key.split("__")
            arow = a2[(a2["cls"] == cls) & (a2["variant"] == "main")
                      & (a2["tau1"].astype(str) == TAU_OF[tk]) & (a2["method"] == meth)].iloc[0]
            res[f"{st}|{cls}"] = {"m": metrics(src, cls, names, members),
                                  "vs": versus(src, base, cls, names, np.random.default_rng(0)),
                                  "support": float(arow["support"]),
                                  "auc": float(arow["auc_lgbm"]),
                                  "auc_lr": float(arow["auc_logreg"])}
    a64 = {cls: pd.read_csv(f"results/stage65/auc_{cls}.csv").set_index("method")
           .loc["m5_prop", "auc_lgbm"] for cls in ("c2", "c1")}
    sup64 = {cls: float(pd.read_csv(f"results/stage64/plans_{cls}.csv").set_index("setting")
                        .loc["semi_t0.01_e0.001", "support_med"]) for cls in ("c2", "c1")}

    def sym(s: list[int]) -> str:
        return " ".join({1: "+", -1: "−", 0: "0"}[int(v)] for v in s)

    def ci(t: tuple[float, float, float]) -> str:
        return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"

    comp = ["# ステージ70 B.2: 生成の比較", "",
            "差は設定 1(ステージ69 の最終設定)との企業単位のペア比較(2,000 回のブートストラップ)。"
            "記号 = 純資産の改善・解消率・順位相関。実効サポートと識別 AUC は目標(QP の前)の"
            "値。", ""]
    for cls in ("c2", "c1"):
        comp += [f"## {cls}", "",
                 "| 設定 | 実効サポート | λ̃*_0 | 違反 | 純資産の改善 | 設定 1 との差 [95%] "
                 "| 解消率 "
                 "| 払込資本↓ | 払込資本↑ | 変更科目数 | 順位相関 | 設定 1 との差 [95%] | 記号 "
                 "| 目標の AUC(LightGBM) |",
                 "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|---:|"]
        b = res[f"s1|{cls}"]["m"]
        comp.append(f"| 1 従来(ステージ69 の最終設定) | {sup64[cls]:.2f} | 1 | {b['viol']} "
                    f"| {b['nai']:.3f} | — | {b['res']:.2f} | {b['paid_dec']:.0%} "
                    f"| {b['paid_inc']:.0%} | {b['n_changed']:.0f} | {b['rho']:+.3f} | — | — "
                    f"| {a64[cls]:.3f} |")
        for st in SETTINGS:
            if f"{st}|{cls}" not in res:
                comp.append(f"| {lab_of(st)} | — | なし | 較正できない |" + " — |" * 10)
                continue
            r = res[f"{st}|{cls}"]
            v = r["m"]
            comp.append(f"| {lab_of(st)} | {r['support']:.2f} | {v['lam']} | {v['viol']} "
                        f"| {v['nai']:.3f} | {ci(r['vs']['nai'])} | {v['res']:.2f} "
                        f"| {v['paid_dec']:.0%} | {v['paid_inc']:.0%} | {v['n_changed']:.0f} "
                        f"| {v['rho']:+.3f} | {ci(r['vs']['rho'])} | {sym(r['vs']['sign'])} "
                        f"| {r['auc']:.3f} |")
        comp.append("")
    # 判定(criteria §4)
    jl = ["# ステージ70 B: 判定(criteria.md §4)", "",
          f"条件 3 の上限: class 2 で {a64['c2'] + 0.05:.3f}、class 1 で {a64['c1'] + 0.05:.3f}"
          "(設定 1 の目標の AUC + 0.05)。", "",
          "| 設定 | 1 違反 0 | 2 実効サポート ≥ 3 | 3 AUC | 4 主指標が劣らない | class 1 の L "
          "| 採用候補 |",
          "|---|---|---|---|---|---:|---|"]
    cands: list[tuple[float, int, str]] = []
    for st in SETTINGS:
        if st.rsplit("_", 1)[0] not in CANDIDATES:
            continue
        if f"{st}|c2" not in res or f"{st}|c1" not in res:
            jl.append(f"| {lab_of(st)} | 較正できない | — | — | — | — | 否 |")
            continue
        rr = {cls: res[f"{st}|{cls}"] for cls in ("c2", "c1")}
        c1_ = all(rr[c]["m"]["viol"] == 0 for c in rr)
        c2_ = all(rr[c]["support"] >= 3 for c in rr)
        c3_ = all(rr[c]["auc"] <= a64[c] + 0.05 for c in rr)
        c4_ = all(-1 not in [int(v) for v in rr[c]["vs"]["sign"]] for c in rr)
        ok = c1_ and c2_ and c3_ and c4_

        def yn(b_: bool) -> str:
            return "成立" if b_ else "**不成立**"
        jl.append(f"| {lab_of(st)} | {yn(c1_)} | {yn(c2_)} "
                  f"| {yn(c3_)}({rr['c2']['auc']:.3f} / {rr['c1']['auc']:.3f}) "
                  f"| {yn(c4_)}({sym(rr['c2']['vs']['sign'])} / {sym(rr['c1']['vs']['sign'])}) "
                  f"| {lval[st]:.2f} | {'候補' if ok else '否'} |")
        if ok:
            cands.append((lval[st], 0 if st.endswith("_m1") else 1, st))
    adopted = min(cands)[2] if cands else None
    verdict = lab_of(adopted) if adopted else "なし(ステージ69 の最終設定のまま)"
    jl += ["", f"**採用: {verdict}**", ""]
    # 設定 2(コストの純粋化)
    jl += ["## 設定 2(新設計のコスト・凍結、argmax)の効果(判定には使わない)", "",
           "| 設定 | class | 違反 | 純資産の改善の差 [95%] | 記号 |", "|---|---|---:|---|---|"]
    for st in ("s2_m10", "s2_m1"):
        for cls in ("c2", "c1"):
            if f"{st}|{cls}" in res:
                r = res[f"{st}|{cls}"]
                jl.append(f"| {lab_of(st)} | {cls} | {r['m']['viol']} | {ci(r['vs']['nai'])} "
                          f"| {sym(r['vs']['sign'])} |")
    jl.append("")
    (OUT / "calibration.md").write_text("\n".join(cal) + "\n", encoding="utf-8")
    (OUT / "comparison.md").write_text("\n".join(comp) + "\n", encoding="utf-8")
    (OUT / "judgement.md").write_text("\n".join(jl) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(
        {"star": star, "L": lval, "adopted": adopted,
         "res": {k: {"m": v["m"], "vs": v.get("vs"), "support": v.get("support"),
                     "auc": v.get("auc")} for k, v in res.items()}},
        ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (OUT / "bs.md").write_text("\n".join(bs_lines(names, members, star)) + "\n", encoding="utf-8")
    for fn in ("calibration.md", "comparison.md", "judgement.md"):
        print((OUT / fn).read_text(encoding="utf-8"))


def bs_lines(names: tuple[str, ...], members: dict[str, np.ndarray],
             star: dict[str, float | None]) -> list[str]:
    """BS 表(class 2。ステージ65〜68 と同じ選び方の企業)。始点・設定 1・2・3a・5a と M1 の目標。"""
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    e51._GEN = INTERIM / "stage67/all"
    g1 = (lambda g: g[g["seed"] == 0].set_index("docID"))(e51.load_gen("c2", "s1__l1"))
    e51._GEN = INTERIM / "stage70/gen"
    show = [st for st in ("s2_m10", "s3a_m10", "s5a_m10") if star.get(st) is not None]
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
    z = np.load(INTERIM / "stage70/targets_c2_main.npz", allow_pickle=True)
    m1 = dict(zip([str(v) for v in z["source_docs"]], z["t0.05__m1"], strict=True))
    labels = load_labels(Path("conf/accounts/labels_ja.yaml"))
    _x, _lab, _d, _ = load_tilde(names)
    out = ["# ステージ70 B: BS 表(class 2、百万円、生成の終点は 5 年後とみなす)", "",
           "企業はステージ65〜68 と同じ規則(乱数 seed 0、始点で債務超過から 2 社、残りから 3 社)。"
           "列は始点、設定 1・2・3a・5a(m = 10、λ は class 1 で較正)の終点、M1 の目標"
           "(τ1 = 0.05 の重み付き中央値。始点の総資産で戻した値。貸借は一致しない)。", ""]
    for d in picks:
        r0 = g1.loc[d]
        a0 = float(r0["assets_start"])
        cols = [("始点", r0[x0c].to_numpy(float) * a0), ("設定 1", r0[x1c].to_numpy(float) * a0),
                *((LABELS[st.rsplit("_", 1)[0]].split(" ")[0], gens[st].loc[d][x1c]
                   .to_numpy(float) * a0) for st in show),
                ("M1 の目標", m1[str(d)] * a0)]
        tag = "(始点で債務超過)" if d in pick_ins else ""
        out += bs_table(spec, names, labels, f"{r0['filerName']}({d}){tag}", cols)
    return out


if __name__ == "__main__":
    main()
