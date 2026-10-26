#!/usr/bin/env python
"""ステージ72 追補: 後の版の条件 3・F1 と設定 2 の比較・F1 の τ1 の感度・対応先の BS 表。

規則は ``results/stage72/addendum_criteria.md``。lightgbm・sklearn のプロセス。事前に
``results/stage72/run_addendum.sh`` と ``stage72_evaluate.py`` を実行する。

出力: ``results/stage72/addendum.md``・``bs_addendum.md``。

    uv run python scripts/stage72_addendum.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import numpy as np
from evaluate_stage61 import CAL, gen_firms, loss, real_targets
from evaluate_stage65 import bs_table
from stage71_evaluate import endpoints
from stage72_evaluate import LAMS, S72, ZERO, ci, sym
from tables_stage69 import INTERIM, S67, Src, ctx, metrics, setup_excl, versus
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.plausibility import subtotal_members
from fsgen.viz.balance_sheet import load_labels

OUT = Path("results/stage72")
STAU = (INTERIM / "stage72/metrics_tau", INTERIM / "stage72/gen_tau")


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    cmat = build_constraint_matrix(spec, include_subtotals=False)[:, 0]
    setup_excl(names)
    sm: dict[str, Any] = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    star: dict[str, float] = {k: float(v) for k, v in sm["star"].items() if v is not None}
    r1, se1, _ = real_targets("c1", names)
    w1 = {m: 1.0 / se1[m] for m in CAL}
    cal = ["## F1 の τ1 の感度: λ の較正(class 1 の 128 社)", "",
           "| 設定 | " + " | ".join(f"{v:g}" for v in LAMS) + " | λ̃*_0 |",
           "|---|" + "---:|" * (len(LAMS) + 1)]
    for st, lab_s in (("f1t001", "F1(τ1 = 0.01)"), ("f1t01", "F1(τ1 = 0.1)")):
        cells: list[str] = []
        best: tuple[float | None, float] = (None, np.inf)
        for lt in LAMS:
            src = Src(*STAU, f"{st}__l{lt:g}", f"{lt:g}")
            c = ctx(src, "c1", names)
            nv = c.viol[c.viol["setting"] == src.setting]["docID"].nunique()
            f = gen_firms("c1", src.setting, names, members)
            f = f[~f.index.isin(e51.EXCL["c1"])]
            lv = loss({"paid_dec": float(f["paid_dec"].mean()), "re_up": float(f["re_up"].mean()),
                       "d_assets": float(f["d_assets"].median())}, r1, w1)
            cells.append(f"{lv:.2f}({nv})")
            if nv == 0 and lv < best[1]:
                best = (lt, lv)
        if best[0] is None:
            raise ValueError(f"{st}: 違反 0 の λ̃ がない")
        star[st] = best[0]
        cal.append(f"| {lab_s} | " + " | ".join(cells) + f" | **{best[0]:g}** |")
    cal.append("")
    base = Src(*S67, "s1__l1", "1")
    srcs = {"a2": Src(*S72, f"a2__l{star['a2']:g}", f"{star['a2']:g}"),
            "f1t001": Src(*STAU, f"f1t001__l{star['f1t001']:g}", f"{star['f1t001']:g}"),
            "f1": Src(*S72, f"f1__l{star['f1']:g}", f"{star['f1']:g}"),
            "f1t01": Src(*STAU, f"f1t01__l{star['f1t01']:g}", f"{star['f1t01']:g}"),
            "f1bal": Src(*S72, f"f1bal__l{star['f1bal']:g}", f"{star['f1bal']:g}"),
            "f1m1": Src(*S72, f"f1m1__l{star['f1m1']:g}", f"{star['f1m1']:g}")}
    plans = {"f1t001": ("stage72/tau", "t0.01"), "f1": ("stage71/plan", "t0.05"),
             "f1t01": ("stage72/tau", "t0.1"), "f1bal": ("stage72/plan", "bal"),
             "f1m1": ("stage71/plan", "t0.05")}
    res: dict[str, dict[str, Any]] = {}
    for cls in ("c2", "c1"):
        res[f"s1|{cls}"] = {"m": metrics(base, cls, names, members)}
        for st, src in srcs.items():
            y = endpoints(src, cls, names)
            info: dict[str, Any] = {"m": metrics(src, cls, names, members),
                                    "vs": versus(src, base, cls, names, np.random.default_rng(0)),
                                    "e_zero": float((np.abs(y) < ZERO).sum(axis=1).mean()),
                                    "e_resid": float(np.median(np.abs(y @ cmat))), "multi": 1.0}
            if st in plans:
                z = np.load(INTERIM / f"{plans[st][0]}_{cls}.npz", allow_pickle=True)
                pt = np.asarray(z[f"{plans[st][1]}__pt"])
                info["multi"] = float(np.median(1.0 / (pt ** 2).sum(axis=1)))
                info["vs_a2"] = versus(src, srcs["a2"], cls, names, np.random.default_rng(0))
            res[f"{st}|{cls}"] = info
    lines = ["# ステージ72 追補(後の版の指示、`addendum_criteria.md`)", "",
             "採用の判定は `criteria.md`(最初の版、計算の前に記録)のものを維持する。"
             "この追補は記述。",
             "", "## 1. 後の版の条件 3(終点の貸借の残差 ≤ 10⁻⁶、0 とみなす科目数 ≥ 6.70)", "",
             "| 設定 | 残差 c2 / c1 | 0 とみなす科目数 c2 / c1 | 条件 3(後の版) | 条件 1・2・4 "
             "(`judgement.md`) |", "|---|---|---|---|---|"]
    jm = (OUT / "judgement.md").read_text(encoding="utf-8")
    for st, lab_s in (("f1", "F1(τ1 = 0.05、m = 10)"), ("f1m1", "F1(m = 1)")):
        rr = [res[f"{st}|{c}"] for c in ("c2", "c1")]
        ok = all(r["e_resid"] <= 1e-6 and r["e_zero"] >= 6.70 for r in rr)
        other = "成立(採用)" if st == "f1" and "採用: 1 F1" in jm else "不成立(条件 1・4)"
        lines.append(f"| {lab_s} | {rr[0]['e_resid']:.1e} / {rr[1]['e_resid']:.1e} "
                     f"| {rr[0]['e_zero']:.2f} / {rr[1]['e_zero']:.2f} "
                     f"| {'成立' if ok else '不成立'} | {other} |")
    lines += ["", "## 2. F1 と設定 2(同じ新設計・𝓕⁺ の 1 社選択、τ1 = 0.01 の argmax)", "",
              "差 = F1 − 設定 2(企業単位のペア比較、2,000 回のブートストラップ)。", "",
              "| class | 設定 | λ̃*_0 | 純資産の改善 | 解消率 | 順位相関 | 払込資本↓ / ↑ "
              "| 変更科目数 "
              "| F1 − 設定 2: 純資産の改善 | F1 − 設定 2: 順位相関 | 記号 |",
              "|---|---|---:|---:|---:|---:|---|---:|---|---|---|"]
    for cls in ("c2", "c1"):
        for st, lab_s in (("a2", "設定 2(argmax)"), ("f1", "F1")):
            r = res[f"{st}|{cls}"]
            v = r["m"]
            d = (f"{ci(r['vs_a2']['nai'])} | {ci(r['vs_a2']['rho'])} | {sym(r['vs_a2']['sign'])}"
                 if "vs_a2" in r else "— | — | —")
            lines.append(f"| {cls} | {lab_s} | {v['lam']} | {v['nai']:.3f} | {v['res']:.2f} "
                         f"| {v['rho']:+.3f} | {v['paid_dec']:.0%} / {v['paid_inc']:.0%} "
                         f"| {v['n_changed']:.0f} | {d} |")
    lines += ["", "## 3. F1 の τ1 の感度(1 社に近い ↔ 半不均衡 ↔ バランス OT)", "",
              "差は設定 1(ステージ69)との企業単位のペア比較。記号 = 純資産の改善・解消率・"
              "順位相関。"
              "統合した企業数は p̃ の実効サポートの中央値。", "",
              "| class | 設定 | 統合した企業数 | λ̃*_0 | 違反 | 純資産の改善 | 設定 1 との差 [95%] "
              "| 解消率 | 順位相関 | 順位相関の差 [95%] | 記号 | 終点: 0 / 残差 |",
              "|---|---|---:|---:|---:|---:|---|---:|---:|---|---|---|"]
    for cls in ("c2", "c1"):
        b = res[f"s1|{cls}"]["m"]
        lines.append(f"| {cls} | 設定 1(ステージ69) | 1 | 1 | {b['viol']} | {b['nai']:.3f} | — "
                     f"| {b['res']:.2f} | {b['rho']:+.3f} | — | — | — |")
        for st, lab_s in (("a2", "設定 2(argmax、τ1 = 0.01)"), ("f1t001", "F1 τ1 = 0.01"),
                          ("f1", "**F1 τ1 = 0.05**"), ("f1t01", "F1 τ1 = 0.1"),
                          ("f1bal", "F1 バランス OT(上位 10 社)")):
            r = res[f"{st}|{cls}"]
            v = r["m"]
            lines.append(f"| {cls} | {lab_s} | {r['multi']:.2f} | {v['lam']} | {v['viol']} "
                         f"| {v['nai']:.3f} | {ci(r['vs']['nai'])} | {v['res']:.2f} "
                         f"| {v['rho']:+.3f} | {ci(r['vs']['rho'])} | {sym(r['vs']['sign'])} "
                         f"| {r['e_zero']:.2f} / {r['e_resid']:.0e} |")
    lines.append("")
    (OUT / "addendum.md").write_text("\n".join([*lines, *cal]) + "\n", encoding="utf-8")
    (OUT / "addendum.json").write_text(json.dumps({"star": star, "res": res}, ensure_ascii=False,
                                                  indent=1, default=float), encoding="utf-8")
    print("\n".join([*lines, *cal]))
    (OUT / "bs_addendum.md").write_text("\n".join(bs_lines(names, members, star)) + "\n",
                                        encoding="utf-8")


def bs_lines(names: tuple[str, ...], members: dict[str, np.ndarray],
             star: dict[str, float]) -> list[str]:
    """BS 表(class 2、同じ 5 社)。始点・対応先の上位 3 社・設定 1・設定 2・F1。"""
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    x, lab, _d, _ = load_tilde(names)
    xt = x[lab == 0]
    e51._GEN = INTERIM / "stage67/all"
    g1 = (lambda g: g[g["seed"] == 0].set_index("docID"))(e51.load_gen("c2", "s1__l1"))
    e51._GEN = INTERIM / "stage72/gen"
    ga2 = (lambda g: g[g["seed"] == 0].set_index("docID"))(
        e51.load_gen("c2", f"a2__l{star['a2']:g}"))
    gf1 = (lambda g: g[g["seed"] == 0].set_index("docID"))(
        e51.load_gen("c2", f"f1__l{star['f1']:g}"))
    z = np.load(INTERIM / "stage71/plan_c2.npz", allow_pickle=True)
    pos = {str(d): i for i, d in enumerate(z["source_docs"])}
    idx, pt = np.asarray(z["t0.05__idx"]), np.asarray(z["t0.05__pt"])
    x0c, x1c = [f"x0_{n}" for n in names], [f"x1_{n}" for n in names]
    na0 = g1[x0c].to_numpy() @ members["jppfs_cor_NetAssets"]
    docs = np.array(g1.index)
    r_ = np.random.default_rng(0)
    ins = docs[na0 < 0]
    pick_ins = list(r_.choice(ins, size=min(2, len(ins)), replace=False))
    rest = [d for d in docs if d not in pick_ins]
    picks = pick_ins + list(r_.choice(rest, size=3, replace=False))
    labels = load_labels(Path("conf/accounts/labels_ja.yaml"))
    out = ["# ステージ72 追補: BS 表(class 2、百万円、生成の終点は 5 年後とみなす)", "",
           "企業はステージ65〜72 と同じ 5 社。対応先は τ1 = 0.05 の S(i) の p̃ の大きい順の 3 社"
           "(総資産比を始点の総資産で戻した値)。設定 2 = 新設計 + 𝓕⁺ の argmax(τ1 = 0.01)、"
           "F1 = τ1 = 0.05 の事後アンサンブル(λ はそれぞれ class 1 で較正)。", ""]
    for d in picks:
        r0 = g1.loc[d]
        a0 = float(r0["assets_start"])
        i = pos[str(d)]
        cand = [(f"対応先 {k + 1}(p̃ {pt[i, k]:.2f})", xt[idx[i, k]] * a0)
                for k in range(3) if pt[i, k] > 0]
        cols = [("始点", r0[x0c].to_numpy(float) * a0), *cand,
                ("設定 1", r0[x1c].to_numpy(float) * a0),
                ("設定 2", ga2.loc[d][x1c].to_numpy(float) * a0),
                ("F1", gf1.loc[d][x1c].to_numpy(float) * a0)]
        tag = "(始点で債務超過)" if d in pick_ins else ""
        out += bs_table(spec, names, labels, f"{r0['filerName']}({d}){tag}、上位 3 社の p̃ の合計 "
                        f"{pt[i, :3].sum():.2f}", cols)
    return out


if __name__ == "__main__":
    main()
