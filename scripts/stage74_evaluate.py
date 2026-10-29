#!/usr/bin/env python
"""ステージ74: リークのない F1 と相対コストの評価・判定(lightgbm・sklearn のプロセス)。

規則は ``results/stage74/criteria.md``。事前に ``results/stage74/run.sh`` を実行する。
基準はステージ69 の設定(ステージ67 の ``s1__l1``)。参考にステージ72 の F1(72 社)を並べる。

出力: ``results/stage74/report_tables.md``・``summary.json``。

    uv run python scripts/stage74_evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import numpy as np
from evaluate_stage61 import CAL, gen_firms, loss, real_targets
from stage71_evaluate import endpoints
from stage72_evaluate import ZERO, ci, sym
from stage74_costs import SETTINGS, pool_of
from tables_stage69 import INTERIM, S67, Src, ctx, metrics, setup_excl, versus
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.plausibility import subtotal_members
from fsgen.evaluation.identification import identification_auc_repeated

OUT = Path("results/stage74")
S74 = (INTERIM / "stage74/metrics", INTERIM / "stage74/gen")
S72 = (INTERIM / "stage72/metrics", INTERIM / "stage72/gen")
LAMS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
LABELS = {"A1": "A1 F1-abs(57 / 10、リークなし)", "A2": "A2 F1-abs(57 / 15)",
          "A3": "A3 F1-abs(57 / 57、リークあり)", "B1": "B1 F1-rel(57 / 10)",
          "B2": "B2 F1-hyb(57 / 10)"}
AUC_MAX = {"c2": 0.762, "c1": 0.736}
ZERO_MIN = 6.70


def inflation(y: np.ndarray, x0: np.ndarray) -> float:
    """始点で 10⁻⁶ ≤ |x| < 10⁻³ の(企業, 科目)のうち、終点で |y| > 10⁻³ になった割合。"""
    small = (np.abs(x0) >= 1e-6) & (np.abs(x0) < 1e-3)
    return float((np.abs(y)[small] > 1e-3).mean()) if small.any() else float("nan")


def start_points(src: Src, cls: str, names: tuple[str, ...]) -> np.ndarray:
    e51._GEN = src.gen
    g = e51.load_gen(cls, src.setting)
    out: np.ndarray = g[g["seed"] == 0][[f"x0_{n}" for n in names]].to_numpy(float)
    return out


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
    cal = ["## λ の較正(class 1 の 128 社だけ)", "",
           "| 設定 | " + " | ".join(f"{v:g}" for v in LAMS) + " | λ̃*_0 |",
           "|---|" + "---:|" * (len(LAMS) + 1)]
    star: dict[str, float | None] = {}
    lval: dict[str, float] = {}
    for st, lab_s in LABELS.items():
        cells: list[str] = []
        best: tuple[float | None, float] = (None, np.inf)
        for lt in LAMS:
            src = Src(*S74, f"{st}__l{lt:g}", f"{lt:g}")
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
    cal.append("")
    base = Src(*S67, "s1__l1", "1")
    f72 = Src(*S72, "f1__l0.3", "0.3")
    plans = {c: np.load(INTERIM / f"stage74/plan_{c}.npz", allow_pickle=True) for c in ("c2", "c1")}
    res: dict[str, dict[str, Any]] = {}
    for cls in ("c2", "c1"):
        for key, src in (("s1", base), ("f72", f72)):
            y = endpoints(src, cls, names)
            res[f"{key}|{cls}"] = {"m": metrics(src, cls, names, members),
                                   "infl": inflation(y, start_points(src, cls, names)),
                                   "e_zero": float((np.abs(y) < ZERO).sum(axis=1).mean())}
        res[f"f72|{cls}"]["vs"] = versus(f72, base, cls, names, np.random.default_rng(0))
        for st in LABELS:
            if star[st] is None:
                continue
            src = Src(*S74, f"{st}__l{star[st]:g}", f"{star[st]:g}")
            y = endpoints(src, cls, names)
            pk = f"{SETTINGS[st][0]}{pool_of(st, cls)}"
            idx, pt = np.asarray(plans[cls][f"{pk}__idx"]), np.asarray(plans[cls][f"{pk}__pt"])
            ra = identification_auc_repeated(xt[idx[pt > 0]], xt, 3)
            info: dict[str, Any] = {
                "m": metrics(src, cls, names, members),
                "vs": versus(src, base, cls, names, np.random.default_rng(0)),
                "e_zero": float((np.abs(y) < ZERO).sum(axis=1).mean()),
                "e_resid": float(np.median(np.abs(y @ cmat))),
                "infl": inflation(y, start_points(src, cls, names)),
                "multi": float(np.median(1.0 / (pt ** 2).sum(axis=1))),
                "t_auc": float(ra["lightgbm"].mean())}
            if st != "A1" and star.get("A1") is not None:
                a1 = Src(*S74, f"A1__l{star['A1']:g}", f"{star['A1']:g}")
                info["vs_a1"] = versus(src, a1, cls, names, np.random.default_rng(0))
            if st == "A1":
                info["vs_f72"] = versus(src, f72, cls, names, np.random.default_rng(0))
            res[f"{st}|{cls}"] = info
            print(f"  {cls} {st}: {info['m']['nai']:.3f}", flush=True)

    def conds(st: str) -> tuple[bool, bool, bool, bool]:
        rr = [res[f"{st}|{c}"] for c in ("c2", "c1")]
        c1_ = all(r["m"]["viol"] == 0 for r in rr)
        c2_ = all(r["multi"] >= 3 for r in rr)
        c3_ = all(r["t_auc"] <= AUC_MAX[c] and r["e_zero"] >= ZERO_MIN
                  for r, c in zip(rr, ("c2", "c1"), strict=True))
        c4_ = all(-1 not in [int(v) for v in r["vs"]["sign"]]
                  and 1 in [int(v) for v in r["vs"]["sign"]] for r in rr)
        return c1_, c2_, c3_, c4_

    def yn(b_: bool) -> str:
        return "成立" if b_ else "**不成立**"

    lines = ["# ステージ74: 表(`criteria.md` の規則)", "", *cal,
             "## 比較(基準 = ステージ69 の設定)", "",
             "差は基準との企業単位のペア比較(2,000 回のブートストラップ)。記号 = 純資産の改善・"
             "解消率・順位相関。膨張 = 始点で 10⁻⁶ ≤ |x| < 10⁻³ の組のうち、終点で |y| > 10⁻³ に"
             "なった割合。", ""]
    for cls in ("c2", "c1"):
        lines += [f"### {cls}", "",
                  "| 設定 | 統合 | λ̃*_0 | 違反 | 純資産の改善 | 差 [95%] | 解消率 | 順位相関 "
                  "| 差 [95%] | 記号 | 払込資本↓ / ↑ | 変更科目数 | 終点のゼロ | 膨張 |",
                  "|---|---:|---:|---:|---:|---|---:|---:|---|---|---|---:|---:|---:|"]
        b = res[f"s1|{cls}"]
        bm = b["m"]
        lines.append(f"| 基準(ステージ69) | 1 | 1 | {bm['viol']} | {bm['nai']:.3f} | — "
                     f"| {bm['res']:.2f} | {bm['rho']:+.3f} | — | — "
                     f"| {bm['paid_dec']:.0%} / {bm['paid_inc']:.0%} | {bm['n_changed']:.0f} "
                     f"| {b['e_zero']:.2f} | {b['infl']:.1%} |")
        f = res[f"f72|{cls}"]
        fm = f["m"]
        lines.append(f"| 参考: ステージ72 の F1(72 社、リークあり) | 6.93 | 0.3 | {fm['viol']} "
                     f"| {fm['nai']:.3f} | {ci(f['vs']['nai'])} | {fm['res']:.2f} "
                     f"| {fm['rho']:+.3f} | {ci(f['vs']['rho'])} | {sym(f['vs']['sign'])} "
                     f"| {fm['paid_dec']:.0%} / {fm['paid_inc']:.0%} | {fm['n_changed']:.0f} "
                     f"| {f['e_zero']:.2f} | {f['infl']:.1%} |")
        for st, lab_s in LABELS.items():
            r = res.get(f"{st}|{cls}")
            if r is None:
                lines.append(f"| {lab_s} | — | なし | 較正できない |" + " — |" * 10)
                continue
            v = r["m"]
            lines.append(f"| {lab_s} | {r['multi']:.2f} | {v['lam']} | {v['viol']} "
                         f"| {v['nai']:.3f} | {ci(r['vs']['nai'])} | {v['res']:.2f} "
                         f"| {v['rho']:+.3f} | {ci(r['vs']['rho'])} | {sym(r['vs']['sign'])} "
                         f"| {v['paid_dec']:.0%} / {v['paid_inc']:.0%} | {v['n_changed']:.0f} "
                         f"| {r['e_zero']:.2f} | {r['infl']:.1%} |")
        lines.append("")
    lines += ["## R1: リークのない F1(A1)の判定", "",
              "| 条件 | 判定 | 値(c2 / c1) |", "|---|---|---|"]
    r1_ok = False
    if "A1|c2" in res and "A1|c1" in res:
        a = [res[f"A1|{c}"] for c in ("c2", "c1")]
        cc = conds("A1")
        vals = (f"{a[0]['m']['viol']} / {a[1]['m']['viol']}",
                f"{a[0]['multi']:.2f} / {a[1]['multi']:.2f}",
                f"AUC {a[0]['t_auc']:.3f} / {a[1]['t_auc']:.3f}、ゼロ {a[0]['e_zero']:.2f} / "
                f"{a[1]['e_zero']:.2f}", f"{sym(a[0]['vs']['sign'])} / {sym(a[1]['vs']['sign'])}")
        for lab_c, ok, val in zip(("1 違反 0", "2 統合 ≥ 3", "3 AUC・終点のゼロ",
                                   "4 主指標(劣らず、1 つ以上優れる)"), cc, vals, strict=True):
            lines.append(f"| {lab_c} | {yn(ok)} | {val} |")
        r1_ok = all(cc)
        lines += ["", "A1 − ステージ72 の F1(リークあり)の差: "
                  + "、".join(f"{c} 純資産の改善 {ci(res[f'A1|{c}']['vs_f72']['nai'])}・順位相関 "
                             f"{ci(res[f'A1|{c}']['vs_f72']['rho'])}" for c in ("c2", "c1")) + "。"]
    r1_txt = "成立。A1 の数値を F1 の正式な数値とする" if r1_ok else "不成立。F1 の採用を取り消す"
    lines += ["", f"**R1: {r1_txt}**",
              "", "## R2: 相対コスト(B1・B2)の判定", "",
              "| 設定 | R1 の 4 条件 | A1 との比較(c2 / c1、記号) "
              "| 膨張(設定 / A1、c2 ・ c1) | 置き換え |",
              "|---|---|---|---|---|"]
    repl: list[tuple[float, str]] = []
    for st in ("B1", "B2"):
        if f"{st}|c2" not in res or f"{st}|c1" not in res or "vs_a1" not in res[f"{st}|c2"]:
            lines.append(f"| {LABELS[st]} | 較正できない | — | — | 否 |")
            continue
        rr = [res[f"{st}|{c}"] for c in ("c2", "c1")]
        aa = [res[f"A1|{c}"] for c in ("c2", "c1")]
        cc = conds(st)
        no_worse = all(-1 not in [int(v) for v in r["vs_a1"]["sign"]] for r in rr)
        less_infl = all(r["infl"] < a_["infl"] for r, a_ in zip(rr, aa, strict=True))
        ok = r1_ok and all(cc) and no_worse and less_infl
        if ok:
            repl.append((lval[st], st))
        lines.append(f"| {LABELS[st]} | {yn(all(cc))}({' '.join('○' if v else '×' for v in cc)}) "
                     f"| {yn(no_worse)}({sym(rr[0]['vs_a1']['sign'])} / "
                     f"{sym(rr[1]['vs_a1']['sign'])}) | {yn(less_infl)}({rr[0]['infl']:.1%} / "
                     f"{aa[0]['infl']:.1%}・{rr[1]['infl']:.1%} / {aa[1]['infl']:.1%}) "
                     f"| {'置き換え' if ok else '否'} |")
    final = min(repl)[1] if repl else ("A1" if r1_ok else None)
    fin_txt = LABELS[final] if final else "ステージ69 の設定(F1 の採用を取り消し)"
    lines += ["", f"**最終: {fin_txt}**", ""]
    (OUT / "report_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(
        {"star": star, "L": lval, "r1": r1_ok, "final": final, "res": res},
        ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
