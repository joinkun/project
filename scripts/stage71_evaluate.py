#!/usr/bin/env python
"""ステージ71: 較正・比較・目標と終点の性質・判定(lightgbm・sklearn のプロセス)。

規則は ``results/stage71/criteria.md``。事前に ``metrics_stage39.py --gen-dir
data/interim/stage71/gen --out data/interim/stage71/metrics`` を実行する。
基準は設定 1(ステージ69 の最終設定)。

    uv run python scripts/stage71_evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import numpy as np
import pandas as pd
from evaluate_stage61 import CAL, gen_firms, loss, real_targets
from tables_stage69 import INTERIM, S67, Src, ctx, metrics, setup_excl, versus
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.plausibility import subtotal_members
from fsgen.evaluation.identification import identification_auc_repeated

OUT = Path("results/stage71")
S71 = (INTERIM / "stage71/metrics", INTERIM / "stage71/gen")
S70 = (INTERIM / "stage70/metrics", INTERIM / "stage70/gen")
LAMS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
LABELS = {"a2": "2 新設計 + 𝓕⁺ + argmax(τ1 = 0.01)", "f1": "3 F1 事後アンサンブル(τ1 = 0.05)",
          "f2": "4 F2 期待 L1 接近(τ1 = 0.05)", "f3": "5 F3 ゼロ整合重心(ハミング ≤ 2、τ1 = 0.05)",
          "f3h0": "5' F3(ハミング 0、τ1 = 0.05)", "f3t05": "5'' F3(ハミング ≤ 2、τ1 = 0.5)"}
CANDIDATES = ("f1", "f2", "f3", "f3h0", "f3t05")
FACE = {"f3": ("t0.05", 2), "f3h0": ("t0.05", 0), "f3t05": ("t0.5", 2)}
UG53 = "S100UG53"


def endpoints(src: Src, cls: str, names: tuple[str, ...]) -> np.ndarray:
    e51._GEN = src.gen
    g = e51.load_gen(cls, src.setting)
    g = g[g["seed"] == 0]
    out: np.ndarray = g[[f"x1_{n}" for n in names]].to_numpy(float)
    return out


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    cmat = build_constraint_matrix(spec, include_subtotals=False)[:, 0]
    assets = members["jppfs_cor_Assets"]
    setup_excl(names)
    x, lab, _docs, _ = load_tilde(names)
    xt = x[lab == 0]
    r1, se1, _ = real_targets("c1", names)
    w1 = {m: 1.0 / se1[m] for m in CAL}
    cal = ["# ステージ71: λ の較正(class 1 の 128 社だけ)", "",
           "| 設定 | " + " | ".join(f"{v:g}" for v in LAMS) + " | λ̃*_0 |",
           "|---|" + "---:|" * (len(LAMS) + 1)]
    star: dict[str, float | None] = {}
    lval: dict[str, float] = {}
    ug: dict[str, list[str]] = {}
    for st, lab_s in LABELS.items():
        cells: list[str] = []
        best: tuple[float | None, float] = (None, np.inf)
        ug[st] = []
        for lt in LAMS:
            src = Src(*S71, f"{st}__l{lt:g}", f"{lt:g}")
            c = ctx(src, "c1", names)
            v = c.viol[c.viol["setting"] == src.setting]
            f = gen_firms("c1", src.setting, names, members)
            f = f[~f.index.isin(e51.EXCL["c1"])]
            nv = v["docID"].nunique()
            if (v["docID"] == UG53).any():
                ug[st].append(f"{lt:g}")
            lv = loss({"paid_dec": float(f["paid_dec"].mean()), "re_up": float(f["re_up"].mean()),
                       "d_assets": float(f["d_assets"].median())}, r1, w1)
            cells.append(f"{lv:.2f}({nv})")
            if nv == 0 and lv < best[1]:
                best = (lt, lv)
        star[st], lval[st] = best[0], best[1]
        cal.append(f"| {lab_s} | " + " | ".join(cells)
                   + f" | **{'なし' if best[0] is None else f'{best[0]:g}'}** |")
    cal += ["", "セルは L(違反した企業数)。class 2 には同じ λ を当てはめる。", "",
            "**S100UG53 の違反**(class 1、λ̃ の格子点): "
            + "、".join(f"{LABELS[k].split(' ')[0]} {', '.join(v) or 'なし'}"
                       for k, v in ug.items())
            + "。ステージ70 の設定 2(𝓕⁺ なし、m = 10)では λ̃ = 0.003〜3 のすべてで"
            "違反していた。", ""]
    base = Src(*S67, "s1__l1", "1")
    s70 = Src(*S70, "s2_m10__l10", "10")
    res: dict[str, dict[str, Any]] = {}
    rng0 = np.random.default_rng
    plan = {cls: np.load(INTERIM / f"stage71/plan_{cls}.npz", allow_pickle=True)
            for cls in ("c2", "c1")}
    end_auc: dict[str, dict[str, float]] = {}
    for cls in ("c2", "c1"):
        yb = endpoints(base, cls, names)
        rb = identification_auc_repeated(yb, xt, 3)
        rbn = identification_auc_repeated(yb / (yb @ assets)[:, None], xt, 3)
        end_auc[f"s1|{cls}"] = {"auc": float(rb["lightgbm"].mean()),
                                "auc_n": float(rbn["lightgbm"].mean()),
                                "zero": float((yb == 0).sum(axis=1).mean())}
        res[f"s1|{cls}"] = {"m": metrics(base, cls, names, members)}
        res[f"s70|{cls}"] = {"m": metrics(s70, cls, names, members),
                             "vs": versus(s70, base, cls, names, rng0(0))}
        pt = np.asarray(plan[cls]["t0.05__pt"])
        eff = float(np.median(1.0 / (pt ** 2).sum(axis=1)))
        kmed = float(np.median(plan[cls]["t0.05__k"]))
        for st in LABELS:
            if star[st] is None:
                continue
            src = Src(*S71, f"{st}__l{star[st]:g}", f"{star[st]:g}")
            y = endpoints(src, cls, names)
            ra = identification_auc_repeated(y, xt, 3)
            rn = identification_auc_repeated(y / (y @ assets)[:, None], xt, 3)
            info: dict[str, Any] = {"m": metrics(src, cls, names, members),
                                    "vs": versus(src, base, cls, names, rng0(0)),
                                    "end_auc": float(ra["lightgbm"].mean()),
                                    "end_auc_n": float(rn["lightgbm"].mean()),
                                    "end_zero": float((y == 0).sum(axis=1).mean()),
                                    "end_resid": float(np.median(np.abs(y @ cmat)))}
            if st in ("f1", "f2"):
                info["multi"], info["k"] = eff, kmed
            elif st in FACE:
                tk, h = FACE[st]
                tau = np.asarray(plan[cls][f"{tk}__face{h}"], dtype=float)
                rt = identification_auc_repeated(tau, xt, 3)
                info["multi"] = float(np.median(plan[cls][f"{tk}__face{h}_used"]))
                info["t_auc"] = float(rt["lightgbm"].mean())
                info["t_zero"] = float((tau == 0).sum(axis=1).mean())
                info["t_resid"] = float(np.median(np.abs(tau @ cmat)))
            else:
                info["multi"] = 1.0
            res[f"{st}|{cls}"] = info
    t64 = {cls: float(pd.read_csv(f"results/stage65/auc_{cls}.csv").set_index("method")
                      .loc["m5_prop", "auc_lgbm"]) for cls in ("c2", "c1")}
    real_zero = float((xt == 0).sum(axis=1).mean())

    def sym(s: list[int]) -> str:
        return " ".join({1: "+", -1: "−", 0: "0"}[int(v)] for v in s)

    def ci(t: tuple[float, float, float]) -> str:
        return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"

    comp = ["# ステージ71: 生成の比較", "",
            "差は設定 1(ステージ69 の最終設定)との企業単位のペア比較(2,000 回のブートストラップ)。"
            "記号 = 純資産の改善・解消率・順位相関。「統合した企業数」は F1・F2 では p̃ の"
            "実効サポート"
            "の中央値、F3 では重心に使った企業数の中央値。", ""]
    for cls in ("c2", "c1"):
        comp += [f"## {cls}", "",
                 "| 設定 | 統合した企業数 | λ̃*_0 | 違反 | 純資産の改善 | 設定 1 との差 [95%] "
                 "| 解消率 | 払込資本↓ / ↑ | 変更科目数 | 順位相関 | 順位相関の差 [95%] | 記号 |",
                 "|---|---:|---:|---:|---:|---|---:|---|---:|---:|---|---|"]
        b = res[f"s1|{cls}"]["m"]
        comp.append(f"| 1 従来(ステージ69) | 1 | 1 | {b['viol']} | {b['nai']:.3f} | — "
                    f"| {b['res']:.2f} | {b['paid_dec']:.0%} / {b['paid_inc']:.0%} "
                    f"| {b['n_changed']:.0f} | {b['rho']:+.3f} | — | — |")
        r7 = res[f"s70|{cls}"]
        comp.append(f"| 参考: ステージ70 の設定 2(𝓕⁺ なし) | 1 | 10 | {r7['m']['viol']} "
                    f"| {r7['m']['nai']:.3f} | {ci(r7['vs']['nai'])} | {r7['m']['res']:.2f} "
                    f"| {r7['m']['paid_dec']:.0%} / {r7['m']['paid_inc']:.0%} "
                    f"| {r7['m']['n_changed']:.0f} | {r7['m']['rho']:+.3f} | {ci(r7['vs']['rho'])} "
                    f"| {sym(r7['vs']['sign'])} |")
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
    prop = ["# ステージ71: 目標と終点の性質", "",
            f"健全企業(class 0)のちょうど 0 の科目数は {real_zero:.2f}。識別 AUC は LightGBM"
            "(5-fold、3 回の平均)。終点は λ̃*_0 の生成(seed 0)、始点の総資産比のまま(「総資産で"
            "正規化」は終点を自分の総資産で割った補助の値)。設定 1 の目標の AUC は "
            f"{t64['c2']:.3f} / {t64['c1']:.3f}。", "",
            "| class | 設定 | 目標: ちょうど 0 / 貸借の残差 / AUC | 終点: ちょうど 0 "
            "| 終点: 貸借の残差 "
            "| 終点の AUC | 終点の AUC(総資産で正規化) |", "|---|---|---|---:|---:|---:|---:|"]
    for cls in ("c2", "c1"):
        e = end_auc[f"s1|{cls}"]
        prop.append(f"| {cls} | 1 従来 | — | {e['zero']:.2f} | — | {e['auc']:.3f} "
                    f"| {e['auc_n']:.3f} |")
        for st, lab_s in LABELS.items():
            if f"{st}|{cls}" not in res:
                continue
            r = res[f"{st}|{cls}"]
            tg = (f"{r['t_zero']:.2f} / {r['t_resid']:.1e} / {r['t_auc']:.3f}" if "t_auc" in r
                  else "—")
            prop.append(f"| {cls} | {lab_s} | {tg} | {r['end_zero']:.2f} | {r['end_resid']:.1e} "
                        f"| {r['end_auc']:.3f} | {r['end_auc_n']:.3f} |")
    prop.append("")
    jl = ["# ステージ71: 判定(criteria.md §4)", "",
          "| 設定 | 1 違反 0 | 2 複数企業 | 3 実在企業らしさ | 4 主指標が劣らない | class 1 の L "
          "| 採用候補 |",
          "|---|---|---|---|---|---:|---|"]
    cands: list[tuple[float, str]] = []
    for st in CANDIDATES:
        if f"{st}|c2" not in res or f"{st}|c1" not in res:
            jl.append(f"| {LABELS[st]} | 較正できない | — | — | — | — | 否 |")
            continue
        rr = {cls: res[f"{st}|{cls}"] for cls in ("c2", "c1")}
        c1_ = all(rr[c]["m"]["viol"] == 0 for c in rr)
        c2_ = all(rr[c]["multi"] >= 2 for c in rr)
        c3_ = all(rr[c]["end_auc"] <= end_auc[f"s1|{c}"]["auc"] + 0.05 for c in rr)
        if st in FACE:
            c3_ = c3_ and all(rr[c]["t_auc"] <= t64[c] + 0.05 for c in rr)
        c4_ = all(-1 not in [int(v) for v in rr[c]["vs"]["sign"]] for c in rr)
        ok = c1_ and c2_ and c3_ and c4_

        def yn(b_: bool) -> str:
            return "成立" if b_ else "**不成立**"
        a3 = (f"終点 {rr['c2']['end_auc']:.3f} / {rr['c1']['end_auc']:.3f}(上限 "
              f"{end_auc['s1|c2']['auc'] + 0.05:.3f} / {end_auc['s1|c1']['auc'] + 0.05:.3f})")
        if st in FACE:
            a3 += f"、目標 {rr['c2']['t_auc']:.3f} / {rr['c1']['t_auc']:.3f}"
        jl.append(f"| {LABELS[st]} | {yn(c1_)} | {yn(c2_)}({rr['c2']['multi']:.1f} / "
                  f"{rr['c1']['multi']:.1f}) | {yn(c3_)}({a3}) | {yn(c4_)}"
                  f"({sym(rr['c2']['vs']['sign'])} / {sym(rr['c1']['vs']['sign'])}) "
                  f"| {lval[st]:.2f} | {'候補' if ok else '否'} |")
        if ok:
            cands.append((lval[st], st))
    adopted = min(cands)[1] if cands else None
    verdict = LABELS[adopted] if adopted else "なし(ステージ69 の最終設定のまま)"
    jl += ["", f"**採用: {verdict}**", ""]
    for name_, body in (("calibration.md", cal), ("comparison.md", comp),
                        ("properties.md", prop), ("judgement.md", jl)):
        (OUT / name_).write_text("\n".join(body) + "\n", encoding="utf-8")
        print("\n".join(body))
    (OUT / "summary.json").write_text(json.dumps(
        {"star": star, "L": lval, "adopted": adopted, "ug53": ug, "end_auc_s1": end_auc,
         "res": res}, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
