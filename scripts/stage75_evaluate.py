#!/usr/bin/env python
"""ステージ75: 57 社の交差適合(D1)と役割の分割(D2)の評価・判定(lightgbm・sklearn のプロセス)。

規則は ``results/stage75/criteria.md``。事前に ``results/stage75/run.sh`` を実行する。
基準はステージ69 の設定(ステージ67 の ``s1__l1``)。class 1 の順位相関は、fold ごとの参照 ref_f
(D2 は 7 社)で設定と基準の両方を測り直す。class 2 の参照は従来どおり Y 群 15 社。

出力: ``results/stage75/report_tables.md``・``summary.json``。

``--argmax-data`` で argmax 系の生成と指標だけ別の場所から読む(ステージ78: F1 の尺度 s で
生成し直した argmax)。

    uv run python scripts/stage75_evaluate.py
    uv run python scripts/stage75_evaluate.py --argmax-data data/interim/stage78 \
        --out results/stage78/d1_sF1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import numpy as np
import pandas as pd
from compare_stage39 import boot, corr, verdict
from evaluate_stage61 import CAL, gen_firms, loss, real_firms, real_targets
from stage71_evaluate import endpoints
from stage72_evaluate import ZERO, ci, sym
from stage74_evaluate import AUC_MAX, ZERO_MIN, inflation, start_points
from tables_stage69 import INTERIM, S67, Src, ctx, metrics, setup_excl
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import subtotal_members
from fsgen.evaluation.identification import identification_auc_repeated
from fsgen.evaluation.paired import paired_test

OUT = Path("results/stage75")
DATA = INTERIM / "stage75"
LAMS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
FOLDS = ("f0", "f1", "f2", "f3", "f4")
KINDS = ("abs", "R1", "R2", "R3")
TARGETS = ("argmax", "f1")
SETTINGS = tuple(f"{k}_{t}" for k in KINDS for t in TARGETS)
PRIMARY = ("nai", "res", "rho")


def label(st: str) -> str:
    k, t = st.split("_")
    return f"{k}・{'argmax' if t == 'argmax' else 'F1'}"


def t2_ref(codes: set[str], names: tuple[str, ...]) -> tuple[pd.Series, int]:
    """T2 の組のうち codes の企業だけの 1 年の差分の中央値(``metrics_stage39.real_refs``)。"""
    pan = pd.read_csv("data/interim/stage23/panel/panel.csv", dtype={"edinetCode": str})
    r38 = pd.read_parquet("data/interim/stage38/real_deltas.parquet")
    t2 = r38[(r38["label_prev"] == 1) & (r38["label_cur"] == 0)][["edinetCode", "t"]]
    t2 = t2[t2["edinetCode"].astype(str).isin(codes)]
    pp = pan.merge(t2.astype({"edinetCode": str}), on=["edinetCode", "t"], how="inner")
    if len(pp) != len(t2):
        raise ValueError(f"T2 の組 {len(t2)} のうちパネルにあるのは {len(pp)}")
    d = pd.DataFrame({n: (pp[f"cur_{n}"].fillna(0) - pp[f"prev_{n}"].fillna(0))
                      / pp["prev_total_assets"] for n in names})
    return d.median(), len(pp)


def fold_targets(f: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    """``evaluate_stage61.real_targets`` と同じ値と SE を、与えた企業だけで計算する。"""
    r = {"paid_dec": float(f["paid_dec"].mean()), "re_up": float(f["re_up"].mean()),
         "d_assets": float(f["d_assets"].median())}
    rng = np.random.default_rng(0)
    boots: dict[str, list[float]] = {m: [] for m in CAL}
    for _ in range(2000):
        s = f.iloc[rng.integers(0, len(f), len(f))]
        boots["paid_dec"].append(float(s["paid_dec"].mean()))
        boots["re_up"].append(float(s["re_up"].mean()))
        boots["d_assets"].append(float(s["d_assets"].median()))
    se = {m: float(np.std(v, ddof=1)) for m, v in boots.items()}
    # 追記(criteria.md §6): 割合の SE が 0(fold の全社が同じ値)のときだけ、+0.5 の補正の二項 SE
    for m in ("paid_dec", "re_up"):
        if se[m] == 0.0:
            v = f[m].dropna()
            pt = (float(v.sum()) + 0.5) / (len(v) + 1)
            se[m] = float(np.sqrt(pt * (1.0 - pt) / len(v)))
    if se["d_assets"] == 0.0:
        raise ValueError("総資産の変化の SE が 0")
    return r, se


def versus_ref(a: Src, b: Src, cls: str, names: tuple[str, ...], ref: pd.Series
               ) -> dict[str, Any]:
    """a − b の企業単位のペア比較。順位相関は与えた参照で測る(乱数 0 から)。"""
    rng = np.random.default_rng(0)
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
    d = corr(wa, ref) - corr(wb, ref)
    _, _, lo, hi = boot(wa, ref, wb, ref, rng)
    return {"nai": (r.median_diff, r.ci_low, r.ci_high),
            "res": (rr.median_diff, rr.ci_low, rr.ci_high), "rho": (d, lo, hi),
            "sign": [verdict("大", r.median_diff, r.ci_low, r.ci_high),
                     verdict("大", rr.median_diff, rr.ci_low, rr.ci_high),
                     verdict("大", d, lo, hi)]}


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--argmax-data", type=Path, default=DATA, help="argmax 系の生成・指標の場所")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    setup_excl(names)
    x, lab, _docs, _ = load_tilde(names)
    xt = x[lab == 0]
    wj = json.loads((DATA / "weights.json").read_text(encoding="utf-8"))
    p57 = {c for f in wj["folds"].values() for c in f}
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    p15 = set(pairs[pairs["group"] == "class 2 → 回復"]["edinetCode"])
    r38 = pd.read_parquet("data/interim/stage38/real_deltas.parquet")
    t2c = set(r38[(r38["label_prev"] == 1) & (r38["label_cur"] == 0)]["edinetCode"].astype(str))
    seven = t2c - p57 - p15
    if len(seven) != 7:
        raise ValueError(f"T2 で 57 社にも 15 社にもない企業が {len(seven)} 社")
    rf1 = real_firms("c1", names)
    base = Src(*S67, "s1__l1", "1")
    c2_ref = ctx(base, "c2", names).ref
    by: dict[str, dict[str, Any]] = {cls: {"m": metrics(base, cls, names, members)}
                                     for cls in ("c2", "c1")}
    for cls in ("c2", "c1"):
        y = endpoints(base, cls, names)
        by[cls]["infl"] = inflation(y, start_points(base, cls, names))
        by[cls]["e_zero"] = float((np.abs(y) < ZERO).sum(axis=1).mean())
    auc_cache: dict[tuple[str, str, str, str], float] = {}

    def src_of(cfg: str, st: str, lt: float) -> Src:
        d = args.argmax_data if st.endswith("_argmax") else DATA
        return Src(d / f"metrics_{cfg}", d / f"gen_{cfg}", f"{st}__l{lt:g}", f"{lt:g}")

    def calibrate(cfg: str, st: str, r: dict[str, float], se: dict[str, float]
                  ) -> tuple[float | None, float, list[str]]:
        w = {m: 1.0 / se[m] for m in CAL}
        best: tuple[float | None, float] = (None, np.inf)
        cells: list[str] = []
        for lt in LAMS:
            src = src_of(cfg, st, lt)
            c = ctx(src, "c1", names)
            nv = c.viol[c.viol["setting"] == src.setting]["docID"].nunique()
            f = gen_firms("c1", src.setting, names, members)
            f = f[~f.index.isin(e51.EXCL["c1"])]
            lv = loss({"paid_dec": float(f["paid_dec"].mean()), "re_up": float(f["re_up"].mean()),
                       "d_assets": float(f["d_assets"].median())}, r, w)
            cells.append(f"{lv:.2f}({nv})")
            if nv == 0 and lv < best[1]:
                best = (lt, lv)
        return best[0], best[1], cells

    def evaluate(cfg: str, st: str, lt: float, refs: dict[str, pd.Series]) -> dict[str, Any]:
        src = src_of(cfg, st, lt)
        out: dict[str, Any] = {}
        kind, target = st.split("_")
        for cls in ("c2", "c1"):
            m = metrics(src, cls, names, members)
            m["rho"] = float(corr(ctx(src, cls, names).wide[src.setting], refs[cls]))
            y = endpoints(src, cls, names)
            z = np.load(DATA / f"plan_{cls}_{cfg}.npz", allow_pickle=True)
            idx, pt = np.asarray(z[f"{kind}__idx"]), np.asarray(z[f"{kind}__pt"])
            key = (cls, cfg, kind, target)
            if key not in auc_cache:
                pts = xt[idx[pt > 0]] if target == "f1" else xt[idx[:, 0]]
                auc_cache[key] = float(identification_auc_repeated(pts, xt, 3)["lightgbm"].mean())
            out[cls] = {"m": m, "vs": versus_ref(src, base, cls, names, refs[cls]),
                        "base_rho": float(corr(ctx(base, cls, names).wide[base.setting],
                                               refs[cls])),
                        "infl": inflation(y, start_points(src, cls, names)),
                        "e_zero": float((np.abs(y) < ZERO).sum(axis=1).mean()),
                        "multi": float(np.median(1.0 / (pt ** 2).sum(axis=1)))
                        if target == "f1" else 1.0,
                        "auc": auc_cache[key]}
        return out

    # D1
    cal_lines = ["## D1: λ の較正(fold f の class 1 の格子、目標 L_f = fold f の 57 社の実データ)",
                 "", "セルは L_f(違反企業数)。λ̃*_f = 違反 0 の中で L_f 最小。", ""]
    d1: dict[str, dict[str, Any]] = {st: {"star": {}, "L": {}, "ev": {}} for st in SETTINGS}
    ref_n: dict[str, int] = {}
    base_rho_c1: dict[str, float] = {}
    for fi, cfg in enumerate(FOLDS):
        codes = set(wj["folds"][cfg])
        r, se = fold_targets(rf1.loc[rf1.index.isin(codes)])
        ref1, n_ref = t2_ref((t2c & codes) | seven, names)
        ref_n[cfg] = n_ref
        base_rho_c1[cfg] = float(corr(ctx(base, "c1", names).wide[base.setting], ref1))
        cal_lines += [f"### {cfg}(L_f の目標 {len(codes)} 社、class 1 の参照 {n_ref} 社)", "",
                      "| 設定 | " + " | ".join(f"{v:g}" for v in LAMS) + " | λ̃*_f |",
                      "|---|" + "---:|" * (len(LAMS) + 1)]
        for st in SETTINGS:
            lt, lv, cells = calibrate(cfg, st, r, se)
            d1[st]["star"][cfg], d1[st]["L"][cfg] = lt, lv
            cal_lines.append(f"| {label(st)} | " + " | ".join(cells)
                             + f" | **{'なし' if lt is None else f'{lt:g}'}** |")
            if lt is not None:
                d1[st]["ev"][cfg] = evaluate(cfg, st, lt, {"c2": c2_ref, "c1": ref1})
            print(f"  [{fi}] {cfg} {st}: λ̃* = {lt}", flush=True)
        cal_lines.append("")

    def agg(st: str, cls: str) -> dict[str, Any]:
        ev = d1[st]["ev"]
        if len(ev) < len(FOLDS):
            return {"ok": False}
        rr = [ev[f][cls] for f in FOLDS]
        out: dict[str, Any] = {"ok": True, "viol_max": max(r["m"]["viol"] for r in rr)}
        for k in ("nai", "res", "rho", "paid_dec", "paid_inc", "n_changed"):
            out[k] = float(np.mean([r["m"][k] for r in rr]))
        for k in ("infl", "e_zero", "multi", "auc"):
            out[k] = float(np.median([r[k] for r in rr]))
        sign = []
        for j, k in enumerate(PRIMARY):
            dd = np.array([r["vs"][k][0] for r in rr])
            n_pos = sum(int(r["vs"]["sign"][j]) == 1 for r in rr)
            n_neg = sum(int(r["vs"]["sign"][j]) == -1 for r in rr)
            out[f"d_{k}"] = (float(dd.mean()), n_pos, n_neg)
            sign.append(1 if dd.mean() > 0 and n_pos >= 3 else
                        -1 if dd.mean() < 0 and n_neg >= 3 else 0)
        out["sign"] = sign
        return out

    summ = {st: {cls: agg(st, cls) for cls in ("c2", "c1")} for st in SETTINGS}

    def conds(st: str) -> tuple[bool, bool, bool, bool]:
        a = [summ[st][c] for c in ("c2", "c1")]
        if not all(x_["ok"] for x_ in a):
            return False, False, False, False
        c1_ = all(x_["viol_max"] == 0 for x_ in a)
        c2_ = all(-1 not in x_["sign"] for x_ in a)
        c3_ = all(1 in x_["sign"] for x_ in a)
        if st.endswith("_f1"):
            c4_ = all(x_["multi"] >= 3 and x_["auc"] <= AUC_MAX[c] and x_["e_zero"] >= ZERO_MIN
                      for x_, c in zip(a, ("c2", "c1"), strict=True))
        else:
            c4_ = all(x_["auc"] <= AUC_MAX[c] for x_, c in zip(a, ("c2", "c1"), strict=True))
        return c1_, c2_, c3_, c4_

    def dcell(t: tuple[float, int, int]) -> str:
        return f"{t[0]:+.3f}({t[1]}+ / {t[2]}−)"

    lines = ["# ステージ75: 表(`criteria.md` の規則)", "", *cal_lines,
             "## D1: 基準(ステージ69 の設定)との比較(5 fold)", "",
             "- 値は fold の平均(違反は最大、膨張・統合・AUC・終点のゼロは中央値)。",
             "- 差 = 設定 − 基準の企業単位のペア比較の fold 平均(括弧は 95% 区間が 0 より上 / 下の"
             " fold 数)。記号 = 純資産の改善・解消率・順位相関(「優れる」= 平均が正かつ 3 fold"
             " 以上で上)。",
             "- class 1 の順位相関は fold ごとの参照 ref_f(T2 ∩ fold f と 7 社)で、基準も測り直す。"
             f"基準の class 1 の順位相関(fold 平均)= {np.mean(list(base_rho_c1.values())):+.3f}"
             f"(fold ごと {', '.join(f'{v:+.3f}' for v in base_rho_c1.values())}、参照の社数 "
             f"{', '.join(str(v) for v in ref_n.values())})。", ""]
    for cls in ("c2", "c1"):
        bm = by[cls]["m"]
        brho = np.mean(list(base_rho_c1.values())) if cls == "c1" else bm["rho"]
        lines += [f"### {cls}", "",
                  "| 設定 | λ̃*_f | 違反 | 純資産の改善 | 差 | 解消率 | 差 | 順位相関 | 差 | 記号 "
                  "| 払込資本↓ / ↑ | 変更科目数 | 膨張 | 統合 | AUC | 終点のゼロ |",
                  "|---|---|---:|---:|---|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|",
                  f"| 基準(ステージ69) | 1 | {bm['viol']} | {bm['nai']:.3f} | — | {bm['res']:.2f} "
                  f"| — | {brho:+.3f} | — | — | {bm['paid_dec']:.0%} / {bm['paid_inc']:.0%} "
                  f"| {bm['n_changed']:.0f} | {by[cls]['infl']:.1%} | 1 | — "
                  f"| {by[cls]['e_zero']:.2f} |"]
        for st in SETTINGS:
            s = summ[st][cls]
            stars = ", ".join("—" if d1[st]["star"][f] is None else f"{d1[st]['star'][f]:g}"
                              for f in FOLDS)
            if not s["ok"]:
                lines.append(f"| {label(st)} | {stars} | 較正できない fold あり |" + " — |" * 13)
                continue
            lines.append(
                f"| {label(st)} | {stars} | {s['viol_max']} | {s['nai']:.3f} "
                f"| {dcell(s['d_nai'])} | {s['res']:.2f} | {dcell(s['d_res'])} "
                f"| {s['rho']:+.3f} | {dcell(s['d_rho'])} "
                f"| {sym(s['sign'])} | {s['paid_dec']:.0%} / {s['paid_inc']:.0%} "
                f"| {s['n_changed']:.0f} | {s['infl']:.1%} | {s['multi']:.2f} | {s['auc']:.3f} "
                f"| {s['e_zero']:.2f} |")
        lines.append("")

    def yn(b_: bool) -> str:
        return "○" if b_ else "×"

    lines += ["## D1: 判定", "",
              "条件: 1 全 fold・両 class で違反 0 / 2 主指標のどれも劣らない"
              " / 3 各 class で 1 つ以上"
              "優れる / 4 F1 は統合 ≥ 3・候補の AUC ≤ 0.762 / 0.736・終点のゼロ ≥ 6.70、argmax は"
              "目標の AUC(fold の中央値)。", "",
              "| 設定 | 1 | 2 | 3 | 4 | L_f の fold 平均 | 採用の候補 |",
              "|---|---|---|---|---|---:|---|"]
    cand: list[tuple[float, str]] = []
    for st in SETTINGS:
        cc = conds(st)
        lfs = [v for v in d1[st]["L"].values() if np.isfinite(v)]
        lm = float(np.mean(lfs)) if len(lfs) == len(FOLDS) else float("nan")
        if all(cc):
            cand.append((lm, st))
        lines.append(f"| {label(st)} | " + " | ".join(yn(v) for v in cc)
                     + f" | {lm:.2f} | {'候補' if all(cc) else '—'} |")
    final = min(cand)[1] if cand else None
    fin = f"{label(final)}(ステージ69 の設定を置き換える)" if final else "ステージ69 の設定のまま"
    lines += ["", f"**最終: {fin}**", ""]

    # D2
    r2, se2, _ = real_targets("c2", names)
    ref7, n7 = t2_ref(seven, names)
    refs2 = {"c2": c2_ref, "c1": ref7}
    lines += ["## D2(参考): コスト = 57 社の全体、較正の目標 = class 2 → 回復 15 社、"
              f"class 1 の参照 = 7 社({n7} 組)", "",
              "λ̃* は class 1 の格子で、違反 0 の中で 15 社の目標との L 最小(class 1 だけで"
              "較正する規則を保つ)。", "",
              "| 設定 | " + " | ".join(f"{v:g}" for v in LAMS) + " | λ̃* |",
              "|---|" + "---:|" * (len(LAMS) + 1)]
    d2: dict[str, Any] = {}
    for st in SETTINGS:
        lt, lv, cells = calibrate("full", st, r2, se2)
        lines.append(f"| {label(st)} | " + " | ".join(cells)
                     + f" | **{'なし' if lt is None else f'{lt:g}'}** |")
        d2[st] = {"star": lt, "L": lv,
                  "ev": evaluate("full", st, lt, refs2) if lt is not None else None}
        print(f"  D2 {st}: λ̃* = {lt}", flush=True)
    lines.append("")
    for cls in ("c2", "c1"):
        bm = by[cls]["m"]
        brho = float(corr(ctx(base, cls, names).wide[base.setting], refs2[cls]))
        lines += [f"### D2 {cls}", "",
                  "| 設定 | λ̃* | 違反 | 純資産の改善 | 差 [95%] | 解消率 | 順位相関 "
                  "| 差 [95%] | 記号 "
                  "| 払込資本↓ / ↑ | 変更科目数 | 膨張 | 統合 | AUC | 終点のゼロ |",
                  "|---|---:|---:|---:|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|",
                  f"| 基準(ステージ69) | 1 | {bm['viol']} | {bm['nai']:.3f} | — | {bm['res']:.2f} "
                  f"| {brho:+.3f} | — | — | {bm['paid_dec']:.0%} / {bm['paid_inc']:.0%} "
                  f"| {bm['n_changed']:.0f} | {by[cls]['infl']:.1%} | 1 | — "
                  f"| {by[cls]['e_zero']:.2f} |"]
        for st in SETTINGS:
            ev = d2[st]["ev"]
            if ev is None:
                lines.append(f"| {label(st)} | なし | 較正できない |" + " — |" * 12)
                continue
            e = ev[cls]
            v = e["m"]
            lines.append(
                f"| {label(st)} | {v['lam']} | {v['viol']} | {v['nai']:.3f} "
                f"| {ci(e['vs']['nai'])} | {v['res']:.2f} | {v['rho']:+.3f} "
                f"| {ci(e['vs']['rho'])} | {sym(e['vs']['sign'])} "
                f"| {v['paid_dec']:.0%} / {v['paid_inc']:.0%} | {v['n_changed']:.0f} "
                f"| {e['infl']:.1%} | {e['multi']:.2f} | {e['auc']:.3f} | {e['e_zero']:.2f} |")
        lines.append("")
    (args.out / "report_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(
        {"d1": d1, "d1_summary": summ, "final": final, "base": by, "base_rho_c1": base_rho_c1,
         "ref_n": ref_n, "d2": d2}, ensure_ascii=False, indent=1, default=float),
        encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
