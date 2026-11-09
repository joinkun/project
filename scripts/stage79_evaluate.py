#!/usr/bin/env python
"""ステージ79: A1・A2・スパース OT の D1(5 fold の交差適合)による比較。

lightgbm・sklearn のプロセス。

規則は ``results/stage79/criteria.md``。事前に ``results/stage79/run.sh`` を実行する。
基準は 2 つ: ステージ69 の設定(ステージ75 と同じ比較)と、R1・F1(ステージ75 の生成を再利用)。
較正・参照・集計の規則はステージ75 と同じ(``stage75_evaluate`` の関数を使う)。

出力: ``results/stage79/comparison_report.md``・``summary.json``。

    uv run python scripts/stage79_evaluate.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import numpy as np
import pandas as pd
from compare_stage39 import corr
from evaluate_stage61 import CAL, gen_firms, loss, real_firms
from stage71_evaluate import endpoints
from stage72_evaluate import ZERO, ci, sym
from stage74_evaluate import AUC_MAX, ZERO_MIN, inflation, start_points
from stage75_evaluate import fold_targets, t2_ref, versus_ref
from tables_stage69 import INTERIM, S67, Src, ctx, metrics, setup_excl
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import subtotal_members
from fsgen.evaluation.identification import identification_auc_repeated

OUT = Path("results/stage79")
D75, D79 = INTERIM / "stage75", INTERIM / "stage79"
LAMS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
FOLDS = ("f0", "f1", "f2", "f3", "f4")
PRIMARY = ("nai", "res", "rho")


@dataclass(frozen=True)
class Setting:
    data: Path  # 生成と指標の場所(gen_{cfg}・metrics_{cfg})
    tag: str  # 生成の設定名
    pkey: str  # 計画のキー
    plan_dir: Path | None = None  # 計画の場所(None なら data)
    argmax: bool = False  # 候補の 1 位だけ(AUC は目標、ESS は 1)


SETTINGS: dict[str, Setting] = {
    "R1・F1(基準)": Setting(D75, "R1_f1", "R1"),
    "A1・F1": Setting(D79, "A1_f1", "A1"),
    "A2・F1": Setting(D79, "A2_f1", "A2"),
    "R1・SP-F1": Setting(D79, "R1sp_f1", "R1sp"),
    "A1・SP-F1": Setting(D79, "A1sp_f1", "A1sp"),
}
BASE_KEY = "R1・F1(基準)"
TITLE = "# ステージ79: 区間面積のコスト A1・A2 とスパース OT の比較(D1、5 fold の交差適合)"


def fold_sign(rr: list[dict[str, Any]], vs_key: str) -> tuple[list[int], dict[str, Any]]:
    """fold ごとの比較を、ステージ75 の規則(平均の符号 + 3 fold 以上で有意)で集計する。"""
    sign, out = [], {}
    for j, k in enumerate(PRIMARY):
        dd = np.array([r[vs_key][k][0] for r in rr], dtype=float)
        n_pos = sum(int(r[vs_key]["sign"][j]) == 1 for r in rr)
        n_neg = sum(int(r[vs_key]["sign"][j]) == -1 for r in rr)
        mean = float(np.nanmean(dd)) if np.isfinite(dd).any() else float("nan")
        out[k] = (mean, n_pos, n_neg)
        sign.append(1 if mean > 0 and n_pos >= 3 else -1 if mean < 0 and n_neg >= 3 else 0)
    return sign, out


def main() -> None:
    run(SETTINGS, OUT, TITLE, "`criteria.md`(§7 の追記を含む)")


def run(settings: dict[str, Setting], out_dir: Path, title: str, rules: str,  # noqa: PLR0915
        diag: tuple[str, ...] = ()) -> None:
    """D1 の比較を計算し、``out_dir`` に ``comparison_report.md``・``summary.json`` を書く。"""
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    setup_excl(names)
    x, lab, _docs, _ = load_tilde(names)
    xt = x[lab == 0]
    wj = json.loads((D75 / "weights.json").read_text(encoding="utf-8"))
    p57 = {c for f in wj["folds"].values() for c in f}
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    p15 = set(pairs[pairs["group"] == "class 2 → 回復"]["edinetCode"])
    r38 = pd.read_parquet("data/interim/stage38/real_deltas.parquet")
    t2c = set(r38[(r38["label_prev"] == 1) & (r38["label_cur"] == 0)]["edinetCode"].astype(str))
    seven = t2c - p57 - p15
    rf1 = real_firms("c1", names)
    base = Src(*S67, "s1__l1", "1")
    c2_ref = ctx(base, "c2", names).ref

    def src_of(st: str, cfg: str, lt: float) -> Src:
        d, tag = settings[st].data, settings[st].tag
        return Src(d / f"metrics_{cfg}", d / f"gen_{cfg}", f"{tag}__l{lt:g}", f"{lt:g}")

    res: dict[str, dict[str, Any]] = {st: {"star": {}, "L": {}, "ev": {}} for st in settings}
    cal_lines: list[str] = []
    for cfg in FOLDS:
        codes = set(wj["folds"][cfg])
        r, se = fold_targets(rf1.loc[rf1.index.isin(codes)])
        wl = {m: 1.0 / se[m] for m in CAL}
        ref1, n_ref = t2_ref((t2c & codes) | seven, names)
        refs = {"c2": c2_ref, "c1": ref1}
        cal_lines += [f"### {cfg}(L_f の目標 {len(codes)} 社、class 1 の参照 {n_ref} 社)", "",
                      "| 設定 | " + " | ".join(f"{v:g}" for v in LAMS) + " | λ̃*_f |",
                      "|---|" + "---:|" * (len(LAMS) + 1)]
        for st in settings:
            best: tuple[float | None, float] = (None, np.inf)
            cells = []
            for lt in LAMS:
                src = src_of(st, cfg, lt)
                c = ctx(src, "c1", names)
                nv = c.viol[c.viol["setting"] == src.setting]["docID"].nunique()
                f = gen_firms("c1", src.setting, names, members)
                f = f[~f.index.isin(e51.EXCL["c1"])]
                lv = loss({"paid_dec": float(f["paid_dec"].mean()),
                           "re_up": float(f["re_up"].mean()),
                           "d_assets": float(f["d_assets"].median())}, r, wl)
                cells.append(f"{lv:.2f}({nv})")
                if nv == 0 and lv < best[1]:
                    best = (lt, lv)
            res[st]["star"][cfg], res[st]["L"][cfg] = best
            cal_lines.append(f"| {st} | " + " | ".join(cells)
                             + f" | **{'なし' if best[0] is None else f'{best[0]:g}'}** |")
        cal_lines.append("")
        base_r1 = (src_of(BASE_KEY, cfg, res[BASE_KEY]["star"][cfg])
                   if res[BASE_KEY]["star"][cfg] is not None else None)
        for st, sp in settings.items():
            lt = res[st]["star"][cfg]
            if lt is None:
                continue
            src = src_of(st, cfg, lt)
            ev: dict[str, Any] = {}
            for cls in ("c2", "c1"):
                m = metrics(src, cls, names, members)
                m["rho"] = float(corr(ctx(src, cls, names).wide[src.setting], refs[cls]))
                y = endpoints(src, cls, names)
                z = np.load((sp.plan_dir or sp.data) / f"plan_{cls}_{cfg}.npz",
                            allow_pickle=True)
                idx, pt = np.asarray(z[f"{sp.pkey}__idx"]), np.asarray(z[f"{sp.pkey}__pt"])
                if sp.argmax:
                    pt = np.zeros_like(pt, dtype=float)
                    pt[:, 0] = 1.0
                nz = (pt > 0).sum(axis=1)
                e = {"m": m, "vs69": versus_ref(src, base, cls, names, refs[cls]),
                     "infl": inflation(y, start_points(src, cls, names)),
                     "e_zero": float((np.abs(y) < ZERO).sum(axis=1).mean()),
                     "s_med": float(np.median(nz)), "s_max": int(nz.max()),
                     "ess": float(np.median(1.0 / (pt ** 2).sum(axis=1))),
                     "auc": float(identification_auc_repeated(
                         xt[idx[pt > 0]], xt, 3)["lightgbm"].mean())}
                if base_r1 is not None and st != BASE_KEY:
                    e["vsR1"] = versus_ref(src, base_r1, cls, names, refs[cls])
                ev[cls] = e
            res[st]["ev"][cfg] = ev
            print(f"  {cfg} {st}: λ̃* = {lt:g}", flush=True)

    def agg(st: str, cls: str) -> dict[str, Any]:
        evs = res[st]["ev"]
        if len(evs) < len(FOLDS):
            return {"ok": False}
        rr = [evs[f][cls] for f in FOLDS]
        out: dict[str, Any] = {"ok": True, "viol_max": max(r["m"]["viol"] for r in rr)}
        for k in ("nai", "res", "rho", "paid_dec", "paid_inc", "n_changed"):
            out[k] = float(np.mean([r["m"][k] for r in rr]))
        for k in ("infl", "e_zero", "ess", "auc", "s_med"):
            out[k] = float(np.median([r[k] for r in rr]))
        out["s_max"] = max(r["s_max"] for r in rr)
        out["sign69"], out["d69"] = fold_sign(rr, "vs69")
        if st != BASE_KEY and all("vsR1" in r for r in rr):
            out["signR1"], out["dR1"] = fold_sign(rr, "vsR1")
        return out

    summ = {st: {cls: agg(st, cls) for cls in ("c2", "c1")} for st in settings}

    def conds(st: str) -> tuple[bool, ...]:
        a = [summ[st][c] for c in ("c2", "c1")]
        if not all(x_["ok"] for x_ in a):
            return (False,) * 6
        c1_ = all(x_["viol_max"] == 0 for x_ in a)
        c2_ = all(-1 not in x_["sign69"] for x_ in a)
        c3_ = all(1 in x_["sign69"] for x_ in a)
        if settings[st].argmax:  # argmax は目標の AUC だけ(ステージ75 §5)
            c4_ = all(x_["auc"] <= AUC_MAX[c] for x_, c in zip(a, ("c2", "c1"), strict=True))
        else:
            c4_ = all(x_["ess"] >= 3 and x_["auc"] <= AUC_MAX[c] and x_["e_zero"] >= ZERO_MIN
                      for x_, c in zip(a, ("c2", "c1"), strict=True))
        if st == BASE_KEY:
            return c1_, c2_, c3_, c4_, True, True
        r_no_worse = all(-1 not in x_["signR1"] for x_ in a)
        r_better = any(1 in x_["signR1"] for x_ in a)
        return c1_, c2_, c3_, c4_, r_no_worse, r_better

    def dcell(t: tuple[float, int, int]) -> str:
        return f"{t[0]:+.3f}({t[1]}+ / {t[2]}−)"

    lines = [title, "", f"規則は {rules}。",
             "値は fold の平均"
             "(違反は最大、膨張・ESS・|S(i)| の中央値・AUC・ゼロは fold の中央値)。",
             "差は企業単位のペア比較の fold 平均で、括弧は 95% 区間が 0 より上 / 下の fold 数。",
             "記号 = 純資産の改善・解消率・順位相関(「優れる」= 平均が正かつ 3 fold 以上で上)。",
             "", "## 1. 主指標", ""]
    bm = {c: metrics(base, c, names, members) for c in ("c2", "c1")}
    for cls in ("c2", "c1"):
        lines += [f"### {cls}", "",
                  "| 設定 | λ̃*_f | 違反 | \\|S(i)\\| 中央値 / 最大 | ESS | 純資産の改善 "
                  "| ステージ69 との差 | 解消率 | 順位相関 | ステージ69 との差 | 記号(対 69) "
                  "| R1・F1 との差: 純資産 / 順位相関 | 記号(対 R1・F1) | 払込資本↓ / ↑ "
                  "| 変更科目数 | 膨張 | AUC | 終点のゼロ |",
                  "|---|---|---:|---|---:|---:|---|---:|---:|---|---|---|---|---|---:|---:|---:"
                  "|---:|",
                  f"| ステージ69 の設定 | 1 | {bm[cls]['viol']} | 1 / 1 | 1 | {bm[cls]['nai']:.3f} "
                  "| — | " + f"{bm[cls]['res']:.2f} | — | — | — | — | — "
                  f"| {bm[cls]['paid_dec']:.0%} / {bm[cls]['paid_inc']:.0%} "
                  f"| {bm[cls]['n_changed']:.0f} | — | — | — |"]
        for st in settings:
            s = summ[st][cls]
            stars = ", ".join("—" if res[st]["star"][f] is None else f"{res[st]['star'][f]:g}"
                              for f in FOLDS)
            if not s["ok"]:
                lines.append(f"| {st} | {stars} | 較正できない fold あり |" + " — |" * 15)
                continue
            vr = (f"{dcell(s['dR1']['nai'])} / {dcell(s['dR1']['rho'])} | {sym(s['signR1'])}"
                  if "dR1" in s else "— | —")
            lines.append(
                f"| {st} | {stars} | {s['viol_max']} | {s['s_med']:.0f} / {s['s_max']} "
                f"| {s['ess']:.2f} | {s['nai']:.3f} | {dcell(s['d69']['nai'])} | {s['res']:.2f} "
                f"| {s['rho']:+.3f} | {dcell(s['d69']['rho'])} | {sym(s['sign69'])} | {vr} "
                f"| {s['paid_dec']:.0%} / {s['paid_inc']:.0%} | {s['n_changed']:.0f} "
                f"| {s['infl']:.1%} | {s['auc']:.3f} | {s['e_zero']:.2f} |")
        lines.append("")
    lines += ["## 2. 条件(報告だけ)", "",
              "1〜4 = ステージ75 の条件"
              "(4 は F1・SP-F1 が ESS ≥ 3・候補の AUC ≤ 0.762 / 0.736・終点のゼロ ≥ 6.70、"
              "argmax は目標の AUC だけ)、"
              "5 = R1・F1 より両 class で劣る指標がない、6 = R1・F1 よりどこかで優れる。", "",
              "| 設定 | 1 | 2 | 3 | 4 | 5 | 6 | L_f の fold 平均 | 候補 |",
              "|---|---|---|---|---|---|---|---:|---|"]
    cand: list[tuple[float, str]] = []
    for st in settings:
        cc = conds(st)
        lfs = [v for v in res[st]["L"].values() if np.isfinite(v)]
        lm = float(np.mean(lfs)) if len(lfs) == len(FOLDS) else float("nan")
        ok = all(cc) and st != BASE_KEY
        if ok:
            cand.append((lm, st))
        lines.append(f"| {st} | " + " | ".join("○" if v else "×" for v in cc)
                     + f" | {lm:.2f} | {'候補' if ok else '—'} |")
    lines += ["", "**条件 1〜6 をすべて満たす設定: " + ("、".join(st for _, st in sorted(cand))
                                                if cand else "なし") + "**", ""]
    if diag:
        lines += ["## 3. fold ごとの診断(ステージ69 の設定との差、企業単位のペア比較)", "",
                  "| 設定 | fold | λ̃*_f | c2 純資産の改善の差 [95%] | c1 純資産の改善の差 [95%] "
                  "| c1 順位相関(ref_f) | c1 順位相関の差 [95%] |",
                  "|---|---|---:|---|---|---:|---|"]
        for st in diag:
            for f in FOLDS:
                ev = res[st]["ev"].get(f)
                if ev is None:
                    lines.append(f"| {st} | {f} | なし | — | — | — | — |")
                    continue
                c2v, c1v = ev["c2"]["vs69"], ev["c1"]["vs69"]
                lines.append(f"| {st} | {f} | {res[st]['star'][f]:g} | {ci(c2v['nai'])} "
                             f"| {ci(c1v['nai'])} | {ev['c1']['m']['rho']:+.3f} "
                             f"| {ci(c1v['rho'])} |")
        lines.append("")
    lines += ["## λ の較正(fold f の class 1 の格子、セルは L_f(違反企業数))", "", *cal_lines]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(
        {"res": res, "summary": summ, "candidates": [st for _, st in sorted(cand)]},
        ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
