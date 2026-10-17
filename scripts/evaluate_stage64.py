#!/usr/bin/env python
"""ステージ64: 輸送計画の健全性・最近傍との一致(A)、τ の作り方ごとの生成の比較(B)。

規則は ``results/stage64/criteria.md``(実行の前に記録)。事前に ``ot_stage64.py``・
``auc_stage64.py``・``qp_stage61.py --tau-file`` を実行し、``data/interim/stage64/all`` に
``metrics_stage39.py`` をかける。

    uv run python scripts/evaluate_stage64.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import numpy as np
import pandas as pd
from compare_stage39 import corr
from evaluate_stage61 import CAL, gen_firms, loss, real_targets

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import subtotal_members

OUT = Path("results/stage64")
DATA = Path("data/interim/stage64")
LAMS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
TAUS = {"t1_nn": "1 最近傍", "t2_bal": "2 バランス OT の argmax", "t3_semi": "3 不均衡 OT の arg"
                                                                     "max",
        "t4_bary": "4 バランス OT の重心(対照)"}
CLS_N = {"c2": "class 2", "c1": "class 1"}


def excl() -> None:
    codes = set(pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})["edinetCode"])
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    for cls in ("c2", "c1"):
        g = pd.read_parquet(sorted((e51._GEN / cls / "t1_nn__l1a_l1").glob("*__seed0.parquet"))[0])
        drop = [d for d in g["docID"].unique() if code_of.get(d, "") in codes]
        e51.EXCL[cls] = tuple(dict.fromkeys((*e51.EXCL[cls], *drop)))


def section_a(reps: dict[str, dict[str, str]]) -> list[str]:
    lines = ["# ステージ64 A: 輸送計画の健全性と最近傍との一致", "",
             "反復: バランス OT は POT の誤差の記録回数(10 反復ごと)、不均衡 OT は反復回数。"
             "一致 = 最大質量の対応先が最近傍と同じ行。採用候補 = 質量 0 の行なし・輸送質量 ≥ 0."
             "5。", ""]
    for cls in ("c2", "c1"):
        p = pd.read_csv(OUT / f"plans_{cls}.csv")
        n0 = int(p.loc[p["setting"] == "nn", "nn_match"].iloc[0])
        lines += [f"## {CLS_N[cls]}(n0 = {n0})", "",
                  "| 方式 | ε | τ1 | 状態 | 反復 | 質量 0 の行 | 最小行質量 | 輸送質量 "
                  "| 実効サポート 中央値 / p90 / 最小 / 最大 | 最近傍と一致 | 採用候補 | 時間(秒"
                  ") |",
                  "|---|---:|---:|---|---:|---:|---:|---:|---|---:|---|---:|"]
        for r in p.itertuples():
            if r.status != "ok":
                lines.append(f"| {r.method} | {r.eps} | {r.tau1} | 失敗 | — | — | — | — | — | — "
                             f"| × "
                             f"| {r.seconds} |")
                continue
            e = "—" if pd.isna(r.eps) else f"{r.eps:g}"
            t = "—" if pd.isna(r.tau1) else f"{r.tau1:g}"
            lines.append(f"| {r.method} | {e} | {t} | ok | {int(r.n_iter)} | {int(r.n_zero_rows)} "
                         f"| {r.min_row_mass:.4f} | {r.mass:.3f} | {r.support_med:.1f} / "
                         f"{r.support_p90:.1f} / {r.support_min:.1f} / {r.support_max:.1f} "
                         f"| **{int(r.nn_match)} / {n0}** | {'○' if r.eligible else '×'} "
                         f"| {r.seconds} |")
        mm = pd.read_csv(OUT / f"mismatch_{cls}.csv")
        lines += ["", f"代表(事前の規則): バランス = `{reps[cls]['bal']}`、不均衡 = "
                      f"`{reps[cls]['semi']}`。",
                  "", "**一致しなかった行の性質**(代表の 2 設定):", "",
                  "| 設定 | 行数 | C̃ の差(argmax − 最近傍)中央値 | 払込資本比率 最近傍 / argmax"
                  "(中央値) |",
                  "|---|---:|---:|---|"]
        for k in (reps[cls]["bal"], reps[cls]["semi"]):
            m = mm[mm["setting"] == k]
            if m.empty:
                lines.append(f"| {k} | 0 | — | — |")
                continue
            lines.append(f"| {k} | {len(m)} | {m['cost_diff'].median():.4f} "
                         f"| {m['paid_nn'].median():.3f} / {m['paid_argmax'].median():.3f} |")
        m = mm[mm["setting"] == reps[cls]["semi"]]
        if 0 < len(m) <= 30:
            lines += ["", f"不均衡 OT の代表で一致しなかった企業({len(m)} 行):", "",
                      "| 始点 | 最近傍の対応先 | argmax の対応先 | C̃ の差 | 払込資本比率 最近傍"
                      " / argmax |",
                      "|---|---|---|---:|---|"]
            for r in m.itertuples():
                lines.append(f"| {r.source} | {r.nn_target} | {r.argmax_target} |"
                             f" {r.cost_diff:.4f} "
                             f"| {r.paid_nn:.3f} / {r.paid_argmax:.3f} |")
        lines.append("")
    return lines


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    e51._M = DATA / "metrics"
    e51._GEN = DATA / "all"
    excl()
    reps = json.loads((OUT / "representatives.json").read_text(encoding="utf-8"))
    (OUT / "plans.md").write_text("\n".join(section_a(reps)) + "\n", encoding="utf-8")
    ctx = {c: e51.Ctx(c, names) for c in ("c2", "c1")}
    rng = np.random.default_rng(0)
    lines = ["# ステージ64 B: τ の作り方ごとの生成", "",
             "第 1 項: 払込資本 × 10、制約に現金の下限、移動コストは非対称 L1。",
             "λ̃*_0 = 違反 0 の中で L 最小(τ ごと)。差は設定 1(最近傍、その λ̃*_0)に対する企業"
             "単位のペア比較。"
             "総資産は年率(生成 ÷ 5)の中央値。proximity は L2(中央値)。", ""]
    summary: dict[str, Any] = {}
    for cls in ("c2", "c1"):
        r, se, n_real = real_targets(cls, names)
        w = {m: 1.0 / se[m] for m in CAL}
        star: dict[str, float | None] = {}
        firms: dict[str, pd.DataFrame] = {}
        lam_rows = []
        for t in TAUS:
            best: tuple[float | None, float] = (None, float("inf"))
            for lt in LAMS:
                s = f"{t}__l1a_l{lt:g}"
                f = gen_firms(cls, s, names, members)
                f = f[~f.index.isin(e51.EXCL[cls])]
                v = ctx[cls].viol[ctx[cls].viol["setting"] == s]
                nviol = int(v["docID"].nunique())
                g = {"paid_dec": float(f["paid_dec"].mean()), "re_up": float(f["re_up"].mean()),
                     "d_assets": float(f["d_assets"].median())}
                lv = loss(g, r, w)
                lam_rows.append({"cls": cls, "tau": t, "lam": lt, "viol": nviol, "loss": lv})
                if nviol == 0 and lv < best[1]:
                    best = (lt, lv)
            star[t] = best[0]
            if best[0] is not None:
                firms[t] = gen_firms(cls, f"{t}__l1a_l{best[0]:g}", names, members)
        summary[cls] = {"star": star}
        lines += [f"## {CLS_N[cls]}(実データ: 払込資本↓ {r['paid_dec']:.0%}、RE↑ {r['re_up']:.0%}、"
                  f"変更科目数 {n_real:.0f})", "",
                  "| τ | λ̃*_0 | L | 違反(社) | 純資産の改善 | 解消率 | 払込資本↓ | RE↑ | 変更"
                  "科目数 "
                  "| 順位相関 | 総資産(年率) | proximity |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        lr = pd.DataFrame(lam_rows)
        for t, lab in TAUS.items():
            if star[t] is None:
                lines.append(f"| {lab} | なし | — | — | — | — | — | — | — | — | — | — |")
                continue
            s = f"{t}__l1a_l{star[t]:g}"
            f = firms[t][~firms[t].index.isin(e51.EXCL[cls])]
            cr = ctx[cls].comp[ctx[cls].comp["version"] == s]
            sel = (lr["cls"] == cls) & (lr["tau"] == t) & (lr["lam"] == star[t])
            lv = float(lr[sel]["loss"].iloc[0])
            res = cr.loc[cr["insolvent_start"].astype(bool), "insolvency_resolved"].mean()
            lines.append(f"| {lab} | {star[t]:g} | {lv:.2f} | 0 "
                         f"| {cr['net_asset_improvement'].median():.3f} | {res:.2f} "
                         f"| {f['paid_dec'].mean():.0%} | {f['re_up'].mean():.0%} "
                         f"| {f['n_changed'].median():.0f} "
                         f"| {corr(ctx[cls].wide[s], ctx[cls].ref):+.3f} |"
                         f" {f['d_assets'].median():+.4f} "
                         f"| {cr['proximity_l2'].median():.3f} |")
        head = " | ".join(f"{x:g}" for x in LAMS)
        lines += ["", "**違反企業数(λ̃ ごと)**", "", f"| τ | {head} |",
                  "|---|" + "---:|" * len(LAMS)]
        for t, lab in TAUS.items():
            v = lr[(lr["cls"] == cls) & (lr["tau"] == t)].sort_values("lam")["viol"].tolist()
            lines.append(f"| {lab} | " + " | ".join(str(x) for x in v) + " |")
        if star["t1_nn"] is not None:
            base = f"t1_nn__l1a_l{star['t1_nn']:g}"
            fb = firms["t1_nn"][~firms["t1_nn"].index.isin(e51.EXCL[cls])]
            lines += ["", "**設定 1(最近傍)との比較**(記号 = 純資産の改善・解消率・順位相関)", "",
                      "| τ | 払込資本↓ の差 [95%] | 純資産の改善の差 | 順位相関の差 | 記号 |",
                      "|---|---|---|---|---|"]
            for t in ("t2_bal", "t3_semi", "t4_bary"):
                if star[t] is None:
                    continue
                s = f"{t}__l1a_l{star[t]:g}"
                fa = firms[t][~firms[t].index.isin(e51.EXCL[cls])]
                dd = (fa["paid_dec"] - fb["paid_dec"].reindex(fa.index)).to_numpy()
                bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                c = ctx[cls].compare(s, base, rng)
                sym = " ".join({1: "+", -1: "−", 0: "0"}[int(c[k][3])]
                               for k in ("純資産の改善", "解消率", "順位相関"))
                lines.append(f"| {TAUS[t]} | {dd.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] "
                             f"| {e51.fc(c['純資産の改善'])} | {e51.fc(c['順位相関'])} | {sym} |")
        lines.append("")
    # B.4: 目標点
    lines += ["# B.4: 目標点の性質", "",
              "| class | τ | 識別 AUC(LightGBM / LogReg) | 異なる τ | τ の払込資本比率の中央値 |",
              "|---|---|---|---:|---:|"]
    for cls in ("c2", "c1"):
        auc = pd.read_csv(OUT / f"auc_{cls}.csv").set_index("key")
        z = np.load(DATA / f"targets_{cls}.npz", allow_pickle=True)
        keys = {"t1_nn": "nn", "t2_bal": f"{reps[cls]['bal']}__argmax",
                "t3_semi": f"{reps[cls]['semi']}__argmax", "t4_bary": f"{reps[cls]['bal']}__bary"}
        x, lab, _d, _k = __import__("tilde_stage51").load_tilde(names)
        xt = x[lab == 0]
        pi = [names.index("jppfs_cor_CapitalStock"), names.index("jppfs_cor_CapitalSurplus")]
        for t, k in keys.items():
            v = z[k]
            tau = xt[v.astype(int)] if v.ndim == 1 else np.asarray(v, dtype=float)
            a = auc.loc[k]
            lines.append(f"| {CLS_N[cls]} | {TAUS[t]} | {a['auc_lgbm']:.3f} /"
                         f" {a['auc_logreg']:.3f} "
                         f"| {int(a['n_distinct'])} | {np.median(tau[:, pi].sum(axis=1)):.3f} |")
    lines.append("")
    (OUT / "generation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print((OUT / "plans.md").read_text(encoding="utf-8")[:3000])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
