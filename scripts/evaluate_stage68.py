#!/usr/bin/env python
"""ステージ68: A(ステージ65 の未訂正の数値)と B(遷移ベースの目標)の評価。

規則は ``results/stage68/criteria.md``(B の生成の前に記録)。事前に ``metrics_stage39.py`` を
A は ``--gen-dir data/interim/stage68/all_a --out data/interim/stage68/metrics_a``、B は
``--gen-dir data/interim/stage68/all --out data/interim/stage68/metrics`` で実行する。

    uv run python scripts/evaluate_stage68.py --part a
    uv run python scripts/evaluate_stage68.py --part b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import numpy as np
import pandas as pd
from compare_stage39 import corr
from evaluate_stage61 import CAL, gen_firms, loss, real_targets
from evaluate_stage65 import bs_table, real_firm_values
from evaluate_stage67 import extra_cols, real_ratio
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import CHECK_NAMES, EXTRA_CHECK_NAMES, subtotal_members
from fsgen.viz.balance_sheet import load_labels

OUT = Path("results/stage68")
DATA = Path("data/interim/stage68")
LAMS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
CHECKS = (*CHECK_NAMES, *EXTRA_CHECK_NAMES)
# A: 設定 → (表示名, ステージ65 の値(リークあり))
A_SETTINGS = {
    "m2_l1only__l1": "2 第 1 層のみ(λ̃ = 1、提案の値)",
    "m5_prop__l1": "5 提案(λ̃ = 1)",
    "m6_balarg__l30": "6 バランス OT の argmax(λ̃ = 30、class 1 の λ̃*_0)",
    "m6_balarg__l1": "6 バランス OT の argmax(λ̃ = 1、指示文どおり)",
    "m7_bary__l3": "7 バランス OT の重心(λ̃ = 3、class 1 の λ̃*_0)",
    "m7_bary__l1": "7 バランス OT の重心(λ̃ = 1、指示文どおり)"}
B_SETTINGS = {"t0m10": "1 T0・m = 10(現行)", "t0m1": "2 T0・m = 1",
              "t1m1": "3 T1(再建終点)・m = 1", "t2m1": "4 T2(遷移加算)・m = 1",
              "t2m1re": "5 T2・m = 1・c_RE > 0"}


def setup(metrics: Path, gen: Path, names: tuple[str, ...]) -> None:
    e51._M = metrics
    e51._GEN = gen
    codes = set(pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})["edinetCode"])
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    x, lab, docs, _ = load_tilde(names)
    drop = [d for d in docs if code_of.get(d, "") in codes]
    for cls in ("c2", "c1"):
        e51.EXCL[cls] = tuple(dict.fromkeys((*e51.EXCL[cls], *drop)))


def part_a(names: tuple[str, ...], members: dict[str, np.ndarray]) -> None:
    setup(DATA / "metrics_a", DATA / "all_a", names)
    ctx = e51.Ctx("c2", names)
    rng = np.random.default_rng(0)
    lines = ["# ステージ68 A: ステージ65 の未訂正の数値(class 2、class 1 で較正した λ)", "",
             "λ は手法ごとに class 1 で較正した λ̃*_0 × s_c1 を class 2 に当てはめた(class 2 の"
             "グリッドでは λ̃ × 1.09179)。手法 6・7 は指示文どおりの λ̃ = 1 も並べる。"
             "評価対象は class 2 の 55 社。", "",
             "| 手法 | 違反(社) | 違反の項目(行) | 純資産の改善 | 解消率 | 払込資本↓ | RE↑ "
             "| 変更科目数 | 順位相関 |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for s, lab in A_SETTINGS.items():
        f = gen_firms("c2", s, names, members)
        f = f[~f.index.isin(e51.EXCL["c2"])]
        cr = ctx.comp[ctx.comp["version"] == s]
        v = ctx.viol[ctx.viol["setting"] == s]
        items = {c.split("_")[0]: int(v[v["check"] == c]["rows"].sum()) for c in CHECKS}
        it = "、".join(f"{k}: {n}" for k, n in items.items() if n) or "—"
        res = cr.loc[cr["insolvent_start"].astype(bool), "insolvency_resolved"].mean()
        lines.append(f"| {lab} | {v['docID'].nunique()} | {it} "
                     f"| {cr['net_asset_improvement'].median():.3f} | {res:.2f} "
                     f"| {f['paid_dec'].mean():.0%} | {f['re_up'].mean():.0%} "
                     f"| {f['n_changed'].median():.0f} | {corr(ctx.wide[s], ctx.ref):+.3f} |")
    lines += ["", "違反の項目は 15 項目のチェックの番号と、違反した行数(55 社 × 3 seed)。", "",
              "## 提案(5)との企業単位のペア比較", "",
              "差 = 他の手法 − 提案。記号 = 純資産の改善・解消率・順位相関。", "",
              "| 手法 | 純資産の改善の差 [95%] | 順位相関の差 [95%] | 記号 |", "|---|---|---|---|"]
    for s, lab in A_SETTINGS.items():
        if s == "m5_prop__l1":
            continue
        c = ctx.compare(s, "m5_prop__l1", rng)
        sym = " ".join({1: "+", -1: "−", 0: "0"}[int(c[k][3])]
                       for k in ("純資産の改善", "解消率", "順位相関"))
        lines.append(f"| {lab} | {e51.fc(c['純資産の改善'])} | {e51.fc(c['順位相関'])} | {sym} |")
    (OUT / "correction_a.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def boot_mean(d: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def part_b(names: tuple[str, ...], members: dict[str, np.ndarray]) -> None:  # noqa: PLR0915
    setup(DATA / "metrics", DATA / "gen", names)
    ctx = {c: e51.Ctx(c, names) for c in ("c2", "c1")}
    rng = np.random.default_rng(0)
    r1, se1, _ = real_targets("c1", names)
    w1 = {m: 1.0 / se1[m] for m in CAL}
    star: dict[str, float | None] = {}
    cal = ["# ステージ68 B: λ の較正(class 1 の 128 社だけ)", "",
           "| 設定 | " + " | ".join(f"{v:g}" for v in LAMS) + " | λ̃*_0 |",
           "|---|" + "---:|" * (len(LAMS) + 1)]
    for st, lab in B_SETTINGS.items():
        cells: list[str] = []
        best: tuple[float | None, float] = (None, np.inf)
        for lt in LAMS:
            s = f"{st}__l{lt:g}"
            f = gen_firms("c1", s, names, members)
            f = f[~f.index.isin(e51.EXCL["c1"])]
            nv = ctx["c1"].viol[ctx["c1"].viol["setting"] == s]["docID"].nunique()
            g = {"paid_dec": float(f["paid_dec"].mean()), "re_up": float(f["re_up"].mean()),
                 "d_assets": float(f["d_assets"].median())}
            lv = loss(g, r1, w1)
            cells.append(f"{lv:.2f}({nv})")
            if nv == 0 and lv < best[1]:
                best = (lt, lv)
        star[st] = best[0]
        cal.append(f"| {lab} | " + " | ".join(cells)
                   + f" | **{'なし' if best[0] is None else f'{best[0]:g}'}** |")
    cal += ["", "セルは L(違反した企業数)。class 2 には同じ λ(class 1 の s × λ̃*_0)を当てはめる。",
            ""]
    lines = ["# ステージ68 B.4: 目標の定義の比較", ""]
    items = ["# 項目別の違反(企業数)", ""]
    firms: dict[tuple[str, str], pd.DataFrame] = {}
    verdict: dict[str, dict[str, Any]] = {st: {} for st in ("t1m1", "t2m1", "t2m1re")}
    for cls in ("c2", "c1"):
        r, se, n_real = real_targets(cls, names)
        w = {m: 1.0 / se[m] for m in CAL}
        lines += [f"## {cls}(実データ: ΔRE/ΔNA {np.median(real_ratio(cls, names, members)):.2f}、"
                  f"払込資本↓ {r['paid_dec']:.0%}、変更科目数 {n_real:.0f}、総資産(年率) "
                  f"{r['d_assets']:+.4f})", "",
                  "| 設定 | λ̃ | 違反 | ΔRE/ΔNA | < 0.99 | γ が拘束 | 払込資本↑ | うち γ が拘束 "
                  "| 払込資本↓ | RE↑ | 変更科目数 | 総資産(年率) | 純資産の改善 | 解消率 "
                  "| 順位相関 | L |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        items += [f"## {cls}", "", "| 設定 | " + " | ".join(c.split("_")[0] for c in CHECKS) + " |",
                  "|---|" + "---:|" * len(CHECKS)]
        for st, lab in B_SETTINGS.items():
            lt_s = star[st]
            if lt_s is None:
                lines.append(f"| {lab} | — | 較正できない |" + " — |" * 13)
                continue
            s = f"{st}__l{lt_s:g}"
            f = gen_firms(cls, s, names, members).join(extra_cols(cls, s, names, members))
            f = f[~f.index.isin(e51.EXCL[cls])]
            firms[(cls, st)] = f
            cr = ctx[cls].comp[ctx[cls].comp["version"] == s]
            v = ctx[cls].viol[ctx[cls].viol["setting"] == s].drop_duplicates(["docID", "check"])
            g = {"paid_dec": float(f["paid_dec"].mean()), "re_up": float(f["re_up"].mean()),
                 "d_assets": float(f["d_assets"].median())}
            res = cr.loc[cr["insolvent_start"].astype(bool), "insolvency_resolved"].mean()
            nviol = v["docID"].nunique()
            if st in verdict:
                verdict[st][f"viol_{cls}"] = int(nviol)
            lines.append(f"| {lab} | {lt_s:g} | {nviol} | {f['re_share'].median():.2f} "
                         f"| {f['re_lt1'].mean():.0%} | {int(f['gamma_active'].sum())} "
                         f"| {f['paid_inc'].mean():.0%} "
                         f"| {int((f['paid_inc'] * f['gamma_active']).sum())} / "
                         f"{int(f['paid_inc'].sum())} | {g['paid_dec']:.0%} | {g['re_up']:.0%} "
                         f"| {f['n_changed'].median():.0f} | {g['d_assets']:+.4f} "
                         f"| {cr['net_asset_improvement'].median():.3f} | {res:.2f} "
                         f"| {corr(ctx[cls].wide[s], ctx[cls].ref):+.3f} | {loss(g, r, w):.2f} |")
            items.append(f"| {lab} | " + " | ".join(str(int((v["check"] == c).sum()))
                                                    for c in CHECKS) + " |")
        items.append("")
        lines += ["", "**判定の比較**(差 = 設定 − 基準、企業単位のペア、2,000 回の"
                  "ブートストラップ。"
                  "< 0.99 は両方で ΔNA > 0 の企業。記号 = 純資産の改善・解消率・順位相関)", "",
                  "| 設定 | 基準 | 払込資本↑ の差 [95%] | ΔRE/ΔNA < 0.99 の差 [95%] "
                  "| 純資産の改善の差 [95%] | 順位相関の差 [95%] | 記号 |",
                  "|---|---|---|---|---|---|---|"]
        for st in ("t0m1", "t1m1", "t2m1", "t2m1re"):
            for base in ("t0m10", "t0m1"):
                if st == base or (cls, st) not in firms or (cls, base) not in firms:
                    continue
                fa = firms[(cls, st)]
                fb = firms[(cls, base)].reindex(fa.index)
                pi = boot_mean((fa["paid_inc"] - fb["paid_inc"]).to_numpy(), rng)
                both = fa["re_lt1"].notna() & fb["re_lt1"].notna()
                rl = boot_mean((fa.loc[both, "re_lt1"] - fb.loc[both, "re_lt1"]).to_numpy(), rng)
                c = ctx[cls].compare(f"{st}__l{star[st]:g}", f"{base}__l{star[base]:g}", rng)
                signs = [int(c[k][3]) for k in ("純資産の改善", "解消率", "順位相関")]
                sym = " ".join({1: "+", -1: "−", 0: "0"}[v_] for v_ in signs)
                lines.append(f"| {B_SETTINGS[st]} | {B_SETTINGS[base]} "
                             f"| {pi[0]:+.2f} [{pi[1]:+.2f}, {pi[2]:+.2f}] "
                             f"| {rl[0]:+.2f} [{rl[1]:+.2f}, {rl[2]:+.2f}] "
                             f"| {e51.fc(c['純資産の改善'])} | {e51.fc(c['順位相関'])} | {sym} |")
                if base == "t0m10" and st in verdict:
                    verdict[st][cls] = {"paid_inc": pi, "re_lt1": rl, "signs": signs,
                                        "j1": pi[1] > 0 and rl[1] > 0,
                                        "j2": all(v_ != -1 for v_ in signs)}
        lines.append("")
    # 判定(criteria §4)
    jl = ["# ステージ68 B.5: 判定(criteria.md §4、設定 1 との比較)", "",
          "| 設定 | 判定 1(払込資本↑ と < 0.99 が両 class で有意に増加) "
          "| 判定 2(主指標が両 class で"
          "有意に劣らない) | 判定 3(違反 0) | class 1 の L | 採用候補 |",
          "|---|---|---|---|---:|---|"]
    cands = []
    r1_, se1_, _ = real_targets("c1", names)
    for st, vd in verdict.items():
        if "c2" not in vd or "c1" not in vd:
            jl.append(f"| {B_SETTINGS[st]} | 較正できない | — | — | — | 否 |")
            continue
        j1 = vd["c2"]["j1"] and vd["c1"]["j1"]
        j2 = vd["c2"]["j2"] and vd["c1"]["j2"]
        j3 = vd.get("viol_c2", 1) == 0 and vd.get("viol_c1", 1) == 0
        f1 = firms[("c1", st)]
        lv = loss({"paid_dec": float(f1["paid_dec"].mean()), "re_up": float(f1["re_up"].mean()),
                   "d_assets": float(f1["d_assets"].median())}, r1_,
                  {m: 1.0 / se1_[m] for m in CAL})
        ok = j1 and j2 and j3
        if ok:
            cands.append((lv, st))

        def yn(b: bool) -> str:
            return "成立" if b else "**不成立**"
        jl.append(f"| {B_SETTINGS[st]} | {yn(j1)} | {yn(j2)} | {yn(j3)} | {lv:.2f} "
                  f"| {'候補' if ok else '否'} |")
    adopted = min(cands)[1] if cands else None
    jl += ["", f"**採用: {B_SETTINGS[adopted] if adopted else 'なし(ステージ65 の訂正版で確定)'}**",
           ""]
    # 実データとの比較(参考、循環の注意あり)
    real = ["# 実データとの比較(参考。T1・T2 は D_rec から作るので一部循環する)", "",
            "差 = 設定 − 実データ。95% 区間は、生成と実データを別々に復元抽出(2,000 回)。", ""]
    paid = pd.read_csv("results/stage56/paidin.csv", dtype={"edinetCode": str})
    for cls in ("c2", "c1"):
        rf = real_firm_values(cls, names, members)
        pg = "再建群(class 2 → 回復)" if cls == "c2" else "再建群(class 1 → class 0)"
        pp = paid[paid["group"] == pg]["annual_d_paid"]
        rr = real_ratio(cls, names, members)
        real += [f"## {cls}(実データ {len(rf)} 社)", "",
                 "| 項目 | " + " | ".join(B_SETTINGS[st] for st in ("t0m10", "t1m1", "t2m1"))
                 + " | 実データ |", "|---|---:|---:|---:|---:|"]
        rows_: dict[str, list[str]] = {}
        for st in ("t0m10", "t1m1", "t2m1"):
            if (cls, st) not in firms:
                continue
            fa = firms[(cls, st)].copy()
            cv = ctx[cls].comp
            cr = cv[cv["version"] == f"{st}__l{star[st]:g}"].set_index("docID")
            fa["nai_annual"] = cr["net_asset_improvement"].reindex(fa.index) / 5.0
            sh = fa["re_share"].dropna().to_numpy()
            spec_ = (("ΔRE/ΔNA < 0.99 の割合", (sh < 0.99).astype(float),
                      (rr < 0.99).astype(float), np.mean),
                     ("払込資本を増やした割合", fa["paid_inc"].to_numpy(),
                      (pp > 0.001).astype(float).to_numpy(), np.mean),
                     ("払込資本を減らした割合", fa["paid_dec"].to_numpy(),
                      rf["paid_dec"].dropna().to_numpy(), np.mean),
                     ("変更された科目数(中央値)", fa["n_changed"].to_numpy(),
                      rf["n_changed"].dropna().to_numpy(), np.median),
                     ("純資産の改善(年率、中央値)", fa["nai_annual"].dropna().to_numpy(),
                      rf["nai_annual"].dropna().to_numpy(), np.median),
                     ("総資産変化(年率、中央値)", fa["d_assets"].to_numpy(),
                      rf["d_assets"].dropna().to_numpy(), np.median))
            for lab, a_, b_, fn in spec_:
                ds = [fn(a_[rng.integers(0, len(a_), len(a_))])
                      - fn(b_[rng.integers(0, len(b_), len(b_))]) for _ in range(2000)]
                lo, hi = np.percentile(ds, [2.5, 97.5])
                sig = "**" if lo > 0 or hi < 0 else ""
                rows_.setdefault(lab, [f"{fn(b_):.3f}"]).insert(
                    -1, f"{sig}{fn(a_):.3f}{sig}")
        for lab, cells in rows_.items():
            real.append(f"| {lab} | " + " | ".join(cells) + " |")
        real += ["", "(太字は 95% 区間で実データと有意に違う)", ""]
    (OUT / "calibration.md").write_text("\n".join(cal) + "\n", encoding="utf-8")
    (OUT / "comparison.md").write_text("\n".join([*lines, *items]) + "\n", encoding="utf-8")
    (OUT / "judgement.md").write_text("\n".join(jl) + "\n", encoding="utf-8")
    (OUT / "realdata.md").write_text("\n".join(real) + "\n", encoding="utf-8")
    (OUT / "bs.md").write_text("\n".join(bs_b(names, members, star)) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({"star": star, "adopted": adopted}, indent=1),
                                      encoding="utf-8")
    for fn_ in ("calibration.md", "comparison.md", "judgement.md", "realdata.md"):
        print((OUT / fn_).read_text(encoding="utf-8"))


def bs_b(names: tuple[str, ...], members: dict[str, np.ndarray],
         star: dict[str, float | None]) -> list[str]:
    """BS 表(class 2。ステージ65・67 と同じ選び方の企業で、始点・設定 1・3・4 と T2 の目標)。"""
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    keys = {st: f"{st}__l{star[st]:g}" for st in ("t0m10", "t1m1", "t2m1") if star[st] is not None}
    gens = {st: (lambda g: g[g["seed"] == 0].set_index("docID"))(e51.load_gen("c2", k))
            for st, k in keys.items()}
    g1 = gens["t0m10"]
    x0c, x1c = [f"x0_{n}" for n in names], [f"x1_{n}" for n in names]
    na0 = g1[x0c].to_numpy() @ members["jppfs_cor_NetAssets"]
    docs = np.array(g1.index)
    r_ = np.random.default_rng(0)
    ins = docs[na0 < 0]
    pick_ins = list(r_.choice(ins, size=min(2, len(ins)), replace=False))
    rest = [d for d in docs if d not in pick_ins]
    picks = pick_ins + list(r_.choice(rest, size=3, replace=False))
    z = np.load(DATA / "targets_c2.npz", allow_pickle=True)
    t2 = dict(zip([str(v) for v in z["source_docs"]], z["t2"], strict=True))
    k2 = dict(zip([str(v) for v in z["source_docs"]], z["k_t2"], strict=True))
    labels = load_labels(Path("conf/accounts/labels_ja.yaml"))
    out = ["# ステージ68 B: BS 表(class 2、百万円、生成の終点は 5 年後とみなす)", "",
           "企業はステージ65・67 と同じ規則(乱数 seed 0、始点で債務超過から 2 社、残りから 3 社)。"
           "列は始点、設定 1・3・4 の終点(λ は class 1 で較正)、T2 の目標(始点 + 対応した"
           "再建企業の実際の変化、始点の総資産で戻した値)。", ""]
    for d in picks:
        r0 = g1.loc[d]
        a0 = float(r0["assets_start"])
        cols = [("始点", r0[x0c].to_numpy(float) * a0),
                *((B_SETTINGS[st].split("(")[0].split("・")[0], gens[st].loc[d][x1c]
                   .to_numpy(float) * a0) for st in keys),
                ("T2 の目標", t2[str(d)] * a0)]
        tag = "(始点で債務超過)" if d in pick_ins else ""
        out += bs_table(spec, names, labels,
                        f"{r0['filerName']}({d}){tag}、T2 の対応先 {k2[str(d)]}", cols)
    return out


def part_grid(names: tuple[str, ...], members: dict[str, np.ndarray]) -> None:
    """結果を見てから追加した参考(判定には使わない)。

    設定 1〜4 の全 λ̃ の指標。違反 > 0 の点も含む。
    """
    setup(DATA / "metrics", DATA / "gen", names)
    ctx = {c: e51.Ctx(c, names) for c in ("c2", "c1")}
    lines = ["# 参考: 全 λ̃ の指標(違反のある点も含む。判定には使わない)", "",
             "違反のある点は 15 項目の必須条件を満たさないので、採用の対象にならない。"
             "目標の向きが QP の出力にどう出るかを見るためだけに示す。", ""]
    for cls in ("c2", "c1"):
        lines += [f"## {cls}", "", "| 設定 | λ̃ | 違反(社) | ΔRE/ΔNA < 0.99 | 払込資本↑ | 払込資本↓ "
                  "| 変更科目数 | 総資産(年率) | 純資産の改善 | 順位相関 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for st in ("t0m10", "t0m1", "t1m1", "t2m1"):
            for lt in LAMS:
                s = f"{st}__l{lt:g}"
                f = gen_firms(cls, s, names, members).join(extra_cols(cls, s, names, members))
                f = f[~f.index.isin(e51.EXCL[cls])]
                cr = ctx[cls].comp[ctx[cls].comp["version"] == s]
                nv = ctx[cls].viol[ctx[cls].viol["setting"] == s]["docID"].nunique()
                lines.append(f"| {B_SETTINGS[st]} | {lt:g} | {nv} | {f['re_lt1'].mean():.0%} "
                             f"| {f['paid_inc'].mean():.0%} | {f['paid_dec'].mean():.0%} "
                             f"| {f['n_changed'].median():.0f} | {f['d_assets'].median():+.4f} "
                             f"| {cr['net_asset_improvement'].median():.3f} "
                             f"| {corr(ctx[cls].wide[s], ctx[cls].ref):+.3f} |")
        lines.append("")
    (OUT / "grid.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", choices=("a", "b", "grid"), required=True)
    args = ap.parse_args()
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.part == "a":
        part_a(names, members)
    elif args.part == "b":
        part_b(names, members)
    else:
        part_grid(names, members)


if __name__ == "__main__":
    main()
