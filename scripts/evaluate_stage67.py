#!/usr/bin/env python
"""ステージ67 C: 利益剰余金の移動コストの効果。

λ は class 1 だけで較正し、class 2 には当てはめるだけ。

規則は ``results/stage67/criteria.md``。生成は ``data/interim/stage67/all/<cls>/<設定>__l<λ̃>``
(設定 1 はステージ66 の s1)。事前に ``metrics_stage39.py --gen-dir data/interim/stage67/all
--out data/interim/stage67/metrics`` を実行する。

    uv run python scripts/evaluate_stage67.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import numpy as np
import pandas as pd
from changes_stage57 import deltas as nominal_deltas
from compare_stage39 import corr
from evaluate_stage61 import CAL, gen_firms, loss, real_targets
from evaluate_stage65 import bs_table, real_firm_values
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import CHECK_NAMES, EXTRA_CHECK_NAMES, subtotal_members
from fsgen.viz.balance_sheet import load_labels

OUT = Path("results/stage67")
DATA = Path("data/interim/stage67")
LAMS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
SETTINGS = {"s1": "1 現行(ステージ65、class 1 で較正)", "s1r": "1r 設定 1 の再現(RecourseQP)",
            "s2": "**2 + 利益剰余金にコスト**", "s3": "3 2 + 階層分離",
            "s4": "4 2 + Elastic-Net", "s5": "5 2 + 階層分離 + Elastic-Net"}
STEPS = (("s1r", "s1", "再現の確認"), ("s2", "s1", "利益剰余金にコスト"), ("s3", "s2", "階層分離"),
         ("s4", "s2", "Elastic-Net"), ("s5", "s3", "Elastic-Net(階層分離あり)"))
CHECKS = (*CHECK_NAMES, *EXTRA_CHECK_NAMES)
GAMMA = 0.7565192502545388
_CS, _CSUR, _RE = ("jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus",
                   "jppfs_cor_RetainedEarnings")


def excl() -> None:
    codes = set(pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})["edinetCode"])
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    for cls in ("c2", "c1"):
        g = pd.read_parquet(sorted((e51._GEN / cls / "s2__l1").glob("*__seed0.parquet"))[0])
        drop = [d for d in g["docID"].unique() if code_of.get(d, "") in codes]
        e51.EXCL[cls] = tuple(dict.fromkeys((*e51.EXCL[cls], *drop)))


def extra_cols(cls: str, s: str, names: tuple[str, ...], members: dict[str, np.ndarray]
               ) -> pd.DataFrame:
    g = e51.load_gen(cls, s)
    g = g[g["seed"] == 0].set_index("docID")
    x0 = g[[f"x0_{n}" for n in names]].to_numpy()
    x1 = g[[f"x1_{n}" for n in names]].to_numpy()
    a0 = x0 @ members["jppfs_cor_Assets"]
    d = x1 - x0
    dna = d @ members["jppfs_cor_NetAssets"] / a0
    dre = d[:, names.index(_RE)] / a0
    dp = (d[:, names.index(_CS)] + d[:, names.index(_CSUR)]) / a0 / 5.0
    share = ratio(dre, dna)
    return pd.DataFrame({"paid_inc": (dp > 0.001).astype(float), "re_share": share,
                         "re_lt1": np.where(np.isnan(share), np.nan, (share < 0.99).astype(float)),
                         "gamma_active": (dre >= GAMMA - 1e-6).astype(float)}, index=g.index)


def ratio(dre: np.ndarray, dna: np.ndarray) -> np.ndarray:
    """ΔRE / ΔNA(ΔNA > 1e-6 の企業だけ、他は NaN)。"""
    pos = dna > 1e-6
    out: np.ndarray = np.where(pos, dre / np.where(pos, dna, 1.0), np.nan)
    return out


def real_ratio(cls: str, names: tuple[str, ...], members: dict[str, np.ndarray]) -> np.ndarray:
    """実データの企業ごとの ΔRE / ΔNA(名目、2022 → 2024。ΔNA > 0 の企業)。"""
    grp = "class 2 → 回復" if cls == "c2" else "class 1 → class 0"
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    dn = nominal_deltas(pairs[pairs["group"] == grp]["edinetCode"].tolist(), names).to_numpy()
    r = ratio(dn[:, names.index(_RE)], dn @ members["jppfs_cor_NetAssets"])
    out: np.ndarray = r[~np.isnan(r)]
    return out


def target_ratio(cls: str, names: tuple[str, ...], members: dict[str, np.ndarray]
                 ) -> tuple[np.ndarray, np.ndarray]:
    """対応先 τ(ステージ64 の argmax)に完全に到達した場合の ΔRE / ΔNA と、払込資本の差 τ − x
    (診断用。結果を見てから追加した)。"""
    x, lab, docs, _ = load_tilde(names)
    z = np.load(f"data/interim/stage64/targets_{cls}.npz", allow_pickle=True)
    pos = {d: i for i, d in enumerate(docs)}
    xs = x[[pos[str(d)] for d in z["source_docs"]]]
    d = x[lab == 0][z["semi_t0.01_e0.001__argmax"].astype(int)] - xs
    keep = ~np.isin([str(v) for v in z["source_docs"]], e51.EXCL[cls])
    r = ratio(d[keep, names.index(_RE)], (d @ members["jppfs_cor_NetAssets"])[keep])
    dpaid = (d[:, names.index(_CS)] + d[:, names.index(_CSUR)])[keep]
    return r[~np.isnan(r)], dpaid


def bs_lines(spec: Any, names: tuple[str, ...], members: dict[str, np.ndarray],
             star: dict[str, float | None]) -> list[str]:
    """C: BS 表(class 2。ステージ65 と同じ選び方の企業で、始点・設定 1〜3・対応先 τ)。"""
    keys = {st: f"{st}__l{star[st]:g}" for st in ("s1", "s2", "s3")}
    gens = {st: (lambda g: g[g["seed"] == 0].set_index("docID"))(e51.load_gen("c2", k))
            for st, k in keys.items()}
    g1 = gens["s1"]
    x0c, x1c = [f"x0_{n}" for n in names], [f"x1_{n}" for n in names]
    na0 = g1[x0c].to_numpy() @ members["jppfs_cor_NetAssets"]
    docs = np.array(g1.index)
    r_ = np.random.default_rng(0)
    ins = docs[na0 < 0]
    pick_ins = list(r_.choice(ins, size=min(2, len(ins)), replace=False))
    rest = [d for d in docs if d not in pick_ins]
    picks = pick_ins + list(r_.choice(rest, size=3, replace=False))
    x, lab, tdocs, _ = load_tilde(names)
    z = np.load("data/interim/stage64/targets_c2.npz", allow_pickle=True)
    tau = dict(zip([str(v) for v in z["source_docs"]],
                   x[lab == 0][z["semi_t0.01_e0.001__argmax"].astype(int)], strict=True))
    labels = load_labels(Path("conf/accounts/labels_ja.yaml"))
    out = ["# ステージ67 C: BS 表(class 2、百万円、生成の終点は 5 年後とみなす)", "",
           "企業はステージ65 と同じ規則(乱数 seed 0、始点で債務超過から 2 社、残りから 3 社)。"
           "列は始点、設定 1・2・3 の終点(λ は class 1 で較正)、対応先 τ(総資産比を始点の"
           "総資産で戻した値)。", ""]
    for d in picks:
        r0 = g1.loc[d]
        a0 = float(r0["assets_start"])
        cols = [("始点", r0[x0c].to_numpy(float) * a0),
                *((f"設定 {st[1]}", gens[st].loc[d][x1c].to_numpy(float) * a0)
                  for st in ("s1", "s2", "s3")),
                ("対応先 τ", tau[str(d)] * a0)]
        tag = "(始点で債務超過)" if d in pick_ins else ""
        out += bs_table(spec, names, labels, f"{r0['filerName']}({d}){tag}", cols)
    return out


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    e51._M = DATA / "metrics"
    e51._GEN = DATA / "all"
    excl()
    ctx = {c: e51.Ctx(c, names) for c in ("c2", "c1")}
    rng = np.random.default_rng(0)
    r1, se1, _ = real_targets("c1", names)
    w1 = {m: 1.0 / se1[m] for m in CAL}
    star: dict[str, float | None] = {}
    cal = ["# ステージ67 C: λ の較正(class 1 の 128 社だけ)", "",
           "| 設定 | " + " | ".join(f"{v:g}" for v in LAMS) + " | λ̃*_0 |",
           "|---|" + "---:|" * (len(LAMS) + 1)]
    for st, lab in SETTINGS.items():
        cells: list[str] = []
        best: tuple[float | None, float] = (None, np.inf)
        for lt in LAMS:
            s = f"{st}__l{lt:g}"
            if not (e51._GEN / "c1" / s).exists():
                cells.append("—")
                continue
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
        sv = "なし" if best[0] is None else f"{best[0]:g}"
        cal.append(f"| {lab} | " + " | ".join(cells) + f" | **{sv}** |")
    cal += ["", "セルは L(違反した企業数)。class 2 には同じ λ(class 1 の s × λ̃*_0)を当てはめる。"
            "設定 1 の class 2 はステージ66 で生成した 1 点だけ。", ""]
    lines = ["# ステージ67 C.2: 利益剰余金のコストの効果", ""]
    items = ["# 項目別の違反(企業数)", ""]
    firms: dict[tuple[str, str], pd.DataFrame] = {}
    realrows = ["# C.4 提案(設定 2)と実データ", "",
                "差 = 設定 2 − 実データ。95% 区間は、生成と実データを別々に復元抽出"
                "(2,000 回)。", ""]
    for cls in ("c2", "c1"):
        r, se, n_real = real_targets(cls, names)
        w = {m: 1.0 / se[m] for m in CAL}
        paid = pd.read_csv("results/stage56/paidin.csv", dtype={"edinetCode": str})
        pg = "再建群(class 2 → 回復)" if cls == "c2" else "再建群(class 1 → class 0)"
        pp = paid[paid["group"] == pg]["annual_d_paid"]
        rf = real_firm_values(cls, names, members)
        real_re = float(np.median(real_ratio(cls, names, members)))
        lines += [f"## {cls}(実データ: ΔRE/ΔNA {real_re:.2f}、払込資本↑ {(pp > 0.001).mean():.0%}、"
                  f"↓ {r['paid_dec']:.0%}、変更科目数 {n_real:.0f})", "",
                  "| 設定 | λ̃ | 違反 | ΔRE/ΔNA | ΔRE/ΔNA < 0.99 | γ が拘束 | 払込資本↑ "
                  "| うち γ が拘束 | 払込資本↓ | RE↑ | 変更科目数 "
                  "| 純資産の改善 | 解消率 | 順位相関 | L |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        items += [f"## {cls}", "", "| 設定 | " + " | ".join(c.split("_")[0] for c in CHECKS) + " |",
                  "|---|" + "---:|" * len(CHECKS)]
        for st, lab in SETTINGS.items():
            lt_s = star[st]
            s = f"{st}__l{lt_s:g}" if lt_s is not None else ""
            if lt_s is None or not (e51._GEN / cls / s).exists():
                lines.append(f"| {lab} | {lt_s} | 生成なし |" + " — |" * 12)
                continue
            f = gen_firms(cls, s, names, members).join(extra_cols(cls, s, names, members))
            f = f[~f.index.isin(e51.EXCL[cls])]
            firms[(cls, st)] = f
            cr = ctx[cls].comp[ctx[cls].comp["version"] == s]
            v = ctx[cls].viol[ctx[cls].viol["setting"] == s].drop_duplicates(["docID", "check"])
            g = {"paid_dec": float(f["paid_dec"].mean()), "re_up": float(f["re_up"].mean()),
                 "d_assets": float(f["d_assets"].median())}
            res = cr.loc[cr["insolvent_start"].astype(bool), "insolvency_resolved"].mean()
            nviol = v["docID"].nunique()
            lines.append(f"| {lab} | {lt_s:g} | {nviol} | {f['re_share'].median():.2f} "
                         f"| {f['re_lt1'].mean():.0%} "
                         f"| {int(f['gamma_active'].sum())} | {f['paid_inc'].mean():.0%} "
                         f"| {int((f['paid_inc'] * f['gamma_active']).sum())} / "
                         f"{int(f['paid_inc'].sum())} "
                         f"| {g['paid_dec']:.0%} | {g['re_up']:.0%} |"
                         f" {f['n_changed'].median():.0f} "
                         f"| {cr['net_asset_improvement'].median():.3f} | {res:.2f} "
                         f"| {corr(ctx[cls].wide[s], ctx[cls].ref):+.3f} | {loss(g, r, w):.2f} |")
            items.append(f"| {lab} | " + " | ".join(str(int((v["check"] == c).sum()))
                                                    for c in CHECKS) + " |")
        items.append("")
        lines += ["", "**段階の効果**(差 = 後 − 前。ΔRE/ΔNA は企業単位の差の中央値 [95%]、払込"
                      "資本は"
                  "ブートストラップ、記号 = 純資産の改善・解消率・順位相関)", "",
                  "| 変更 | ΔRE/ΔNA の差 | ΔRE/ΔNA < 0.99 の差 | 払込資本↑ の差 | 変更科目数 "
                  "| 純資産の改善の差 | 記号 |",
                  "|---|---|---|---|---|---|---|"]
        for a, b, lab in STEPS:
            if (cls, a) not in firms or (cls, b) not in firms:
                continue
            fa, fb = firms[(cls, a)], firms[(cls, b)].reindex(firms[(cls, a)].index)
            dre = (fa["re_share"] - fb["re_share"]).dropna().to_numpy()
            bs = [np.median(dre[rng.integers(0, len(dre), len(dre))]) for _ in range(2000)]
            rl, rh = np.percentile(bs, [2.5, 97.5])
            d1 = (fa["re_lt1"] - fb["re_lt1"]).dropna().to_numpy()
            bs1 = [d1[rng.integers(0, len(d1), len(d1))].mean() for _ in range(2000)]
            ql, qh = np.percentile(bs1, [2.5, 97.5])
            dd = (fa["paid_inc"] - fb["paid_inc"]).to_numpy()
            bs2 = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
            pl, ph = np.percentile(bs2, [2.5, 97.5])
            sa, sb = f"{a}__l{star[a]:g}", f"{b}__l{star[b]:g}"
            c = ctx[cls].compare(sa, sb, rng)
            sym = " ".join({1: "+", -1: "−", 0: "0"}[int(c[k][3])]
                           for k in ("純資産の改善", "解消率", "順位相関"))
            lines.append(f"| {lab} | {np.median(dre):+.3f} [{rl:+.3f}, {rh:+.3f}] "
                         f"| {d1.mean():+.2f} [{ql:+.2f}, {qh:+.2f}] "
                         f"| {dd.mean():+.2f} [{pl:+.2f}, {ph:+.2f}] "
                         f"| {fb['n_changed'].median():.0f} → {fa['n_changed'].median():.0f} "
                         f"| {e51.fc(c['純資産の改善'])} | {sym} |")
        lines.append("")
        if (cls, "s2") in firms:
            prop = firms[(cls, "s2")].copy()
            sp = f"s2__l{star['s2']:g}"
            cp_ = ctx[cls].comp[ctx[cls].comp["version"] == sp].set_index("docID")
            prop["nai_annual"] = cp_["net_asset_improvement"].reindex(prop.index) / 5.0
            realrows += [f"## {cls}(実データ {len(rf)} 社)", "",
                         "| 項目 | 設定 2(社数) | 実データ(社数) | 差 [95%] |",
                         "|---|---:|---:|---|"]
            rr = real_ratio(cls, names, members)
            sh = prop["re_share"].dropna().to_numpy()
            rows_ = (("純資産の改善(年率、中央値)", prop["nai_annual"].dropna().to_numpy(),
                      rf["nai_annual"].dropna().to_numpy(), np.median),
                     ("ΔRE/ΔNA(中央値、ΔNA > 0)", sh, rr, np.median),
                     ("ΔRE/ΔNA < 0.99 の割合(ΔNA > 0)", (sh < 0.99).astype(float),
                      (rr < 0.99).astype(float), np.mean),
                     ("払込資本を増やした割合", prop["paid_inc"].to_numpy(),
                      (pp > 0.001).astype(float).to_numpy(), np.mean),
                     ("払込資本を減らした割合", prop["paid_dec"].to_numpy(),
                      rf["paid_dec"].dropna().to_numpy(), np.mean),
                     ("変更された科目数(中央値)", prop["n_changed"].to_numpy(),
                      rf["n_changed"].dropna().to_numpy(), np.median))
            for lab, a_, b_, fn in rows_:
                ds = [fn(a_[rng.integers(0, len(a_), len(a_))])
                      - fn(b_[rng.integers(0, len(b_), len(b_))]) for _ in range(2000)]
                lo, hi = np.percentile(ds, [2.5, 97.5])
                realrows.append(f"| {lab} | {fn(a_):.3f}({len(a_)}) | {fn(b_):.3f}({len(b_)}) "
                                f"| {fn(a_) - fn(b_):+.3f} [{lo:+.3f}, {hi:+.3f}] |")
            tr, tp = target_ratio(cls, names, members)
            realrows += ["", f"対応先 τ に完全に到達した場合の ΔRE/ΔNA(診断、{len(tr)} 社): 中央値 "
                         f"{np.median(tr):.2f}、p25 {np.percentile(tr, 25):.2f}、p75 "
                         f"{np.percentile(tr, 75):.2f}、< 0.99 の割合 "
                         f"{(tr < 0.99).mean():.0%}。対応先の払込資本が始点より小さい企業 "
                         f"{(tp < 0).mean():.0%}({len(tp)} 社)、払込資本の差 τ − x の中央値 "
                         f"{np.median(tp):+.3f}(総資産比)。", ""]
    # D.1: ステージ65 の訂正(class 2 の提案を、class 1 で較正した λ に差し替える)
    s1 = f"s1__l{star['s1']:g}"
    f = firms[("c2", "s1")]
    cr = ctx["c2"].comp[ctx["c2"].comp["version"] == s1].set_index("docID")
    res = cr.loc[cr["insolvent_start"].astype(bool), "insolvency_resolved"].mean()
    rho = corr(ctx["c2"].wide[s1], ctx["c2"].ref)
    corr_ = ["# ステージ67 D.1: ステージ65 の class 2 の数値の訂正", "",
             "ステージ65 の提案(手法 5)の class 2 は、class 2 の L を見て λ̃*_0 = 0.1 を選んでいた"
             "(リーク)。訂正後は、class 1 で較正した λ̃*_0 = 1(λ = 1 × s、s は class 1 の尺度)を"
             "class 2 にそのまま当てはめた値(ステージ66 の設定 1 と同じ生成)。class 1 の数値は"
             "もともと class 1 で較正しているので変わらない。", "",
             "| 項目 | ステージ65(リークあり、λ̃ = 0.1) | 訂正後(class 1 で較正) |",
             "|---|---:|---:|",
             f"| 払込資本を減らした割合 | 38% | {f['paid_dec'].mean():.0%} |",
             f"| 払込資本を増やした割合 | — | {f['paid_inc'].mean():.0%} |",
             f"| 純資産の改善(5 年、中央値) | 0.327 | {cr['net_asset_improvement'].median():.3f} |",
             f"| 債務超過の解消率 | 1.00 | {res:.2f} |",
             f"| 符号付き順位相関 | +0.386 | {rho:+.3f} |",
             f"| 変更された科目数(中央値) | 12 | {f['n_changed'].median():.0f} |", "",
             "ステージ65 の値は `results/stage65/metrics.md`(c2、5 提案)から転記。", "",
             "## 訂正後の class 2: 提案 対 実データ(class 2 → 回復、15 社)", "",
             "差 = 提案 − 実データ。95% 区間は、生成と実データの企業を別々に復元抽出(2,000 回)。",
             "", "| 項目 | 提案 | 実データ | 差 [95%] |", "|---|---:|---:|---|"]
    rf = real_firm_values("c2", names, members)
    prop = f.copy()
    prop["nai_annual"] = cr["net_asset_improvement"].reindex(prop.index) / 5.0
    for lab, col, how in (("純資産の改善(年率、中央値)", "nai_annual", "median"),
                          ("払込資本を減らした割合", "paid_dec", "mean"),
                          ("利益剰余金が増えた割合", "re_up", "mean"),
                          ("変更された科目数(中央値)", "n_changed", "median"),
                          ("総資産変化(年率、中央値)", "d_assets", "median")):
        a_ = prop[col].dropna().to_numpy()
        b_ = rf[col].dropna().to_numpy()
        fn = np.median if how == "median" else np.mean
        ds = [fn(a_[rng.integers(0, len(a_), len(a_))])
              - fn(b_[rng.integers(0, len(b_), len(b_))]) for _ in range(2000)]
        lo, hi = np.percentile(ds, [2.5, 97.5])
        fmt = "{:.0%}" if how == "mean" else ("{:.0f}" if col == "n_changed" else "{:+.4f}")
        corr_.append(f"| {lab} | {fmt.format(fn(a_))} | {fmt.format(fn(b_))} "
                     f"| {fn(a_) - fn(b_):+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    (OUT / "correction_stage65.md").write_text("\n".join(corr_) + "\n", encoding="utf-8")
    (OUT / "bs.md").write_text("\n".join(bs_lines(spec, names, members, star)) + "\n",
                               encoding="utf-8")
    (OUT / "calibration.md").write_text("\n".join(cal) + "\n", encoding="utf-8")
    (OUT / "comparison.md").write_text("\n".join([*lines, *items]) + "\n", encoding="utf-8")
    (OUT / "realdata.md").write_text("\n".join(realrows) + "\n", encoding="utf-8")
    summary: dict[str, Any] = {"star": star}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for fn_ in ("calibration.md", "comparison.md", "realdata.md", "correction_stage65.md"):
        print((OUT / fn_).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
