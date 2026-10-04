#!/usr/bin/env python
"""ステージ53: 健全企業の密度 PL = log q(x) による生成結果の評価。

sklearn のプロセス。新しい生成はしない。

規則は ``results/stage53/criteria.md``(評価の前に記録)。q はステージ52 の版 I。
状態は各時点の総資産で割り直す。ゼロの扱いは criteria.md §5(訂正)。分位点は class 0 の
5 分割交差検証の hold-out の PL を基準にする。

出力(``results/stage53/``): ``baseline.md``(B.1)、``relations.md``(B.2〜B.4)、``settings.md``
(A.3・C.2・C.3)、``pl_rows.parquet``(行ごとの PL)、``summary.json``。

    uv run python scripts/pl_stage53.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from compare_stage39 import comp_level
from density_stage51 import load_tilde
from density_stage52 import fit_indep
from scipy import stats
from sklearn.model_selection import KFold

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    load_canonical_spec,
    orthonormal_basis,
    state_vector_names,
)
from fsgen.accounting.plausibility import (
    evaluate_checks,
    evaluate_extra_checks,
    subtotal_members,
)
from fsgen.models.zero_density import IndepZeroGaussian

ZERO = 1e-6
SOLVER_ZERO = 1e-9  # 実データの非 0 の最小(class 0 で 1e-9 未満はない)
RAW_SNAP = {"on": False}
TS = (0.0, 0.25, 0.5, 0.75, 1.0)
ALPHA = 0.0019850022055580064
EXCL = {"c2": ("S100TSS9", "S100TWFQ"), "c1": ("S100VGDJ",)}
_RE, _CS, _CSUR = ("jppfs_cor_RetainedEarnings", "jppfs_cor_CapitalStock",
                   "jppfs_cor_CapitalSurplus")
# 設定 → (class → (ディレクトリ, method, 指標ファイルのディレクトリ, 指標の version))
SETTINGS: dict[str, dict[str, tuple[str, str, str | None, str | None]]] = {
    "最終設定 s47_final": {c: (f"data/interim/stage47/gen/{c}/final", "PCFM",
                              "data/interim/stage47/metrics", "final") for c in ("c2", "c1")},
    "R1 s47_r1": {c: (f"data/interim/stage47/gen/{c}/r1", "PCFM",
                      "data/interim/stage47/metrics", "r1") for c in ("c2", "c1")},
    "射影前の出力(FM)": {c: (f"data/interim/stage47/gen/{c}/final", "FM", None, None)
                     for c in ("c2", "c1")},
    "密度 + 距離 版 I(ステージ52)": {c: (f"data/interim/stage52/gen/{c}/indep", "PCFM",
                                    "data/interim/stage52/metrics", "indep") for c in ("c2", "c1")},
    "尤度コスト t(ステージ50)": {c: (f"data/interim/stage50/gen/{c}/t", "PCFM",
                                "data/interim/stage50/metrics", "t") for c in ("c2", "c1")},
    "ステージ24 等重み(R1 旧設定)": {"c2": ("data/interim/stage24/equal", "PCFM",
                                    "data/interim/stage51/metrics/s24", "equal")},
    "ステージ24 推定ベース": {"c2": ("data/interim/stage24/estimated", "PCFM",
                              "data/interim/stage51/metrics/s24", "estimated")},
    "ステージ24 定性ベース": {"c2": ("data/interim/stage24/qualitative", "PCFM",
                              "data/interim/stage51/metrics/s24", "qualitative")},
    "R4 γ = 0(ステージ29、参考)": {"c2": ("data/interim/stage29/r4", "PCFM", None, None)},
}


class PL:
    def __init__(self, model: IndepZeroGaussian, assets_vec: np.ndarray, ref: np.ndarray) -> None:
        self.model, self.a, self.ref = model, assets_vec, np.sort(ref)

    def __call__(self, x: np.ndarray, start_zero: np.ndarray | None = None) -> np.ndarray:
        """x は任意の尺度の状態(行)。各行の総資産で割り直して評価する。総資産 ≤ 0 は nan。

        実データは丸めない(密度の推定・交差検証の基準と同じ)。生成の状態は ``start_zero``
        (始点の実データでちょうど 0 の科目)を渡し、その科目で |x| < 1e-6 の値と、全科目で
        |x| < 1e-9(実データの非 0 の最小より小さい)の値を 0 にする(criteria.md §5 の訂正)。
        ``RAW_SNAP`` が真なら、記録した当初の規則(全科目で |x| < 1e-6 を 0)で評価する。
        """
        a = x @ self.a
        out = np.full(len(x), np.nan)
        ok = a > 0
        if ok.any():
            xn = x[ok] / a[ok, None]
            if RAW_SNAP["on"]:
                xn = np.where(np.abs(xn) < ZERO, 0.0, xn)
            elif start_zero is not None:
                snap = (start_zero[ok] & (np.abs(xn) < ZERO)) | (np.abs(xn) < SOLVER_ZERO)
                xn = np.where(snap, 0.0, xn)
            out[ok] = self.model.logpdf(xn)
        return out

    def pct(self, v: np.ndarray) -> np.ndarray:
        out: np.ndarray = np.searchsorted(self.ref, v) / len(self.ref) * 100
        return np.where(np.isnan(v), np.nan, out)


def q3(v: np.ndarray, d: int = 1) -> str:
    v = v[np.isfinite(v)]
    if not len(v):
        return "—"
    a, b, c = np.percentile(v, [25, 50, 75])
    return f"{b:.{d}f} [{a:.{d}f}, {c:.{d}f}]"


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage53"))
    ap.add_argument("--raw-snap", action="store_true",
                    help="当初の規則(全科目で |x| < 1e-6 を 0)で評価する(訂正前の結果の再現用)")
    args = ap.parse_args()
    RAW_SNAP["on"] = bool(args.raw_snap)
    args.out.mkdir(parents=True, exist_ok=True)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    u = orthonormal_basis(build_constraint_matrix(spec, include_subtotals=False))
    d = np.load("data/interim/stage52/density/indep.npz")
    model = IndepZeroGaussian(d["pi0"], d["u"], d["mu"], d["cov"])
    x, label, docs, _keep = load_tilde(names)
    docs_a = np.array(docs)
    x0 = x[label == 0]
    cv = np.empty(len(x0))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(x0):
        cv[te] = fit_indep(x0[tr], u)[0].logpdf(x0[te])
    pl = PL(model, members["jppfs_cor_Assets"], cv)
    start_zero = {dd: r == 0 for dd, r in zip(docs, x, strict=True)}  # 始点の実データのゼロ
    summary: dict[str, Any] = {}

    # ---------------- B.1 基準線 ----------------
    panel = pd.read_csv("data/interim/stage23/panel/panel.csv", dtype={"edinetCode": str})
    grp = pd.read_csv("results/stage51/asset_change.csv", dtype={"edinetCode": str})
    rows: list[dict[str, Any]] = []
    groups = [("class 0(学習に使った全社)", pl(x0)), ("class 0(交差検証の hold-out)", cv),
              ("class 1", pl(x[label == 1])), ("class 2", pl(x[label == 2])),
              ("class 2(外れ値を除く)", pl(x[(label == 2) & ~np.isin(docs_a, EXCL["c2"])]))]
    for g, lab in (("再建群(class 2 → 回復)", "class 2 → 回復"),
                   ("再建群(class 1 → class 0)", "class 1 → class 0")):
        codes = grp[grp["group"] == g]["edinetCode"].tolist()
        a = panel[(panel["t"] == 2023) & panel["edinetCode"].isin(codes)]
        b = panel[(panel["t"] == 2024) & panel["edinetCode"].isin(codes)]
        groups.append((f"{lab}・始点(2022 年度末、{len(a)} 社)",
                       pl(a[[f"prev_{n}" for n in names]].fillna(0.0).to_numpy())))
        groups.append((f"{lab}・終点(2024 年度末、{len(b)} 社)",
                       pl(b[[f"cur_{n}" for n in names]].fillna(0.0).to_numpy())))
    lines = ["# ステージ53 B.1: PL の基準線", "",
             "PL = log q(x)(ステージ52 の版 I、class 0 3,768 書類で学習)。分位点 = class 0 の交"
             "差検証の"
             "hold-out の分布での位置(%)。中央値 [四分位]。", "",
             "| 集団 | 社数 | PL | 分位点 |", "|---|---:|---|---|"]
    for lab, v in groups:
        lines.append(f"| {lab} | {np.isfinite(v).sum()} | {q3(v)} | {q3(pl.pct(v), 0)} |")
        rows.append({"group": lab, "median": float(np.nanmedian(v)),
                     "pct_median": float(np.nanmedian(pl.pct(v)))})
    med: dict[str, float] = {str(r["group"]): float(r["median"]) for r in rows}
    order = (med["class 0(交差検証の hold-out)"] > med["class 1"]
             > med["class 2(外れ値を除く)"])
    lines += ["", f"順序(hold-out class 0 > class 1 > class 2(55)):"
                  f" {'成立' if order else '成立しない'}", ""]
    summary["B1"] = {"rows": rows, "order": bool(order)}
    (args.out / "baseline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- C 既存設定 ----------------
    allrows = []
    for sname, per in SETTINGS.items():
        for cls, (gdir, method, _mdir, _ver) in per.items():
            g = pd.concat([pd.read_parquet(f) for f in sorted(Path(gdir).glob("*__seed*.parquet"))])
            g = g[(g["method"] == method)]
            if method == "PCFM":
                g = g[g["config"] != "raw"]
            g = g[~g["docID"].isin(EXCL[cls] if not sname.startswith("ステージ24") else ())]
            g = g.reset_index(drop=True)
            xa = g[[f"x0_{n}" for n in names]].to_numpy()
            xb = g[[f"x1_{n}" for n in names]].to_numpy()
            sz = np.array([start_zero.get(dd, np.abs(r) < SOLVER_ZERO)
                           for dd, r in zip(g["docID"], xa, strict=True)])
            pls = np.column_stack([pl((1 - t) * xa + t * xb, sz) for t in TS])
            flags = np.column_stack([evaluate_checks(xa, xb, spec, names),
                                     evaluate_extra_checks(xb, xa, spec, names)])
            df = pd.DataFrame({"setting": sname, "cls": cls, "docID": g["docID"], "seed": g["seed"],
                               **{f"pl_{t}": pls[:, i] for i, t in enumerate(TS)},
                               "n_viol": flags.sum(axis=1), "any_viol": flags.any(axis=1),
                               "re_transfer": g.get("re_transfer", np.nan)})
            allrows.append(df)
    pr = pd.concat(allrows, ignore_index=True)
    pr["dpl"] = pr["pl_1.0"] - pr["pl_0.0"]
    pr["pct1"] = pl.pct(pr["pl_1.0"].to_numpy())
    steps = np.diff(pr[[f"pl_{t}" for t in TS]].to_numpy(), axis=1)
    pr["monotone"] = np.all(steps >= -1e-6, axis=1) & np.all(np.isfinite(steps), axis=1)
    pr.to_parquet(args.out / "pl_rows.parquet", index=False)
    firm = pr.groupby(["setting", "cls", "docID"]).agg(
        pl1=("pl_1.0", "median"), pl0=("pl_0.0", "median"), dpl=("dpl", "median"),
        pct1=("pct1", "median"), mono=("monotone", "mean"), viol=("n_viol", "sum"),
        nan1=("pl_1.0", lambda s: int(s.isna().sum()))).reset_index()
    lines = ["# ステージ53 A.3・C.2・C.3: 既存設定の PL", "",
             "企業ごとの seed 中央値の中央値 [四分位]。分位点 = class 0(hold-out)の分布での位置("
             "%)。"
             "単調 = t = 0, 0.25, 0.5, 0.75, 1 の PL が減らない(3 seed の 2/3 以上)。"
             "違反 = 15 項目の違反の延べ件数(行 × 項目)。総資産 ≤ 0 の行は PL を計算できない(nan"
             ")。", "",
             "| 設定 | class | 社数 | 終点の PL | 分位点 | ΔPL(始点から) | 単調増加 | 15 項目の"
             "違反(延べ) "
             "| PL を計算できない行 |", "|---|---|---:|---|---|---|---:|---:|---:|"]
    cset: dict[str, Any] = {}
    for (sname, cls), f in firm.groupby(["setting", "cls"], sort=False):
        mono = int((f["mono"] >= 2 / 3).sum())
        lines.append(f"| {sname} | {cls} | {len(f)} | {q3(f['pl1'].to_numpy())} "
                     f"| {q3(f['pct1'].to_numpy(), 0)} | {q3(f['dpl'].to_numpy())} | {mono} /"
                     f" {len(f)} "
                     f"| {int(f['viol'].sum())} | {int(f['nan1'].sum())} |")
        cset[f"{sname}|{cls}"] = {"pl1": float(f["pl1"].median()), "dpl": float(f["dpl"].median()),
                                  "pct1": float(f["pct1"].median()), "mono": mono, "n": len(f)}
    # C.3: 行単位の PL と違反の関係(射影前の出力と最終設定)
    sub = pr[pr["setting"].isin(["射影前の出力(FM)", "最終設定 s47_final"])]
    rho_v = stats.spearmanr(sub["pl_1.0"], sub["n_viol"], nan_policy="omit")
    fm = pr[pr["setting"] == "射影前の出力(FM)"]
    v_pl = fm.loc[fm["any_viol"], "pl_1.0"]
    nv_pl = fm.loc[~fm["any_viol"], "pl_1.0"]
    lines += ["", "## C.3 PL と 15 項目の違反(行単位)", "",
              f"- 射影前の出力: 違反のある行 {int(fm['any_viol'].sum())} / {len(fm)}、"
              f"PL の中央値 違反あり {np.nanmedian(v_pl):.1f}、違反なし {np.nanmedian(nv_pl):.1f}、"
              f"総資産 ≤ 0 で PL を計算できない行 {int(fm['pl_1.0'].isna().sum())}",
              f"- 射影前の出力と最終設定をまとめた行で、終点の PL と違反の件数の Spearman ρ = "
              f"{rho_v.statistic:.3f}(p = {rho_v.pvalue:.2g})", ""]
    summary["C"] = cset
    (args.out / "settings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- B.2 他の指標との関係 ----------------
    mets = []
    for sname, per in SETTINGS.items():
        gdir, method, mdir, ver = per["c2"]
        if mdir is None:
            continue
        r = pd.read_parquet(Path(mdir) / "rows_c2.parquet")
        r = r[r["version"] == ver]
        c = comp_level(r) if "proximity_32" in r.columns else r.groupby("docID").median(
            numeric_only=True).reset_index()
        f = firm[(firm["setting"] == sname) & (firm["cls"] == "c2")]
        mets.append(c.merge(f, on="docID").assign(setting=sname))
    mm = pd.concat(mets, ignore_index=True)
    lines = ["# ステージ53 B.2〜B.4: PL と他の指標、循環性、振替", "",
             "## B.2 他の指標との順位相関(class 2、企業 × 設定、seed の中央値)", "",
             f"対象: {', '.join(mm['setting'].unique())}(validity を計算した設定)。", "",
             "| 指標 | まとめた全体(終点の PL の分位点と) | 最終設定の中だけ | ΔPL と(まとめた"
             "全体) |",
             "|---|---:|---:|---:|"]
    b2: dict[str, float] = {}
    for col, lab in (("validity_lightgbm", "validity LightGBM"),
                     ("validity_random_forest", "validity RF"),
                     ("validity_logreg", "validity LogReg"),
                     ("net_asset_improvement", "純資産の改善"), ("proximity_l2", "proximity")):
        a = stats.spearmanr(mm["pct1"], mm[col], nan_policy="omit").statistic
        fin = mm[mm["setting"] == "最終設定 s47_final"]
        w = stats.spearmanr(fin["pct1"], fin[col], nan_policy="omit").statistic
        dd = stats.spearmanr(mm["dpl"], mm[col], nan_policy="omit").statistic
        b2[col] = float(a)
        lines.append(f"| {lab} | {a:+.3f} | {w:+.3f} | {dd:+.3f} |")
    strong = abs(b2["validity_lightgbm"]) >= 0.7
    lines += ["", f"validity LightGBM との |ρ| ≥ 0.7(強く相関):"
                  f" {'該当' if strong else '該当しない'}", ""]

    # ---------------- B.3 循環性 ----------------
    plan = np.load("data/interim/stage28/plans_equal/plans_nearest_na0.npz")["plan_full"]
    tg = np.argmax(plan, axis=1)
    docs2 = docs_a[label == 2]
    pl_t = pd.Series(pl(x0[tg]), index=docs2)
    fin = firm[(firm["setting"] == "最終設定 s47_final") & (firm["cls"] == "c2")].set_index("docID")
    both = pd.DataFrame({"end": fin["pl1"], "target": pl_t.reindex(fin.index),
                         "start": fin["pl0"]}).dropna()
    rho_t = stats.spearmanr(both["end"], both["target"]).statistic
    rho_s = stats.spearmanr(both["end"], both["start"]).statistic
    circ = rho_t >= 0.8
    lines += ["## B.3 循環性(最終設定、class 2)", "",
              f"対応先(全社の計画の最近傍、実在の class 0 企業)の PL と、生成の終点の PL"
              f"({len(both)} 社)。", "",
              "| 量 | 値 |", "|---|---:|",
              f"| 対応先の PL(中央値) | {both['target'].median():.1f} |",
              f"| 終点の PL(中央値) | {both['end'].median():.1f} |",
              f"| 始点の PL(中央値) | {both['start'].median():.1f} |",
              f"| 終点 − 対応先(中央値) | {(both['end'] - both['target']).median():+.1f} |",
              f"| 終点が対応先を上回る社数 | {int((both['end'] > both['target']).sum())} /"
              f" {len(both)} |",
              f"| Spearman ρ(終点, 対応先) | **{rho_t:+.3f}** |",
              f"| Spearman ρ(終点, 始点)(参考) | {rho_s:+.3f} |", "",
              f"循環(ρ ≥ 0.8): {'該当' if circ else '該当しない'}", ""]

    # ---------------- B.4 振替 ----------------
    x2 = x[(label == 2) & ~np.isin(docs_a, EXCL["c2"])]
    ire, ics, icsur = names.index(_RE), names.index(_CS), names.index(_CSUR)
    neg = x2[:, ire] < 0
    xs = x2[neg].copy()
    t_amt = np.minimum(-xs[:, ire], xs[:, icsur] + (1 - ALPHA) * xs[:, ics])
    from_sur = np.minimum(t_amt, xs[:, icsur])
    xt = xs.copy()
    xt[:, icsur] -= from_sur
    xt[:, ics] -= t_amt - from_sur
    xt[:, ire] += t_amt
    dpl_t = pl(xt) - pl(xs)
    w = stats.wilcoxon(dpl_t[np.isfinite(dpl_t)], alternative="greater")
    up = bool(np.nanmedian(dpl_t) > 0 and w.pvalue < 0.05)
    # 参考 1: 利益剰余金がちょうど 0 になる影響を除く(欠損の 99% だけ填補)
    t99 = 0.99 * t_amt
    f99 = np.minimum(t99, xs[:, icsur])
    x99 = xs.copy()
    x99[:, icsur] -= f99
    x99[:, ics] -= t99 - f99
    x99[:, ire] += t99
    dpl_99 = pl(x99) - pl(xs)
    re0 = int(np.sum(np.abs(xt[:, ire]) < ZERO))
    # 参考 2: 生成の出力で |ΔNA| / A < 0.01 の企業(スペックの方法)
    na = subtotal_members(spec, names)["jppfs_cor_NetAssets"]
    pr_c2 = pr[pr["cls"] == "c2"].copy()
    lines += ["## B.4 振替の検出", "",
              f"### 主の検定: 会計の操作で作った「振替だけの状態」(class 2 で利益剰余金が負の"
              f" {neg.sum()} 社)", "",
              f"- 欠損を資本剰余金、次に資本金(下限 α まで)から填補。純資産は不変。"
              f"利益剰余金がちょうど 0 になった社 {re0} / {neg.sum()}。",
              f"- **ΔPL の中央値 {np.nanmedian(dpl_t):+.2f} [{np.nanpercentile(dpl_t, 25):+.2f}, "
              f"{np.nanpercentile(dpl_t, 75):+.2f}]、上がった社 {int((dpl_t > 0).sum())} /"
              f" {neg.sum()}、"
              f"片側 Wilcoxon p = {w.pvalue:.2g}**",
              f"- **判定(criteria.md §3): 振替で上がる → {'該当' if up else '該当しない'}**",
              f"- 参考(判定に使わない): 欠損の 99% だけ填補(利益剰余金を 0 にしない)と ΔPL の中"
              f"央値 "
              f"{np.nanmedian(dpl_99):+.2f}、上がった社 {int((dpl_99 > 0).sum())} / {neg.sum()}"
              "(利益剰余金のゼロ率は class 0 でほぼ 0% なので、ちょうど 0 になると離散部分で大"
              "きく下がる)", ""]
    lines += ["### 参考: 生成の出力で純資産がほとんど動かない企業(スペックの方法、判定に使わない)",
              "",
              "| 設定 | \\|ΔNA\\| / A < 0.01 の行 | その ΔPL の中央値 | 他の行の ΔPL の中央値 |",
              "|---|---:|---:|---:|"]
    for sname, g in pr_c2.groupby("setting", sort=False):
        src = SETTINGS[sname]["c2"]
        gg = pd.concat([pd.read_parquet(f) for f in sorted(Path(src[0]).glob("*__seed*.parquet"))])
        gg = gg[gg["method"] == src[1]]
        if src[1] == "PCFM":
            gg = gg[gg["config"] != "raw"]
        gg = gg[~gg["docID"].isin(EXCL["c2"] if not sname.startswith("ステージ24") else ())]
        dna = (gg[[f"x1_{n}" for n in names]].to_numpy()
               - gg[[f"x0_{n}" for n in names]].to_numpy()) @ na
        small = np.abs(dna) < 0.01
        dv = g["dpl"].to_numpy()
        lines.append(f"| {sname} | {int(small.sum())} | "
                     f"{np.nanmedian(dv[small]) if small.any() else float('nan'):+.2f} "
                     f"| {np.nanmedian(dv[~small]) if (~small).any() else float('nan'):+.2f} |")
    lines.append("")
    summary["B2"] = {"rho": b2, "strong": bool(strong)}
    summary["B3"] = {"rho_target": float(rho_t), "circular": bool(circ)}
    summary["B4"] = {"median": float(np.nanmedian(dpl_t)), "p": float(w.pvalue), "up": up,
                     "median_99": float(np.nanmedian(dpl_99))}
    if circ:
        verdict = "採用しない(循環)"
    elif up:
        verdict = "採用しない(振替で上がる)"
    elif not order:
        verdict = "採用しない(順序が成立しない)"
    elif strong:
        verdict = "副指標として併記(validity と強く相関)"
    else:
        verdict = "主指標に加える"
    summary["verdict"] = verdict
    lines += ["## B.5 判定(`criteria.md` §3 の優先順)", "", f"**{verdict}**", ""]
    (args.out / "relations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                      default=float), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
