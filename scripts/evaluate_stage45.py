#!/usr/bin/env python
"""ステージ45: 制約の更新((5) を削除、自己株式の上限を追加、方向制約を削除)の前後比較。

入力: ``metrics_stage39.py --gen-dir data/interim/stage45/gen`` の出力(``data/interim/stage45/metrics``)。
``prev``・``prev_r1`` は更新前(ステージ43 の最終設定と R1 + 不活性制約)、``new``・``new_r1`` は更新後。

出力(``results/stage45/``): ``results.md``、``summary.json``。

    uv run python scripts/evaluate_stage45.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from compare_stage39 import boot, comp_level, corr, fmt_ci
from criteria_stage31 import wilson
from evaluate_stage34 import special_reason
from recovery_stage30 import changes
from run_stage25 import CONTINUOUS, summarize

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names

_M = Path("data/interim/stage45/metrics")
_GEN = Path("data/interim/stage45/gen")
_PANEL = Path("data/interim/stage23/panel/panel.csv")
OUTLIERS = {"c2": ("S100TSS9", "S100TWFQ"), "c1": ("S100VGDJ",)}
CLS = {"c2": "class 2(55 社)", "c1": "class 1(131 社)"}
_RE, _CS, _CSUR, _TS = ("jppfs_cor_RetainedEarnings", "jppfs_cor_CapitalStock",
                        "jppfs_cor_CapitalSurplus", "jppfs_cor_TreasuryStock")
GEN_YEARS, THR, MATERIAL, HOLD, TOL = 5.0, 0.001, 0.001, 1e-6, 1e-4
PAIRS = (("prev", "new", "最終設定(R3)"), ("prev_r1", "new_r1", "R1(感度分析)"))


def load_gen(cls: str, s: str) -> pd.DataFrame:
    g = pd.concat([pd.read_parquet(f) for f in
                   sorted((_GEN / cls / s).glob("*__seed*.parquet"))])
    return g[(g["method"] == "PCFM") & ~g["docID"].isin(OUTLIERS[cls])].reset_index(drop=True)


def firm_frac(flag: pd.Series, doc: pd.Series) -> tuple[int, int]:
    f = pd.Series(np.asarray(flag), index=np.asarray(doc)).groupby(level=0).mean()
    return int((f >= 2 / 3).sum()), len(f)


def halved(g: pd.DataFrame) -> tuple[int, int]:
    """資本金が始点の 50% を明確に下回った企業(3 seed の 2/3 以上)。"""
    x0, x1 = g[f"x0_{_CS}"], g[f"x1_{_CS}"]
    held = x0 > HOLD
    lim = 0.5 * x0 - np.maximum(0.5 * x0 * TOL, 1e-6)
    return firm_frac(((x1 < lim) & held)[held], g.loc[held, "docID"])


def ts_up(g: pd.DataFrame) -> tuple[int, int]:
    """自己株式を保有する企業で、|自己株式| が総資産比 0.001 を超えて増えた企業。"""
    t0, t1 = g[f"x0_{_TS}"].abs(), g[f"x1_{_TS}"].abs()
    held = t0 >= HOLD
    return firm_frac(((t1 - t0) > MATERIAL)[held], g.loc[held, "docID"])


def annual(g: pd.DataFrame, col: str) -> pd.Series:
    d = (g[f"x1_{col}"] - g[f"x0_{col}"]).groupby(g["docID"]).median() / GEN_YEARS
    out: pd.Series = d
    return out


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage45"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    summary: dict[str, object] = {}
    lines = ["# ステージ45: 制約の更新の前後比較", "",
             "更新前 = ステージ43 の最終設定(α = 0.5、方向制約あり、自己株式の上限なし)。"
             "更新後 = (5) を削除、自己株式の上限(分配可能額の代理)を追加、ステージ7 の方向制約を削除。"
             "他は同じ(R3 + 上限 γ = 0.757、β = 0.3、不活性制約 規則 M、K = 13)。チェックは更新後の 15 項目"
             "(8 = 自己株式が上限超)で数える。", ""]

    # ---------------- D.2 全指標 ----------------
    for cls in ("c2", "c1"):
        rows = pd.read_parquet(_M / f"rows_{cls}.parquet")
        rows = rows[~rows["docID"].isin(OUTLIERS[cls])]
        comp = comp_level(rows)
        viol = pd.read_csv(_M / f"viol_{cls}.csv", dtype={"docID": str})
        viol = viol[~viol["docID"].isin(OUTLIERS[cls])]
        dels = pd.read_parquet(_M / f"deltas_{cls}.parquet")
        dels = dels[(dels["space"] == "native") & ~dels["docID"].isin(OUTLIERS[cls])]
        refs = pd.read_parquet(_M / "refs.parquet")
        ref = refs[(refs["cls"] == cls) & (refs["space"] == "native")].set_index(
            "account")["median"]

        def wide(s: str, dels: pd.DataFrame = dels) -> pd.DataFrame:
            w = dels[dels["setting"] == s]
            out: pd.DataFrame = w.pivot(index="docID", columns="account",
                                        values="delta")[list(names)]
            return out

        for before, after, title in PAIRS:
            summ, pr = summarize(comp, (), (before, after), before)
            s_ = summ.set_index("version")
            pr = pr.set_index("metric")
            wb, wa = wide(before), wide(after)
            cb, ca = corr(wb, ref), corr(wa, ref)
            _, _, dlo, dhi = boot(wa, ref, wb.loc[wa.index], ref, rng)
            vb = viol[viol["setting"] == before].groupby("check")["rows"].sum()
            va = viol[viol["setting"] == after].groupby("check")["rows"].sum()
            n = int(s_.loc[after, "n"])
            lines += [f"## D.2 {title}、{CLS[cls]}", "",
                      "| 指標 | 更新前 | 更新後 | 差(更新後 − 更新前)[95% CI] |", "|---|---:|---:|---|",
                      f"| 15 項目の違反(行) | {int(vb.sum())} / {3 * n}"
                      + (f"({'、'.join(f'{k} {int(v)}' for k, v in vb.items())})" if len(vb) else "")
                      + f" | **{int(va.sum())} / {3 * n}** | — |"]
            labels = {m: lab for m, lab, _ in CONTINUOUS}
            for m in ("net_asset_improvement", "abs_asset_change", "sparsity", "validity_lightgbm",
                      "proximity_l2"):
                q = pr.loc[labels[m]]
                dg = 0 if m == "sparsity" else 3
                lab_m = labels[m].replace("|", "\\|")
                lines.append(f"| {lab_m} | {s_.loc[before, m]:.{dg}f} | {s_.loc[after, m]:.{dg}f} "
                             f"| {fmt_ci(q['median_diff(version-equal)'], q['ci_low'], q['ci_high'])} |")
                if m == "net_asset_improvement":
                    q2 = pr.loc["債務超過の解消率(社ごとの 3 seed 平均)"]
                    lines.append(
                        f"| 債務超過の解消(2/3 seed 以上) "
                        f"| {int(s_.loc[before, 'resolved_firms(>=2/3 seeds)'])}/"
                        f"{int(s_.loc[before, 'insolvent_start'])} "
                        f"| {int(s_.loc[after, 'resolved_firms(>=2/3 seeds)'])}/"
                        f"{int(s_.loc[after, 'insolvent_start'])} "
                        f"| 解消率 {fmt_ci(q2['median_diff(version-equal)'], q2['ci_low'], q2['ci_high'])} |")
            lines += [f"| 符号付き順位相関 | {cb:+.3f} | {ca:+.3f} | {fmt_ci(ca - cb, dlo, dhi, 3)} |", ""]
            summary[f"{cls}_{after}"] = {
                "violations": int(va.sum()), "violations_before": int(vb.sum()),
                **{m: float(s_.loc[after, m]) for m in ("net_asset_improvement",
                                                        "abs_asset_change", "sparsity",
                                                        "validity_lightgbm", "proximity_l2")},
                "resolved": int(s_.loc[after, "resolved_firms(>=2/3 seeds)"]),
                "signed_corr": ca}
        # R3 − R1(更新後)
        _su, pr = summarize(comp, (), ("new_r1", "new"), "new_r1")
        q = pr.set_index("metric").loc["純資産の改善"]
        lines += [f"更新後の R3 − R1(純資産の改善): "
                  f"{fmt_ci(q['median_diff(version-equal)'], q['ci_low'], q['ci_high'])}", ""]

    # ---------------- B.3・D.3 実データとの比較 ----------------
    fy = pd.read_csv("data/interim/stage30/firm_years.csv", dtype={"edinetCode": str})
    wl = fy.pivot(index="edinetCode", columns="year", values="label")
    pan = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    a23 = pan[pan["t"] == 2023].set_index("edinetCode")
    b24 = pan[pan["t"] == 2024].set_index("edinetCode")
    both = [c for c in a23.index.intersection(b24.index) if wl.loc[c, 2022] in (1, 2)]
    reason = special_reason(fy, both)
    press = [c for c in both if reason.get(c, "") == ""]
    cs0 = a23.loc[press, f"prev_{_CS}"].fillna(0).to_numpy()
    cs1 = b24.loc[press, f"cur_{_CS}"].fillna(0).to_numpy()
    real_half = (cs1[cs0 > 0] < 0.5 * cs0[cs0 > 0])
    t0 = a23.loc[press, f"prev_{_TS}"].fillna(0).abs().to_numpy()
    t1 = b24.loc[press, f"cur_{_TS}"].fillna(0).abs().to_numpy()
    a0 = a23.loc[press, "prev_total_assets"].to_numpy()
    held = t0 >= HOLD * a0
    real_ts = ((t1 - t0) > MATERIAL * a0)[held]
    rf = pd.read_csv("results/stage30/recovery_firms.csv", dtype={"edinetCode": str})
    rec2 = rf[(rf["reason"].fillna("") == "") & rf["in_panel"].astype(bool) & (rf["group"] == "Y")]
    c1 = wl[wl[2022] == 1].copy()
    c1["reason"] = special_reason(fy, c1.index.tolist())
    c1 = c1.join(changes(_PANEL.parent, c1.index.tolist(), names), how="left")
    c1 = c1[(c1["reason"] == "") & c1["n_years"].notna()]
    rec1 = c1[wl.loc[c1.index, 2024] == 0]
    rec = {"c2": rec2, "c1": rec1}
    rec_lab = {"c2": "再建群 class 2 → 注記なし 15 社", "c1": "class 1 → class 0 の 57 社"}

    def pct(k: int, m: int) -> str:
        lo, hi = wilson(k, m) if m else (float("nan"), float("nan"))
        return f"{k}/{m}({k / m:.0%}、[{lo:.0%}, {hi:.0%}])" if m else "—"

    lines += ["## B.3・D.3 実データとの比較", "",
              "資本金の半減と自己株式の増加の実データは、class 1・2(2022 年度、特殊業種を除く)"
              f"{len(press)} 社の 2022 → 2024 年度末(2 年)。払込資本・利益剰余金は、実際に再建した企業(年率)。"
              "生成は 5 年(払込資本・利益剰余金は年率に直した)。企業は 3 seed の 2/3 以上で数える。", ""]
    for cls in ("c2", "c1"):
        lines += [f"### {CLS[cls]}", "",
                  "| 項目 | 生成(更新前) | 生成(更新後) | 実データ |", "|---|---|---|---|"]
        gp, gn = load_gen(cls, "prev"), load_gen(cls, "new")
        r = rec[cls]
        rows_t = [
            ("資本金を半減させた企業の割合", pct(*halved(gp)), pct(*halved(gn)),
             pct(int(real_half.sum()), len(real_half))),
            ("自己株式が増えた企業の割合(保有企業)", pct(*ts_up(gp)), pct(*ts_up(gn)),
             pct(int(real_ts.sum()), len(real_ts))),
        ]
        for lab, col_fn, real_col in (
            ("払込資本を増やした企業の割合", lambda g: annual(g, _CS) + annual(g, _CSUR), "d_paid"),
            ("利益剰余金が増えた企業の割合", lambda g: annual(g, _RE), "d_re"),
        ):
            vp, vn = col_fn(gp), col_fn(gn)
            rv = r[real_col].dropna()
            rows_t.append((lab, pct(int((vp > THR).sum()), len(vp)),
                           pct(int((vn > THR).sum()), len(vn)),
                           pct(int((rv > THR).sum()), len(rv)) + f"({rec_lab[cls]})"))
        for lab, a_, b_, c_ in rows_t:
            lines.append(f"| {lab} | {a_} | {b_} | {c_} |")
        neg = int((load_gen(cls, "new")[f"x1_{_CS}"] < -1e-9).sum())
        lines += ["", f"資本金が負になった行(更新後): {neg}。資本金は canonical で `sign: nonneg`"
                  "(第 1 層の符号制約)で、(5) を外しても非負は保たれる。", ""]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
