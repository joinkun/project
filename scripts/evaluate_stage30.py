#!/usr/bin/env python
"""ステージ30 D.1: 6 版を再建群から作った基準(``results/stage30/criteria.json``)で判定する。

numpy/pandas のみ。判定の手順はステージ29(``evaluate_stage29.py``)と同じで、基準 5(増資の規模)を
加えた。企業ごとに seed の中央値を取る。基準 1 は ``run_stage25.py`` の出力から読む。

    uv run python scripts/evaluate_stage30.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from evaluate_stage29 import load_version

from fsgen.accounting.constraints import (
    load_canonical_spec,
    state_vector_names,
)

VERSIONS = ("r1", "r3", "r3_cap", "r4", "r4_g040", "r4_g239")
LABELS = {
    "r1": "R1(RE 固定)",
    "r3": "R3(固定なし)",
    "r3_cap": "R3 + 上限 0.5",
    "r4": "R4(γ = 0)",
    "r4_g040": "R4(γ = 0.40)",
    "r4_g239": "R4(γ = 2.39、再建群)",
}
_TOL = 1e-6
_CS, _CSUR = "jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus"


def _band_check(values: pd.Series, band: list[float] | None) -> tuple[bool | None, float, float]:
    if len(values) == 0 or band is None:
        return None, float("nan"), float("nan")
    med, p95 = float(values.median()), float(values.quantile(0.95))
    lo, hi = band
    return (lo <= med <= hi) and (lo <= p95 <= hi), med, p95


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/interim/stage30"))
    ap.add_argument("--results", type=Path, default=Path("results/stage30"))
    args = ap.parse_args()
    crit = json.loads((args.results / "criteria.json").read_text(encoding="utf-8"))
    years, thr, gamma = float(crit["years"]), float(crit["threshold_annual"]), float(crit["gamma"])
    lo2, hi2 = crit["share_re_up_band"]
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    s55 = pd.read_csv(args.results / "metrics" / "summary_excl_outliers55.csv").set_index("version")
    s57 = pd.read_csv(args.results / "metrics" / "summary_all57.csv").set_index("version")

    rows = []
    for v in VERSIONS:
        df = load_version(args.dir / v, names)
        # 増資の規模のため Δ払込資本も読む(load_version は RE の分解のみ)
        raw = pd.concat(
            [pd.read_parquet(args.dir / v / f"nearest_na0__seed{s}.parquet") for s in (0, 1, 2)]
        )
        raw = raw[raw["method"] == "PCFM"]
        d_paid = (
            raw[f"x1_{_CS}"] + raw[f"x1_{_CSUR}"] - raw[f"x0_{_CS}"] - raw[f"x0_{_CSUR}"]
        ).to_numpy()
        df["d_paid"] = d_paid
        firm = df.groupby("docID")[["d_re", "transfer", "profit", "d_paid"]].median()
        up = firm["d_re"] / years > thr
        c1 = int(s55.loc[v, "violations(rows)"]) == 0
        c2 = lo2 <= up.mean() <= hi2
        c3 = bool((firm["profit"] <= gamma + _TOL).all())
        c4, t_med, t_p95 = _band_check(
            firm.loc[firm["transfer"] / years > thr, "transfer"], crit["transfer_5y_band"]
        )
        c5, i_med, i_p95 = _band_check(
            firm.loc[firm["d_paid"] / years > thr, "d_paid"], crit["issue_5y_band"]
        )
        ok = c1 and c2 and c3 and (c4 is not False) and (c5 is not False)
        rows.append({
            "version": v, "violations55": int(s55.loc[v, "violations(rows)"]),
            "violations57": int(s57.loc[v, "violations(rows)"]),
            "share_re_up": float(up.mean()), "profit_max": float(firm["profit"].max()),
            "n_transfer": int((firm["transfer"] / years > thr).sum()),
            "transfer_median": t_med, "transfer_p95": t_p95,
            "n_issue": int((firm["d_paid"] / years > thr).sum()),
            "issue_median": i_med, "issue_p95": i_p95,
            "n_profit_pos": int((firm["profit"] / years > thr).sum()),
            "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5, "all": ok,
            "net_asset_improvement": float(s55.loc[v, "net_asset_improvement"]),
            "resolved": int(s55.loc[v, "resolved_firms(>=2/3 seeds)"]),
        })
    res = pd.DataFrame(rows)
    res.to_csv(args.results / "criteria_evaluation.csv", index=False)

    def mark(b: bool | None) -> str:
        return "該当なし" if b is None else ("○" if b else "×")

    lo4, hi4 = crit["transfer_5y_band"]
    lo5, hi5 = crit["issue_5y_band"]
    lines = [
        "# ステージ30 D.1: 再建群から作った基準による再判定", "",
        f"閾値(`criteria.json`、判定の前にコミット): 基準 2 = {lo2:.3f}〜{hi2:.3f}、"
        f"基準 3 = γ {gamma:.3f}、基準 4 = {lo4:.3f}〜{hi4:.3f}、基準 5 = {lo5:.3f}〜{hi5:.2f}"
        "(5 年、総資産比)。", "",
        "| 版 | 違反(55/57) | 2 RE が増える社 | 3 P の最大 | 4 振替(社・中央値・p95) "
        "| 5 増資(社・中央値・p95) | 1 | 2 | 3 | 4 | 5 | すべて | 純資産の改善 | 解消(12 社中) |",
        "|---|---|---:|---:|---|---|---|---|---|---|---|---|---:|---:|",
    ]
    for r in rows:
        tr = (f"{r['n_transfer']}・{r['transfer_median']:.3f}・{r['transfer_p95']:.3f}"
              if r["n_transfer"] else "0")
        iss = (f"{r['n_issue']}・{r['issue_median']:.3f}・{r['issue_p95']:.3f}"
               if r["n_issue"] else "0")
        lines.append(
            f"| {LABELS[r['version']]} | {r['violations55']}/{r['violations57']} "
            f"| {r['share_re_up']:.1%} | {r['profit_max']:+.3f} | {tr} | {iss} "
            f"| {mark(r['c1'])} | {mark(r['c2'])} | {mark(r['c3'])} | {mark(r['c4'])} "
            f"| {mark(r['c5'])} | **{mark(r['all'])}** | {r['net_asset_improvement']:.3f} "
            f"| {r['resolved']} |"
        )
    (args.results / "criteria_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
