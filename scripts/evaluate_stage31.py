#!/usr/bin/env python
"""ステージ31 B: 標本誤差を入れた基準(``results/stage31/criteria.json``)で全版を判定する。

numpy/pandas のみ。手順はステージ30(``evaluate_stage30.py``)と同じで、帯だけを替えた:
基準 2 = Wilson 95% 区間、基準 3 = 5 × p90 のブートストラップ区間の上端、基準 4・5 = 広げた帯。
上限に達した社数も数える(R3 + 上限は ΔRE、R4 は利益部分 P が γ に達した社)。

    uv run python scripts/evaluate_stage31.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from evaluate_stage29 import load_version
from evaluate_stage30 import _band_check

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names

VERSIONS = ("r1", "r3", "r3_cap", "r3_cap076", "r4", "r4_g040", "r4_g076", "r4_g239")
LABELS = {
    "r1": "R1(RE 固定)",
    "r3": "R3(固定なし)",
    "r3_cap": "R3 + 上限 0.5",
    "r3_cap076": "**R3 + 上限 0.757(最終)**",
    "r4": "R4(γ = 0)",
    "r4_g040": "R4(γ = 0.40)",
    "r4_g076": "R4(γ = 0.757、参考)",
    "r4_g239": "R4(γ = 2.39)",
}
CAPS = {  # 版 -> (上限をかける量, γ)
    "r3_cap": ("d_re", 0.5),
    "r3_cap076": ("d_re", None),
    "r4": ("profit", 0.0),
    "r4_g040": ("profit", 0.40),
    "r4_g076": ("profit", None),
    "r4_g239": ("profit", 2.390866719697921),
}
_TOL = 1e-6
_COUNT_TOL = 1e-3
_CS, _CSUR = "jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus"


def firm_table(directory: Path, names: tuple[str, ...]) -> pd.DataFrame:
    df = load_version(directory, names)
    raw = pd.concat(
        [pd.read_parquet(directory / f"nearest_na0__seed{s}.parquet") for s in (0, 1, 2)]
    )
    raw = raw[raw["method"] == "PCFM"]
    df["d_paid"] = (
        raw[f"x1_{_CS}"] + raw[f"x1_{_CSUR}"] - raw[f"x0_{_CS}"] - raw[f"x0_{_CSUR}"]
    ).to_numpy()
    return df.groupby("docID")[["d_re", "transfer", "profit", "d_paid"]].median()


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/interim/stage31"))
    ap.add_argument("--results", type=Path, default=Path("results/stage31"))
    args = ap.parse_args()
    crit = json.loads((args.results / "criteria.json").read_text(encoding="utf-8"))
    years, thr, gamma = float(crit["years"]), float(crit["threshold_annual"]), float(crit["gamma"])
    lo2, hi2 = crit["share_band_wilson"]
    p_hi = float(crit["gamma_5y_p90_all_boot95"][1])
    band4 = [crit["transfer_5y"]["band_lo"], crit["transfer_5y"]["band_hi"]]
    band5 = [crit["issue_5y"]["band_lo"], crit["issue_5y"]["band_hi"]]
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    s55 = pd.read_csv(args.results / "metrics" / "summary_excl_outliers55.csv").set_index("version")
    s57 = pd.read_csv(args.results / "metrics" / "summary_all57.csv").set_index("version")

    rows = []
    for v in VERSIONS:
        firm = firm_table(args.dir / v, names)
        up = firm["d_re"] / years > thr
        c1 = int(s55.loc[v, "violations(rows)"]) == 0
        c2 = lo2 <= up.mean() <= hi2
        c3 = bool((firm["profit"] <= p_hi + _TOL).all())
        c4, t_med, t_p95 = _band_check(firm.loc[firm["transfer"] / years > thr, "transfer"], band4)
        c5, i_med, i_p95 = _band_check(firm.loc[firm["d_paid"] / years > thr, "d_paid"], band5)
        ok = c1 and c2 and c3 and (c4 is not False) and (c5 is not False)
        at_cap: int | None = None
        if v in CAPS:
            col, g = CAPS[v]
            g = gamma if g is None else g
            at_cap = int((firm[col] >= g - _COUNT_TOL).sum())
        rows.append({
            "version": v, "violations55": int(s55.loc[v, "violations(rows)"]),
            "violations57": int(s57.loc[v, "violations(rows)"]),
            "n_up": int(up.sum()), "n_firms": len(firm), "share_re_up": float(up.mean()),
            "profit_max": float(firm["profit"].max()),
            "n_transfer": int((firm["transfer"] / years > thr).sum()),
            "transfer_median": t_med, "transfer_p95": t_p95,
            "n_issue": int((firm["d_paid"] / years > thr).sum()),
            "issue_median": i_med, "issue_p95": i_p95, "at_cap": at_cap,
            "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5, "all": ok,
            "net_asset_improvement": float(s55.loc[v, "net_asset_improvement"]),
            "resolved": int(s55.loc[v, "resolved_firms(>=2/3 seeds)"]),
        })
    pd.DataFrame(rows).to_csv(args.results / "criteria_evaluation.csv", index=False)

    def mark(b: bool | None) -> str:
        return "該当なし" if b is None else ("○" if b else "×")

    lines = [
        "# ステージ31 B.1: 標本誤差を入れた基準による再判定", "",
        "違反は外れ値 2 社を除く 55 社(と 57 社)。基準 2〜5 はステージ29・30 と同じく"
        "除外後の 57 社で数える。", "",
        f"基準 2 = Wilson {lo2:.3f}〜{hi2:.3f}、基準 3 = P の最大 ≤ {p_hi:.2f}、"
        f"基準 4 = {band4[0]:.3f}〜{band4[1]:.3f}、基準 5 = {band5[0]:.3f}〜{band5[1]:.2f}"
        f"(5 年、総資産比)。γ(最終設定の上限)= {gamma:.3f}。", "",
        "| 版 | 違反(55/57) | 2 RE が増える社 | 3 P の最大 | 4 振替(社・中央値・p95) "
        "| 5 増資(社・中央値・p95) | 上限に達した社 | 1 | 2 | 3 | 4 | 5 | すべて "
        "| 純資産の改善 | 解消(12 社中) |",
        "|---|---|---:|---:|---|---|---:|---|---|---|---|---|---|---:|---:|",
    ]
    for r in rows:
        tr = (f"{r['n_transfer']}・{r['transfer_median']:.3f}・{r['transfer_p95']:.3f}"
              if r["n_transfer"] else "0")
        iss = (f"{r['n_issue']}・{r['issue_median']:.3f}・{r['issue_p95']:.3f}"
               if r["n_issue"] else "0")
        cap = "—" if r["at_cap"] is None else str(r["at_cap"])
        lines.append(
            f"| {LABELS[r['version']]} | {r['violations55']}/{r['violations57']} "
            f"| {r['n_up']}/{r['n_firms']} ({r['share_re_up']:.1%}) | {r['profit_max']:+.3f} "
            f"| {tr} | {iss} "
            f"| {cap} | {mark(r['c1'])} | {mark(r['c2'])} | {mark(r['c3'])} | {mark(r['c4'])} "
            f"| {mark(r['c5'])} | **{mark(r['all'])}** | {r['net_asset_improvement']:.3f} "
            f"| {r['resolved']} |"
        )
    (args.results / "criteria_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
