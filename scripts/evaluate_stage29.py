#!/usr/bin/env python
"""ステージ29 C・D: 各版を A.2 の基準(``results/stage29/criteria.json``)で判定し、R4 の分解を出す。

numpy/pandas のみ。生成(``data/interim/stage29/<版>``)の PCFM の行を読み、企業ごとに seed の
中央値を取る。分解は実データと同じ規則(``decompose_re_change``: 払込資本の減少分を振替)。
基準 1(違反)は ``run_stage25.py`` の出力(``metrics/summary_*.csv``)から読む。

    uv run python scripts/evaluate_stage29.py
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import decompose_re_change

VERSIONS = ("r1", "r3", "r3_cap", "r4", "r4_g040")
LABELS = {
    "r1": "R1(RE 固定)",
    "r3": "R3(固定なし)",
    "r3_cap": "R3 + 上限 0.5",
    "r4": "R4(γ = 0)",
    "r4_g040": "R4(γ = 0.40、参考)",
}
GAMMA_OF = {"r3_cap": 0.5, "r4_g040": 0.40}  # 版ごとの設計上の上限(基準 3 とは別)
_TOL = 1e-6  # 基準 3 の判定(criteria.md に記録した値)
# 「利益部分 = 0」「上限に張り付き」を数える許容(総資産(0)比)。射影 QP の収束判定は企業の規模で
# 正規化した許容なので、総資産比では最大 3e-4 程度の誤差が残る(サイバーステップ)
_COUNT_TOL = 1e-3
_RE = "jppfs_cor_RetainedEarnings"


def load_version(directory: Path, names: tuple[str, ...]) -> pd.DataFrame:
    files = sorted(glob.glob(str(directory / "nearest_na0__seed*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[df["method"] == "PCFM"]
    if df["config"].nunique() != 1:
        raise ValueError(f"{directory}: PCFM の設定が 1 つでない {sorted(df['config'].unique())}")
    x0 = df[[f"x0_{n}" for n in names]].to_numpy()
    x1 = df[[f"x1_{n}" for n in names]].to_numpy()
    transfer, profit = decompose_re_change(x0, x1, names)
    i_re = names.index(_RE)
    out = pd.DataFrame(
        {
            "docID": df["docID"].to_numpy(),
            "filerName": df["filerName"].to_numpy(),
            "seed": df["seed"].to_numpy(),
            "d_re": x1[:, i_re] - x0[:, i_re],
            "transfer": transfer,
            "profit": profit,
            "branch": df["re_branch"].to_numpy() if "re_branch" in df else -1,
        }
    )
    return out


def _q(s: pd.Series, qs: tuple[int, ...] = (5, 25, 50, 75, 95)) -> str:
    return " | ".join(f"{s.quantile(q / 100):+.4f}" for q in qs)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/interim/stage29"))
    ap.add_argument("--results", type=Path, default=Path("results/stage29"))
    args = ap.parse_args()
    crit = json.loads((args.results / "criteria.json").read_text(encoding="utf-8"))
    years = float(crit["years"])
    thr = float(crit["threshold_annual"])
    lo2, hi2 = crit["share_re_up_band"]
    gamma = float(crit["gamma"])
    lo4, hi4 = crit["transfer_5y_band"]
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    s55 = pd.read_csv(args.results / "metrics" / "summary_excl_outliers55.csv").set_index("version")
    s57 = pd.read_csv(args.results / "metrics" / "summary_all57.csv").set_index("version")
    emp = pd.read_csv(args.results / "empirical_gc_firms.csv")

    rows = []
    firm_tables: dict[str, pd.DataFrame] = {}
    for v in VERSIONS:
        df = load_version(args.dir / v, names)
        firm = df.groupby("docID")[["d_re", "transfer", "profit"]].median()
        firm_tables[v] = firm
        up = firm["d_re"] / years > thr
        has_t = firm["transfer"] / years > thr
        t_pos = firm.loc[has_t, "transfer"]
        c1 = int(s55.loc[v, "violations(rows)"]) == 0
        c2 = lo2 <= up.mean() <= hi2
        c3 = bool((firm["profit"] <= gamma + _TOL).all())
        if len(t_pos):
            t_med, t_p95 = float(t_pos.median()), float(t_pos.quantile(0.95))
            c4: bool | None = (lo4 <= t_med <= hi4) and (lo4 <= t_p95 <= hi4)
        else:
            t_med = t_p95 = float("nan")
            c4 = None
        ok = c1 and c2 and c3 and (c4 is not False)
        rows.append(
            {
                "version": v, "violations55": int(s55.loc[v, "violations(rows)"]),
                "violations57": int(s57.loc[v, "violations(rows)"]),
                "share_re_up": float(up.mean()), "profit_max": float(firm["profit"].max()),
                "n_transfer": int(has_t.sum()), "transfer_median": t_med, "transfer_p95": t_p95,
                "c1": c1, "c2": c2, "c3": c3, "c4": c4, "all": ok,
            }
        )
    res = pd.DataFrame(rows)
    res.to_csv(args.results / "criteria_evaluation.csv", index=False)

    def mark(b: bool | None) -> str:
        return "該当なし" if b is None else ("○" if b else "×")

    lines = [
        "# ステージ29 D.1: A.2 の基準による判定", "",
        f"閾値(`criteria.json`、生成前に記録): 基準 2 = {lo2:.2f}〜{hi2:.2f}、"
        f"基準 3 = γ {gamma:g}、基準 4 = {lo4:.3f}〜{hi4:.2f}(5 年、総資産比)。"
        "企業ごとに seed の中央値。", "",
        "| 版 | 1 違反(55 社 / 57 社) | 2 RE が増える社 | 3 利益部分の最大(5 年) "
        "| 4 振替のある社・中央値・p95 | 1 | 2 | 3 | 4 | すべて |",
        "|---|---|---:|---:|---|---|---|---|---|---|",
    ]
    for r in rows:
        tr = (
            f"{r['n_transfer']} 社・{r['transfer_median']:.3f}・{r['transfer_p95']:.3f}"
            if r["n_transfer"] else "0 社"
        )
        lines.append(
            f"| {LABELS[r['version']]} | {r['violations55']} / {r['violations57']} "
            f"| {r['share_re_up']:.1%} | {r['profit_max']:+.4f} | {tr} | {mark(r['c1'])} "
            f"| {mark(r['c2'])} | {mark(r['c3'])} | {mark(r['c4'])} | **{mark(r['all'])}** |"
        )

    # C.3 R4 の分解
    lines += ["", "# C.3 利益剰余金の変化の分解(5 年、総資産比、企業ごとに seed の中央値)", ""]
    lines += [
        "| 系列 | 社数 | 系列の値 | p5 | p25 | p50 | p75 | p95 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    emp5 = emp.assign(transfer=emp["transfer"] * years, profit=emp["profit"] * years,
                      d_re=emp["d_re"] * years)
    for label, frame in (("実データ(年率 × 5)", emp5),
                         *((LABELS[v], firm_tables[v]) for v in ("r4", "r4_g040", "r3_cap", "r3"))):
        for col, cname in (("d_re", "ΔRE"), ("transfer", "振替"), ("profit", "利益部分")):
            lines.append(f"| {label} | {len(frame)} | {cname} | {_q(frame[col])} |")
    lines += [
        "", "| 版 | 社数 | 利益部分 = 0 | 利益部分 > 0 | 上限に張り付き | 振替のある社 "
        "| RE が動かない社 | 分岐 A / B(行) |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for v in ("r4", "r4_g040"):
        f = firm_tables[v]
        g = GAMMA_OF.get(v, gamma)
        df = load_version(args.dir / v, names)
        br = df["branch"].value_counts().to_dict()
        lines.append(
            f"| {LABELS[v]} | {len(f)} | {int((f['profit'].abs() <= _COUNT_TOL).sum())} "
            f"| {int((f['profit'] > _COUNT_TOL).sum())} "
            f"| {int((f['profit'] >= g - _COUNT_TOL).sum())} "
            f"| {int((f['transfer'] / years > thr).sum())} "
            f"| {int((f['d_re'].abs() / years <= thr).sum())} "
            f"| {br.get(0, 0)} / {br.get(1, 0)} |"
        )
    lines += [
        "",
        f"「利益部分 = 0」「上限に張り付き」は許容 {_COUNT_TOL:g}(総資産(0)比)で数えた。",
        f"実データ(50 社): 利益部分 > 0 は {int((emp['profit'] > thr).sum())} 社、"
        f"振替のある社は {int((emp['transfer'] > thr).sum())} 社、RE が増えた社は "
        f"{int((emp['d_re'] > thr).sum())} 社。",
    ]
    (args.results / "criteria_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
