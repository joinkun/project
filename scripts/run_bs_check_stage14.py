#!/usr/bin/env python
"""ステージ14 §A: 推奨設定(最近傍・W1・PCFM・R1)のBS表と、8項目の会計的チェック。

torch/lightgbm/sklearn/POTを使わない(numpy/pandas/yaml)。ステージ13の
生成結果(data/interim/stage13/nearest_na0__W1_uniform__seed*.parquet)を読む。

出力(data/interim/stage14/): bs_nn/*.md(10社)、checks_nn.csv、check_flags_nn.csv
使い方:
    uv run python scripts/run_bs_check_stage14.py
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import (
    CHECK_NAMES,
    EXTRA_CHECK_NAMES,
    evaluate_checks,
    evaluate_extra_checks,
    subtotal_members,
)
from fsgen.models.industry import load_industry_config
from fsgen.viz.balance_sheet import build_bs_lines, load_labels, render_markdown

_PATTERN = "data/interim/stage13/nearest_na0__W1_uniform__seed*.parquet"
_TARGETS = Path("data/interim/stage13/full_targets_nearest_na0.csv")
_OUT = Path("data/interim/stage14")
_PREV = ("S100TBIV", "S100TODR", "S100VIQU", "S100VFZB", "S100VI0I")


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen16", default=None,
                    help="ステージ16以降の生成結果 TAG:CONFIG(例: nearest_na0:W1_S5_a05_b03)")
    ap.add_argument("--gen16-dir", default="data/interim/stage16")
    ap.add_argument("--proj", default=None, help="proj_<X>.parquet(再射影)のX。省略時はステージ13")
    ap.add_argument("--stage", default="S0")
    ap.add_argument("--label", default="nn")
    ap.add_argument("--targets", default=str(_TARGETS), help="対応先のfull_targets_*.csv")
    ap.add_argument("--out-dir", default=str(_OUT))
    args = ap.parse_args()
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    labels_map = load_labels(Path("conf/accounts/labels_ja.yaml"))
    ind = load_industry_config(Path("conf/accounts/industry.yaml"))
    m = subtotal_members(spec, names)
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    info = agg.set_index("docID")[["edinetCode", "total_assets"]]

    if args.gen16 is not None:
        g_tag, g_cfg = args.gen16.split(":")
        files = sorted(glob.glob(f"{args.gen16_dir}/{g_tag}__seed*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = df[(df["method"] == "PCFM") & (df["config"] == g_cfg)]
    elif args.proj is None:
        df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(_PATTERN))], ignore_index=True)
        df = df[df["method"] == "PCFM"]
    else:
        df = pd.read_parquet(Path(args.out_dir) / f"proj_{args.proj}.parquet")
        df = df[df["stage"] == args.stage]
    x0_cols = [f"x0_{n}" for n in names]
    x1_cols = [f"x1_{n}" for n in names]

    # --- 13項目(8項目+追加5項目)のチェック: 全68社(seed0)+ 3seedのいずれかで違反した社数 ---
    all_names = [*CHECK_NAMES, *EXTRA_CHECK_NAMES]

    def flags_of(d: pd.DataFrame) -> np.ndarray:
        x0, x1 = d[x0_cols].to_numpy(), d[x1_cols].to_numpy()
        return np.column_stack([evaluate_checks(x0, x1, spec, names),
                                evaluate_extra_checks(x1, x0, spec, names)])

    s0 = df[df["seed"] == 0].sort_values("docID").reset_index(drop=True)
    flags0 = flags_of(s0)
    allseed = df.sort_values(["docID", "seed"]).reset_index(drop=True)
    any_seed = pd.DataFrame(flags_of(allseed), columns=all_names).assign(docID=allseed["docID"])
    n_any = any_seed.groupby("docID").any().sum()
    summ = pd.DataFrame([{
        "check": nm, "n_violations": int(flags0[:, j].sum()),
        "n_violations_any_of_3seeds": int(n_any[nm]),
        "docIDs": ",".join(d for d, f in zip(s0["docID"], flags0[:, j], strict=True) if f),
    } for j, nm in enumerate(all_names)])
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    summ.to_csv(Path(args.out_dir) / f"checks_{args.label}.csv", index=False)
    pd.DataFrame(flags0, columns=all_names).assign(docID=s0["docID"]).to_csv(
        Path(args.out_dir) / f"check_flags_{args.label}.csv", index=False
    )
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 90)
    print(f"=== 13項目チェック({args.label}、68社、seed0) ===")
    cols = ["check", "n_violations", "n_violations_any_of_3seeds", "docIDs"]
    print(summ[cols].to_string(index=False))
    print(f"\nいずれかで違反した企業(seed0): {int(flags0.any(axis=1).sum())}/68社")

    # --- §A.1: BS表10社 ---
    rng = np.random.default_rng(0)
    others = sorted(set(s0["docID"]) - set(_PREV))
    chosen = list(_PREV) + [str(d) for d in rng.choice(others, size=5, replace=False)]
    tg = pd.read_csv(args.targets).set_index("source_docID")
    out_dir = Path(args.out_dir) / f"bs_{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for doc in chosen:
        r = s0[s0["docID"] == doc].iloc[0]
        asset = float(r["assets_start"])
        xs = r[x0_cols].to_numpy(dtype=float) * asset
        xe = r[x1_cols].to_numpy(dtype=float) * asset
        lines = build_bs_lines(spec, names, [xs, xe], labels_map)
        t = tg.loc[doc]
        t_code = info.loc[t["target_docID"], "edinetCode"]
        s_code = info.loc[doc, "edinetCode"]
        t_ind = ind.labels([str(t_code)])[0]
        s_ind = ind.labels([str(s_code)])[0]
        t_assets = float(info.loc[t["target_docID"], "total_assets"])
        head = (
            f"始点の業種: {s_ind} / 対応先: {t['target_name']}(docID {t['target_docID']}、"
            f"業種 {t_ind}、RE/資産 {t['target_RE_over_assets']:.2f}、"
            f"総資産 {t_assets / 1e6:,.0f}百万円、始点比 {t_assets / asset:.2f}倍)"
        )
        md = render_markdown(lines, (0.0, 1.0), f"{r['filerName']}", doc)
        (out_dir / f"{doc}.md").write_text(head + "\n\n" + md, encoding="utf-8")
        g = {k: float(v) for k, v in {
            "assets": xs @ m["jppfs_cor_Assets"], "assets_end": xe @ m["jppfs_cor_Assets"],
            "liab": xs @ m["jppfs_cor_Liabilities"], "liab_end": xe @ m["jppfs_cor_Liabilities"],
            "na": xs @ m["jppfs_cor_NetAssets"], "na_end": xe @ m["jppfs_cor_NetAssets"],
        }.items()}
        i_cash = names.index("jppfs_cor_CashAndDeposits")
        rows.append({
            "docID": doc, "企業": str(r["filerName"])[:14], "業種": s_ind,
            "対応先": str(t["target_name"])[:12], "対応先業種": t_ind,
            "対応先RE/資産": t["target_RE_over_assets"],
            "資産": f"{g['assets'] / 1e6:,.0f}→{g['assets_end'] / 1e6:,.0f}",
            "負債": f"{g['liab'] / 1e6:,.0f}→{g['liab_end'] / 1e6:,.0f}",
            "純資産": f"{g['na'] / 1e6:,.0f}→{g['na_end'] / 1e6:,.0f}",
            "現金": f"{xs[i_cash] / 1e6:,.0f}→{xe[i_cash] / 1e6:,.0f}",
        })
    print("\n=== §A.1 BS表10社(百万円、始点→終点)。前回5社=先頭5行 ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nBS表: {out_dir}/<docID>.md")


if __name__ == "__main__":
    main()
