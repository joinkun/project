#!/usr/bin/env python
"""ステージ14 §B/C: 段階別・案別の、会計的チェック違反数(主指標)と各指標(副指標)。

lightgbm/sklearn/scipyのプロセス(torch/cvxpy/POTを使わない)。
run_projection_stage14.pyのparquetを読む。

主指標: A.2の8項目の違反社数 + 総負債が負の社数。副指標: proximity・validity等。
B.2: infeasible件数、動けない企業数(始点からの移動<0.02)、アクティブ制約の集計。

使い方:
    uv run python scripts/run_metrics_stage14.py --files nearest_na0_all:R1 \\
        --out stageB
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import glob  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fsgen.accounting.constraints import (  # noqa: E402
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.plausibility import (  # noqa: E402
    CHECK_NAMES,
    EXTRA_CHECK_NAMES,
    evaluate_checks,
    evaluate_extra_checks,
    subtotal_members,
)
from fsgen.evaluation.classifiers import train_classifiers  # noqa: E402
from fsgen.models.cost import cost_vectors, load_cost_config  # noqa: E402

_DIR = Path("data/interim/stage14")
_STUCK_MOVE = 0.02  # 始点からの移動(tilde空間のL2)がこれ未満なら「動けない」


def load_real(names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    lab = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str})
    lab = lab[pd.to_numeric(lab["label"], errors="coerce").notna()]
    m = agg.merge(lab[["docID", "label"]], on="docID", how="inner")
    m = m[m["total_assets"].notna() & (m["total_assets"] != 0)]
    x = m[list(names)].fillna(0.0).to_numpy(dtype=float)
    x = x / m["total_assets"].to_numpy(dtype=float)[:, None]
    return x, m["label"].astype(int).to_numpy() == 2


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", required=True, help="proj_<X>.parquetのX:ラベル をカンマ区切り")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    m = subtotal_members(spec, names)
    cc = load_cost_config(Path("conf/accounts/costs.yaml"))
    w_plus, w_minus = cost_vectors(cc, names, non_actionable_mode="keep")
    x_all, is2 = load_real(names)
    clfs = train_classifiers(x_all, is2, seed=42)
    x0c = [f"x0_{n}" for n in names]
    x1c = [f"x1_{n}" for n in names]
    i_cash = names.index("jppfs_cor_CashAndDeposits")
    signs = [b.sign for b in spec.basis]

    rows = []
    for spec_s in args.files.split(","):
        fname, label = spec_s.split(":")
        df = pd.read_parquet(_DIR / f"proj_{fname}.parquet")
        for stage, sub in df.groupby("stage", sort=False):
            sub = sub.sort_values(["seed", "docID"]).reset_index(drop=True)
            x0, x1 = sub[x0c].to_numpy(), sub[x1c].to_numpy()
            flags = evaluate_checks(x0, x1, spec, names)
            s0 = (sub["seed"] == 0).to_numpy()
            extra = evaluate_extra_checks(x1, x0, spec, names)
            move = np.linalg.norm(x1 - x0, axis=1)
            delta = x1 - x0
            viol = np.zeros(len(sub), dtype=int)
            for j, sg in enumerate(signs):
                if sg == "nonneg":
                    viol += (x1[:, j] < -1e-4).astype(int)
                elif sg == "nonpos":
                    viol += (x1[:, j] > 1e-4).astype(int)
            a0 = x0 @ m["jppfs_cor_Assets"]
            a1 = x1 @ m["jppfs_cor_Assets"]
            na0, na1 = x0 @ m["jppfs_cor_NetAssets"], x1 @ m["jppfs_cor_NetAssets"]
            row: dict[str, object] = {"案": label, "段階": stage}
            for j, nm in enumerate(CHECK_NAMES):
                row[nm] = int(flags[s0, j].sum())
            for j, nm in enumerate(EXTRA_CHECK_NAMES):
                row[nm] = int(extra[s0, j].sum())
            row["致命的(1〜4,9)"] = int((flags[s0][:, :4].any(axis=1) | extra[s0, 0]).sum())
            row["何らか違反(1〜10)"] = int((flags[s0].any(axis=1) | extra[s0].any(axis=1)).sum())
            row["infeasible(204中)"] = int((sub["status"] == "infeasible").sum())
            row["動けない(移動<0.02)"] = int((move < _STUCK_MOVE).sum())
            row["proximity"] = float(np.median(move))
            row["prox_cost_w"] = float(
                np.median(np.sum(np.where(delta >= 0, w_plus, w_minus) * np.abs(delta), axis=1))
            )
            row["総資産変化"] = float(np.median((a1 - a0) / a0))
            row["純資産変化/資産"] = float(np.median((na1 - na0) / a0))
            j_re = names.index("jppfs_cor_RetainedEarnings")
            re_d = x1[:, j_re] - x0[:, j_re]
            row["RE変化/資産(中央値)"] = float(np.median(re_d))
            row["RE変化/資産(|.|90%)"] = float(np.percentile(np.abs(re_d), 90))
            row["現金ゼロ率"] = float(np.mean(x1[:, i_cash] / a0 < 1e-3))
            row["validity_lgbm"] = float(
                np.median(1 - clfs["lightgbm"].predict_proba_class2(x1))
            )
            row["恒等式残差"] = float(np.max(np.abs(x1 @ c)))
            row["符号違反(基礎)"] = int(viol.sum())
            act = Counter(
                a for s in sub["active"] for a in (s.split(";") if s else [])
            )
            row["拘束制約Top"] = "; ".join(f"{k}:{v}" for k, v in act.most_common(4))
            rows.append(row)
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.float_format", "{:.3g}".format)
    out.to_csv(_DIR / f"summary_{args.out}.csv", index=False)
    chk = [*CHECK_NAMES, *EXTRA_CHECK_NAMES, "致命的(1〜4,9)", "何らか違反(1〜10)"]
    print("=== 会計的チェックの違反社数(68社、seed0) ===")
    print(out[["案", "段階", *chk]].to_string(index=False))
    print("\n=== 実行可能性・動きの大きさ・副指標 ===")
    cols = ["案", "段階", "infeasible(204中)", "動けない(移動<0.02)", "proximity", "prox_cost_w",
            "総資産変化", "純資産変化/資産", "RE変化/資産(中央値)",
            "RE変化/資産(|.|90%)", "現金ゼロ率", "validity_lgbm", "恒等式残差",
            "符号違反(基礎)"]
    print(out[cols].to_string(index=False))
    print("\n=== 拘束している制約(204行中の件数) ===")
    print(out[["案", "段階", "拘束制約Top"]].to_string(index=False))

    # S0 の再現確認(ステージ13のPCFM出力と一致するか)
    if glob.glob("data/interim/stage13/nearest_na0__W1_uniform__seed0.parquet"):
        ref = pd.read_parquet("data/interim/stage13/nearest_na0__W1_uniform__seed0.parquet")
        ref = ref[ref["method"] == "PCFM"].set_index("docID")
        first = args.files.split(",")[0].split(":")[0]
        if first.startswith("nearest_na0_all"):
            s0df = pd.read_parquet(_DIR / f"proj_{first}.parquet")
            s0df = s0df[(s0df["stage"] == "S0") & (s0df["seed"] == 0)].set_index("docID")
            d = np.abs(s0df[x1c].to_numpy() - ref.loc[s0df.index, x1c].to_numpy()).max()
            print(f"\nS0再射影とステージ13のPCFM出力の最大差(seed0): {d:.2e}")


if __name__ == "__main__":
    main()
