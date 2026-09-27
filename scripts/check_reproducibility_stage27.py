#!/usr/bin/env python
"""ステージ27 A: 乱数種の修正の前後で、主要な結論が保たれるかを確かめる。

lightgbm/sklearn/scipy/cvxpy のプロセス(torch・POT なし)。修正前・修正後のコードをそれぞれ 2 回
実行した生成結果(``data/interim/stage27/{s13,s16}_{before,after}_{a,b}``)を読む。データは現在の
canonical(ステージ21〜22 の統合後)、重みは旧 ``costs.yaml``(ステージ13〜19 と同じ)。

1. CAFM < PCFM(ステージ13 の 4 比較: CAFM_scratch・CAFM_warm × W1_uniform・W3_eq_only、
   validity LightGBM)
2. PCFM > 非学習ベース(prior_work・convex_qp、ステージ10。乱数を使わないので 1 回だけ作る)
3. W1 > W3(ステージ16: W3_S5_a05_b03 − W1_S5_a05_b03 の |総資産変化|)
4. 第 1 層のみ(W1_S0)で 13 項目のいずれかに違反する社数(ステージ17: 63 / 68)
5. 同じコードの 2 回の実行の一致(最大差)

    uv run python scripts/check_reproducibility_stage27.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import glob  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.accounting.plausibility import (  # noqa: E402
    evaluate_checks,
    evaluate_extra_checks,
    subtotal_members,
)
from fsgen.accounting.projection_qp import load_direction_constraints  # noqa: E402
from fsgen.errors import SolverNotConvergedError  # noqa: E402
from fsgen.evaluation.baselines import ConvexQpBaseline, build_prior_work_baseline  # noqa: E402
from fsgen.evaluation.classifiers import train_classifiers  # noqa: E402
from fsgen.evaluation.paired import aggregate_by_company, paired_test  # noqa: E402

RUNS = ("before_a", "before_b", "after_a", "after_b")
_TS = (0.0, 1.0)


def load_real(names: tuple[str, ...]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    lab = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str})
    lab = lab[pd.to_numeric(lab["label"], errors="coerce").notna()]
    m = agg.merge(lab[["docID", "label"]], on="docID", how="inner")
    m = m[m["total_assets"].notna() & (m["total_assets"] != 0)].reset_index(drop=True)
    x = m[list(names)].fillna(0.0).to_numpy(dtype=float)
    return m, x, m["total_assets"].to_numpy(dtype=float)


def _read(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/interim/stage27"))
    ap.add_argument("--out", type=Path, default=Path("results/stage27/reproducibility.md"))
    args = ap.parse_args()

    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    meta, x_all, assets_all = load_real(names)
    labels = meta["label"].astype(int).to_numpy()
    x_tilde = x_all / assets_all[:, None]
    clfs = train_classifiers(x_tilde, labels == 2, seed=42)
    x0c = [f"x0_{n}" for n in names]
    x1c = [f"x1_{n}" for n in names]

    def validity(x1: np.ndarray) -> np.ndarray:
        result: np.ndarray = 1 - clfs["lightgbm"].predict_proba_class2(x1)
        return result

    lines = ["# ステージ27 A: 乱数種の修正の前後で結論が保たれるか", ""]
    lines += [
        "データは現在の canonical(統合後)、重みは旧 costs.yaml(ステージ13〜19 と同じ)。",
        "修正前・修正後のコードをそれぞれ 2 回(a・b)実行した。差は企業単位(n=68、seed の中央値)の",
        "ペア Wilcoxon の差の中央値と 95% ブートストラップ CI。",
        "",
    ]

    # --- 5. 再現性 ---
    lines += ["## 同じコードの 2 回の実行の一致", "", "| 生成 | 修正前 a と b の最大差 | 修正後 |"]
    lines.append("|---|---:|---:|")
    for kind, pat in (("ステージ13(FM/PCFM/CAFM)", "s13"), ("ステージ16", "s16")):
        diffs = []
        for ver in ("before", "after"):
            a = _read(str(args.dir / f"{pat}_{ver}_a" / "*.parquet"))
            b = _read(str(args.dir / f"{pat}_{ver}_b" / "*.parquet"))
            keys = [c for c in ("condition", "config", "method", "seed", "docID") if c in a]
            a, b = (
                a.sort_values(keys).reset_index(drop=True),
                b.sort_values(keys).reset_index(drop=True),
            )
            diffs.append(float(np.abs(a[x1c].to_numpy() - b[x1c].to_numpy()).max()))
        lines.append(f"| {kind} | {diffs[0]:.3g} | {diffs[1]:.3g} |")

    # --- 1. CAFM − PCFM ---
    lines += [
        "",
        "## 1. CAFM − PCFM(validity LightGBM、ステージ13)",
        "",
        "| 条件 | CAFM | 実行 | 差の中央値 | 95% CI | p | CAFM が劣る(CI < 0) |",
        "|---|---|---|---:|---|---:|---|",
    ]
    s13 = {r: _read(str(args.dir / f"s13_{r}" / "*.parquet")) for r in RUNS}
    for cond in ("W1_uniform", "W3_eq_only"):
        for cafm in ("CAFM_scratch", "CAFM_warm"):
            for r in RUNS:
                df = s13[r]
                df = df[df["condition"] == cond].copy()
                df["v"] = validity(df[x1c].to_numpy())
                agg = aggregate_by_company(df, "v")
                t = paired_test(agg[cafm].to_numpy(), agg["PCFM"].to_numpy())
                worse = "○" if t.ci_high < 0 else "×"
                lines.append(
                    f"| {cond} | {cafm} | {r} | {t.median_diff:+.4f} "
                    f"| [{t.ci_low:+.4f}, {t.ci_high:+.4f}] | {t.p_value:.2g} | {worse} |"
                )

    # --- 2. PCFM − 非学習ベース ---
    idx0, idx2 = np.where(labels == 0)[0], np.where(labels == 2)[0]
    median0 = np.median(x_all[idx0], axis=0)
    prior_work = build_prior_work_baseline(median0, names)
    dcs = load_direction_constraints(Path("conf/accounts/core.yaml"))
    convex_qp = ConvexQpBaseline(
        median_tilde=np.median(x_tilde[idx0], axis=0), spec=spec, direction_constraints=dcs
    )
    base_rows = []
    base_failed: dict[str, list[str]] = {"prior_work": [], "convex_qp": []}
    for i in idx2:
        for name, bl in (("prior_work", prior_work), ("convex_qp", convex_qp)):
            try:
                path = bl.generate(x_all[i], float(assets_all[i]), _TS)
            except SolverNotConvergedError:
                # 厳しい収束検査(ステージ13)に通らない社は比較から除き、件数を報告する
                base_failed[name].append(str(meta.at[i, "docID"]))
                continue
            base_rows.append(
                {
                    "docID": meta.at[i, "docID"],
                    "method": name,
                    "x0": x_tilde[i],
                    "x1": path[-1] / assets_all[i],
                }
            )
    base = pd.DataFrame(base_rows)
    bx0, bx1 = np.stack(base["x0"].to_numpy()), np.stack(base["x1"].to_numpy())
    base["v"] = validity(bx1)
    base["prox"] = np.linalg.norm(bx1 - bx0, axis=1)
    ba0, ba1 = bx0 @ members["jppfs_cor_Assets"], bx1 @ members["jppfs_cor_Assets"]
    base["aac"] = np.abs((ba1 - ba0) / ba0)
    lines += [
        "",
        "## 2. PCFM(W1_uniform)− 非学習ベース",
        "",
        "非学習ベースは乱数を使わない。射影 QP の収束検査に通らず除いた社: "
        + "、".join(f"{k} {len(v)} 社({', '.join(v) or 'なし'})" for k, v in base_failed.items()),
        "",
        "| ベース | 指標 | 実行 | PCFM の中央値 | ベースの中央値 | 差(PCFM − ベース) | 95% CI "
        "| PCFM が良い |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    metric_defs = (
        ("v", "validity LightGBM", "大"),
        ("prox", "等重み proximity", "小"),
        ("aac", "|総資産変化|", "小"),
    )
    for bname in ("prior_work", "convex_qp"):
        for col, label, better_dir in metric_defs:
            bv = base[base["method"] == bname].set_index("docID")[col]
            for r in RUNS:
                df = s13[r]
                df = df[(df["condition"] == "W1_uniform") & (df["method"] == "PCFM")].copy()
                x0p, x1p = df[x0c].to_numpy(), df[x1c].to_numpy()
                if col == "v":
                    df[col] = validity(x1p)
                elif col == "prox":
                    df[col] = np.linalg.norm(x1p - x0p, axis=1)
                else:
                    a0p = x0p @ members["jppfs_cor_Assets"]
                    df[col] = np.abs((x1p @ members["jppfs_cor_Assets"] - a0p) / a0p)
                pv = df.groupby("docID")[col].median()
                pv = pv[pv.index.isin(bv.index)]
                t = paired_test(pv.to_numpy(), bv.reindex(pv.index).to_numpy())
                ok = t.ci_low > 0 if better_dir == "大" else t.ci_high < 0
                lines.append(
                    f"| {bname} | {label}({better_dir}が良い) | {r} | {pv.median():.3f} "
                    f"| {bv.median():.3f} | {t.median_diff:+.4f} "
                    f"| [{t.ci_low:+.4f}, {t.ci_high:+.4f}] | {'○' if ok else '×'} |"
                )

    # --- 3・4. ステージ16・17 ---
    lines += [
        "",
        "## 3. W3 − W1(|総資産変化|、ステージ16。正なら W3 が悪い)と 4. 第 1 層のみの違反",
        "",
        "| 実行 | W3 − W1 差の中央値 | 95% CI | W1 が上(CI > 0) | 第 1 層のみで違反した社"
        "(seed 0) | 3 seed のいずれかで違反 |",
        "|---|---:|---|---|---:|---:|",
    ]
    for r in RUNS:
        df = _read(str(args.dir / f"s16_{r}" / "*.parquet"))
        df = df[df["method"] == "PCFM"].copy()
        x0, x1 = df[x0c].to_numpy(), df[x1c].to_numpy()
        a0, a1 = x0 @ members["jppfs_cor_Assets"], x1 @ members["jppfs_cor_Assets"]
        df["abs_asset_change"] = np.abs((a1 - a0) / a0)
        agg = aggregate_by_company(df, "abs_asset_change", group_cols=("config",))
        t = paired_test(agg["W3_S5_a05_b03"].to_numpy(), agg["W1_S5_a05_b03"].to_numpy())
        l1 = df[df["config"] == "W1_S0"].reset_index(drop=True)
        f = np.column_stack(
            [
                evaluate_checks(l1[x0c].to_numpy(), l1[x1c].to_numpy(), spec, names),
                evaluate_extra_checks(l1[x1c].to_numpy(), l1[x0c].to_numpy(), spec, names),
            ]
        ).any(axis=1)
        s0 = int(f[l1["seed"].to_numpy() == 0].sum())
        any3 = int(pd.Series(f).groupby(l1["docID"]).any().sum())
        better = "○" if t.ci_low > 0 else "×"
        lines.append(
            f"| {r} | {t.median_diff:+.4f} | [{t.ci_low:+.4f}, {t.ci_high:+.4f}] | {better} "
            f"| {s0} / 68 | {any3} / 68 |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
