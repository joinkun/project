#!/usr/bin/env python
"""ステージ11-A: 指標計算専用スクリプト(lightgbm/sklearnのみ、torchは一切importしない)。

run_generation.py(torch専用プロセス)が書き出したparquetを読み込み、
validity(§A.5再現確認)と識別AUC(§B.2)を計算する。状態ベクトルは
parquetの時点ですでに総資産で正規化済み(tilde空間)なので、ここで
Assetsによる再正規化を絶対に行わないこと(ステージ10.6でこの
二重正規化バグを一度踏んでいる——§1.2参照)。

本スクリプトはtorchを一切importしない(§A.4でvmmap確認済み)が、
lightgbm(Homebrew版libomp.dylib)とsklearn(独自同梱のlibomp.dylib)は
両方importするため、libomp.dylibが2つ同一プロセスに同居する
(vmmapで確認済み)。lightgbm側はclassifiers.pyで既にn_jobs=1固定済みで、
sklearn側(RandomForest/LogisticRegression)もn_jobs既定値=1のため
実測ではクラッシュしないが、念のため防御的にOpenMPのスレッド数を
プロセス全体で1に制限しておく(``KMP_DUPLICATE_LIB_OK``は2つ目の
ランタイムを無視して続行させるだけで危険なため使わない、というA.3の
方針に反しない範囲での追加対策)。

使い方:
    uv run python scripts/run_metrics.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.evaluation.classifiers import train_classifiers  # noqa: E402
from fsgen.evaluation.identification import identification_auc_repeated  # noqa: E402

_DEFAULT_CANONICAL = Path("conf/accounts/core.yaml")
_DEFAULT_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_DEFAULT_LABELS_CSV = Path("data/interim/labels/labels.csv")
_DEFAULT_GEN_DIR = Path("data/interim/generation_11a")

_SEED = 42
_N_SPLITS = 5
_N_REPEATS = 20  # §B.2: class0の抽出を20回変えて分布を出す

# §A.5: ステージ10.5で報告された既報数値(W1条件、5-fold×3-seedの中央値)
_REPORTED = {
    ("PCFM", "proximity_l2"): 0.347,
    ("PCFM", "validity"): 0.948,
    ("CAFM", "proximity_l2"): 0.395,
    ("CAFM", "validity"): 0.397,
}
_REPRODUCE_TOL = 0.05  # 絶対誤差でこれを超えたら「再現しない」と判定する


def load_real_dataset(
    aggregated_csv: Path, labels_csv: Path, names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """(x_tilde, is_class2) を返す(§A.5のvalidity分類器学習・§B.2の識別AUC用)。"""
    agg = pd.read_csv(aggregated_csv, dtype={"docID": str})
    is_financial = agg["is_financial"].astype(str)
    agg = agg[(agg["parse_status"] == "ok") & (is_financial != "True")]
    labels_df = pd.read_csv(labels_csv, dtype={"docID": str})
    labels_df = labels_df[pd.to_numeric(labels_df["label"], errors="coerce").notna()]
    merged = agg.merge(labels_df[["docID", "label"]], on="docID", how="inner")
    merged = merged[merged["total_assets"].notna() & (merged["total_assets"] != 0)]
    x = merged[list(names)].fillna(0.0).to_numpy(dtype=float)
    assets = merged["total_assets"].to_numpy(dtype=float)
    labels = merged["label"].astype(int).to_numpy()
    x_tilde = x / assets[:, None]
    is_class2 = labels == 2
    return x_tilde, is_class2


def discriminative_auc_repeated(
    generated: np.ndarray,
    x0_tilde: np.ndarray,
    n_repeats: int,
    n_splits: int,
    seed: int,
) -> dict[str, list[float]]:
    """共通実装(``fsgen.evaluation.identification``、収束チェック付き)への薄いラッパー。"""
    res = identification_auc_repeated(
        generated, x0_tilde, n_repeats, n_splits=n_splits, seed=seed
    )
    return {k: list(v) for k, v in res.items()}


def main() -> None:
    spec = load_canonical_spec(_DEFAULT_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)

    gen_path = _DEFAULT_GEN_DIR / "generated_endpoints.parquet"
    if not gen_path.exists():
        raise FileNotFoundError(
            f"{gen_path} が見つかりません。先に "
            "`uv run python scripts/precompute_transport.py` と "
            "`uv run python scripts/run_generation.py` を実行してください。"
        )
    gen_df = pd.read_parquet(gen_path)
    print(f"読み込み: {gen_path}({len(gen_df)}行)")

    x0_tilde, is_class2 = load_real_dataset(
        _DEFAULT_AGGREGATED_CSV, _DEFAULT_LABELS_CSV, names
    )
    x_real0_tilde = x0_tilde[~is_class2]

    # --- validity分類器(§A.5・stage8.5以降と同じ向き: health = 1 - P(class2)) ---
    print("\n=== validity分類器の学習 ===")
    classifiers = train_classifiers(x0_tilde, is_class2, seed=_SEED)
    lgb_clf = classifiers["lightgbm"]

    x0_cols = [f"x0_{n}" for n in names]
    x1_cols = [f"x1_{n}" for n in names]

    def _proximity_and_validity(sub: pd.DataFrame) -> pd.DataFrame:
        x0 = sub[x0_cols].to_numpy()
        x1 = sub[x1_cols].to_numpy()
        # x0/x1はすでに総資産で正規化済み(tilde空間)なので、Assetsで割り直さない。
        proximity = np.linalg.norm(x1 - x0, axis=1)
        health = 1.0 - lgb_clf.predict_proba_class2(x1)
        out = sub[["seed", "fold", "docID", "method"]].copy()
        out["proximity_l2"] = proximity
        out["validity"] = health
        return out

    metrics_df = _proximity_and_validity(gen_df)
    metrics_df.to_csv(_DEFAULT_GEN_DIR / "metrics.csv", index=False)

    # --- §A.5: 既報数値の再現確認(PCFM/CAFM、W1条件、3 seed×5foldの中央値) ---
    print("\n=== §A.5: 既報数値の再現確認 ===")
    reproduction_rows = []
    for method in ("PCFM", "CAFM"):
        sub = metrics_df[metrics_df["method"] == method]
        for metric_name in ("proximity_l2", "validity"):
            observed = float(sub[metric_name].median())
            reported = _REPORTED[(method, metric_name)]
            diff = observed - reported
            reproduced = abs(diff) <= _REPRODUCE_TOL
            reproduction_rows.append({
                "method": method, "metric": metric_name,
                "reported": reported, "observed": observed, "diff": diff,
                "reproduced": reproduced,
            })
            mark = "OK" if reproduced else "**再現しない**"
            print(
                f"  {method:<6} {metric_name:<14} 既報={reported:.3f} "
                f"実測={observed:.3f} 差={diff:+.3f} [{mark}]"
            )
    pd.DataFrame(reproduction_rows).to_csv(_DEFAULT_GEN_DIR / "reproduction_check.csv", index=False)

    # --- §B.2: 識別AUC(全手法、LightGBM/LogReg、5-fold CV×20回) ---
    print(f"\n=== §B.2: 識別AUC(全手法、5-fold CV×{_N_REPEATS}回) ===")
    seed0 = gen_df[gen_df["seed"] == 0]
    method_names = ("prior_work", "convex_qp", "stage9_proposed", "FM", "PCFM", "CAFM")
    auc_summary_rows = []
    for method in method_names:
        sub = seed0[seed0["method"] == method]
        generated = sub[x1_cols].to_numpy()
        n_points = len(generated)
        auc_results = discriminative_auc_repeated(
            generated, x_real0_tilde, n_repeats=_N_REPEATS, n_splits=_N_SPLITS, seed=_SEED
        )
        for clf_name, scores in auc_results.items():
            arr = np.array(scores)
            auc_summary_rows.append({
                "method": method, "classifier": clf_name, "n_points": n_points,
                "auc_mean": float(arr.mean()), "auc_std": float(arr.std()),
                "auc_min": float(arr.min()), "auc_max": float(arr.max()),
            })
        print(
            f"  {method:<16} n={n_points:<3} "
            f"LightGBM AUC={np.mean(auc_results['lightgbm']):.3f}"
            f"±{np.std(auc_results['lightgbm']):.3f}  "
            f"LogReg AUC={np.mean(auc_results['logreg']):.3f}"
            f"±{np.std(auc_results['logreg']):.3f}"
        )
    auc_df = pd.DataFrame(auc_summary_rows)
    auc_df.to_csv(_DEFAULT_GEN_DIR / "discriminative_auc.csv", index=False)

    print(f"\n出力先: {_DEFAULT_GEN_DIR}")


if __name__ == "__main__":
    main()
