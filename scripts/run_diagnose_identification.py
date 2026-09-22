#!/usr/bin/env python
"""ステージ13 §D: 識別AUC≈0.99の原因分解(lightgbm/sklearn/matplotlibのプロセス、torchなし)。

生成された終点(x1、総資産で正規化済み)と class 0 実データを比べ、
D.1 どの科目が「生成物らしさ」を決めているか(LightGBMのgain重要度、
    標準化したロジスティック回帰の係数の絶対値)
D.2 上位10科目の単変量分布(Kolmogorov-Smirnov統計量、ヒストグラムの重ね描き)
D.3 科目間の相関構造(Frobenius距離、最も食い違う科目ペア上位10)
を出す。制約は会計恒等式と符号だけで、科目間の相関は課していない。

使い方:
    uv run python scripts/run_diagnose_identification.py --label bal_W1 \\
        --glob "data/interim/stage13/balanced__W1_uniform__seed*.parquet" --method PCFM
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import glob  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.evaluation.classifiers import strict_sklearn_convergence  # noqa: E402
from fsgen.evaluation.metrics import correlation_frobenius_distance  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Hiragino Sans", *plt.rcParams["font.sans-serif"]]

_CANONICAL = Path("conf/accounts/core.yaml")
_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_LABELS_CSV = Path("data/interim/labels/labels.csv")
_OUT = Path("data/interim/stage13/diagnosis")
_SEED = 42
_REPEATS = 20
_C_GEN, _C_REAL = "#eb6834", "#2a78d6"  # 生成=橙、実データ=青(参照パレットslot2/slot1)
_INK, _MUTED, _GRID = "#0b0b0b", "#898781", "#e1e0d9"


def load_class0(names: tuple[str, ...]) -> np.ndarray:
    agg = pd.read_csv(_AGGREGATED_CSV, dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    lab = pd.read_csv(_LABELS_CSV, dtype={"docID": str})
    lab = lab[pd.to_numeric(lab["label"], errors="coerce") == 0]
    m = agg.merge(lab[["docID"]], on="docID", how="inner")
    m = m[m["total_assets"].notna() & (m["total_assets"] != 0)]
    x = m[list(names)].fillna(0.0).to_numpy(dtype=float)
    result: np.ndarray = x / m["total_assets"].to_numpy(dtype=float)[:, None]
    return result


def importances(
    gen: np.ndarray, real0: np.ndarray, names: tuple[str, ...]
) -> pd.DataFrame:
    """D.1: 生成点 vs class0(68点を20回抽出)の識別器の科目別重要度の平均。"""
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(_SEED)
    gain = np.zeros(len(names))
    coef = np.zeros(len(names))
    for rep in range(_REPEATS):
        real = real0[rng.choice(len(real0), size=len(gen), replace=False)]
        x = np.concatenate([gen, real])
        y = np.concatenate([np.ones(len(gen)), np.zeros(len(real))])
        gbm = lgb.LGBMClassifier(
            n_estimators=200, num_leaves=15, random_state=_SEED + rep, n_jobs=1,
            class_weight="balanced", verbosity=-1, importance_type="gain",
        ).fit(x, y)
        g = gbm.feature_importances_.astype(float)
        gain += g / max(g.sum(), 1e-12)
        with strict_sklearn_convergence():
            lin = LogisticRegression(max_iter=2000, random_state=_SEED + rep).fit(
                StandardScaler().fit_transform(x), y
            )
        c = np.abs(lin.coef_[0])
        coef += c / c.sum()
    out = pd.DataFrame({
        "account": names, "lgbm_gain_share": gain / _REPEATS,
        "logreg_abs_coef_share": coef / _REPEATS,
    })
    return out.sort_values("lgbm_gain_share", ascending=False).reset_index(drop=True)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--glob", required=True)
    ap.add_argument("--method", required=True)
    args = ap.parse_args()
    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    real0 = load_class0(names)
    files = sorted(glob.glob(args.glob))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[df["method"] == args.method]
    gen_all = df[[f"x1_{n}" for n in names]].to_numpy()
    gen0 = df[df["seed"] == 0][[f"x1_{n}" for n in names]].to_numpy()
    out_dir = _OUT / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {args.label}: {args.method} 生成点{len(gen_all)}(seed0={len(gen0)})、"
          f"class0={len(real0)} ===")

    # --- D.1 ---
    imp = importances(gen0, real0, names)
    imp.to_csv(out_dir / "importance.csv", index=False)
    print("\n--- D.1 識別器の科目別重要度(上位10、20回平均) ---")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(imp.head(10).to_string(index=False))
    top = list(imp["account"].head(10))

    # --- D.2 ---
    from scipy.stats import ks_2samp

    rows = []
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.5), facecolor="#fcfcfb")
    for ax, acct in zip(axes.ravel(), top, strict=True):
        j = names.index(acct)
        g, r = gen_all[:, j], real0[:, j]
        ks = ks_2samp(g, r)
        rows.append({
            "account": acct, "KS": float(ks.statistic), "gen_median": float(np.median(g)),
            "real_median": float(np.median(r)), "gen_share_zero": float(np.mean(np.abs(g) < 1e-6)),
            "real_share_zero": float(np.mean(np.abs(r) < 1e-6)),
        })
        lo, hi = np.percentile(np.concatenate([g, r]), [1, 99])
        bins = np.linspace(lo, hi, 40) if hi > lo else 20
        ax.hist(r, bins=bins, density=True, color=_C_REAL, alpha=0.55, label="class0実データ")
        ax.hist(g, bins=bins, density=True, color=_C_GEN, alpha=0.55, label="生成")
        ax.set_title(f"{acct}\nKS={ks.statistic:.2f}", fontsize=8.5, color=_INK, loc="left")
        ax.tick_params(colors=_MUTED, labelsize=8)
        ax.grid(axis="y", color=_GRID, linewidth=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"上位10科目の分布比較(総資産で正規化、{args.label})", x=0.02,
                 ha="left", color=_INK)
    fig.tight_layout()
    fig.savefig(out_dir / "univariate.png", dpi=130)
    plt.close(fig)
    ks_df = pd.DataFrame(rows)
    ks_df.to_csv(out_dir / "ks.csv", index=False)
    print("\n--- D.2 単変量分布(上位10科目、KS統計量) ---")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(ks_df.to_string(index=False))

    # --- D.3 ---
    std_g, std_r = gen_all.std(axis=0), real0.std(axis=0)
    keep = (std_g > 0) & (std_r > 0)
    dropped = [n for n, k in zip(names, keep, strict=True) if not k]
    dist = correlation_frobenius_distance(gen_all, real0)
    cg = np.corrcoef(gen_all[:, keep], rowvar=False)
    cr = np.corrcoef(real0[:, keep], rowvar=False)
    kn = [n for n, k in zip(names, keep, strict=True) if k]
    diff = np.abs(cg - cr)
    iu = np.triu_indices(len(kn), k=1)
    order = np.argsort(-diff[iu])[:10]
    pairs = pd.DataFrame([
        {"account_a": kn[iu[0][o]], "account_b": kn[iu[1][o]],
         "corr_gen": cg[iu[0][o], iu[1][o]], "corr_real": cr[iu[0][o], iu[1][o]],
         "abs_diff": diff[iu[0][o], iu[1][o]]} for o in order
    ])
    pairs.to_csv(out_dir / "corr_top_pairs.csv", index=False)
    print(f"\n--- D.3 相関構造: Frobenius距離={dist:.2f}(比較対象の科目={len(kn)}/{len(names)}、"
          f"分散0で除外={dropped}) ---")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(pairs.to_string(index=False))
    print(f"\n出力先: {out_dir}")


if __name__ == "__main__":
    main()
