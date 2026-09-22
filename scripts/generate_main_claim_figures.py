#!/usr/bin/env python
"""ステージ8.5 §C: 「標準指標だけでは会計的破綻を見逃す」ことを示す図と表を作る。

``data/interim/evaluation/metrics.csv``(scripts/run_evaluation.py の出力、
validityは健全確率(高いほど良い)に統一済み)を読み込み、
main_claim.png / metric_blindness.png / summary_main.md を出力する。

使い方:
    uv run python scripts/generate_main_claim_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_DEFAULT_METRICS_CSV = Path("data/interim/evaluation/metrics.csv")
_DEFAULT_OUTPUT_DIR = Path("data/interim/evaluation")
_METHOD_ORDER = ("prior_work", "convex_qp", "naive_ot", "gaussian", "proposed")
_METHOD_LABELS = {
    "prior_work": "Prior work",
    "convex_qp": "Direct QP",
    "naive_ot": "OT + linear interp.",
    "gaussian": "Gaussian (Bures-Wasserstein)",
    "proposed": "Proposed (OT + projection)",
}
_METHOD_COLORS = {
    "prior_work": "tab:orange",
    "convex_qp": "tab:green",
    "naive_ot": "tab:brown",
    "gaussian": "tab:gray",
    "proposed": "tab:cyan",
}
_RESIDUAL_FLOOR = 1e-16  # 対数軸に0を乗せるためのクリップ下限(§C.2)


def _clip_residual(values: pd.Series) -> np.ndarray:
    result: np.ndarray = np.clip(values.to_numpy(dtype=float), _RESIDUAL_FLOOR, None)
    return result


def make_main_claim_figure(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for method in _METHOD_ORDER:
        sub = df[df["method"] == method]
        residual = _clip_residual(sub["endpoint_balance_residual_ratio"])
        ax.scatter(
            sub["proximity_l2"],
            residual,
            label=_METHOD_LABELS[method],
            color=_METHOD_COLORS[method],
            alpha=0.55,
            s=22,
        )
        ax.scatter(
            [sub["proximity_l2"].median()],
            [np.median(residual)],
            color=_METHOD_COLORS[method],
            edgecolor="black",
            linewidth=1.2,
            s=160,
            marker="D",
            zorder=5,
        )
    ax.set_yscale("log")
    ax.set_ylim(_RESIDUAL_FLOOR / 2, 10)
    ax.set_xlabel("proximity: ||x(1)-x(0)|| / Assets_0  (lower is better)")
    ax.set_ylabel("balance-identity residual: ||C^T x(1)|| / Assets_0  (log, lower is better)")
    ax.set_title(
        "Standard CFE metrics vs accounting feasibility\n"
        f"(y clipped at {_RESIDUAL_FLOOR:.0e}; diamonds = per-method median; ideal = bottom-left)"
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_metric_blindness_figure(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    panels = [
        ("proximity_l2", "proximity (lower is better)"),
        ("sparsity", "sparsity: # accounts changed (lower is better)"),
        ("validity_lightgbm", "validity: health probability (higher is better)"),
    ]
    for ax, (col, xlabel) in zip(axes, panels, strict=True):
        for method in _METHOD_ORDER:
            sub = df[df["method"] == method]
            residual = _clip_residual(sub["endpoint_balance_residual_ratio"])
            ax.scatter(
                sub[col], residual, label=_METHOD_LABELS[method],
                color=_METHOD_COLORS[method], alpha=0.55, s=20,
            )
        ax.set_yscale("log")
        ax.set_ylim(_RESIDUAL_FLOOR / 2, 10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("balance residual / Assets_0 (log)")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Do standard metrics point the same direction as accounting feasibility?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _fmt(median: float, q1: float, q3: float, best: bool, fmt: str = "{:.3g}") -> str:
    body = f"{fmt.format(median)} [{fmt.format(q1)}, {fmt.format(q3)}]"
    return f"**{body}**" if best else body


def write_summary_table(df: pd.DataFrame, out_path: Path) -> None:
    cols = [
        "proximity_l2",
        "sparsity",
        "validity_lightgbm",
        "validity_logreg",
        "validity_random_forest",
        "endpoint_balance_residual_ratio",
        "sign_violations",
    ]
    med = df.groupby("method")[cols].median().loc[list(_METHOD_ORDER)]
    q1 = df.groupby("method")[cols].quantile(0.25).loc[list(_METHOD_ORDER)]
    q3 = df.groupby("method")[cols].quantile(0.75).loc[list(_METHOD_ORDER)]

    # 低いほど良い指標と高いほど良い指標(太字判定の向き)。
    lower_is_better = {
        "proximity_l2",
        "sparsity",
        "endpoint_balance_residual_ratio",
        "sign_violations",
    }
    # タイ(ほぼ同値)の場合は複数手法を太字にする(idxmin/idxmaxだと最初の1つしか選ばれない)。
    best_value: dict[str, float] = {}
    for col in cols:
        best_value[col] = med[col].min() if col in lower_is_better else med[col].max()

    def _is_best(col: str, method: str) -> bool:
        v = med.loc[method, col]
        b = best_value[col]
        tol = max(abs(b), 1.0) * 1e-9
        return bool(abs(v - b) <= tol)

    header = (
        "| method | proximity | sparsity | validity(LightGBM/LogReg/RF) "
        "| **balance residual (t=1)** | **sign violations** |"
    )
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for method in _METHOD_ORDER:
        prox = _fmt(
            med.loc[method, "proximity_l2"], q1.loc[method, "proximity_l2"],
            q3.loc[method, "proximity_l2"], _is_best("proximity_l2", method),
        )
        spar = _fmt(
            med.loc[method, "sparsity"], q1.loc[method, "sparsity"], q3.loc[method, "sparsity"],
            _is_best("sparsity", method), fmt="{:.0f}",
        )
        validity_cells = []
        for col in ("validity_lightgbm", "validity_logreg", "validity_random_forest"):
            validity_cells.append(
                _fmt(
                    med.loc[method, col], q1.loc[method, col], q3.loc[method, col],
                    _is_best(col, method), fmt="{:.3f}",
                )
            )
        validity_str = " / ".join(validity_cells)
        resid = _fmt(
            med.loc[method, "endpoint_balance_residual_ratio"],
            q1.loc[method, "endpoint_balance_residual_ratio"],
            q3.loc[method, "endpoint_balance_residual_ratio"],
            _is_best("endpoint_balance_residual_ratio", method), fmt="{:.3e}",
        )
        sign_v = _fmt(
            med.loc[method, "sign_violations"], q1.loc[method, "sign_violations"],
            q3.loc[method, "sign_violations"], _is_best("sign_violations", method),
            fmt="{:.0f}",
        )
        lines.append(
            f"| {_METHOD_LABELS[method]} | {prox} | {spar} | {validity_str} | {resid} | {sign_v} |"
        )

    text = (
        "# Stage 8.5: does the accounting identity survive standard CFE metrics?\n\n"
        "Median [Q1, Q3] over 68 leave-one-out companies. Bold = best (median) per column.\n"
        "validity is the health probability P(class0|x(1)) = 1 - P(class2|x(1))"
        " (higher is better; see Stage 8.5 report §A for the sign-fix history).\n\n"
        + "\n".join(lines)
        + "\n"
    )
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, default=_DEFAULT_METRICS_CSV)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    df = pd.read_csv(args.metrics_csv)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    make_main_claim_figure(df, figures_dir / "main_claim.png")
    make_metric_blindness_figure(df, figures_dir / "metric_blindness.png")
    write_summary_table(df, args.output_dir / "summary_main.md")
    print(f"出力しました: {figures_dir / 'main_claim.png'}")
    print(f"出力しました: {figures_dir / 'metric_blindness.png'}")
    print(f"出力しました: {args.output_dir / 'summary_main.md'}")


if __name__ == "__main__":
    main()
