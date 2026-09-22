#!/usr/bin/env python
"""ステージ10.5 §2: 微分可能射影層の数値的健全性を診断する。

3つの重み条件(W1=一様, W2=床のみ delta=1e-2, W3=現行)について、
- 呼び出しごとのソルバstatus・求解時間(§2.1)
- 解析勾配(cvxpylayers)と数値勾配(中心差分)の相対誤差(§2.2、最重要)
を測る。W1では frozen_accounts による等式制約は維持する
(ステージ10 §0.1で判明した通り、重みだけでは固定できないため)。

使い方:
    uv run python scripts/verify_cafm_gradient.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.projection_qp import load_direction_constraints
from fsgen.models.cost import combined_projection_weights, cost_vectors, load_cost_config
from fsgen.models.differentiable_projection import ProjectionLayer, build_projection_layer

_SEED = 42
_N_POINTS = 20
_N_COMPANIES = 5  # 複数企業で診断し、企業依存性も見る
_FD_EPS_REL = 1e-4
_PERTURB_SCALE = 0.3  # 始点からのずれの大きさ(正規化空間)


def load_dataset(
    aggregated_csv: Path, labels_csv: Path, names: tuple[str, ...]
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, list[int]]:
    agg = pd.read_csv(aggregated_csv, dtype={"docID": str})
    is_financial = agg["is_financial"].astype(str)
    agg = agg[(agg["parse_status"] == "ok") & (is_financial != "True")]
    labels_df = pd.read_csv(labels_csv, dtype={"docID": str})
    labels_df = labels_df[pd.to_numeric(labels_df["label"], errors="coerce").notna()]
    merged = agg.merge(labels_df[["docID", "filerName", "label"]], on="docID", how="inner")
    merged = merged[merged["total_assets"].notna() & (merged["total_assets"] != 0)]
    x = merged[list(names)].fillna(0.0).to_numpy(dtype=float)
    assets = merged["total_assets"].to_numpy(dtype=float)
    labels = merged["label"].astype(int).tolist()
    return merged["docID"].tolist(), merged["filerName"].tolist(), x, assets, labels


@dataclass(frozen=True)
class WeightCondition:
    name: str
    weights: np.ndarray


def build_conditions(
    x_start: np.ndarray, w_plus_proj: np.ndarray, w_minus_proj: np.ndarray
) -> list[WeightCondition]:
    dim = len(x_start)
    w1 = np.ones(dim)
    w2 = combined_projection_weights(
        x_start, 1.0, np.ones(dim), np.ones(dim), delta=1e-2
    )
    w3 = combined_projection_weights(x_start, 1.0, w_plus_proj, w_minus_proj)
    return [
        WeightCondition("W1_uniform", w1),
        WeightCondition("W2_floor_only", w2),
        WeightCondition("W3_current", w3),
    ]


def numerical_gradient(
    layer: ProjectionLayer, y: np.ndarray, direction: np.ndarray, eps_rel: float = _FD_EPS_REL
) -> np.ndarray | None:
    """loss(y) = project(y) @ direction のyに関する中心差分勾配。

    片側でもソルバが失敗したら、その点自体を無効(None)として呼び出し側で
    スキップする(W3では極端な重みでソルバが infeasible/max_iters に達する
    ことがあるため)。
    """
    dim = len(y)
    grad = np.zeros(dim)
    for i in range(dim):
        step = eps_rel * max(abs(y[i]), 1.0)
        y_plus = y.copy()
        y_plus[i] += step
        y_minus = y.copy()
        y_minus[i] -= step
        try:
            with torch.no_grad():
                out_plus = float(
                    (layer.project(torch.tensor(y_plus, dtype=torch.float64)) * direction).sum()
                )
                out_minus = float(
                    (layer.project(torch.tensor(y_minus, dtype=torch.float64)) * direction).sum()
                )
        except Exception:  # noqa: BLE001
            return None
        grad[i] = (out_plus - out_minus) / (2 * step)
    return grad


def analytic_gradient(layer: ProjectionLayer, y: np.ndarray, direction: np.ndarray) -> np.ndarray:
    y_t = torch.tensor(y, dtype=torch.float64, requires_grad=True)
    direction_t = torch.tensor(direction, dtype=torch.float64)
    out = (layer.project(y_t) * direction_t).sum()
    out.backward()  # type: ignore[no-untyped-call]
    assert y_t.grad is not None
    grad: np.ndarray = y_t.grad.numpy()
    return grad


def main() -> None:
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    build_constraint_matrix(spec, include_subtotals=False)
    dcs = load_direction_constraints(Path("conf/accounts/core.yaml"))
    cost_config = load_cost_config(Path("conf/accounts/costs.yaml"))
    w_plus_proj, w_minus_proj = cost_vectors(cost_config, names, non_actionable_mode="freeze")
    non_actionable = tuple(cost_config.non_actionable)

    doc_ids, filer_names, x, assets, labels = load_dataset(
        Path("data/interim/values/aggregated_core.csv"),
        Path("data/interim/labels/labels.csv"),
        names,
    )
    x_tilde = x / assets[:, None]
    idx2 = [i for i, lab in enumerate(labels) if lab == 2]

    rng = np.random.default_rng(_SEED)
    company_sample = rng.choice(idx2, size=_N_COMPANIES, replace=False)

    solver_rows = []
    grad_rows = []
    dim = len(names)

    for p in company_sample:
        x_start = x_tilde[p]
        doc_id = doc_ids[p]
        conditions = build_conditions(x_start, w_plus_proj, w_minus_proj)
        for cond in conditions:
            ratio = float(cond.weights.max() / cond.weights.min())
            layer = build_projection_layer(
                spec, x_start, cond.weights, dcs, non_actionable
            )
            rel_errors = []
            for _pt in range(_N_POINTS):
                y = x_start + rng.normal(scale=_PERTURB_SCALE, size=dim) * np.abs(x_start).mean()
                direction = rng.normal(size=dim)
                direction /= np.linalg.norm(direction)

                t0 = time.perf_counter()
                try:
                    a_grad = analytic_gradient(layer, y, direction)
                    status = "solved"
                except Exception as exc:  # noqa: BLE001
                    status = f"failed:{type(exc).__name__}"
                    a_grad = None
                elapsed = time.perf_counter() - t0
                solver_rows.append(
                    {
                        "docID": doc_id,
                        "condition": cond.name,
                        "status": status,
                        "solve_time": elapsed,
                    }
                )
                if a_grad is None:
                    continue
                n_grad = numerical_gradient(layer, y, direction)
                if n_grad is None:
                    continue
                denom = max(float(np.linalg.norm(n_grad)), 1e-8)
                rel_err = float(np.linalg.norm(a_grad - n_grad) / denom)
                rel_errors.append(rel_err)

            rel_errors_arr = np.array(rel_errors)
            has_errors = len(rel_errors) > 0
            grad_rows.append(
                {
                    "docID": doc_id,
                    "condition": cond.name,
                    "weight_ratio": ratio,
                    "rel_error_median": float(np.median(rel_errors_arr)) if has_errors else np.nan,
                    "rel_error_max": float(np.max(rel_errors_arr)) if has_errors else np.nan,
                    "frac_above_0.1": (
                        float(np.mean(rel_errors_arr > 0.1)) if has_errors else np.nan
                    ),
                }
            )
            print(
                f"{doc_id} {cond.name:<14} weight_ratio={ratio:.3e} "
                f"rel_err_median={grad_rows[-1]['rel_error_median']:.4f} "
                f"rel_err_max={grad_rows[-1]['rel_error_max']:.4f} "
                f"frac>0.1={grad_rows[-1]['frac_above_0.1']:.2f}"
            )

    solver_df = pd.DataFrame(solver_rows)
    grad_df = pd.DataFrame(grad_rows)
    out_dir = Path("data/interim/flow_matching")
    out_dir.mkdir(parents=True, exist_ok=True)
    solver_df.to_csv(out_dir / "gradient_check_solver.csv", index=False)
    grad_df.to_csv(out_dir / "gradient_check_summary.csv", index=False)

    print("\n=== §2.1 ソルバ状態(条件別集計) ===")
    for cond_name in ("W1_uniform", "W2_floor_only", "W3_current"):
        sub = solver_df[solver_df["condition"] == cond_name]
        n_total = len(sub)
        n_solved = int((sub["status"] == "solved").sum())
        n_failed = int((sub["status"] != "solved").sum())
        mean_time = sub["solve_time"].mean()
        print(
            f"{cond_name:<14} 総呼び出し={n_total}, solved={n_solved}, "
            f"failed={n_failed}, 平均求解時間={mean_time * 1000:.2f}ms"
        )

    print("\n=== §2.2 勾配検証(条件別、企業横断の中央値) ===")
    for cond_name in ("W1_uniform", "W2_floor_only", "W3_current"):
        sub = grad_df[grad_df["condition"] == cond_name]
        print(
            f"{cond_name:<14} 重み比中央値={sub['weight_ratio'].median():.3e}, "
            f"相対誤差中央値={sub['rel_error_median'].median():.4f}, "
            f"相対誤差最大={sub['rel_error_max'].max():.4f}, "
            f"frac>0.1平均={sub['frac_above_0.1'].mean():.3f}"
        )


if __name__ == "__main__":
    main()
