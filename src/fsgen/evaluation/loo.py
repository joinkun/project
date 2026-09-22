"""Leave-one-out(LOO)による評価の実行(ステージ8 §B.3)。

現在の提案手法(OT対応付け → 低次元線形補間 → QP射影)で企業iの経路を作る
とき、i自身の情報は次の2箇所に入り込む:

- 低次元基底B(全社で学習 → iも含む)
- OTの始点分布μ0(class 2 全体 → iを含む)

このため、LOOでは i を除いて B と輸送計画を作り直す。μ0={i}のみでは
OTが(周辺分布の制約で)コスト行列と無関係に退化する
(``baselines.ot_target_for_new_point`` のdocstring参照)ため、実際には
μ0 = class2\\{i} で計画を作り、iは低次元空間で最も近いtrain2企業の対応先を
借りる。

``prior_work``・``convex_qp`` は class 0 の統計量(中央値)しか使わないため
LOOは不要(全社で1回だけ構築すればよい)。``gaussian`` は class 2 の平均・
共分散を使うためLOOが必要。

validity判定用の分類器は「学習に使っていない別の分類器」であることが
目的であり、iのOT対応先とは無関係な情報(class0/class2の統計的な分離)を
学習するだけなので、LOOの外側で1回だけ学習する
(4,019社中1社の除外が学習全体に与える影響は無視できるという判断)。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fsgen.accounting.constraints import CanonicalSpec
from fsgen.accounting.projection_qp import (
    DirectionConstraintSpec,
    default_weights,
    evaluate_condition,
)
from fsgen.evaluation import metrics as m
from fsgen.evaluation.baselines import (
    ConvexQpBaseline,
    ProposedMethod,
    RecoursePath,
    build_gaussian_baseline,
    build_naive_ot_baseline,
    build_prior_work_baseline,
)

_VIOLATION_RELATIVE_TOL = 1e-4  # ステージ7 generate_projected_paths.py と同じ基準


@dataclass
class LooDataset:
    doc_ids: list[str]
    filer_names: list[str]
    x: np.ndarray  # (n, D)
    assets: np.ndarray  # (n,)
    labels: list[int]
    names: tuple[str, ...]
    spec: CanonicalSpec
    c: np.ndarray  # (D, 1) 会計恒等式の制約行列
    direction_constraints: tuple[DirectionConstraintSpec, ...]
    idx0: list[int] = field(init=False)
    idx2: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.idx0 = [i for i, lab in enumerate(self.labels) if lab == 0]
        self.idx2 = [i for i, lab in enumerate(self.labels) if lab == 2]


@dataclass
class CompanyResult:
    doc_id: str
    filer_name: str
    method: str
    path: np.ndarray  # (T, D)
    x_start: np.ndarray
    assets_start: float


def _direction_triggered(
    x_start: np.ndarray,
    spec: CanonicalSpec,
    direction_constraints: tuple[DirectionConstraintSpec, ...],
) -> dict[str, bool]:
    return {
        dc.name: evaluate_condition(dc.condition, x_start, spec) for dc in direction_constraints
    }


def compute_company_metrics(
    result: CompanyResult, ds: LooDataset
) -> dict[str, float | int | str]:
    path = result.path
    a0 = result.assets_start
    weights = default_weights(result.x_start, a0)  # 科目の規模差を吸収した重み付きproximity用(§B.2)
    row: dict[str, float | int | str] = {
        "docID": result.doc_id,
        "filerName": result.filer_name,
        "method": result.method,
        "proximity_l2": m.proximity_l2(path, a0),
        "proximity_l1": m.proximity_l1(path, a0),
        "proximity_l2_weighted": m.proximity_l2_weighted(path, weights),
        "proximity_l1_weighted": m.proximity_l1_weighted(path, weights),
        "sparsity": m.sparsity(path, a0),
        "balance_residual_ratio": m.balance_residual(path, ds.c, a0),
        "endpoint_balance_residual_ratio": m.endpoint_balance_residual(path, ds.c, a0),
        "sign_violations": m.sign_violation_count(
            path, ds.spec, ds.names, a0, _VIOLATION_RELATIVE_TOL
        ),
        "intermediate_violations": m.intermediate_feasibility_violations(
            path, ds.spec, ds.names, a0, _VIOLATION_RELATIVE_TOL
        ),
        "monotonicity_reversals": m.monotonicity_reversals(path),
    }
    triggered = _direction_triggered(result.x_start, ds.spec, ds.direction_constraints)
    total_direction_violations = 0
    for dc in ds.direction_constraints:
        total_direction_violations += m.direction_violation_count(
            path, ds.names, dc.name, triggered[dc.name], a0, _VIOLATION_RELATIVE_TOL
        )
    row["direction_violations"] = total_direction_violations
    return row


@dataclass
class LooRunResult:
    metrics_df: pd.DataFrame
    endpoints: dict[str, np.ndarray]  # method -> (n2, D) 正規化済み終端(x/assets_start)
    paths: dict[str, list[np.ndarray]]  # method -> [(T,D), ...] (企業順=ds.idx2順)


def run_loo(
    ds: LooDataset,
    ts: tuple[float, ...],
    k: int,
    epsilon: float,
    weighted: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> LooRunResult:
    x0 = ds.x[ds.idx0]
    assets0 = ds.assets[ds.idx0]
    median0 = np.median(x0, axis=0)
    median0_tilde = np.median(x0 / assets0[:, None], axis=0)

    prior_work = build_prior_work_baseline(median0, ds.names)
    convex_qp = ConvexQpBaseline(
        median_tilde=median0_tilde,
        spec=ds.spec,
        direction_constraints=ds.direction_constraints,
        weighted=weighted,
    )

    rows: list[dict[str, float | int | str]] = []
    endpoints: dict[str, list[np.ndarray]] = {
        name: [] for name in ("prior_work", "convex_qp", "naive_ot", "gaussian", "proposed")
    }
    paths: dict[str, list[np.ndarray]] = {name: [] for name in endpoints}

    n2 = len(ds.idx2)
    for pos, i in enumerate(ds.idx2):
        x_start = ds.x[i]
        assets_start = float(ds.assets[i])
        doc_id = ds.doc_ids[i]
        filer = ds.filer_names[i]

        train2_global = [j for j in ds.idx2 if j != i]
        pool_idx = ds.idx0 + train2_global
        x_pool = ds.x[pool_idx]
        assets_pool = ds.assets[pool_idx]
        x_tilde_pool = x_pool / assets_pool[:, None]
        idx0_in_pool = np.arange(len(ds.idx0))
        idx_train2_in_pool = np.arange(len(ds.idx0), len(ds.idx0) + len(train2_global))

        naive_ot = build_naive_ot_baseline(
            x_tilde_pool, ds.c, k, idx0_in_pool, idx_train2_in_pool, epsilon
        )
        proposed = ProposedMethod(
            naive=naive_ot,
            spec=ds.spec,
            direction_constraints=ds.direction_constraints,
            weighted=weighted,
        )

        gaussian = build_gaussian_baseline(naive_ot.b, naive_ot.z0, naive_ot.z_train2)

        methods: dict[str, RecoursePath] = {
            "prior_work": prior_work,
            "convex_qp": convex_qp,
            "naive_ot": naive_ot,
            "gaussian": gaussian,
            "proposed": proposed,
        }
        for method_name, generator in methods.items():
            path = generator.generate(x_start, assets_start, ts)
            result = CompanyResult(
                doc_id=doc_id,
                filer_name=filer,
                method=method_name,
                path=path,
                x_start=x_start,
                assets_start=assets_start,
            )
            rows.append(compute_company_metrics(result, ds))
            endpoints[method_name].append(path[-1] / assets_start)
            paths[method_name].append(path)

        if on_progress is not None:
            on_progress(pos + 1, n2)

    endpoints_arr = {name: np.array(v) for name, v in endpoints.items()}
    return LooRunResult(metrics_df=pd.DataFrame(rows), endpoints=endpoints_arr, paths=paths)


__all__ = ["CompanyResult", "LooDataset", "LooRunResult", "compute_company_metrics", "run_loo"]
