#!/usr/bin/env python
"""ステージ28 B.2: 非決定性の候補を 1 つずつ確かめる(torch/cvxpy のプロセス。POT・lightgbm なし)。

1. 射影 QP(OSQP ほか、``project_onto_feasible``): 同じ入力を 10 回解き、解の最大差
2. 微分可能射影(SCS、``ProjectionLayer.project``): 同じ入力を 10 回解き、解の最大差
3. PyTorch の演算: 同じ種で FM を 2 回学習した重みの最大差と、
   ``torch.use_deterministic_algorithms(True)`` の下で学習した重みとの最大差
4. CAFM の学習(cvxpylayers 経由): 同じ種で 2 回学習した重みの最大差

データは除外後(``--exclude``)の fold 0。3・4 は時間を抑えるためエポック数を減らす。

    uv run python scripts/determinism_stage28.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from run_generation_stage13 import (  # noqa: E402
    _CANONICAL,
    _COSTS,
    _FOLDS,
    _HIDDEN,
    _K,
    _LR,
    _N_LAYERS,
    load_dataset,
)

from fsgen.accounting.constraints import (  # noqa: E402
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.projection_qp import (  # noqa: E402
    load_direction_constraints,
    project_onto_feasible,
)
from fsgen.accounting.subspace import build_reduced_basis  # noqa: E402
from fsgen.datasets.exclusion import load_excluded_docs, subset_folds  # noqa: E402
from fsgen.models.cost import combined_projection_weights, load_cost_config  # noqa: E402
from fsgen.models.differentiable_projection import build_projection_layer  # noqa: E402
from fsgen.models.flow_matching import (  # noqa: E402
    VelocityField,
    seeded_velocity_field,
    train_cafm,
    train_fm,
)

_REPEATS = 10


def _weights_diff(a: VelocityField, b: VelocityField) -> float:
    return max(
        float((pa - pb).abs().max()) for pa, pb in zip(a.parameters(), b.parameters(), strict=True)
    )


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--plans", type=Path, default=Path("data/interim/stage28/plans_old/plans_nearest_na0.npz")
    )
    ap.add_argument("--exclude", type=Path, default=Path("data/interim/stage28/excluded.csv"))
    ap.add_argument("--fm-epochs", type=int, default=10)
    ap.add_argument("--cafm-epochs", type=int, default=2)
    ap.add_argument("--cafm-firms", type=int, default=8, help="CAFM の学習に使う始点の数")
    ap.add_argument("--out", type=Path, default=Path("results/stage28/determinism.md"))
    args = ap.parse_args()
    torch.set_num_threads(1)

    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    dcs = load_direction_constraints(_CANONICAL)
    frozen = tuple(load_cost_config(_COSTS).non_actionable)
    dim = len(names)
    ones = np.ones(dim)

    _docs, _names, x, assets, labels, src_keep = load_dataset(
        names, load_excluded_docs(args.exclude)
    )
    x_tilde = x / assets[:, None]
    idx0 = [i for i, lab in enumerate(labels) if lab == 0]
    idx2 = [i for i, lab in enumerate(labels) if lab == 2]
    x0_tilde, x2_tilde = x_tilde[idx0], x_tilde[idx2]
    folds = subset_folds(np.load(_FOLDS), src_keep)
    plans = np.load(args.plans)
    train_pos, test_pos = folds["train_pos_0"], folds["test_pos_0"]
    plan_train, plan_test = plans["plan_train_0"], plans["plan_test_0"]
    rng = np.random.default_rng(0)
    lines = ["# ステージ28 B.2: 非決定性の候補の確認", ""]
    lines += ["| 候補 | 確認 | 最大差 | 一致 |", "|---|---|---:|---|"]

    def row(name: str, how: str, diff: float) -> None:
        lines.append(f"| {name} | {how} | {diff:.3g} | {'○' if diff == 0.0 else '×'} |")
        print(lines[-1], flush=True)

    # 1. 射影 QP
    worst = 0.0
    for p in test_pos[:5]:
        x_start = x2_tilde[p]
        y = x_start + rng.normal(scale=0.05, size=dim)
        w = combined_projection_weights(x_start, 1.0, ones, ones, 1e-3)
        sols = [
            project_onto_feasible(y, spec, x_start=x_start, weights=w,
                                  direction_constraints=dcs, frozen_accounts=frozen)[0]
            for _ in range(_REPEATS)
        ]
        worst = max(worst, max(float(np.abs(s - sols[0]).max()) for s in sols))
    row("射影 QP(OSQP ほか)", f"5 社 × 同じ入力を {_REPEATS} 回", worst)

    # 2. 微分可能射影(SCS)
    worst = 0.0
    for p in train_pos[:3]:
        x_start = x2_tilde[p]
        w = combined_projection_weights(x_start, 1.0, ones, ones, 1e-3)
        layer = build_projection_layer(spec, x_start, w, dcs, frozen)
        y_t = torch.tensor(x_start + rng.normal(scale=0.05, size=dim), dtype=torch.float64)
        sols_t = [layer.project(y_t).detach().numpy() for _ in range(_REPEATS)]
        worst = max(worst, max(float(np.abs(s - sols_t[0]).max()) for s in sols_t))
    row("微分可能射影(SCS)", f"3 社 × 同じ入力を {_REPEATS} 回", worst)

    # 3. PyTorch(FM の学習)
    pool_idx = idx0 + [idx2[p] for p in train_pos]
    b = build_reduced_basis(c, x_tilde[pool_idx], k=_K)
    z0 = x0_tilde @ b
    z_train, z_test = x2_tilde[train_pos] @ b, x2_tilde[test_pos] @ b

    def fit_fm() -> VelocityField:
        net = seeded_velocity_field(1000, dim=_K, hidden=_HIDDEN, n_layers=_N_LAYERS)
        train_fm(z_train, z0, plan_train, net, epochs=args.fm_epochs, lr=_LR, seed=1,
                 z_val=z_test, plan_val=plan_test)
        return net

    fm_a, fm_b = fit_fm(), fit_fm()
    row("PyTorch(FM の学習)", f"同じ種で 2 回({args.fm_epochs} エポック)",
        _weights_diff(fm_a, fm_b))
    torch.use_deterministic_algorithms(True)
    try:
        fm_det = fit_fm()
    finally:
        torch.use_deterministic_algorithms(False)
    row("PyTorch の決定的演算", "use_deterministic_algorithms(True) で学習し、通常と比較",
        _weights_diff(fm_a, fm_det))

    # 4. CAFM(cvxpylayers 経由)
    n = args.cafm_firms
    fit_pos = train_pos[:n]
    layers = [
        build_projection_layer(
            spec, x2_tilde[p], combined_projection_weights(x2_tilde[p], 1.0, ones, ones, 1e-3),
            dcs, frozen,
        )
        for p in fit_pos
    ]

    def fit_cafm() -> VelocityField:
        net = seeded_velocity_field(1000, dim=_K, hidden=_HIDDEN, n_layers=_N_LAYERS)
        train_cafm(z_train[:n], z0, plan_train[:n], net, layers, x2_tilde[fit_pos], b,
                   epochs=args.cafm_epochs, lr=_LR, seed=1)
        return net

    row("CAFM の学習(cvxpylayers)", f"同じ種で 2 回({n} 社、{args.cafm_epochs} エポック)",
        _weights_diff(fit_cafm(), fit_cafm()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
