#!/usr/bin/env python
"""ステージ34 A: class 1(重要事象のみ)を始点に、最終設定の経路を生成する(torch/cvxpy のプロセス)。

class 1 を hold-out に保つため、速度場は class 2(除外後 57 社)→ class 0 の輸送計画
(``plan_full``、等重み最近傍)で学習し、それを class 1 の始点に当てる。class 1 は学習にも
目標分布にも使わない(目標分布は class 0 のまま)。最終設定は 5-fold で学習するが、class 1 は
どの fold にも属さないので、class 2 の全社で 1 本学習する(seed ごと)。

射影は最終設定と同じ(W1、第 2 層、α = 0.5、β = 0.3、R3 + 上限 γ = 0.757、毎ステップ射影)。
始点の class 1 は、ステージ28 の規則で除外(金融・ゴルフ場)した ``class1_docs.csv`` の残り。

    uv run python scripts/run_generation_stage34.py --seed 0
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from run_generation_stage16 import (  # noqa: E402
    _CANONICAL,
    _DELTA,
    _EPOCHS,
    _HIDDEN,
    _K,
    _LR,
    _N_LAYERS,
    _TS,
    inactive_targets,
    load_dataset,
    make_configs,
)

from fsgen.accounting.constraints import (  # noqa: E402
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.plausibility import (  # noqa: E402
    ConstraintLayers,
    build_linear_constraints,
    decompose_re_change,
    inactive_accounts,
)
from fsgen.accounting.projection_qp import (  # noqa: E402
    load_direction_constraints,
    project_onto_feasible,
)
from fsgen.accounting.subspace import build_reduced_basis  # noqa: E402
from fsgen.datasets.exclusion import load_excluded_docs  # noqa: E402
from fsgen.errors import InfeasibleProjectionError  # noqa: E402
from fsgen.models.cost import combined_projection_weights, load_cost_config  # noqa: E402
from fsgen.models.flow_matching import (  # noqa: E402
    euler_sample_batch,
    seeded_velocity_field,
    train_fm,
)
from fsgen.models.transport import check_plan_health  # noqa: E402

_GAMMA = 0.7565192502545388
_CONFIG = "W1_S5_a05_b03_re0.756519"
_FULL_FOLD = 5  # 種 = seed × 1000 + 5(5-fold の 0〜4 と重ならない)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--plans", type=Path,
                    default=Path("data/interim/stage28/plans_equal/plans_nearest_na0.npz"))
    ap.add_argument("--costs", type=Path, default=Path("conf/accounts/costs_equal.yaml"))
    ap.add_argument("--exclude", type=Path, default=Path("data/interim/stage28/excluded.csv"))
    ap.add_argument("--class1", type=Path, default=Path("data/interim/stage34/class1_docs.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/interim/stage34/class1"))
    ap.add_argument("--tag", default="nearest_na0", help="出力ファイル名の設計名(ステージ39)")
    ap.add_argument("--freeze", choices=("no_re", "all"), default="no_re",
                    help="no_re = R3(最終設定)、all = R1(利益剰余金も固定。ステージ43)")
    ap.add_argument("--inactive", default="none",
                    help="不活性制約: none / selected / all / カンマ区切りの科目(ステージ41)")
    ap.add_argument("--allow-dropped-sources", action="store_true",
                    help="不均衡 OT で質量 0 になった class 2 始点を許し、学習から除く(ステージ39)")
    args = ap.parse_args()
    torch.set_num_threads(1)

    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    dcs = load_direction_constraints(_CANONICAL)
    frozen = tuple(n for n in load_cost_config(args.costs).non_actionable
                   if args.freeze == "all" or n != "jppfs_cor_RetainedEarnings")  # R3 / R1
    dim = len(names)
    targets = inactive_targets(args.inactive, names)
    # ステージ43: R1(利益剰余金を固定、上限なし)は W1_S5_a05_b03。R3 は上限 γ 付き
    config = _CONFIG if args.freeze == "no_re" else "W1_S5_a05_b03"
    _cond, cfg = make_configs(ConstraintLayers(), _GAMMA)[config]

    doc_ids, filer, x, assets, labels, _keep = load_dataset(
        names, load_excluded_docs(args.exclude))
    x_tilde = x / assets[:, None]
    idx0 = [i for i, lab in enumerate(labels) if lab == 0]
    idx2 = [i for i, lab in enumerate(labels) if lab == 2]
    c1 = pd.read_csv(args.class1, dtype={"docID": str})
    use1 = set(c1.loc[c1["reason"].fillna("") == "", "docID"])
    idx1 = [i for i, (d, lab) in enumerate(zip(doc_ids, labels, strict=True))
            if lab == 1 and d in use1]
    plan = np.load(args.plans)["plan_full"]
    if plan.shape != (len(idx2), len(idx0)):
        raise ValueError(f"輸送計画の形 {plan.shape} が (class 2, class 0) = "
                         f"({len(idx2)}, {len(idx0)}) と合わない")
    check_plan_health(plan, np.full(len(idx2), 1.0 / len(idx2)), None, check_marginals=False,
                      allow_zero_mass_rows=args.allow_dropped_sources)
    alive = plan.sum(axis=1) > 1e-12  # ステージ39: 質量 0 の class 2 始点は学習から除く

    b = build_reduced_basis(c, x_tilde[idx0 + idx2], k=_K)
    z0 = x_tilde[idx0] @ b
    z2 = x_tilde[idx2] @ b
    net = seeded_velocity_field(args.seed * 1000 + _FULL_FOLD, dim=_K, hidden=_HIDDEN,
                                n_layers=_N_LAYERS)
    train_fm(z2[alive], z0, plan[alive], net, epochs=_EPOCHS, lr=_LR, seed=args.seed)
    x1s = x_tilde[idx1]
    z_start = x1s @ b
    z_path = euler_sample_batch(net, z_start, _TS)
    raw = x1s[:, None, :] + np.einsum("dk,ntk->ntd", b, z_path - z_start[:, None, :])

    ones = np.ones(dim)
    rows: list[dict[str, float | int | str]] = []
    for j, i in enumerate(idx1):
        x_start = x_tilde[i]
        inactive = inactive_accounts(x_start, names, targets)  # ステージ41
        frozen_i = frozen + tuple(a for a in inactive if a not in frozen)
        weights = combined_projection_weights(x_start, 1.0, ones, ones, _DELTA)
        extra = build_linear_constraints(spec, names, x_start, cfg)
        out = raw[j].copy()
        status, active = "ok", ""
        try:
            for t_i in range(out.shape[0]):
                out[t_i], info = project_onto_feasible(
                    raw[j][t_i], spec, x_start=x_start, weights=weights,
                    direction_constraints=dcs, frozen_accounts=frozen_i, extra_constraints=extra)
            active = ";".join(info.active_constraints)
        except InfeasibleProjectionError:
            out = np.repeat(x_start[None, :], out.shape[0], axis=0)
            status = "infeasible"
        for method, path in (("FM", raw[j]), ("PCFM", out)):
            transfer, profit = decompose_re_change(path[0], path[-1], names)
            row: dict[str, float | int | str] = {
                "tag": "class1", "config": config, "seed": args.seed, "fold": -1,
                "method": method, "docID": doc_ids[i], "filerName": filer[i],
                "assets_start": float(assets[i]), "status": status if method == "PCFM" else "ok",
                "active": active if method == "PCFM" else "", "re_branch": -1,
                "re_transfer": float(transfer), "re_profit": float(profit),
                "n_inactive": len(inactive),
            }
            for k, n in enumerate(names):
                row[f"x0_{n}"] = float(path[0, k])
                row[f"x1_{n}"] = float(path[-1, k])
            rows.append(row)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.tag}__seed{args.seed}.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    n_inf = sum(1 for r in rows if r["status"] == "infeasible")
    print(f"[class1 seed{args.seed}] 始点 {len(idx1)} 社、学習 class 2 {len(idx2)} 社、"
          f"学習から除いた質量 0 の始点 {int((~alive).sum())} 社、実行不可能 {n_inf} 社 "
          f"→ {out_path}", flush=True)


if __name__ == "__main__":
    main()
