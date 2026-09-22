#!/usr/bin/env python
"""ステージ10: Flow Matching / PCFM / CAFM の学習・評価。

5-fold CV(§4.3、理由は本文参照)・複数seedで、低次元(z, 13次元)空間の
FM/PCFM(endpoint_only, every_step)/CAFM(scratch, warm start)を、
ステージ8/9のベースラインと比較する。

使い方:
    uv run python scripts/run_flow_matching_evaluation.py
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Hiragino Sans", *plt.rcParams["font.sans-serif"]]

from fsgen.accounting.constraints import (  # noqa: E402
    CanonicalSpec,
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.projection_qp import (  # noqa: E402
    DirectionConstraintSpec,
    load_direction_constraints,
    project_onto_feasible,
)
from fsgen.accounting.subspace import build_reduced_basis  # noqa: E402
from fsgen.evaluation import metrics as m  # noqa: E402
from fsgen.evaluation.baselines import (  # noqa: E402
    ConvexQpBaseline,
    build_prior_work_baseline,
    ot_target_for_new_point,
)
from fsgen.evaluation.classifiers import train_classifiers  # noqa: E402
from fsgen.models.cost import (  # noqa: E402
    combined_projection_weights,
    cost_vectors,
    load_cost_config,
    weighted_asymmetric_cost_matrix,
)
from fsgen.models.differentiable_projection import (  # noqa: E402
    ProjectionLayer,
    build_projection_layer,
)
from fsgen.models.flow_matching import (  # noqa: E402
    TrainResult,
    VelocityField,
    euler_sample_batch,
    train_cafm,
    train_fm,
)
from fsgen.models.transport import solve_ot_from_cost  # noqa: E402
from fsgen.viz.balance_sheet import (  # noqa: E402
    build_bs_lines,
    load_labels,
    render_markdown,
    render_text,
)

_DEFAULT_CANONICAL = Path("conf/accounts/core.yaml")
_DEFAULT_COSTS = Path("conf/accounts/costs.yaml")
_DEFAULT_LABELS_JA = Path("conf/accounts/labels_ja.yaml")
_DEFAULT_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_DEFAULT_LABELS_CSV = Path("data/interim/labels/labels.csv")
_DEFAULT_OUTPUT_DIR = Path("data/interim/flow_matching")

_K = 13
_EPSILON = 0.05
_TS = (0.0, 0.25, 0.5, 0.75, 1.0)
_N_FOLDS = 5
_SEEDS = (0, 1, 2)
_EPOCHS = 50  # 検証損失は数epoch以内に頭打ち・過学習気味になることを実測で確認済み(§7参照)
_LR = 1e-3
_HIDDEN = 128
_N_LAYERS = 3
_VIOLATION_RELATIVE_TOL = 1e-4


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


def make_folds(n2: int, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """class2企業(0..n2-1)をn_folds分割する。返り値は(train_pos, test_pos)のリスト。"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n2)
    chunks = np.array_split(order, n_folds)
    folds = []
    for i in range(n_folds):
        test_pos = chunks[i]
        train_pos = np.concatenate([chunks[j] for j in range(n_folds) if j != i])
        folds.append((train_pos, test_pos))
    return folds


@dataclass
class Dataset:
    doc_ids2: list[str]
    filer2: list[str]
    x2: np.ndarray
    assets2: np.ndarray
    x0: np.ndarray
    assets0: np.ndarray
    names: tuple[str, ...]
    spec: CanonicalSpec
    c: np.ndarray
    direction_constraints: tuple[DirectionConstraintSpec, ...]
    w_plus_proj: np.ndarray
    w_minus_proj: np.ndarray
    w_plus_ot: np.ndarray
    w_minus_ot: np.ndarray
    non_actionable: tuple[str, ...]
    proj_delta: float = 1e-3  # combined_projection_weightsの床(ステージ10.5 §1、W2は1e-2)


def build_fold_projection_layers(
    ds: Dataset, x2_tilde: np.ndarray, pos: np.ndarray
) -> list[ProjectionLayer]:
    layers = []
    for p in pos:
        x_start = x2_tilde[p]
        weights = combined_projection_weights(
            x_start, 1.0, ds.w_plus_proj, ds.w_minus_proj, ds.proj_delta
        )
        layers.append(
            build_projection_layer(
                ds.spec, x_start, weights, ds.direction_constraints, ds.non_actionable
            )
        )
    return layers


def sample_and_convert(
    net: VelocityField, x2_tilde: np.ndarray, pos: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """test companies(posで指定)についてEuler法でサンプリングし、正規化x空間の経路(n,T,D)を返す。"""
    z_start = x2_tilde[pos] @ b
    z_path = euler_sample_batch(net, z_start, _TS)  # (n, T, k)
    displacement = z_path - z_start[:, None, :]
    x_tilde_path: np.ndarray = x2_tilde[pos][:, None, :] + np.einsum(
        "dk,ntk->ntd", b, displacement
    )
    return x_tilde_path


def project_path(
    ds: Dataset,
    x_tilde_path: np.ndarray,
    x_start_tilde: np.ndarray,
    mode: str,
) -> np.ndarray:
    """mode="none"|"endpoint_only"|"every_step" でサンプリング時射影する(正規化x空間)。"""
    if mode == "none":
        return x_tilde_path
    weights = combined_projection_weights(
        x_start_tilde, 1.0, ds.w_plus_proj, ds.w_minus_proj, ds.proj_delta
    )
    out = x_tilde_path.copy()
    t_len = x_tilde_path.shape[0]
    t_indices = range(t_len) if mode == "every_step" else (t_len - 1,)
    for t_i in t_indices:
        x_proj, _info = project_onto_feasible(
            x_tilde_path[t_i],
            ds.spec,
            x_start=x_start_tilde,
            weights=weights,
            direction_constraints=ds.direction_constraints,
            frozen_accounts=ds.non_actionable,
        )
        out[t_i] = x_proj
    return out


def count_sign_violations_path(
    path_tilde: np.ndarray, ds: Dataset, expected: dict[str, str]
) -> int:
    count = 0
    tol = _VIOLATION_RELATIVE_TOL
    for t_i in range(path_tilde.shape[0]):
        for d, name in enumerate(ds.names):
            sign = expected[name]
            if sign == "free":
                continue
            v = path_tilde[t_i, d]
            if (sign == "nonneg" and v < -tol) or (sign == "nonpos" and v > tol):
                count += 1
    return count


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=_DEFAULT_CANONICAL)
    parser.add_argument("--costs", type=Path, default=_DEFAULT_COSTS)
    parser.add_argument("--labels-ja", type=Path, default=_DEFAULT_LABELS_JA)
    parser.add_argument("--aggregated-csv", type=Path, default=_DEFAULT_AGGREGATED_CSV)
    parser.add_argument("--labels-csv", type=Path, default=_DEFAULT_LABELS_CSV)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=_EPOCHS)
    parser.add_argument(
        "--weight-condition",
        choices=["W1_uniform", "W2_floor_only", "W3_current"],
        default="W3_current",
        help="射影の重み条件(ステージ10.5 §1)。CAFM/PCFMの射影にのみ影響し、"
        "輸送コスト(ignore mode)は変えない。",
    )
    args = parser.parse_args()

    spec = load_canonical_spec(args.canonical)
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    dcs = load_direction_constraints(args.canonical)
    cost_config = load_cost_config(args.costs)
    dim = len(names)
    # ステージ10.5 §1: 射影の重み条件を切り替える(frozen_accountsは常に維持する)。
    if args.weight_condition == "W1_uniform":
        w_plus_proj, w_minus_proj, proj_delta = np.ones(dim), np.ones(dim), 1e-3
    elif args.weight_condition == "W2_floor_only":
        w_plus_proj, w_minus_proj, proj_delta = np.ones(dim), np.ones(dim), 1e-2
    else:
        w_plus_proj, w_minus_proj = cost_vectors(cost_config, names, non_actionable_mode="freeze")
        proj_delta = 1e-3
    w_plus_ot, w_minus_ot = cost_vectors(cost_config, names, non_actionable_mode="ignore")
    non_actionable = tuple(cost_config.non_actionable)
    print(f"重み条件: {args.weight_condition}")

    doc_ids, filer_names, x, assets, labels = load_dataset(
        args.aggregated_csv, args.labels_csv, names
    )
    x_tilde = x / assets[:, None]
    idx0 = [i for i, lab in enumerate(labels) if lab == 0]
    idx2 = [i for i, lab in enumerate(labels) if lab == 2]

    ds = Dataset(
        doc_ids2=[doc_ids[i] for i in idx2],
        filer2=[filer_names[i] for i in idx2],
        x2=x[idx2],
        assets2=assets[idx2],
        x0=x[idx0],
        assets0=assets[idx0],
        names=names,
        spec=spec,
        c=c,
        direction_constraints=dcs,
        w_plus_proj=w_plus_proj,
        w_minus_proj=w_minus_proj,
        w_plus_ot=w_plus_ot,
        w_minus_ot=w_minus_ot,
        non_actionable=non_actionable,
        proj_delta=proj_delta,
    )
    x2_tilde = ds.x2 / ds.assets2[:, None]
    x0_tilde = ds.x0 / ds.assets0[:, None]
    n2 = len(idx2)
    print(f"class0: {len(idx0)}社, class2: {n2}社")

    expected = {b_.name: b_.sign for b_ in spec.basis}
    re_idx = names.index("jppfs_cor_RetainedEarnings")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    folds = make_folds(n2, _N_FOLDS, seed=0)
    rows: list[dict[str, float | int | str]] = []
    endpoints_norm: dict[tuple[int, str], list[np.ndarray]] = {}
    docids_by_seed: dict[int, list[str]] = {}
    timing_rows: list[dict[str, float | str]] = []
    curve_sample: dict[str, TrainResult] = {}

    # BS表用サンプル(§7): 東邦亜鉛(S100TWQJ)は除外し、対応先が変わった実例として
    # ステージ9で確認済みのS100VJ5Aを含める(§4節参照)。残りは固定seedで無作為抽出。
    bs_candidates = [d for d in ds.doc_ids2 if d != "S100TWQJ"]
    rng_pick = np.random.default_rng(_SEEDS[0])
    must_include_bs = [d for d in ("S100VJ5A",) if d in bs_candidates]
    remaining_bs = [d for d in bs_candidates if d not in must_include_bs]
    extra_bs = list(
        rng_pick.choice(remaining_bs, size=max(0, 5 - len(must_include_bs)), replace=False)
    )
    picked_bs = must_include_bs + extra_bs
    print(f"BS表対象(東邦亜鉛は除外): {picked_bs}")
    bs_paths: dict[str, dict[str, np.ndarray]] = {}

    def _metrics_row(
        seed: int, fold_i: int, doc_id: str, filer: str, method: str,
        path_tilde: np.ndarray, x_start: np.ndarray, assets_start: float,
    ) -> dict[str, float | int | str]:
        path_raw = path_tilde * assets_start
        row: dict[str, float | int | str] = {
            "seed": seed, "fold": fold_i, "docID": doc_id, "filerName": filer, "method": method,
            "proximity_l2": m.proximity_l2(path_raw, assets_start),
            "proximity_l1": m.proximity_l1(path_raw, assets_start),
            "proximity_cost_weighted": m.proximity_cost_weighted(
                path_raw, assets_start, w_plus_proj, w_minus_proj
            ),
            "sparsity": m.sparsity(path_raw, assets_start),
            "balance_residual_ratio": m.balance_residual(path_raw, c, assets_start),
            "endpoint_balance_residual_ratio": m.endpoint_balance_residual(
                path_raw, c, assets_start
            ),
            "sign_violations": count_sign_violations_path(path_tilde, ds, expected),
            "monotonicity_reversals": m.monotonicity_reversals(path_raw),
            "re_change": abs(path_raw[-1, re_idx] - path_raw[0, re_idx]) / assets_start,
        }
        return row

    print(f"\n=== 5-fold CV({_N_FOLDS}分割)× seed{_SEEDS} で FM/PCFM/CAFM を学習・評価 ===")
    for seed in _SEEDS:
        docids_by_seed[seed] = []
        for fold_i, (train_pos, test_pos) in enumerate(folds):
            pool_idx = idx0 + [idx2[p] for p in train_pos]
            x_pool_tilde = x_tilde[pool_idx]
            b = build_reduced_basis(c, x_pool_tilde, k=_K)
            z0 = x0_tilde @ b
            z_train = x2_tilde[train_pos] @ b
            z_test = x2_tilde[test_pos] @ b

            cost_train = weighted_asymmetric_cost_matrix(
                x2_tilde[train_pos], x0_tilde, w_plus_ot, w_minus_ot
            )
            plan_train = solve_ot_from_cost(cost_train, _EPSILON)
            cost_test = weighted_asymmetric_cost_matrix(
                x2_tilde[test_pos], x0_tilde, w_plus_ot, w_minus_ot
            )
            plan_test = solve_ot_from_cost(cost_test, _EPSILON)

            # --- FM ---
            net_fm = VelocityField(dim=_K, hidden=_HIDDEN, n_layers=_N_LAYERS).double()
            t0 = time.perf_counter()
            result_fm = train_fm(
                z_train, z0, plan_train, net_fm, epochs=args.epochs, lr=_LR, seed=seed,
                z_val=z_test, plan_val=plan_test,
            )
            fm_train_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            x_tilde_path_fm = sample_and_convert(net_fm, x2_tilde, test_pos, b)
            fm_sample_time = (time.perf_counter() - t0) / max(len(test_pos), 1)
            timing_rows.append({
                "seed": seed, "fold": fold_i, "method": "FM",
                "train_seconds": fm_train_time, "sample_seconds_per_company": fm_sample_time,
            })

            # --- CAFM (scratch / warm start) ---
            layers_train = build_fold_projection_layers(ds, x2_tilde, train_pos)
            net_cafm_scratch = VelocityField(dim=_K, hidden=_HIDDEN, n_layers=_N_LAYERS).double()
            t0 = time.perf_counter()
            x_start_train = x2_tilde[train_pos]
            result_cafm_scratch = train_cafm(
                z_train, z0, plan_train, net_cafm_scratch, layers_train,
                x_start_train, b,
                epochs=args.epochs, lr=_LR, seed=seed, z_val=z_test, plan_val=plan_test,
            )
            cafm_scratch_time = time.perf_counter() - t0

            net_cafm_warm = VelocityField(dim=_K, hidden=_HIDDEN, n_layers=_N_LAYERS).double()
            net_cafm_warm.load_state_dict(net_fm.state_dict())
            t0 = time.perf_counter()
            result_cafm_warm = train_cafm(
                z_train, z0, plan_train, net_cafm_warm, layers_train,
                x_start_train, b,
                epochs=args.epochs, lr=_LR, seed=seed, z_val=z_test, plan_val=plan_test,
            )
            cafm_warm_time = time.perf_counter() - t0
            timing_rows.append({
                "seed": seed, "fold": fold_i, "method": "CAFM_scratch",
                "train_seconds": cafm_scratch_time, "sample_seconds_per_company": np.nan,
                "solver_failures": result_cafm_scratch.solver_failures,
            })
            timing_rows.append({
                "seed": seed, "fold": fold_i, "method": "CAFM_warmstart",
                "train_seconds": cafm_warm_time, "sample_seconds_per_company": np.nan,
                "solver_failures": result_cafm_warm.solver_failures,
            })
            if result_cafm_scratch.solver_failures or result_cafm_warm.solver_failures:
                print(
                    f"    [注] 微分可能射影の求解失敗(スキップ): "
                    f"scratch={result_cafm_scratch.solver_failures}件, "
                    f"warmstart={result_cafm_warm.solver_failures}件"
                    f"(延べepoch×企業ペア数のうち)"
                )

            if seed == _SEEDS[0] and fold_i == 0:
                curve_sample["FM"] = result_fm
                curve_sample["CAFM_scratch"] = result_cafm_scratch
                curve_sample["CAFM_warmstart"] = result_cafm_warm

            x_tilde_path_cafm_scratch = sample_and_convert(net_cafm_scratch, x2_tilde, test_pos, b)
            x_tilde_path_cafm_warm = sample_and_convert(net_cafm_warm, x2_tilde, test_pos, b)

            # --- stage9_proposed(参考: OT+線形補間+射影、ニューラルネットなし) ---
            targets_train = np.argmax(plan_train, axis=1)

            for i_local, p in enumerate(test_pos):
                doc_id = ds.doc_ids2[p]
                filer = ds.filer2[p]
                x_start_tilde = x2_tilde[p]
                assets_start = float(ds.assets2[p])
                docids_by_seed[seed].append(doc_id)

                raw_path = x_tilde_path_fm[i_local]
                variants = {
                    "FM": raw_path,
                    "PCFM_endpoint": project_path(ds, raw_path, x_start_tilde, "endpoint_only"),
                    "PCFM_every": project_path(ds, raw_path, x_start_tilde, "every_step"),
                    "CAFM_scratch": project_path(
                        ds, x_tilde_path_cafm_scratch[i_local], x_start_tilde, "every_step"
                    ),
                    "CAFM_warmstart": project_path(
                        ds, x_tilde_path_cafm_warm[i_local], x_start_tilde, "every_step"
                    ),
                }
                for method_name, path_tilde in variants.items():
                    rows.append(
                        _metrics_row(
                            seed, fold_i, doc_id, filer, method_name,
                            path_tilde, x_start_tilde, assets_start,
                        )
                    )
                    endpoints_norm.setdefault((seed, method_name), []).append(path_tilde[-1])

                # stage9_proposed: 低次元空間で最近傍のtrain企業のOT対応先を借りる(ステージ8§B.3)
                ot = ot_target_for_new_point(ds.x2[p], assets_start, b, z0, z_train, targets_train)
                ts_arr = np.asarray(_TS)
                disp_z = ts_arr[:, None] * (ot.z_target - ot.z_start)[None, :]
                disp_x = np.einsum("dk,tk->td", b, disp_z)
                path_s9_tilde = x_start_tilde[None, :] + disp_x
                path_s9_proj = project_path(ds, path_s9_tilde, x_start_tilde, "every_step")
                rows.append(
                    _metrics_row(
                        seed, fold_i, doc_id, filer, "stage9_proposed",
                        path_s9_proj, x_start_tilde, assets_start,
                    )
                )
                endpoints_norm.setdefault((seed, "stage9_proposed"), []).append(path_s9_proj[-1])

                if seed == _SEEDS[0] and doc_id in picked_bs:
                    bs_paths[doc_id] = {**variants, "stage9_proposed": path_s9_proj}

            print(f"  seed={seed} fold={fold_i + 1}/{_N_FOLDS} 完了"
                  f"(train={len(train_pos)}社, test={len(test_pos)}社,"
                  f" FM={fm_train_time:.1f}s, CAFM(scratch)={cafm_scratch_time:.1f}s,"
                  f" CAFM(warm)={cafm_warm_time:.1f}s)")

    # --- prior_work / convex_qp(class0統計量のみ、foldに依存しないので1回だけ) ---
    print("\n=== prior_work / convex_qp(参考ベースライン、seed非依存) ===")
    median0 = np.median(ds.x0, axis=0)
    median0_tilde = np.median(x0_tilde, axis=0)
    prior_work = build_prior_work_baseline(median0, names)
    convex_qp = ConvexQpBaseline(
        median_tilde=median0_tilde, spec=spec, direction_constraints=dcs, weighted=True
    )
    for seed in (_SEEDS[0],):  # class0統計量のみでseed非依存、代表として1回だけ計算
        for p in range(n2):
            doc_id = ds.doc_ids2[p]
            filer = ds.filer2[p]
            x_start = ds.x2[p]
            assets_start = float(ds.assets2[p])
            for method_name, baseline in (("prior_work", prior_work), ("convex_qp", convex_qp)):
                path_raw = baseline.generate(x_start, assets_start, _TS)
                path_tilde = path_raw / assets_start
                rows.append(
                    _metrics_row(
                        seed, -1, doc_id, filer, method_name,
                        path_tilde, x_start / assets_start, assets_start,
                    )
                )
                endpoints_norm.setdefault((seed, method_name), []).append(path_tilde[-1])

    metrics_df = pd.DataFrame(rows)
    timing_df = pd.DataFrame(timing_rows)

    # --- validity: LOOの外側で1回だけ学習した3分類器で健全確率を測る(ステージ8.5と同じ向き) ---
    print("\n=== validity分類器の学習 ===")
    x_all_tilde = x_tilde
    is_class2 = np.array([lab == 2 for lab in labels])
    classifiers = train_classifiers(x_all_tilde, is_class2, seed=_SEEDS[0])
    for (seed, method_name), pts in endpoints_norm.items():
        arr = np.array(pts)
        mask = (metrics_df["seed"] == seed) & (metrics_df["method"] == method_name)
        for clf_name, clf in classifiers.items():
            health = 1.0 - clf.predict_proba_class2(arr)
            metrics_df.loc[mask, f"validity_{clf_name}"] = health
    metrics_df["validity"] = metrics_df["validity_lightgbm"]
    metrics_df.to_csv(args.output_dir / "metrics.csv", index=False)
    timing_df.to_csv(args.output_dir / "timing.csv", index=False)
    print(f"metrics.csv / timing.csv を書き出しました: {args.output_dir}")

    # --- plausibility(プールレベル、seedごとに集計してばらつきも見る) ---
    print("\n=== plausibility(手法ごと、3seed平均±標準偏差) ===")
    real_nn_dist = m.pairwise_nearest_neighbor_distances(x0_tilde)
    rng_sample = np.random.default_rng(_SEEDS[0])
    sample_idx = rng_sample.choice(len(x0_tilde), size=min(n2, len(x0_tilde)), replace=False)
    real_sample = x0_tilde[sample_idx]
    plausibility_rows = []
    method_names = (
        "FM", "PCFM_endpoint", "PCFM_every", "CAFM_scratch", "CAFM_warmstart",
        "stage9_proposed", "prior_work", "convex_qp",
    )
    for method_name in method_names:
        has_multi_seed = any((s, method_name) in endpoints_norm for s in _SEEDS[1:])
        seeds_here = _SEEDS if has_multi_seed else (_SEEDS[0],)
        aucs, corrs, nn_pcts = [], [], []
        for seed in seeds_here:
            key = (seed, method_name)
            if key not in endpoints_norm:
                continue
            gen = np.array(endpoints_norm[key])
            nn_pct = [
                m.percentile_rank(m.nearest_neighbor_distance(g, x0_tilde), real_nn_dist)
                for g in gen
            ]
            nn_pcts.append(float(np.median(nn_pct)))
            aucs.append(m.discriminative_auc(gen, real_sample, seed=_SEEDS[0]))
            corrs.append(m.correlation_frobenius_distance(gen, x0_tilde))
        plausibility_rows.append({
            "method": method_name,
            "nn_percentile_mean": float(np.mean(nn_pcts)) if nn_pcts else float("nan"),
            "nn_percentile_std": float(np.std(nn_pcts)) if len(nn_pcts) > 1 else 0.0,
            "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
            "auc_std": float(np.std(aucs)) if len(aucs) > 1 else 0.0,
            "corr_dist_mean": float(np.mean(corrs)) if corrs else float("nan"),
            "corr_dist_std": float(np.std(corrs)) if len(corrs) > 1 else 0.0,
        })
        r = plausibility_rows[-1]
        nn_str = f"{r['nn_percentile_mean']:.1f}±{r['nn_percentile_std']:.1f}"
        auc_str = f"{r['auc_mean']:.3f}±{r['auc_std']:.3f}"
        corr_str = f"{r['corr_dist_mean']:.3f}±{r['corr_dist_std']:.3f}"
        print(f"  {method_name:<16} NN%ile={nn_str}, AUC={auc_str}, corr距離={corr_str}")
    plausibility_df = pd.DataFrame(plausibility_rows)
    plausibility_df.to_csv(args.output_dir / "plausibility.csv", index=False)

    # --- 手法×指標 集計表(3seed込みの中央値・四分位) ---
    print("\n=== 手法×指標 集計表(中央値[Q1,Q3]、全seed込み) ===")
    summary_cols = [
        "proximity_l2", "proximity_cost_weighted", "sparsity",
        "endpoint_balance_residual_ratio", "sign_violations", "re_change",
        "validity_lightgbm", "validity_logreg", "validity_random_forest",
    ]
    med = metrics_df.groupby("method")[summary_cols].median()
    q1 = metrics_df.groupby("method")[summary_cols].quantile(0.25)
    q3 = metrics_df.groupby("method")[summary_cols].quantile(0.75)
    present_methods = [mth for mth in method_names if mth in med.index]
    for mth in present_methods:
        print(f"  {mth}:")
        for col in summary_cols:
            print(
                f"    {col:<32} {med.loc[mth, col]:.4g} "
                f"[{q1.loc[mth, col]:.4g}, {q3.loc[mth, col]:.4g}]"
            )

    # --- CAFM vs PCFM(every_step)の直接比較(会社×seedペアで対応比較) ---
    print("\n=== CAFM vs PCFM_every の直接比較 ===")
    from scipy.stats import wilcoxon

    key_cols = ["seed", "fold", "docID"]
    for cafm_variant in ("CAFM_scratch", "CAFM_warmstart"):
        pcfm_rows = metrics_df[metrics_df["method"] == "PCFM_every"].set_index(key_cols)
        cafm_rows = metrics_df[metrics_df["method"] == cafm_variant].set_index(key_cols)
        common = pcfm_rows.index.intersection(cafm_rows.index)
        prox_diff = cafm_rows.loc[common, "proximity_l2"] - pcfm_rows.loc[common, "proximity_l2"]
        valid_diff = (
            cafm_rows.loc[common, "validity_lightgbm"] - pcfm_rows.loc[common, "validity_lightgbm"]
        )
        try:
            stat_p, p_p = wilcoxon(prox_diff)
            stat_v, p_v = wilcoxon(valid_diff)
        except ValueError:
            p_p = p_v = float("nan")
        print(
            f"  {cafm_variant} - PCFM_every: proximity差 中央値={prox_diff.median():.4f} "
            f"(Wilcoxon p={p_p:.4f}), validity差 中央値={valid_diff.median():.4f} (p={p_v:.4f})"
        )

    # --- warm start あり/なし比較 ---
    print("\n=== warm start あり/なし比較(CAFM) ===")
    for col in ("proximity_l2", "validity_lightgbm", "endpoint_balance_residual_ratio"):
        scratch_med = metrics_df[metrics_df["method"] == "CAFM_scratch"][col].median()
        warm_med = metrics_df[metrics_df["method"] == "CAFM_warmstart"][col].median()
        print(f"  {col:<32} scratch={scratch_med:.4g}, warm_start={warm_med:.4g}")

    # --- 学習・サンプリング時間 ---
    print("\n=== 学習・サンプリング時間 ===")
    time_summary = timing_df.groupby("method")["train_seconds"].agg(["mean", "std"])
    print(time_summary)

    # --- 学習曲線 ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, result in curve_sample.items():
        axes[0].plot(result.train_losses, label=f"{name} (train)")
        if result.val_losses:
            axes[1].plot(result.val_losses, label=f"{name} (val)")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("training loss")
    axes[0].legend(fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation loss")
    axes[1].legend(fontsize=8)
    fig.suptitle("学習曲線(seed=0, fold=0)")
    fig.tight_layout()
    fig.savefig(figures_dir / "training_curves.png", dpi=150)
    plt.close(fig)
    print(f"\n学習曲線を出力: {figures_dir / 'training_curves.png'}")

    # --- BS表(5社: FM / PCFM_every / CAFM_warmstart / stage9_proposed) ---
    print("\n=== BS表出力 ===")
    labels_map = load_labels(args.labels_ja)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    bs_methods = ("FM", "PCFM_every", "CAFM_warmstart", "stage9_proposed")
    for doc_id, paths_by_method in bs_paths.items():
        p = ds.doc_ids2.index(doc_id)
        assets_start = float(ds.assets2[p])
        filer = ds.filer2[p]
        for method_name in bs_methods:
            path_tilde = paths_by_method[method_name]
            path_raw = path_tilde * assets_start
            x_by_t = [path_raw[t_i] for t_i in range(path_raw.shape[0])]
            lines = build_bs_lines(spec, names, x_by_t, labels_map)
            text = render_text(lines, _TS, f"{filer} [{method_name}]", doc_id)
            md = render_markdown(lines, _TS, f"{filer} [{method_name}]", doc_id)
            print("\n" + text)
            (samples_dir / f"{doc_id}_{method_name}.md").write_text(md, encoding="utf-8")

    print(f"\n出力先: {args.output_dir}")


if __name__ == "__main__":
    main()
