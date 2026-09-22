#!/usr/bin/env python
"""ステージ10.6: PCFMの指標改善(ステージ10→10.5, -15%/+15%)が

「重み設計による正当な違い」か「QPの前進解の破綻」かを切り分ける。

§1.1: 同一のFM重み(seedごとに1回だけ学習)に対して射影条件のみを
      W1/W2/W3で変え、条件間の差とseed間のばらつきを比較する。
§1.2: FMの生出力340点(68社×5時点、seed=0)に対する3条件の射影解を
      直接比較する。
§1.3(最重要): x_W1もx_W3も同じ実行可能集合の元なので、x_W3が真に
      最適なら f_W3(x_W3) <= f_W3(x_W1) が必ず成り立つ。逆転を数える。
§1.4: 白色化を無効にした場合も同じ検証を行う。
§1.5: 逆転が出た場合のみ、独立ソルバ(ECOS/CLARABEL/SCS)で検算する。

CAFMの学習は本ステージでは不要(PCFMはサンプリング時にしか射影を
使わないため)。FM学習のみ(3 seed × 5 fold、ステージ10と同一seed)。

使い方:
    uv run python scripts/stage106_pcfm_diagnosis.py
"""

from __future__ import annotations

import os

# torch(libomp)とnumpy/scipyのBLAS(Accelerate)・LightGBMが同一プロセス内で
# 別々のOpenMPランタイムを読み込むと、mac OSで__kmp_fork_barrier内のSIGSEGVが
# 発生することを本ステージで確認した(train_fm呼び出し中に発生、docstringの
# n_jobs=1だけでは防げない場合があった)。importより前に環境変数で単一スレッド化
# し、ランタイム間の競合を避ける。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

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
from fsgen.evaluation.classifiers import train_classifiers  # noqa: E402
from fsgen.models.cost import (  # noqa: E402
    CostConfig,
    combined_projection_weights,
    cost_vectors,
    load_cost_config,
    weighted_asymmetric_cost_matrix,
)
from fsgen.models.flow_matching import VelocityField, euler_sample_batch, train_fm  # noqa: E402
from fsgen.models.transport import solve_ot_from_cost  # noqa: E402

_DEFAULT_CANONICAL = Path("conf/accounts/core.yaml")
_DEFAULT_COSTS = Path("conf/accounts/costs.yaml")
_DEFAULT_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_DEFAULT_LABELS_CSV = Path("data/interim/labels/labels.csv")
_DEFAULT_OUTPUT_DIR = Path("data/interim/flow_matching_diag106")

_K = 13
_EPSILON = 0.05
_TS = (0.0, 0.25, 0.5, 0.75, 1.0)
_N_FOLDS = 5
_SEEDS = (0, 1, 2)  # ステージ10/10.5と同一
_EPOCHS = 50
_LR = 1e-3
_HIDDEN = 128
_N_LAYERS = 3
_VIOLATION_RELATIVE_TOL = 1e-4
_REVERSAL_REL_TOL = 1e-6  # f(x_other)/f(x_own) - 1 がこれを超えたら「逆転」とみなす


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
    rng = np.random.default_rng(seed)
    order = rng.permutation(n2)
    chunks = np.array_split(order, n_folds)
    folds = []
    for i in range(n_folds):
        test_pos = chunks[i]
        train_pos = np.concatenate([chunks[j] for j in range(n_folds) if j != i])
        folds.append((train_pos, test_pos))
    return folds


def sample_and_convert(
    net: VelocityField, x2_tilde: np.ndarray, pos: np.ndarray, b: np.ndarray
) -> np.ndarray:
    z_start = x2_tilde[pos] @ b
    z_path = euler_sample_batch(net, z_start, _TS)
    displacement = z_path - z_start[:, None, :]
    x_tilde_path: np.ndarray = x2_tilde[pos][:, None, :] + np.einsum(
        "dk,ntk->ntd", b, displacement
    )
    return x_tilde_path


@dataclass(frozen=True)
class WeightCond:
    """射影の重み条件。run_flow_matching_evaluation.pyのW1/W2/W3定義と厳密に一致させる。

    ステージ10.5の報告は、勾配検証スクリプト(verify_cafm_gradient.py)の
    「W1=np.ones(dim)そのもの(比=1.0)」という定義と、実際の指標比較を
    行ったrun_flow_matching_evaluation.pyの「W1_uniform=ones/(|x|+1e-3*Assets)
    (比は科目の規模比で決まり1.0ではない)」という定義を混同して報告して
    いた。本ステージでは後者(実際に15%の差を生んだ側)を正として使う。
    """

    name: str
    w_plus: np.ndarray
    w_minus: np.ndarray
    delta: float


@dataclass(frozen=True)
class RawPoint:
    doc_id: str
    t: float
    x_start: np.ndarray
    y: np.ndarray
    assets: float


def build_conditions(cost_config: CostConfig, names: tuple[str, ...]) -> list[WeightCond]:
    dim = len(names)
    w3_plus, w3_minus = cost_vectors(cost_config, names, non_actionable_mode="freeze")
    # W3_eq_only(§3.2のボーナス検証): non_actionableのfreeze上書き(1e6)を外し、
    # costs.yamlの素の参考値(すべて1.0)を使う。frozen_accountsの等式制約は
    # 全条件で常に維持するので、この条件でも非操作科目の値は変わらないはず。
    weq_plus, weq_minus = cost_vectors(cost_config, names, non_actionable_mode="keep")
    return [
        WeightCond("W1_uniform", np.ones(dim), np.ones(dim), 1e-3),
        WeightCond("W2_floor_only", np.ones(dim), np.ones(dim), 1e-2),
        WeightCond("W3_current", w3_plus, w3_minus, 1e-3),
        WeightCond("W3_eq_only", weq_plus, weq_minus, 1e-3),
    ]


def objective_value(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * (x - y) ** 2))


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(_DEFAULT_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    dcs = load_direction_constraints(_DEFAULT_CANONICAL)
    cost_config = load_cost_config(_DEFAULT_COSTS)
    non_actionable = tuple(cost_config.non_actionable)
    w_plus_ot, w_minus_ot = cost_vectors(cost_config, names, non_actionable_mode="ignore")
    conditions = build_conditions(cost_config, names)
    expected = {b_.name: b_.sign for b_ in spec.basis}

    out_dir = _DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_ids, filer_names, x, assets, labels = load_dataset(
        _DEFAULT_AGGREGATED_CSV, _DEFAULT_LABELS_CSV, names
    )
    x_tilde = x / assets[:, None]
    idx0 = [i for i, lab in enumerate(labels) if lab == 0]
    idx2 = [i for i, lab in enumerate(labels) if lab == 2]
    x0_tilde = x_tilde[idx0]
    x2_tilde = x_tilde[idx2]
    doc_ids2 = [doc_ids[i] for i in idx2]
    assets2 = assets[idx2]
    n2 = len(idx2)
    print(f"class0: {len(idx0)}社, class2: {n2}社")

    print("\n=== 重み比(68社全体、条件ごとの中央値・最大) ===")
    ratio_rows = []
    for cond in conditions:
        ratios = []
        for p in range(n2):
            w = combined_projection_weights(
                x2_tilde[p], 1.0, cond.w_plus, cond.w_minus, cond.delta
            )
            ratios.append(float(np.max(w) / np.min(w)))
        ratio_rows.append(
            {"condition": cond.name, "ratio_median": float(np.median(ratios)),
             "ratio_max": float(np.max(ratios))}
        )
        print(f"  {cond.name:<14} 中央値={np.median(ratios):.3e} 最大={np.max(ratios):.3e}")
    pd.DataFrame(ratio_rows).to_csv(out_dir / "weight_ratios.csv", index=False)

    re_idx = names.index("jppfs_cor_RetainedEarnings")

    folds = make_folds(n2, _N_FOLDS, seed=0)

    section11_rows: list[dict[str, float | int | str]] = []
    # section11_rowsと1対1対応(validityは後段でまとめて計算、下のコメント参照)
    section11_endpoints: list[np.ndarray] = []
    raw_points: list[RawPoint] = []  # §1.2-1.4用(seed=0のみ)

    cond_names_1_1 = [cnd.name for cnd in conditions[:3]]
    print(f"\n=== §1.1: FM学習(seed×fold)+ 3条件({cond_names_1_1})で射影 ===")
    for seed in _SEEDS:
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

            net_fm = VelocityField(dim=_K, hidden=_HIDDEN, n_layers=_N_LAYERS).double()
            train_fm(
                z_train, z0, plan_train, net_fm, epochs=_EPOCHS, lr=_LR, seed=seed,
                z_val=z_test, plan_val=plan_test,
            )
            x_tilde_path_fm = sample_and_convert(net_fm, x2_tilde, test_pos, b)

            for i_local, p in enumerate(test_pos):
                doc_id = doc_ids2[p]
                filer = filer_names[idx2[p]]
                x_start_tilde = x2_tilde[p]
                assets_start = float(assets2[p])
                raw_path = x_tilde_path_fm[i_local]  # (T, D) 生FM出力(未射影)

                if seed == _SEEDS[0]:
                    for t_i, t in enumerate(_TS):
                        raw_points.append(
                            RawPoint(
                                doc_id=doc_id, t=t, x_start=x_start_tilde,
                                y=raw_path[t_i], assets=assets_start,
                            )
                        )

                for cond in conditions[:3]:  # W1/W2/W3のみ(§1.1はこの3条件が対象)
                    weights = combined_projection_weights(
                        x_start_tilde, 1.0, cond.w_plus, cond.w_minus, cond.delta
                    )
                    path_proj = raw_path.copy()
                    for t_i in range(len(_TS)):
                        x_proj, _info = project_onto_feasible(
                            raw_path[t_i], spec, x_start=x_start_tilde, weights=weights,
                            direction_constraints=dcs, frozen_accounts=non_actionable,
                        )
                        path_proj[t_i] = x_proj
                    path_raw = path_proj * assets_start
                    sign_viol = 0
                    for t_i in range(path_proj.shape[0]):
                        for d, name in enumerate(names):
                            sgn = expected[name]
                            if sgn == "free":
                                continue
                            v = path_proj[t_i, d]
                            tol = _VIOLATION_RELATIVE_TOL
                            if (sgn == "nonneg" and v < -tol) or (sgn == "nonpos" and v > tol):
                                sign_viol += 1
                    balance_resid = float(
                        np.max(np.abs(c.T @ path_raw[-1])) / assets_start
                    )
                    section11_rows.append({
                        "seed": seed, "fold": fold_i, "docID": doc_id, "filerName": filer,
                        "condition": cond.name,
                        "proximity_l2": float(
                            np.linalg.norm(path_raw[-1] - path_raw[0]) / assets_start
                        ),
                        "sign_violations": sign_viol,
                        "endpoint_balance_residual_ratio": balance_resid,
                        "re_change": float(
                            abs(path_raw[-1, re_idx] - path_raw[0, re_idx]) / assets_start
                        ),
                    })
                    section11_endpoints.append(path_proj[-1].copy())
            print(f"  seed={seed} fold={fold_i + 1}/{_N_FOLDS} 完了(test={len(test_pos)}社)")

    # LightGBM+torchをmacOSで同一プロセス内混在させるとセグフォルトする既知の問題
    # (ステージ10、§classifiers.pyのn_jobs=1コメント参照)があり、n_jobs=1でも
    # 「LightGBMを先に・torch計算を後に」の順だと再現することが本ステージで判明した。
    # ステージ10/10.5と同じ「torch計算を先に済ませ、LightGBMは最後に1回だけ」の
    # 順序に合わせ、分類器の学習と適用をFM学習ループの外(ここ)にまとめて行う。
    is_class2 = np.array([lab == 2 for lab in labels])
    classifiers = train_classifiers(x_tilde, is_class2, seed=_SEEDS[0])
    endpoints_arr = np.stack(section11_endpoints, axis=0)
    health_arr = 1.0 - classifiers["lightgbm"].predict_proba_class2(endpoints_arr)
    section11_df = pd.DataFrame(section11_rows)
    section11_df["validity_lightgbm"] = health_arr
    section11_df.to_csv(out_dir / "section1_1_metrics.csv", index=False)
    print(f"\n{len(raw_points)}点のFM生出力を収集(seed=0)")
    # FM学習をやり直さずに§1.2-1.5を再分析できるよう、生出力を保存しておく。
    np.savez(
        out_dir / "raw_points.npz",
        doc_id=np.array([p.doc_id for p in raw_points]),
        t=np.array([p.t for p in raw_points]),
        x_start=np.stack([p.x_start for p in raw_points]),
        y=np.stack([p.y for p in raw_points]),
        assets=np.array([p.assets for p in raw_points]),
    )

    # --- §1.1 報告: 条件間の差 vs seed間のばらつき ---
    print("\n=== §1.1: 条件間の差 vs seed間のばらつき(proximity_l2, validity) ===")
    for col in ("proximity_l2", "validity_lightgbm"):
        piv = section11_df.pivot_table(
            index=["fold", "docID"], columns=["seed", "condition"], values=col
        )
        cond_diffs = []
        for seed in _SEEDS:
            if (seed, "W3_current") in piv.columns and (seed, "W1_uniform") in piv.columns:
                d = (piv[(seed, "W3_current")] - piv[(seed, "W1_uniform")]).dropna()
                cond_diffs.append(d)
        cond_diff_all = pd.concat(cond_diffs) if cond_diffs else pd.Series(dtype=float)

        seed_diffs = []
        for s_a, s_b in ((0, 1), (0, 2), (1, 2)):
            if (s_a, "W1_uniform") in piv.columns and (s_b, "W1_uniform") in piv.columns:
                d = (piv[(s_a, "W1_uniform")] - piv[(s_b, "W1_uniform")]).dropna()
                seed_diffs.append(d)
        seed_diff_all = pd.concat(seed_diffs) if seed_diffs else pd.Series(dtype=float)

        print(
            f"  {col:<20} 条件差(W3-W1)|中央値|={cond_diff_all.abs().median():.4f} "
            f"vs seed差(W1内)|中央値|={seed_diff_all.abs().median():.4f} "
            f"(比={cond_diff_all.abs().median() / max(seed_diff_all.abs().median(), 1e-12):.2f}倍)"
        )

    # ================= §1.2 / §1.3 / §1.4 =================
    print("\n=== §1.2/1.3/1.4: 射影解の直接比較・逆転検証(340点、whiten有無) ===")
    for whiten in (True, False):
        tag = "whiten_on" if whiten else "whiten_off"
        print(f"\n--- {tag} ---")
        solved: dict[tuple[int, str], np.ndarray] = {}
        for pt_i, pt in enumerate(raw_points):
            x_start_tilde = pt.x_start
            y = pt.y
            for cond in conditions:
                weights = combined_projection_weights(
                    x_start_tilde, 1.0, cond.w_plus, cond.w_minus, cond.delta
                )
                x_proj, _info = project_onto_feasible(
                    y, spec, x_start=x_start_tilde, weights=weights,
                    direction_constraints=dcs, frozen_accounts=non_actionable, whiten=whiten,
                    strict=whiten,  # whiten=Falseは既知の不良版の比較(意図的に許容)
                )
                solved[(pt_i, cond.name)] = x_proj

        # --- §1.2: x_W3 - x_W1 の乖離 ---
        # solved[...]はすでに総資産で正規化済み(pt.y自体がx2_tildeベース、
        # project_onto_feasibleはyと同じ単位系で返す)なので、ここでassetsに
        # よる除算を重ねてはいけない(重ねると10^-9桁まで潰れる、実際に
        # 本ステージでこのバグを検出し修正した)。
        diffs_l2 = []
        per_account_diff = np.zeros(len(names))
        for pt_i in range(len(raw_points)):
            d = solved[(pt_i, "W3_current")] - solved[(pt_i, "W1_uniform")]
            diffs_l2.append(float(np.linalg.norm(d)))
            per_account_diff += np.abs(d)
        per_account_diff /= len(raw_points)
        diffs_l2_arr = np.array(diffs_l2)
        print(
            f"  §1.2 ||x_W3-x_W1||/Assets: 中央値={np.median(diffs_l2_arr):.4f} "
            f"Q1={np.percentile(diffs_l2_arr, 25):.4f} Q3={np.percentile(diffs_l2_arr, 75):.4f} "
            f"最大={np.max(diffs_l2_arr):.4f}"
        )
        top10_idx = np.argsort(-per_account_diff)[:10]
        print("  科目別の乖離(上位10、総資産比の平均絶対差):")
        for i in top10_idx:
            print(f"    {names[i]:<45} {per_account_diff[i]:.4f}")

        for cond in conditions:
            resids = []
            viol_count = 0
            for pt_i in range(len(raw_points)):
                x_p = solved[(pt_i, cond.name)]
                # x_pはすでに正規化済みなので、C^Tx_pの絶対値自体が「資産比の残差」。
                resids.append(float(np.max(np.abs(c.T @ x_p))))
                for d, name in enumerate(names):
                    sgn = expected[name]
                    if sgn == "free":
                        continue
                    v = x_p[d]
                    tol = _VIOLATION_RELATIVE_TOL
                    if (sgn == "nonneg" and v < -tol) or (sgn == "nonpos" and v > tol):
                        viol_count += 1
            print(
                f"    {cond.name:<14} 恒等式残差(資産比)中央値={np.median(resids):.2e} "
                f"最大={np.max(resids):.2e} 符号違反={viol_count}件"
            )

        # --- §1.3/1.4: 逆転検証 ---
        reversal_rows: list[dict[str, float | int | str | bool]] = []
        # (own_name, pt_i, ratio)。§1.5で実際に逆転した点だけを検算するために保持する。
        all_reversal_points: list[tuple[str, int, float]] = []
        for own_name, other_name in (("W3_current", "W1_uniform"), ("W1_uniform", "W3_current")):
            own_cond = next(cnd for cnd in conditions if cnd.name == own_name)
            n_reversal = 0
            n_reversal_nontrivial = 0  # t>0のみ(t=0はy=x_startで自明に実行可能、後述)
            ratios = []
            reversal_detail = []
            for pt_i, pt in enumerate(raw_points):
                x_start_tilde = pt.x_start
                y = pt.y
                weights_own = combined_projection_weights(
                    x_start_tilde, 1.0, own_cond.w_plus, own_cond.w_minus, own_cond.delta
                )
                f_own = objective_value(solved[(pt_i, own_name)], y, weights_own)
                f_other = objective_value(solved[(pt_i, other_name)], y, weights_own)
                ratio = f_own / max(f_other, 1e-30)
                ratios.append(ratio)
                if ratio > 1.0 + _REVERSAL_REL_TOL:
                    n_reversal += 1
                    if pt.t > 0:
                        n_reversal_nontrivial += 1
                    reversal_detail.append((pt.doc_id, pt.t, ratio))
                    all_reversal_points.append((own_name, pt_i, ratio))
            print(
                f"  §1.3/1.4 [{tag}] f_{own_name}(x_{own_name}) <= f_{own_name}(x_{other_name})? "
                f"逆転={n_reversal}/{len(raw_points)}件"
                f"(うちt=0以外の実質的逆転={n_reversal_nontrivial}件)、"
                f"比率分布: 中央値={np.median(ratios):.4f} 最大={np.max(ratios):.4f}"
            )
            print(
                "    (t=0の逆転はy=x_start(既に実行可能)を射影する自明ケースで、"
                "両辺の目的関数値が~1e-30と機械精度以下になるため比が不安定になる"
                "だけで、実際の最適化の破綻ではない。§1.5で検算する。)"
            )
            if reversal_detail:
                worst = sorted(reversal_detail, key=lambda r: -r[2])[:5]
                print(f"    逆転上位5件(docID, t, ratio): {worst}")
            reversal_rows.append({
                "whiten": whiten, "own": own_name, "other": other_name,
                "n_reversal": n_reversal, "n_reversal_nontrivial": n_reversal_nontrivial,
                "n_total": len(raw_points),
                "ratio_median": float(np.median(ratios)), "ratio_max": float(np.max(ratios)),
            })
        pd.DataFrame(reversal_rows).to_csv(out_dir / f"reversal_{tag}.csv", index=False)

        # --- §1.5: 逆転が出た場合のみ、実際に逆転した点を独立ソルバで検算 ---
        # (whiten_onの生産設定でのみ行う。whiten_offは既に§1.4で明らかに破綻して
        # いる—比が最大1e31—ため、追加の検算は情報を増やさない。)
        if whiten and all_reversal_points:
            print("\n  §1.5: 逆転が検出された点を独立ソルバで検算します(比の大きい順、最大10件)")
            top_reversals = sorted(all_reversal_points, key=lambda r: -r[2])[:10]
            for own_name, pt_i, ratio in top_reversals:
                pt = raw_points[pt_i]
                own_cond = next(cnd for cnd in conditions if cnd.name == own_name)
                weights_own = combined_projection_weights(
                    pt.x_start, 1.0, own_cond.w_plus, own_cond.w_minus, own_cond.delta
                )
                objs = {
                    "OSQP(元の解)": objective_value(solved[(pt_i, own_name)], pt.y, weights_own)
                }
                for solver_name in ("ECOS", "CLARABEL", "SCS"):
                    x_s, _info = project_onto_feasible(
                        pt.y, spec, x_start=pt.x_start, weights=weights_own,
                        direction_constraints=dcs, frozen_accounts=non_actionable,
                        solver=solver_name,
                    )
                    objs[solver_name] = objective_value(x_s, pt.y, weights_own)
                print(
                    f"    {pt.doc_id} t={pt.t} own={own_name} 逆転比={ratio:.4f}: {objs}"
                )
        elif whiten:
            print("\n  §1.5: 逆転0件のためスキップ")

    print(f"\n出力先: {out_dir}")


if __name__ == "__main__":
    main()
