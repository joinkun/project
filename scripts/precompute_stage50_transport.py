#!/usr/bin/env python
"""ステージ50: 分布由来の尤度コストによる対応先(POT のプロセス。torch/sklearn は無効化)。

設計と基準は ``results/stage50/criteria.md``(生成の前に記録)。ステージ49 の当てはめ
(``results/stage49/fits.csv``)とゼロの区分の件数(``zeros.csv``)から科目ごとの確率モデルを作り、
c(x_i, y_j) = Σ_a −log P(y_a | x_a) の行列で対応先を選ぶ(``fsgen.models.likelihood_cost``)。

出力(``--out-dir``、既定 ``data/interim/stage50/plans``):
- ``plans_{版}.npz``: 現行と同じ形(``plan_full``、``plan_{train,test}_{k}``)
- ``full_targets_{版}.csv``: 全社(class 2 57 社)の対応先
- ``results/stage50/cost_checks.csv``: B.3 の数値の確認(有限か、最大 / 中央値、計算時間、脱落)
- ``results/stage50/period.csv``: A.6 の案 1 と案 2(位置 × 5、尺度 × √5)で対応先が変わった社数

    uv run python scripts/precompute_stage50_transport.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("POT_BACKEND_DISABLE_PYTORCH", "1")
sys.modules.setdefault("sklearn", None)  # type: ignore[arg-type]

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from precompute_stage13_transport import load_data  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.datasets.exclusion import load_excluded_docs, subset_folds  # noqa: E402
from fsgen.models.likelihood_cost import (  # noqa: E402
    AccountModel,
    Family,
    conditional_probs,
    likelihood_cost_matrix,
    scale_period,
)
from fsgen.models.transport import check_plan_health, solve_unbalanced_normalized  # noqa: E402

_FITS = Path("results/stage49/fits.csv")
_ZEROS = Path("results/stage49/zeros.csv")
_PANEL = Path("data/interim/stage23/panel/panel.csv")
_FIRM_YEARS = Path("data/interim/stage30/firm_years.csv")
_FOLDS = Path("data/interim/industry_12/folds.npz")
_EXCLUDE = Path("data/interim/stage28/excluded.csv")
_N_FOLDS = 5
_EPS, _TAU = 0.05, 0.1  # 不均衡 OT(ステージ39 と同じ)
_YEARS = 5.0
# コスト 0 の科目: 非操作 3 科目(nearest_na0 と同じ)と 4 つの丸め差の残余(案 A)
NON_ACTIONABLE = ("jppfs_cor_RetainedEarnings", "valuation_difference_securities",
                  "non_controlling_interests")
ROUNDING_RESIDUALS = ("__OTHER__jppfs_cor_Assets", "__OTHER__jppfs_cor_NoncurrentAssets",
                      "__OTHER__jppfs_cor_Liabilities", "__OTHER__jppfs_cor_ShareholdersEquity")
# 版 → (ステージ49 の当てはめの version, family, 当てはめの分布族, π の母集団)
VERSIONS: dict[str, tuple[str, Family, str, str]] = {
    "sub_all": ("全件", "subbotin", "Subbotin", "all"),
    "sub_trim": ("外れ値を除く", "subbotin", "Subbotin", "all"),
    "t": ("全件", "t", "t", "all"),
    "lognorm": ("全件", "normal", "正規", "all"),
    "t_pi12": ("全件", "t", "t", "class12"),
}
CATS = ("both0", "new", "gone", "both", "flip")


def class12_counts(names: tuple[str, ...]) -> tuple[pd.DataFrame, int]:
    """推定用パネルのうち始点の年度が class 1・2 の組で、ゼロの区分を数える。

    定義はステージ49 と同じ。
    """
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    p = p[(p["prev_total_assets"] > 0) & (p["cur_total_assets"] > 0)].reset_index(drop=True)
    fy = pd.read_csv(_FIRM_YEARS, dtype={"edinetCode": str})
    lab = fy.set_index(["edinetCode", "year"])["label"]
    cls = np.array([lab.get((c, t - 1), np.nan) for c, t in zip(p["edinetCode"], p["t"],
                                                                 strict=True)])
    p = p[np.isin(cls, [1, 2])]
    rows = []
    for n in names:
        x0 = p[f"prev_{n}"].fillna(0.0).to_numpy() / p["prev_total_assets"].to_numpy()
        x1 = p[f"cur_{n}"].fillna(0.0).to_numpy() / p["cur_total_assets"].to_numpy()
        nz0, nz1 = x0 != 0, x1 != 0
        same = nz0 & nz1 & (np.sign(x0) == np.sign(x1))
        rows.append({"account": n, "both0": int((~nz0 & ~nz1).sum()),
                     "new": int((~nz0 & nz1).sum()), "gone": int((nz0 & ~nz1).sum()),
                     "both": int(same.sum()), "flip": int((nz0 & nz1 & ~same).sum())})
    return pd.DataFrame(rows).set_index("account"), len(p)


def build_models(version: str, names: tuple[str, ...]) -> tuple[list[AccountModel | None],
                                                                pd.DataFrame]:
    fit_version, family, fit_family, pi_pop = VERSIONS[version]
    fits = pd.read_csv(_FITS)
    fits = fits[(fits["version"] == fit_version) & (fits["family"] == fit_family)]
    fits = fits.set_index("account")
    if pi_pop == "all":
        z = pd.read_csv(_ZEROS).set_index("account")
        counts = pd.DataFrame({c: (z[c] * z["n"]).round().astype(int) for c in CATS})
        n_obs = int(z["n"].iloc[0])
    else:
        counts, n_obs = class12_counts(names)
    pi_min = 1.0 / n_obs
    models: list[AccountModel | None] = []
    info = []
    for n in names:
        if n in NON_ACTIONABLE or n in ROUNDING_RESIDUALS:
            models.append(None)
            continue
        f = fits.loc[n]
        shape = {"subbotin": f["p_b"], "t": f["p_df"], "normal": np.nan}[family]
        cnt = {c: int(counts.loc[n, c]) for c in CATS}
        pr = conditional_probs(cnt["both0"], cnt["new"], cnt["gone"], cnt["both"], cnt["flip"],
                               pi_min)
        models.append(AccountModel(family, float(f["p_loc"]), float(f["p_scale"]),
                                   float(shape), **pr))
        info.append({"version": version, "account": n, "loc": f["p_loc"],
                     "scale": f["p_scale"], "shape": shape, **pr,
                     "n_clipped": int(sum(v == pi_min for v in pr.values())),
                     "pi_min": pi_min})
    return models, pd.DataFrame(info)


def nearest(cost: np.ndarray) -> np.ndarray:
    n0 = cost.shape[0]
    plan = np.zeros_like(cost)
    plan[np.arange(n0), np.argmin(cost, axis=1)] = 1.0 / n0
    check_plan_health(plan, np.full(n0, 1.0 / n0), check_marginals=True)
    return plan


def unbalanced(cost: np.ndarray) -> np.ndarray:
    c = cost - cost.min()
    c = c / np.median(c)
    n0, n1 = c.shape
    return solve_unbalanced_normalized(c, np.full(n0, 1.0 / n0), np.full(n1, 1.0 / n1), _EPS, _TAU,
                                       allow_dropped_sources=True)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("data/interim/stage50/plans"))
    ap.add_argument("--results", type=Path, default=Path("results/stage50"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    x, assets, label, doc, filer, edinet, src_keep = load_data(
        names, load_excluded_docs(_EXCLUDE), 2)
    idx0, idx2 = np.where(label == 0)[0], np.where(label == 2)[0]
    x0, x2 = x[idx0], x[idx2]
    folds = subset_folds(np.load(_FOLDS), src_keep)
    print(f"始点 {len(idx2)} 社、目標 {len(idx0)} 社、科目 {len(names)}", flush=True)

    checks, infos, period_rows = [], [], []
    designs = [(v, "nearest") for v in VERSIONS] + [("t", "unb")]
    for version, kind in designs:
        models, info = build_models(version, names)
        if kind == "nearest":
            infos.append(info)
        tag = version if kind == "nearest" else f"{version}_unb"
        t0 = time.perf_counter()
        cost_full = likelihood_cost_matrix(x2, x0, models)
        t_eval = time.perf_counter() - t0
        solve = nearest if kind == "nearest" else unbalanced
        plans = {"plan_full": solve(cost_full)}
        for k in range(_N_FOLDS):
            for part in ("train", "test"):
                pos = folds[f"{part}_pos_{k}"]
                plans[f"plan_{part}_{k}"] = solve(likelihood_cost_matrix(x2[pos], x0, models))
        np.savez_compressed(args.out_dir / f"plans_{tag}.npz", **plans)  # type: ignore[arg-type]
        pf = plans["plan_full"]
        row_mass = pf.sum(axis=1)
        alive = row_mass > 1e-12
        tg = np.argmax(pf, axis=1)
        shifted = cost_full - cost_full.min()
        dropped = {f"dropped_train_{k}": int((plans[f"plan_train_{k}"].sum(axis=1) <= 1e-12).sum())
                   for k in range(_N_FOLDS)}
        checks.append({
            "design": tag, "finite": bool(np.isfinite(cost_full).all()),
            "cost_min": float(cost_full.min()), "cost_median": float(np.median(cost_full)),
            "cost_max": float(cost_full.max()),
            "max_over_median_shifted": float(shifted.max() / np.median(shifted)),
            "n_accounts_in_cost": sum(m is not None for m in models),
            "n_clipped_probs": int(info["n_clipped"].sum()) if kind == "nearest" else np.nan,
            "seconds_full_cost": t_eval, "seconds_total": time.perf_counter() - t0,
            "n_dropped_full": int((~alive).sum()), "transported_mass": float(pf.sum()),
            "unique_targets": int(len(set(tg[alive].tolist()))), **dropped,
        })
        pd.DataFrame({
            "design": tag, "source_docID": [doc[i] for i in idx2],
            "source_name": [filer[i] for i in idx2], "source_edinet": [edinet[i] for i in idx2],
            "target_docID": [doc[idx0[j]] for j in tg],
            "target_name": [filer[idx0[j]] for j in tg],
            "target_edinet": [edinet[idx0[j]] for j in tg],
            "alive": alive, "row_mass": row_mass * len(idx2),
            "asset_ratio": assets[idx0][tg] / assets[idx2],
        }).to_csv(args.out_dir / f"full_targets_{tag}.csv", index=False)
        if kind == "nearest" and version != "t_pi12":  # A.6 案 2
            m5 = [scale_period(m, _YEARS) if m is not None else None for m in models]
            tg5 = np.argmin(likelihood_cost_matrix(x2, x0, m5), axis=1)
            n_fold_changed = 0
            for k in range(_N_FOLDS):
                pos = folds[f"train_pos_{k}"]
                a1 = np.argmax(plans[f"plan_train_{k}"], axis=1)
                a5 = np.argmin(likelihood_cost_matrix(x2[pos], x0, m5), axis=1)
                n_fold_changed += int((a1 != a5).sum())
            period_rows.append({"version": version, "n_sources": len(idx2),
                                "changed_full": int((tg != tg5).sum()),
                                "changed_train_rows": n_fold_changed,
                                "train_rows": sum(len(folds[f"train_pos_{k}"])
                                                  for k in range(_N_FOLDS))})
        print(f"{tag}: {checks[-1]}", flush=True)
    pd.DataFrame(checks).to_csv(args.results / "cost_checks.csv", index=False)
    pd.concat(infos).to_csv(args.results / "account_models.csv", index=False)
    pd.DataFrame(period_rows).to_csv(args.results / "period.csv", index=False)
    print(pd.DataFrame(period_rows).to_string(index=False))


if __name__ == "__main__":
    main()
