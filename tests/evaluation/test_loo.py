"""loo.py のテスト。合成データで小規模なLOOが最後まで走ることを検証する。"""

from __future__ import annotations

import numpy as np

from fsgen.accounting.constraints import BasisSpec, CanonicalSpec, SubtotalSpec
from fsgen.evaluation.loo import LooDataset, run_loo

_BASIS = (
    # run_loo は build_prior_work_baseline 経由で実際の科目名(inventories等)を
    # 直接参照するため、合成スペックでもこれらの名前をそのまま使う。
    BasisSpec("inventories", "jppfs_cor_CurrentAssets", "assets", sign="nonneg"),
    BasisSpec("jppfs_cor_OtherCA", "jppfs_cor_CurrentAssets", "assets", sign="nonneg"),
    BasisSpec(
        "__OTHER__jppfs_cor_NoncurrentAssets", "jppfs_cor_NoncurrentAssets", "assets", sign="free"
    ),
    BasisSpec("STL", "jppfs_cor_CurrentLiabilities", "liabilities", sign="nonneg"),
    BasisSpec("Treasury", "jppfs_cor_ShareholdersEquity", "net_assets", sign="nonpos"),
    BasisSpec("Other", "jppfs_cor_ShareholdersEquity", "net_assets", sign="free"),
)
_SUBTOTALS = (
    SubtotalSpec("jppfs_cor_Assets", "assets"),
    SubtotalSpec("jppfs_cor_Liabilities", "liabilities"),
    SubtotalSpec("jppfs_cor_NetAssets", "net_assets"),
)
_SPEC = CanonicalSpec(name="synthetic", basis=_BASIS, subtotals=_SUBTOTALS)
_NAMES = (
    "inventories",
    "jppfs_cor_OtherCA",
    "__OTHER__jppfs_cor_NoncurrentAssets",
    "STL",
    "Treasury",
    "Other",
)


def _feasible_point(cash: float, stl: float, treasury: float) -> np.ndarray:
    other_ca = 0.2 * cash
    other_nca = 0.1 * cash
    other = (cash + other_ca + other_nca) - stl - treasury
    return np.array([cash, other_ca, other_nca, stl, treasury, other])


def _make_dataset() -> LooDataset:
    from fsgen.accounting.constraints import build_constraint_matrix

    rng = np.random.default_rng(7)
    c = build_constraint_matrix(_SPEC, include_subtotals=False)

    n0, n2 = 40, 4
    x0 = np.stack(
        [
            _feasible_point(
                100 + rng.normal(scale=5), 30 + rng.normal(scale=3), -10 + rng.normal(scale=2)
            )
            for _ in range(n0)
        ]
    )
    x2 = np.stack(
        [
            _feasible_point(
                50 + rng.normal(scale=5), 80 + rng.normal(scale=3), -3 + rng.normal(scale=1)
            )
            for _ in range(n2)
        ]
    )
    x = np.concatenate([x0, x2], axis=0)
    assets = x[:, 0] + 200.0  # 常に正の適当な資産規模
    labels = [0] * n0 + [2] * n2
    doc_ids = [f"D{i}" for i in range(n0 + n2)]
    filer_names = [f"F{i}" for i in range(n0 + n2)]
    return LooDataset(
        doc_ids=doc_ids,
        filer_names=filer_names,
        x=x,
        assets=assets,
        labels=labels,
        names=_NAMES,
        spec=_SPEC,
        c=c,
        direction_constraints=(),
    )


def test_run_loo_produces_all_methods_for_all_companies() -> None:
    ds = _make_dataset()
    result = run_loo(ds, ts=(0.0, 0.5, 1.0), k=2, epsilon=0.1)
    methods = set(result.metrics_df["method"])
    assert methods == {"prior_work", "convex_qp", "naive_ot", "gaussian", "proposed"}
    assert len(result.metrics_df) == len(ds.idx2) * 5
    for method in methods:
        assert result.endpoints[method].shape == (len(ds.idx2), len(_NAMES))


def test_run_loo_progress_callback_reaches_total() -> None:
    ds = _make_dataset()
    calls: list[tuple[int, int]] = []

    def _record(done: int, total: int) -> None:
        calls.append((done, total))

    run_loo(ds, ts=(0.0, 1.0), k=2, epsilon=0.1, on_progress=_record)
    assert calls[-1] == (len(ds.idx2), len(ds.idx2))


def test_proposed_method_has_no_sign_violations() -> None:
    ds = _make_dataset()
    result = run_loo(ds, ts=(0.0, 0.5, 1.0), k=2, epsilon=0.1)
    proposed = result.metrics_df[result.metrics_df["method"] == "proposed"]
    assert (proposed["sign_violations"] == 0).all()
