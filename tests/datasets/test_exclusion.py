from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fsgen.datasets.exclusion import exclusion_masks, load_excluded_docs, subset_folds


def test_subset_folds_renumbers_positions() -> None:
    folds = {"train_pos_0": np.array([0, 2, 3]), "test_pos_0": np.array([1, 4])}
    keep = np.array([True, False, True, True, False])
    out = subset_folds(folds, keep)
    # 残る始点 0,2,3 は新しい位置 0,1,2
    assert out["train_pos_0"].tolist() == [0, 1, 2]
    assert out["test_pos_0"].tolist() == []


def test_subset_folds_keeps_assignment() -> None:
    folds = {"train_pos_0": np.array([1, 3]), "test_pos_0": np.array([0, 2])}
    keep = np.array([True, True, False, True])
    out = subset_folds(folds, keep)
    assert out["test_pos_0"].tolist() == [0]
    assert out["train_pos_0"].tolist() == [1, 2]


def test_subset_folds_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        subset_folds({"test_pos_0": np.array([5])}, np.array([True, True]))


def test_exclusion_masks() -> None:
    docs = ["a", "b", "c", "d"]
    labels = [0, 2, 2, 0]
    row, src = exclusion_masks(docs, labels, frozenset({"b", "d"}))
    assert row.tolist() == [True, False, True, False]
    assert src.tolist() == [False, True]


def test_load_excluded_docs(tmp_path: Path) -> None:
    assert load_excluded_docs(None) == frozenset()
    p = tmp_path / "ex.csv"
    p.write_text("docID,reason\nS1,golf\nS2,financial\n", encoding="utf-8")
    assert load_excluded_docs(p) == frozenset({"S1", "S2"})
