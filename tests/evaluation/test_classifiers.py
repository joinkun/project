"""classifiers.py のテスト。小さな合成データで3分類器が学習・予測できることを検証する。"""

from __future__ import annotations

import numpy as np

from fsgen.evaluation.classifiers import train_classifiers


def test_train_classifiers_returns_three_models_with_separable_predictions() -> None:
    rng = np.random.default_rng(0)
    x0 = rng.normal(loc=0.0, scale=0.1, size=(200, 5))
    x2 = rng.normal(loc=3.0, scale=0.1, size=(30, 5))  # class0とはっきり分離
    x = np.concatenate([x0, x2], axis=0)
    is_class2 = np.array([False] * len(x0) + [True] * len(x2))

    clfs = train_classifiers(x, is_class2, seed=0)
    assert set(clfs.keys()) == {"lightgbm", "logreg", "random_forest"}

    for clf in clfs.values():
        proba = clf.predict_proba_class2(x)
        assert proba.shape == (len(x),)
        # 明確に分離されたデータなので、class0側は低確率・class2側は高確率になるはず
        assert proba[~is_class2].mean() < 0.3
        assert proba[is_class2].mean() > 0.7
