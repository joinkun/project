"""識別AUC(生成点 vs 実データ)と、その原因分解(ステージ13 §D)。

lightgbm/sklearnを使うので、**torchを使うプロセスからimportしないこと**
(ステージ11-Aのプロセス分離)。両者は関数内で遅延importする。
"""

from __future__ import annotations

import numpy as np

from fsgen.evaluation.classifiers import strict_sklearn_convergence

_SEED = 42


def identification_auc_repeated(
    generated: np.ndarray,
    real0: np.ndarray,
    repeats: int,
    *,
    n_splits: int = 5,
    seed: int = _SEED,
) -> dict[str, np.ndarray]:
    """生成点 vs class0の無作為抽出点を識別するAUC(LightGBM/LogReg、5-fold CV)。

    class0の抽出をrepeats回変える。抽出の乱数列は``seed``のみで決まるので、
    手法間で同じ抽出を使い対応のある比較ができる。sklearnの未収束は例外
    (§A.1)、CV中の例外も握りつぶさない(``error_score="raise"``)。
    """
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    out: dict[str, list[float]] = {"lightgbm": [], "logreg": []}
    for rep in range(repeats):
        real = real0[rng.choice(len(real0), size=min(len(generated), len(real0)), replace=False)]
        x = np.concatenate([generated, real])
        y = np.concatenate([np.ones(len(generated)), np.zeros(len(real))])
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        gbm = lgb.LGBMClassifier(
            n_estimators=200, num_leaves=15, random_state=seed + rep, n_jobs=1,
            class_weight="balanced", verbosity=-1,
        )
        lin = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed + rep)
        )
        with strict_sklearn_convergence():
            for name, model in (("lightgbm", gbm), ("logreg", lin)):
                scores = cross_val_score(model, x, y, cv=cv, scoring="roc_auc", error_score="raise")
                out[name].append(float(np.mean(scores)))
    return {k: np.array(v) for k, v in out.items()}


__all__ = ["identification_auc_repeated"]
