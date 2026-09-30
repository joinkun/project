"""validity 判定用の分類器(ステージ8 §B.2, §B.4)。

経路生成に OT の対応先(class 0 側)を使っているため、生成結果を同じ情報
(OTの対応先)で「健全に近づいたか」を判定すると循環してしまう。そのため
学習に使っていない独立の分類器(class 2 vs class 0 の二値分類)で
P(class 2 | x) を測る。

class 2 (68) : class 0 (~3951) は極端な不均衡なので、各分類器のクラス
重み付け機能を使う。分類器はLOOの外側で1回だけ、全社(class 0 全社 + class
2 全社)で学習する(§B.3: 1社の除外が学習全体に与える影響は無視できるという
判断。妥当性は報告書で議論する)。

Slack et al. (2021) が示した「反実仮想説明は分類器の設計次第で操作可能」
という指摘を踏まえ、性質の異なる3種類(勾配ブースティング・線形・
ランダムフォレスト)を用意し、手法間の優劣が分類器選択に依存しないかを
§B.4 で確認する。
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_SEED = 42
_LOGREG_MAX_ITER = 2000


@contextmanager
def strict_sklearn_convergence() -> Iterator[None]:
    """sklearnのConvergenceWarningを例外にする(ステージ13 §A.1)。

    ``LogisticRegression`` 等は未収束でも警告を出すだけで係数を返す。
    ``cross_val_score`` はさらに例外を握りつぶす(``error_score=nan``)ため、
    呼び出し側は ``error_score="raise"`` も併用すること。
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        yield


class ValidityClassifier(Protocol):
    def predict_proba_class2(self, x_normalized: np.ndarray) -> np.ndarray:
        """(n, D)の正規化済み状態ベクトルを受け取り、P(class 2)の配列(n,)を返す。"""
        ...


@dataclass
class _SklearnLikeClassifier:
    model: object
    name: str

    def predict_proba_class2(self, x_normalized: np.ndarray) -> np.ndarray:
        proba: np.ndarray = self.model.predict_proba(x_normalized)[:, 1]  # type: ignore[attr-defined]
        return proba


def train_classifiers(
    x_normalized: np.ndarray, is_class2: np.ndarray, seed: int = _SEED
) -> dict[str, _SklearnLikeClassifier]:
    """3種類の分類器をそれぞれ1回学習する(§B.4)。

    x_normalized: (n, D) 総資産で正規化した状態ベクトル。
    is_class2: (n,) bool、class2ならTrue。

    注意(ステージ34 で判明): これまでの呼び出し側(各スクリプトの ``load_real``)は、ラベルのある
    書類をすべて渡しており、class 1(重要事象のみ)も「class 2 でない」側に含めて学習している。
    class 1 を除いて学習し直しても、最終設定の validity LightGBM は 0.998 で変わらなかった。
    """
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=15,
        random_state=seed,
        # n_jobs=1: 防御的な設定。torchとlightgbmが同一プロセスに同居すると
        # libomp.dylibが二重ロードされてクラッシュする問題(ステージ11-A §A.1)は、
        # n_jobs=1では防げない。実際の対策はプロセス分離(生成と指標計算を別プロセスに)。
        n_jobs=1,
        class_weight="balanced",
        verbosity=-1,
    )
    lgb_clf.fit(x_normalized, is_class2.astype(int))

    # 特徴量の尺度が科目ごとに大きく異なる(残余科目は最大std~0.35、一部は~0.0002)ため、
    # LogisticRegressionは標準化が必須(木ベースのlightgbm/random_forestは尺度に不変)。
    # 標準化を欠いていたステージ13〜16では、生成点のvalidity(logreg)が始点(class2実企業)
    # 自体より低く出て、識別できていなかった(ステージ17 §A で発見・修正)。
    # identification.py/metrics.pyの臨時LogRegは元々StandardScalerを使っており、ここだけ
    # 抜けていた。
    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=_LOGREG_MAX_ITER, random_state=seed),
    )
    with strict_sklearn_convergence():
        logreg.fit(x_normalized, is_class2.astype(int))

    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=seed
    )
    rf.fit(x_normalized, is_class2.astype(int))

    return {
        "lightgbm": _SklearnLikeClassifier(lgb_clf, "lightgbm"),
        "logreg": _SklearnLikeClassifier(logreg, "logreg"),
        "random_forest": _SklearnLikeClassifier(rf, "random_forest"),
    }


__all__ = ["ValidityClassifier", "strict_sklearn_convergence", "train_classifiers"]
