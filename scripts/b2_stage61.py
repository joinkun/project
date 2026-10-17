#!/usr/bin/env python
"""ステージ61 B.2・B.3: 2 社の平均が実在企業と区別できる理由(lightgbm・sklearn のプロセス)。

規則は ``results/stage61/criteria.md`` §5。class 0 から無作為に作った 2 社の組(seed 0)の平均を、
class 0 の実在企業と比べる(識別 AUC、``identification_auc_repeated``、繰り返し 3 回)。特徴量の版:

- (i) ゼロかどうか(32 科目の 0/1)だけ
- (ii) 平均からのずれを √2 倍して分散の縮みを戻した値
- (iii) 元の値

あわせて、科目ごとの KS 統計量・ゼロの割合・分散の比。参考に、class 0 の実在企業の無作為抽出
(55 社)の AUC(定義上 0.5 付近)も計算する。

    uv run python scripts/b2_stage61.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402
from tilde_stage51 import load_tilde  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.evaluation.identification import identification_auc_repeated  # noqa: E402

OUT = Path("results/stage61")
N_PAIRS = 500


def main() -> None:
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    x, lab, _docs, _keep = load_tilde(names)
    real = x[lab == 0]
    rng = np.random.default_rng(0)
    idx = np.array([rng.choice(len(real), 2, replace=False) for _ in range(N_PAIRS)])
    avg = real[idx].mean(axis=1)
    mu = real.mean(axis=0)
    restored = mu + np.sqrt(2.0) * (avg - mu)
    rand55 = real[rng.choice(len(real), 55, replace=False)]

    def auc(gen: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
        r = identification_auc_repeated(gen, ref, 3)
        return float(r["lightgbm"].mean()), float(r["logreg"].mean())

    rows = [("(iii) 元の値: 2 社の平均", *auc(avg, real)),
            ("(ii) 分散の縮みを戻した 2 社の平均", *auc(restored, real)),
            ("(i) ゼロかどうかだけ: 2 社の平均", *auc((avg == 0).astype(float),
                                                  (real == 0).astype(float))),
            ("参考: class 0 の実在企業の無作為抽出(55 社)", *auc(rand55, real))]
    lines = ["# ステージ61 B.2: 2 社の平均が区別できる理由", "",
             f"class 0({len(real):,} 社)から無作為に {N_PAIRS} 組の 2 社(seed 0)を作り、その平"
             f"均を class 0 の"
             "実在企業と比べた(5-fold CV、3 回の平均、seed 42)。", "",
             "| 特徴量 | AUC LightGBM | AUC LogReg |", "|---|---:|---:|"]
    lines += [f"| {k} | {a:.3f} | {b:.3f} |" for k, a, b in rows]
    acc = []
    for j, n in enumerate(names):
        ks = stats.ks_2samp(real[:, j], avg[:, j]).statistic
        z_r, z_a = float((real[:, j] == 0).mean()), float((avg[:, j] == 0).mean())
        vr = float(avg[:, j].var() / real[:, j].var()) if real[:, j].var() > 0 else float("nan")
        acc.append((ja.get(n, n), ks, z_r, z_a, vr))
    df = pd.DataFrame(acc, columns=["科目", "KS", "ゼロの割合(実データ)", "ゼロの割合(2 社の平均)",
                                    "分散の比(平均 / 実データ)"])
    df.to_csv(OUT / "b2_accounts.csv", index=False)
    lines += ["", "## 科目ごと(KS の大きい順)", "",
              "| 科目 | KS | ゼロの割合 実データ | 2 社の平均 | 分散の比 |", "|---|---:|---:|---"
                                                          ":|---:|"]
    for r in df.sort_values("KS", ascending=False).itertuples(index=False):
        lines.append(f"| {r[0]} | {r[1]:.3f} | {r[2]:.1%} | {r[3]:.1%} | {r[4]:.2f} |")
    zero_gap = (df["ゼロの割合(実データ)"] - df["ゼロの割合(2 社の平均)"])
    lines += ["", f"ゼロの割合の差(実データ − 平均)の中央値: {zero_gap.median():.1%}、最大 "
              f"{zero_gap.max():.1%}。分散の比の中央値:"
              f" {df['分散の比(平均 / 実データ)'].median():.2f}"
              "(独立な 2 社なら 0.5)。", ""]
    (OUT / "b2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
