#!/usr/bin/env python
"""ステージ80: コスト {abs, R1, A1} × 目標 {argmax, F1, SP-F1} の D1 の比較。

lightgbm・sklearn のプロセス。

規則は ``results/stage80/criteria.md`` §2。集計は ``stage79_evaluate.run``
(ステージ75 と同じ較正・参照・条件)。ステージ75・78・79 の生成は再利用し、
新規は A1・argmax と γ' = 0.1 の SP-F1 の 3 設定。

出力: ``results/stage80/d1/comparison_report.md``・``summary.json``。

    uv run python scripts/stage80_evaluate.py
"""

from __future__ import annotations

from pathlib import Path

from stage79_evaluate import BASE_KEY, Setting, run
from tables_stage69 import INTERIM

D75, D78, D79, D80 = (INTERIM / f"stage{n}" for n in (75, 78, 79, 80))
MAIN = "A1・SP-F1(主提案、γ' = 0.1)"
SETTINGS: dict[str, Setting] = {
    "abs・argmax": Setting(D78, "abs_argmax", "abs", plan_dir=D75, argmax=True),
    "R1・argmax": Setting(D78, "R1_argmax", "R1", plan_dir=D75, argmax=True),
    "A1・argmax": Setting(D80, "A1_argmax", "A1", plan_dir=D79, argmax=True),
    "abs・F1": Setting(D75, "abs_f1", "abs"),
    BASE_KEY: Setting(D75, "R1_f1", "R1"),
    "A1・F1": Setting(D79, "A1_f1", "A1"),
    "abs・SP-F1": Setting(D80, "abs_sp", "abssp"),
    "R1・SP-F1": Setting(D80, "R1_sp", "R1sp"),
    MAIN: Setting(D80, "A1_sp", "A1sp"),
    "参考: A1・SP-F1(ステージ79、γ' = 10⁻⁴)": Setting(D79, "A1sp_f1", "A1sp"),
}


def main() -> None:
    run(SETTINGS, Path("results/stage80/d1"),
        "# ステージ80: コスト × 目標の D1 の比較(5 fold の交差適合)",
        "`results/stage80/criteria.md`(A1・SP-F1 の採用はステージ79 を見たあとの決定。"
        "本表は事後の解析)", diag=(MAIN, BASE_KEY))


if __name__ == "__main__":
    main()
