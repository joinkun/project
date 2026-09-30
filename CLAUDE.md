# fsgen

EDINET の貸借対照表から、GC 注記企業の再建経路を勘定科目レベルで生成する研究。
**経緯・最終設定・却下した設計・既知の限界は `HANDOFF.md` にある。作業前に必ず読むこと。**

## ルール

- プロセス分離を守る(POT / torch・cvxpy / lightgbm・sklearn を別プロセスに)。`KMP_DUPLICATE_LIB_OK` は使わない。
- シードは 0/1/2(分類器は 42)。`data/interim/industry_12/folds.npz` は作り直さない。
- `uv run pytest -q`、`uv run mypy --strict src scripts tests`、`uv run ruff check .` をクリーンに保つ
  (`pipelines/p03_build_dataset.py` の E501 のみ既存の例外)。
- 最適化は必ず収束を検査し、未収束は例外にする。
- 14 項目チェックの違反 0 が必須条件(ステージ40 で 14 を追加)。評価は企業単位(n=68)のペア検定 + ブートストラップ CI。
- 報告は日本語。結論が変わることを恐れず、数字を有利に見せない。BS 表は目で読む。
- 論文に使う結果は `results/stageNN/` に置く(`data/` は gitignore 対象)。
