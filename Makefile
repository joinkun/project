.PHONY: sync test lint format typecheck fetch parse build-dataset clean evaluate evaluate-industry evaluate-stage13 evaluate-stage14

sync:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src

fetch:
	uv run python pipelines/p01_fetch.py

parse:
	uv run python pipelines/p02_parse.py

build-dataset:
	uv run python pipelines/p03_build_dataset.py

# ステージ11-A §A: torch(生成)とlightgbm(指標)を同一プロセスに同居させると
# macOSでOpenMPランタイムの二重ロードによりセグフォルトする(§A.1)。
# 3段階に分離して実行する:
#   1. precompute_transport.py: OT(POT)の計算のみ。torch/lightgbmは一切importしない
#      (POTがtorch/sklearnを間接的にimportしようとするのを明示的に無効化している)。
#   2. run_generation.py: torch/cvxpy/cvxpylayersのみ。lightgbm/sklearnは一切importしない。
#   3. run_metrics.py: lightgbm/sklearnのみ。torchは一切importしない。
evaluate:
	uv run python scripts/precompute_transport.py
	uv run python scripts/run_generation.py
	uv run python scripts/run_metrics.py

# ステージ12: 業種を考慮した輸送。ステージ11-Aと同じ3段階のプロセス分離。
#   1. precompute_industry_transport.py: POTのみ(torch/sklearnは間接importも無効化)
#   2. run_generation_industry.py: torch/cvxpyのみ
#   3. run_metrics_industry.py: lightgbm/sklearn/matplotlibのみ
evaluate-industry:
	uv run python scripts/precompute_industry_transport.py
	uv run python scripts/run_generation_industry.py
	uv run python scripts/run_metrics_industry.py

# ステージ13: 収束版OTでの再検証・輸送設計の比較(例: 最近傍ベースライン)。3段階のプロセス分離。
#   1. precompute_stage13_transport.py: POTのみ  2. run_generation_stage13.py: torch/cvxpyのみ
#   3. run_metrics_stage13.py / run_metrics_designs.py / run_diagnose_identification*.py: lightgbm/sklearnのみ
evaluate-stage13:
	uv run python scripts/precompute_stage13_transport.py --designs bal_na0,nearest_na0
	for c in W1_uniform W3_eq_only; do for s in 0 1 2; do \
	  uv run python scripts/run_generation_stage13.py --tag nearest_na0_cafm \
	    --plans data/interim/stage13/plans_nearest_na0.npz --condition $$c --seed $$s --check-marginals 0; \
	done; done
	uv run python scripts/run_metrics_stage13.py --tag nearest_na0_cafm

# ステージ14: 生成BSの会計的妥当性(追加制約S0〜S6、RE固定の再検討)。stage13の生成結果が前提。
#   run_projection_stage14.py: cvxpyのみ / run_metrics_stage14.py・run_bs_check_stage14.py: 指標側
evaluate-stage14:
	uv run python scripts/run_projection_stage14.py --tag nearest_na0 --freeze all
	uv run python scripts/run_projection_stage14.py --tag nearest_na0 --freeze no_re --stages S0,S3,S5
	uv run python scripts/run_metrics_stage14.py --files nearest_na0_all:R1 --out stageB
	uv run python scripts/run_bs_check_stage14.py --proj nearest_na0_all --stage S5 --label nn_s5

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name '__pycache__' -exec rm -rf {} +

.PHONY: evaluate-stage15
evaluate-stage15:  ## ステージ15: 等重み/コスト重み最近傍×W1/W3の再射影・指標・ペア検定
	uv run python scripts/precompute_stage13_transport.py --designs nearestu_na0
	for s in 0 1 2; do uv run python scripts/run_generation_stage13.py --tag nearestu_na0 --plans data/interim/stage13/plans_nearestu_na0.npz --condition W1_uniform --seed $$s --methods FM --check-marginals 0; done
	for t in nearest_na0 nearestu_na0; do for c in W1_uniform W3_eq_only; do uv run python scripts/run_projection_stage14.py --tag $$t --condition $$c --stages S0,S5_old,S5,S5_a1 --out-dir data/interim/stage15; done; done
	uv run python scripts/run_metrics_stage15.py --out C --files nearestu_na0_all:①等重み+W1,nearestu_na0_all_W3_eq_only:②等重み+W3,nearest_na0_all:③重みあり+W1,nearest_na0_all_W3_eq_only:④重みあり+W3
	uv run python scripts/run_paired_stage15.py --stage S5_a1

.PHONY: evaluate-stage16
evaluate-stage16:  ## ステージ16: canonical再集計(32科目)→輸送→生成(全設定)→指標→ペア検定
	uv run python scripts/build_canonical.py --help > /dev/null  # core.yamlは手で追記済み(貸倒引当金2科目)
	uv run python scripts/inspect_values.py
	uv run python scripts/verify_canonical_stage16.py
	uv run python scripts/precompute_stage13_transport.py --designs nearest_na0,nearestu_na0 --out-dir data/interim/stage16
	for t in nearest_na0 nearestu_na0; do for s in 0 1 2; do uv run python scripts/run_generation_stage16.py --tag $$t --plans data/interim/stage16/plans_$$t.npz --seed $$s; done; done
	uv run python scripts/run_metrics_stage16.py --tags nearest_na0,nearestu_na0 --out D --auc
	uv run python scripts/run_paired_stage16.py --beta b03

.PHONY: evaluate-stage17
evaluate-stage17:  ## ステージ17: validity診断(logreg標準化修正)、業種制約の再評価、制約の2層整理
	uv run pytest -q tests/evaluation/test_classifiers.py tests/accounting/test_plausibility.py
	uv run python scripts/precompute_stage13_transport.py --designs nearest_na0 --out-dir data/interim/stage17 --industry hard
	uv run python scripts/precompute_stage13_transport.py --designs nearest_na0 --out-dir data/interim/stage17 --industry soft
	for ind in indhard indsoft; do for s in 0 1 2; do uv run python scripts/run_generation_stage16.py --tag nearest_na0_$$ind --plans data/interim/stage17/plans_nearest_na0_$$ind.npz --seed $$s --configs W1_S0,W1_S5_a05_b03 --out-dir data/interim/stage17; done; done
	uv run python scripts/run_metrics_stage16.py --tags nearest_na0,nearest_na0_indhard,nearest_na0_indsoft --configs W1_S0,W1_S5_a05_b03 --dir data/interim/stage17 --out B --auc

.PHONY: evaluate-stage18
evaluate-stage18:  ## ステージ18: 業種×規模制約の最近傍(5設計)→生成→指標→ペア検定。基準はstage16の生成結果
	uv run python scripts/precompute_stage18_transport.py --tag d2_hard_none --industry hard --size none
	uv run python scripts/precompute_stage18_transport.py --tag d3_none_0.5_2 --industry none --size 0.5,2.0
	uv run python scripts/precompute_stage18_transport.py --tag d4_hard_0.5_2 --industry hard --size 0.5,2.0
	uv run python scripts/precompute_stage18_transport.py --tag d5_hard_0.33_3 --industry hard --size 0.33,3.0
	uv run python scripts/precompute_stage18_transport.py --tag d6_soft_0.5_2 --industry soft --size 0.5,2.0
	for t in d2_hard_none d3_none_0.5_2 d4_hard_0.5_2 d5_hard_0.33_3 d6_soft_0.5_2; do for s in 0 1 2; do uv run python scripts/run_generation_stage16.py --tag $$t --plans data/interim/stage18/plans_$$t.npz --seed $$s --configs W1_S0,W1_S5_a05_b03 --out-dir data/interim/stage18; done; done
	cp data/interim/stage16/nearest_na0__seed*.parquet data/interim/stage18/
	uv run python scripts/run_metrics_stage16.py --tags nearest_na0,d2_hard_none,d3_none_0.5_2,d4_hard_0.5_2,d5_hard_0.33_3,d6_soft_0.5_2 --configs W1_S0,W1_S5_a05_b03 --dir data/interim/stage18 --out B --auc
	uv run python scripts/run_paired_stage18.py

.PHONY: evaluate-stage19
evaluate-stage19:  ## ステージ19: 違反2社の診断(チェック修正)、規模制約のtrade-off、R1/R3比較
	uv run pytest -q tests/accounting/test_plausibility.py
	for s in 0 1 2; do uv run python scripts/run_generation_stage16.py --tag nearest_na0 --plans data/interim/stage16/plans_nearest_na0.npz --seed $$s --configs W1_S5_a05_b03 --freeze no_re --out-dir data/interim/stage18/r3; done
	uv run python scripts/run_metrics_stage16.py --tags nearest_na0,d3_none_0.5_2 --configs W1_S5_a05_b03 --dir data/interim/stage18 --out stage19_B --auc
	uv run python scripts/run_metrics_stage16.py --tags nearest_na0 --configs W1_S5_a05_b03 --dir data/interim/stage18/r3 --out r3
	uv run python scripts/run_tradeoff_stage19.py
