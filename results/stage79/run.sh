#!/bin/bash
# ステージ79: コスト(numpy)→ 輸送計画(POT・numpy)→ 生成(cvxpy)→ 指標(lightgbm)→ 評価(lightgbm)
set -e
mkdir -p data/interim/stage79/logs
uv run python scripts/stage79_weights.py > data/interim/stage79/logs/weights.txt
uv run python scripts/stage79_ot.py --diag > data/interim/stage79/logs/ot_diag.txt 2>&1 &
for c in c1 c2; do for g in f0 f1 f2 f3 f4; do
  uv run python scripts/stage79_ot.py --cls $c --cfgs $g > data/interim/stage79/logs/ot_${c}_${g}.txt 2>&1 &
done; done; wait
for c in c1 c2; do for g in f0 f1 f2 f3 f4; do for s in A1_f1 A2_f1 R1sp_f1 A1sp_f1; do for i in 0 1; do
  echo "$c $g $s $i"; done; done; done; done |
  xargs -P 14 -L 1 bash -c 'uv run python scripts/stage79_qp.py --cls $0 --cfg $1 --setting $2 --shard $3/2 > data/interim/stage79/logs/qp_$0_$1_$2_$3.txt 2>&1; echo "done $0 $1 $2 $3 $?"'
for g in f0 f1 f2 f3 f4; do
  uv run python scripts/metrics_stage39.py --gen-dir data/interim/stage79/gen_$g --out data/interim/stage79/metrics_$g > data/interim/stage79/logs/metrics_$g.txt 2>&1 &
done; wait
uv run python scripts/stage79_evaluate.py > data/interim/stage79/logs/evaluate.txt 2>&1
echo finished
