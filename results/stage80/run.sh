#!/bin/bash
# ステージ80: run_ot.sh の後に、生成(cvxpy)→ 指標(lightgbm)→ D1 の評価(lightgbm)
set -e
mkdir -p data/interim/stage80/logs
{ for c in c1 c2; do for g in f0 f1 f2 f3 f4; do for s in abs_sp R1_sp A1_sp A1_argmax; do for i in 0 1; do
    echo "$c $g $s $i"; done; done; done; done
  for c in c1 c2; do for i in 0 1; do echo "$c full A1_sp $i"; done; done; } |
  xargs -P 14 -L 1 bash -c 'uv run python scripts/stage80_qp.py --cls $0 --cfg $1 --setting $2 --shard $3/2 > data/interim/stage80/logs/qp_$0_$1_$2_$3.txt 2>&1; echo "done $0 $1 $2 $3 $?"'
for g in f0 f1 f2 f3 f4 full; do
  uv run python scripts/metrics_stage39.py --gen-dir data/interim/stage80/gen_$g --out data/interim/stage80/metrics_$g > data/interim/stage80/logs/metrics_$g.txt 2>&1 &
done; wait
uv run python scripts/stage80_evaluate.py > data/interim/stage80/logs/evaluate.txt 2>&1
echo finished
