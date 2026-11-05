#!/bin/bash
# ステージ78: argmax 系を F1 の尺度 s(criteria.md §3、--s-from f1)で生成し直し、指標を計算する。
set -e
mkdir -p data/interim/stage78/logs
for c in c1 c2; do for g in f0 f1 f2 f3 f4 full; do for k in abs R1 R2 R3; do for i in 0 1; do
  echo "$c $g $k $i"; done; done; done; done |
  xargs -P 14 -L 1 bash -c 'uv run python scripts/stage75_qp.py --cls $0 --cfg $1 --kind $2 --target argmax --shard $3/2 --s-from f1 --data-out data/interim/stage78 --out results/stage78 > data/interim/stage78/logs/qp_$0_$1_$2_$3.txt 2>&1; echo "done $0 $1 $2 $3 $?"'
for g in f0 f1 f2 f3 f4 full; do
  uv run python scripts/metrics_stage39.py --gen-dir data/interim/stage78/gen_$g --out data/interim/stage78/metrics_$g > data/interim/stage78/logs/metrics_$g.txt 2>&1 &
done; wait
echo finished
