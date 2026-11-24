#!/bin/bash
# ステージ82 §8: unified_extended の 2 設定を生成し直し、e2 群の指標・評価をやり直してから phase2 を実行する
set -e
L=data/interim/stage82/logs
for c in c1 c2; do for g in f0 f1 f2 f3 f4; do for k in R1p R1ip; do for i in 0 1; do
  echo "$c $g $k $i"; done; done; done; done |
  xargs -P 14 -L 1 bash -c 'uv run python scripts/stage82_qp.py --cls $0 --cfg $1 --group e2 --cost $2 --qp unified --shard $3/2 > data/interim/stage82/logs/qp_e2_$0_$1_$2_prop_unified_t123_$3.txt 2>&1 || echo "FAIL $0 $1 $2 $3"'
for g in f0 f1 f2 f3 f4; do
  uv run python scripts/metrics_stage39.py --gen-dir data/interim/stage82/e2/gen_$g --out data/interim/stage82/e2/metrics_$g > $L/metrics_e2_$g.txt 2>&1 &
done; wait
uv run python scripts/stage82_unified_revision.py --steps evaluate,full,final > $L/phase2.txt 2>&1
echo finished
