#!/bin/bash
# ステージ68 B: 5 設定 × 2 class × 2 分割
mkdir -p data/interim/stage68/logs
for c in c1 c2; do for st in t0m10 t0m1 t1m1 t2m1 t2m1re; do for i in 0 1; do echo "$c $st $i"; done; done; done |
  xargs -P 8 -L 1 bash -c 'uv run python scripts/qp_stage68.py --cls $0 --setting $1 --shard $2/2 > data/interim/stage68/logs/qp_$0_$1_$2.txt 2>&1; echo "done $0 $1 $2 $?"'
