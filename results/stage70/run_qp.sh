#!/bin/bash
# ステージ70 B: 16 設定 × 2 class × 2 分割(cvxpy のプロセス、12 並列)
for c in c1 c2; do for st in s2 s3a s3b s4 s5a s5b s6a s6b; do for m in m10 m1; do for i in 0 1; do
  echo "$c ${st}_$m $i"; done; done; done; done |
  xargs -P 12 -L 1 bash -c 'uv run python scripts/stage70_qp.py --cls $0 --setting $1 --shard $2/2 > data/interim/stage70/logs/qp_$0_$1_$2.txt 2>&1; echo "done $0 $1 $2 $?"'
