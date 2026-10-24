#!/bin/bash
# ステージ71: 輸送計画の候補(POT)→ 6 設定 × 2 class × 2 分割の生成(cvxpy、12 並列)
# uv run python scripts/stage71_ot.py --cls c2 ; uv run python scripts/stage71_ot.py --cls c1
mkdir -p data/interim/stage71/logs
for c in c1 c2; do for st in a2 f1 f2 f3 f3h0 f3t05; do for i in 0 1; do echo "$c $st $i"; done; done; done |
  xargs -P 12 -L 1 bash -c 'uv run python scripts/stage71_qp.py --cls $0 --setting $1 --shard $2/2 > data/interim/stage71/logs/qp_$0_$1_$2.txt 2>&1; echo "done $0 $1 $2 $?"'
