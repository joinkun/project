#!/bin/bash
# ステージ72: 生成(cvxpy、10 並列)。引数に設定名を並べる(省略時は全設定)
# 前提: bash results/stage72/run_ot.sh(f1bal・l2nn・pre05・pre05m1 が使う)
sets=${@:-"f1 f1m1 a2 a2t05 bal_l2 f1bal l2nn pre05 pre05m1"}
for c in c1 c2; do for st in $sets; do for i in 0 1; do echo "$c $st $i"; done; done; done |
  xargs -P 10 -L 1 bash -c 'uv run python scripts/stage72_qp.py --cls $0 --setting $1 --shard $2/2 > data/interim/stage72/logs/qp_$0_$1_$2.txt 2>&1; echo "done $0 $1 $2 $?"'
