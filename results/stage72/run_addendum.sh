#!/bin/bash
# ステージ72 追補: F1 の τ1 の感度(τ1 = 0.01・0.1)。POT → cvxpy → 指標
set -e
uv run python scripts/stage72_tau_ot.py --cls c2 > data/interim/stage72/logs/tau_ot_c2.txt 2>&1 &
uv run python scripts/stage72_tau_ot.py --cls c1 > data/interim/stage72/logs/tau_ot_c1.txt 2>&1 &
wait
for c in c1 c2; do for st in f1t001 f1t01; do for i in 0 1; do echo "$c $st $i"; done; done; done |
  xargs -P 8 -L 1 bash -c 'uv run python scripts/stage72_qp.py --cls $0 --setting $1 --shard $2/2 --gen-root data/interim/stage72/gen_tau > data/interim/stage72/logs/qp_$0_$1_$2.txt 2>&1; echo "done $0 $1 $2 $?"'
uv run python scripts/metrics_stage39.py --gen-dir data/interim/stage72/gen_tau --out data/interim/stage72/metrics_tau > data/interim/stage72/logs/metrics_tau.txt 2>&1
echo finished
