#!/bin/bash
# ステージ74: コスト → 輸送計画(POT)→ F1 の生成(cvxpy、10 並列)→ 指標
set -e
mkdir -p data/interim/stage74/logs
uv run python scripts/stage74_weights.py > data/interim/stage74/logs/weights.txt
uv run python scripts/stage74_ot.py --cls c2 > data/interim/stage74/logs/ot_c2.txt 2>&1 &
uv run python scripts/stage74_ot.py --cls c1 > data/interim/stage74/logs/ot_c1.txt 2>&1 &
wait
for c in c1 c2; do for st in A1 A2 A3 B1 B2; do for i in 0 1; do echo "$c $st $i"; done; done; done |
  xargs -P 10 -L 1 bash -c 'uv run python scripts/stage74_qp.py --cls $0 --setting $1 --shard $2/2 > data/interim/stage74/logs/qp_$0_$1_$2.txt 2>&1; echo "done $0 $1 $2 $?"'
uv run python scripts/metrics_stage39.py --gen-dir data/interim/stage74/gen --out data/interim/stage74/metrics > data/interim/stage74/logs/metrics.txt 2>&1
echo finished
