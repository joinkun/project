#!/bin/bash
# ステージ75: コスト(numpy)→ 輸送計画(POT)→ 生成(cvxpy、14 並列)→ 指標(fold ごと、lightgbm)。argmax の s は記録時の挙動(--s-from target)
set -e
mkdir -p data/interim/stage75/logs
uv run python scripts/stage75_weights.py > data/interim/stage75/logs/weights.txt
for c in c1 c2; do for g in "f0,f1" "f2,f3" "f4,full"; do
  uv run python scripts/stage75_ot.py --cls $c --cfgs $g > "data/interim/stage75/logs/ot_${c}_${g/,/_}.txt" 2>&1 &
done; done; wait
for c in c1 c2; do for g in f0 f1 f2 f3 f4 full; do for k in abs R1 R2 R3; do for t in argmax f1; do for i in 0 1; do
  echo "$c $g $k $t $i"; done; done; done; done; done |
  xargs -P 14 -L 1 bash -c 'uv run python scripts/stage75_qp.py --cls $0 --cfg $1 --kind $2 --target $3 --shard $4/2 --s-from target > data/interim/stage75/logs/qp_$0_$1_$2_$3_$4.txt 2>&1; echo "done $0 $1 $2 $3 $4 $?"'
for g in f0 f1 f2 f3 f4 full; do
  uv run python scripts/metrics_stage39.py --gen-dir data/interim/stage75/gen_$g --out data/interim/stage75/metrics_$g > data/interim/stage75/logs/metrics_$g.txt 2>&1 &
done; wait
echo finished
