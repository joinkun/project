#!/bin/bash
# ステージ69.5 A: class 1 で較正した λ の絶対値(λ̃*_0 × s_c1)を class 2 に当てはめる 2 ジョブ。
# class 2 のグリッドでは λ̃_c2 = λ̃*_0 × s_c1 / s_c2(s はステージ61 の scale_{c1,c2}.json)。
set -e
NN=$(python3 -c "print(repr(1 * 0.3866021397069638 / 0.35409862210986653))")
L2=$(python3 -c "print(repr(10 * 204.3832688321667 / 166.43053828235747))")
echo "nn λ̃_c2 = $NN、l2 λ̃_c2 = $L2"
# 1-NN(非対称 L1): ステージ64 の t1_nn と同じ設定(m = 10、現金の下限、目標 = 最近傍)
uv run python scripts/qp_stage61.py --cls c2 --dists l1a \
  --m-file data/interim/stage62/m/c2_paid10.json --cash-floor 0.11732 \
  --tau-file data/interim/stage64/targets_c2.npz --tau-key nn --lams "$NN" \
  --data data/interim/stage695/nn --out results/stage69/qp/nn \
  > data/interim/stage695/logs/nn.txt 2>&1 &
# 等重み L2 + 最近傍: ステージ61 の格子と同じ設定(m なし、現金の下限なし、目標 = L2 の最近傍)
uv run python scripts/qp_stage61.py --cls c2 --dists l2 --lams "$L2" \
  --data data/interim/stage695/l2 --out results/stage69/qp/l2 \
  > data/interim/stage695/logs/l2.txt 2>&1 &
wait
echo done
# ステージ69.5 B(確認のみ): class 1 の設定 2(利益剰余金にコスト)を、規則どおりの c_RE で解き直す
# uv run python scripts/cre_stage695.py
# uv run python scripts/qp_stage67.py --cls c1 --setting s2 --shard {0,1}/2 \
#   --weights data/interim/stage695/weights_c1_rule.json \
#   --gen-root data/interim/stage695/gen67 --qp-out results/stage69/qp/s2_rule
