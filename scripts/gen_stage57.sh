#!/bin/bash
# ステージ57: 1 設定 × 1 class × 1 seed の生成(最終設定 s47_final、不活性制約あり)。
# 使い方: gen_stage57.sh <c2|c1> <setting> <seed> <plans.npz> <basis: pca|null> <n_steps>
# 出力先は data/interim/stage57/gen/<cls>/<setting>/。経路の全点を npz に保存する。プロジェクトのルートで実行する。
set -e
cls=$1; setting=$2; seed=$3; plans=$4; basis=$5; nsteps=$6
base=data/interim/stage57
mkdir -p $base/gen/$cls/$setting $base/logs
common=(--tag nearest_na0 --plans "$plans" --seed "$seed" --costs conf/accounts/costs_equal.yaml
        --exclude data/interim/stage28/excluded.csv --inactive selected --loss-log
        --basis "$basis" --n-steps "$nsteps" --save-path)
if [ "$cls" = c2 ]; then
  uv run python scripts/run_generation_stage16.py "${common[@]}" --configs s47_final \
    --re-cap 0.7565192502545388 --freeze no_re --out-dir $base/gen/c2/$setting \
    > $base/logs/gen_c2_${setting}_$seed.txt 2>&1
else
  uv run python scripts/run_generation_stage34.py "${common[@]}" --config s47_final \
    --class1 data/interim/stage34/class1_docs.csv --out-dir $base/gen/c1/$setting \
    > $base/logs/gen_c1_${setting}_$seed.txt 2>&1
fi
echo "done $cls $setting $seed"
