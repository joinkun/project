#!/bin/bash
# ステージ44: 1 設定 × 1 seed の生成(最終設定から 1 要素だけ変える)。プロジェクトのルートで実行する。
# 使い方: gen_stage44.sh <c2|c1> <setting> <seed> <config> <inactive> [追加の引数...]
#   config: 射影の設定名(最終設定は W1_S5_a05_b03_re0.756519、変種は s44_*)
#   inactive: --inactive の値(最終設定は selected)
#   追加の引数: --k 8 / --basis null / --freeze all(R1)など。出力先は環境変数 OUT_BASE(既定 data/interim/stage44)
set -e
cls=$1; setting=$2; seed=$3; config=$4; inactive=$5; shift 5
base=${OUT_BASE:-data/interim/stage44}
plans=data/interim/stage28/plans_equal/plans_nearest_na0.npz
mkdir -p $base/gen/$cls/$setting $base/logs
if [ "$cls" = c2 ]; then
  uv run python scripts/run_generation_stage16.py --tag nearest_na0 --plans $plans --seed "$seed" \
    --configs "$config" --re-cap 0.7565192502545388 --freeze no_re \
    --costs conf/accounts/costs_equal.yaml --exclude data/interim/stage28/excluded.csv \
    --inactive "$inactive" --loss-log --out-dir $base/gen/c2/$setting "$@" \
    > $base/logs/gen_c2_${setting}_$seed.txt 2>&1
else
  uv run python scripts/run_generation_stage34.py --tag nearest_na0 --plans $plans --seed "$seed" \
    --config "$config" --costs conf/accounts/costs_equal.yaml \
    --exclude data/interim/stage28/excluded.csv --class1 data/interim/stage34/class1_docs.csv \
    --inactive "$inactive" --loss-log --out-dir $base/gen/c1/$setting "$@" \
    > $base/logs/gen_c1_${setting}_$seed.txt 2>&1
fi
echo "done $cls $setting $seed"
