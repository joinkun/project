#!/bin/bash
# ステージ39: 1 設定 × 1 seed の生成(class 2 は run_generation_stage16、class 1 は run_generation_stage34)
# 使い方: gen.sh <root> <class:c2|c1> <plans_dir> <costs> <setting> <design> <seed>
set -e
root=$1; cls=$2; plans=$3; costs=$4; setting=$5; design=$6; seed=$7
cd "$root"
extra=""
case "$design" in unb*) extra="--allow-dropped-sources";; esac
mkdir -p data/interim/stage39/gen/$cls/$setting data/interim/stage39/logs
if [ "$cls" = c2 ]; then
  uv run --project /home/yuto/project python /home/yuto/project/scripts/run_generation_stage16.py \
    --tag "$design" --plans "$plans/plans_$design.npz" --seed "$seed" \
    --configs W1_S5_a05_b03_re0.756519 --re-cap 0.7565192502545388 --freeze no_re \
    --costs "$costs" --exclude data/interim/stage28/excluded.csv \
    --out-dir data/interim/stage39/gen/c2/$setting $extra \
    > data/interim/stage39/logs/gen_c2_${setting}_$seed.txt 2>&1
else
  uv run --project /home/yuto/project python /home/yuto/project/scripts/run_generation_stage34.py \
    --tag "$design" --plans "$plans/plans_$design.npz" --seed "$seed" --costs "$costs" \
    --exclude data/interim/stage28/excluded.csv --class1 data/interim/stage34/class1_docs.csv \
    --out-dir data/interim/stage39/gen/c1/$setting $extra \
    > data/interim/stage39/logs/gen_c1_${setting}_$seed.txt 2>&1
fi
echo "done $root $cls $setting $seed"
