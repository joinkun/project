#!/bin/bash
# ステージ50: 1 設定 × 1 class × 1 seed の生成(対応先の選択だけを変え、他は最終設定 s47_final)。
# 使い方: gen_stage50.sh <c2|c1> <setting> <seed> <inactive>
#   setting: equal(現行の等重み最近傍)/ sub_all / sub_trim / t / lognorm / t_unb / t_pi12
#   inactive: selected(規則 M)/ none。none の出力は <setting>_noinact に置く
# 出力先は data/interim/stage50/gen/<cls>/<setting>[_noinact]/。プロジェクトのルートで実行する。
set -e
cls=$1; setting=$2; seed=$3; inactive=$4
base=data/interim/stage50
if [ "$setting" = equal ]; then
  plans=data/interim/stage28/plans_equal/plans_nearest_na0.npz; tag=nearest_na0
else
  plans=$base/plans/plans_$setting.npz; tag=$setting
fi
out=$setting; [ "$inactive" = none ] && out=${setting}_noinact
mkdir -p $base/gen/$cls/$out $base/logs
common=(--tag "$tag" --plans "$plans" --seed "$seed" --costs conf/accounts/costs_equal.yaml
        --exclude data/interim/stage28/excluded.csv --inactive "$inactive" --loss-log
        --allow-dropped-sources)
if [ "$cls" = c2 ]; then
  uv run python scripts/run_generation_stage16.py "${common[@]}" --configs s47_final \
    --re-cap 0.7565192502545388 --freeze no_re --out-dir $base/gen/c2/$out \
    > $base/logs/gen_c2_${out}_$seed.txt 2>&1
else
  uv run python scripts/run_generation_stage34.py "${common[@]}" --config s47_final \
    --class1 data/interim/stage34/class1_docs.csv --out-dir $base/gen/c1/$out \
    > $base/logs/gen_c1_${out}_$seed.txt 2>&1
fi
echo "done $cls $out $seed"
