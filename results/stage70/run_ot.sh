#!/bin/bash
# ステージ70 A.2: 輸送計画と目標(POT のプロセス、5 つ並列)
for j in "c2 main" "c2 nci0" "c1 main" "c1 nci0" "c1 p57"; do
  set -- $j
  uv run python scripts/stage70_ot.py --cls $1 --variant $2 > data/interim/stage70/logs/ot_$1_$2.txt 2>&1 &
done
wait
echo done
