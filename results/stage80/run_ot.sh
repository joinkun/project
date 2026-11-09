#!/bin/bash
# ステージ80: γ' = 0.1 のスパース OT(POT・numpy)と命題 1 の検算
set -e
mkdir -p data/interim/stage80/logs
uv run python scripts/stage80_ot.py --verify > data/interim/stage80/logs/verify.txt 2>&1 &
for c in c1 c2; do for g in f0 f1 f2 f3 f4 full; do
  uv run python scripts/stage80_ot.py --cls $c --cfgs $g > data/interim/stage80/logs/ot_${c}_${g}.txt 2>&1 &
done; done; wait
echo finished
