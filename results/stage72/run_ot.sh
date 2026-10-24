#!/bin/bash
# ステージ72: バランス OT の候補集合・事前に混ぜる目標・等重み L2 の最近傍(POT のプロセス)
uv run python scripts/stage72_ot.py --cls c2 > data/interim/stage72/logs/ot_c2.txt 2>&1 &
uv run python scripts/stage72_ot.py --cls c1 > data/interim/stage72/logs/ot_c1.txt 2>&1 &
wait
echo done
