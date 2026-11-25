#!/bin/bash
# ステージ83: run_d1.sh の後に、評価とゲート(lightgbm)→ 代表企業の輸送(numpy)→ バックテスト(numpy・cvxpy・lightgbm)→ 検算と図(lightgbm)
set -e
L=data/interim/stage83/logs
uv run python scripts/stage83_evaluate.py > $L/evaluate.txt 2>&1
uv run python scripts/stage82_ot.py --data data/interim/stage83 --out results/stage83 --cls c2 --cfg full \
  --pairs KDE0p:pure_lp,KDE0p:top5,KDE0p:top8,KDE0p:sinkhorn,KDE0p:row,KDE0p:col > $L/ot_c2_full_modes.txt 2>&1
uv run python scripts/stage83_backtest_ot.py > $L/bt_ot.txt 2>&1
for g in f0 f1 f2 f3 f4 c2rec; do for c in KDE0p KDE0; do echo "$g $c"; done; done |
  xargs -P 12 -L 1 bash -c 'uv run python scripts/stage83_backtest_qp.py --group $0 --cost $1 > data/interim/stage83/logs/bt_qp_$0_$1.txt 2>&1 || echo "FAIL bt_qp $0 $1"'
uv run python scripts/stage83_backtest_eval.py > $L/bt_eval.txt 2>&1
uv run python scripts/stage83_final.py > $L/final.txt 2>&1
echo finished
