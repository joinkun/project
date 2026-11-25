#!/bin/bash
# ステージ83: 輸送(POT)→ D1 と単一の生成(cvxpy)→ 指標(lightgbm)。ステージ82 のスクリプトに --data を渡す
set -e
D=data/interim/stage83; O=results/stage83; L=$D/logs
mkdir -p $L
P="KDE0p:prop,KDE0p:pure_lp,KDE0p:top5,KDE0p:top8,KDE0p:sinkhorn,KDE0p:row,KDE0p:col"
{ for c in c1 c2; do for g in f0 f1 f2 f3 f4; do echo "$c $g $P"; done; echo "$c full KDE0p:prop,KDE0p_all:prop"; done; } |
  xargs -P 12 -L 1 bash -c 'uv run python scripts/stage82_ot.py --data data/interim/stage83 --out results/stage83 --cls $0 --cfg $1 --pairs $2 > data/interim/stage83/logs/ot_$0_$1.txt 2>&1 || echo "FAIL ot $0 $1"'
{ for c in c1 c2; do for g in f0 f1 f2 f3 f4; do for i in 0 1; do
    echo "a KDE0p prop t123 $c $g $i"
    for m in pure_lp top5 top8 sinkhorn row col; do echo "b KDE0p $m t123 $c $g $i"; done
  done; done
  for i in 0 1; do for t in t123 t12 t1; do echo "full KDE0p prop $t $c full $i"; done; echo "full KDE0p_all prop t123 $c full $i"; done; done; } |
  xargs -P 14 -L 1 bash -c 'uv run python scripts/stage82_qp.py --data data/interim/stage83 --out results/stage83 --group $0 --cost $1 --ot $2 --tier $3 --cls $4 --cfg $5 --shard $6/2 > data/interim/stage83/logs/qp_$0_$1_$2_$3_$4_$5_$6.txt 2>&1 || echo "FAIL qp $0 $1 $2 $3 $4 $5 $6"'
for grp in a b; do for g in f0 f1 f2 f3 f4; do
  echo "$grp $g"; done; done | xargs -P 8 -L 1 bash -c 'uv run python scripts/metrics_stage39.py --gen-dir data/interim/stage83/$0/gen_$1 --out data/interim/stage83/$0/metrics_$1 > data/interim/stage83/logs/metrics_$0_$1.txt 2>&1 || echo "FAIL metrics $0 $1"'
uv run python scripts/metrics_stage39.py --gen-dir $D/full/gen_full --out $D/full/metrics_full > $L/metrics_full.txt 2>&1
echo finished
