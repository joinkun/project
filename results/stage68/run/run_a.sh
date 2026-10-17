#!/bin/bash
# ステージ68 A: class 1 で較正した λ(λ_c1 = λ̃*_0 × s_c1)を class 2 に当てはめる。λ̃_c2 = λ_c1 / s_c2
R=1.0917922735858947
c=c2
common="--cls c2 --dists l1a --m-file data/interim/stage62/m/c2_paid10.json --tau-file data/interim/stage64/targets_c2.npz"
uv run python scripts/qp_stage61.py $common --tau-key semi_t0.01_e0.001__argmax --layer1-only --lams $(python3 -c "print(1*$R)") --data data/interim/stage68/a_m2 --out results/stage68/qp_a/m2 > data/interim/stage68/logs/a_m2.txt 2>&1 &
uv run python scripts/qp_stage61.py $common --tau-key bal_e0.001__argmax --cash-floor 0.11732 --lams $(python3 -c "print(30*$R)"),$(python3 -c "print(1*$R)") --data data/interim/stage68/a_m6 --out results/stage68/qp_a/m6 > data/interim/stage68/logs/a_m6.txt 2>&1 &
uv run python scripts/qp_stage61.py $common --tau-key bal_e0.001__bary --cash-floor 0.11732 --lams $(python3 -c "print(3*$R)"),$(python3 -c "print(1*$R)") --data data/interim/stage68/a_m7 --out results/stage68/qp_a/m7 > data/interim/stage68/logs/a_m7.txt 2>&1 &
wait
echo done
