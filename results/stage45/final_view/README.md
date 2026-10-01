# 更新後の最終設定での、ステージ43 と同じ形式の表

`scripts/evaluate_stage43.py --gen-dir data/interim/stage45/view43/gen --metrics-dir data/interim/stage45/view43/metrics`
で作った。見出しは「ステージ43」のままだが、中身は**ステージ45 の更新後の最終設定**((5) を削除、自己株式の上限を追加、
方向制約を削除)の数値。中間報告 第 4 版には、`results/stage43/final_table.md` ではなくこちらの `final_table.md` を使う。

- `final_table.md`: 最終設定の指標(95% 区間つき)と、実際の再建群との比較
- `r1.md`: 「1」= 更新後の最終設定、「2」= 更新後の R1(不活性制約 + 自己株式の上限、(5) なし)、「3」= ステージ31 の R1(参考)
- `direction.md`: class 1 の向きの診断(更新後)
