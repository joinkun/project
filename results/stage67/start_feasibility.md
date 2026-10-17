# ステージ67 D.2: 始点 x_i が実行可能集合 F_i に入るか(定理の但し書き)

第 1 層 = 貸借一致と符号(`is_feasible`、許容 1e-6)。第 2 層 = 最終設定の追加の線形制約を始点で評価。

| class | 貸借一致の残差 > 1e-6 | 残差の最大(総資産比) | 符号の違反 | 第 2 層を満たさない | 理由 |
|---|---:|---:|---:|---:|---|
| class 2(55 社) | 13 | 4.9e-04 | 0 | 1 | {'retention:jppfs_cor_PropertyPlantAndEquipment>=0.3*start': 1} |
| class 1(132 社) | 18 | 4.2e-04 | 0 | 1 | {'retention:jppfs_cor_PropertyPlantAndEquipment>=0.3*start': 1} |
