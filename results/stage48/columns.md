# all_paths.csv の列

1 社 5 行(t = 0, 0.25, 0.5, 0.75, 1)。t = 0.25・0.5・0.75 は始点と終点の線形補間。金額は百万円。

| 列 | 内容 |
|---|---|
| docID | 書類 ID(始点の有価証券報告書) |
| 企業名 | 提出者名 |
| class | 2 または 1 |
| t | 時点(1 = 5 年) |
| `__OTHER__jppfs_cor_Assets` | 資産その他(資産の部) |
| `inventories` | 棚卸資産(資産の部) |
| `receivables_trade` | 売上債権(資産の部) |
| `jppfs_cor_CashAndDeposits` | 現金及び預金(資産の部) |
| `jppfs_cor_OtherCA` | その他流動資産(資産の部) |
| `allowance_doubtful_current` | 貸倒引当金(流動)(資産の部) |
| `__OTHER__jppfs_cor_CurrentAssets` | 流動資産その他(資産の部) |
| `jppfs_cor_PropertyPlantAndEquipment` | 有形固定資産(資産の部) |
| `jppfs_cor_IntangibleAssets` | 無形固定資産(資産の部) |
| `jppfs_cor_InvestmentsAndOtherAssets` | 投資その他の資産(資産の部) |
| `allowance_doubtful_noncurrent` | 貸倒引当金(固定)(資産の部) |
| `__OTHER__jppfs_cor_NoncurrentAssets` | 固定資産その他(資産の部) |
| `short_term_loans` | 短期借入金(負債の部) |
| `jppfs_cor_IncomeTaxesPayable` | 未払法人税等(負債の部) |
| `jppfs_cor_OtherCL` | その他流動負債(負債の部) |
| `payables_trade` | 仕入債務(負債の部) |
| `accounts_payable_other` | 未払金(負債の部) |
| `bonds_current` | 1年内償還予定社債(負債の部) |
| `__OTHER__jppfs_cor_CurrentLiabilities` | 流動負債その他(負債の部) |
| `__OTHER__jppfs_cor_Liabilities` | 負債その他(負債の部) |
| `long_term_loans` | 長期借入金(負債の部) |
| `bonds_noncurrent` | 社債(負債の部) |
| `__OTHER__jppfs_cor_NoncurrentLiabilities` | 固定負債その他(負債の部) |
| `non_controlling_interests` | 非支配株主持分(純資産の部) |
| `__OTHER__jppfs_cor_NetAssets` | 純資産その他(純資産の部) |
| `jppfs_cor_RetainedEarnings` | 利益剰余金(純資産の部) |
| `jppfs_cor_CapitalStock` | 資本金(純資産の部) |
| `jppfs_cor_CapitalSurplus` | 資本剰余金(純資産の部) |
| `jppfs_cor_TreasuryStock` | 自己株式(純資産の部) |
| `__OTHER__jppfs_cor_ShareholdersEquity` | 株主資本その他(純資産の部) |
| `valuation_difference_securities` | その他有価証券評価差額金(純資産の部) |
| `__OTHER__jppfs_cor_ValuationAndTranslationAdjustments` | 評価・換算差額等その他(純資産の部) |
| 資産合計・負債合計・純資産合計 | 基礎科目の和 |
| 貸借差額 | 資産合計 − 負債合計 − 純資産合計(検算用、丸め誤差の範囲で 0) |
| 振替 | 始点からの ΔRE のうち、払込資本の減少分(`decompose_re_change`) |
| 利益部分 | ΔRE − 振替 |
