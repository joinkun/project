# canonical の科目一覧(`conf/accounts/core.yaml`)

基礎科目 32 個(状態ベクトルの順序)、小計 9 個、総合計 4 個。`__OTHER__*` は親小計から他の基礎科目を差し引いた残余(差額定義)で、恒等式が構成的に成り立つ。

## 基礎科目

| # | 基礎科目 | 日本語 | 親小計 | 側 | 符号 | XBRL 要素数 |
|---:|---|---|---|---|---|---:|
| 1 | `__OTHER__jppfs_cor_Assets` | 資産その他 | 資産 | 資産 | 自由 | 0 |
| 2 | `inventories` | 棚卸資産 | 流動資産 | 資産 | ≥ 0 | 16 |
| 3 | `receivables_trade` | 売上債権 | 流動資産 | 資産 | ≥ 0 | 17 |
| 4 | `jppfs_cor_CashAndDeposits` | 現金及び預金 | 流動資産 | 資産 | ≥ 0 | 1 |
| 5 | `jppfs_cor_OtherCA` | その他流動資産 | 流動資産 | 資産 | ≥ 0 | 1 |
| 6 | `allowance_doubtful_current` | 貸倒引当金(流動) | 流動資産 | 資産 | ≤ 0 | 6 |
| 7 | `__OTHER__jppfs_cor_CurrentAssets` | 流動資産その他 | 流動資産 | 資産 | 自由 | 0 |
| 8 | `jppfs_cor_PropertyPlantAndEquipment` | 有形固定資産 | 固定資産 | 資産 | ≥ 0 | 1 |
| 9 | `jppfs_cor_IntangibleAssets` | 無形固定資産 | 固定資産 | 資産 | ≥ 0 | 1 |
| 10 | `jppfs_cor_InvestmentsAndOtherAssets` | 投資その他の資産 | 固定資産 | 資産 | ≥ 0 | 1 |
| 11 | `allowance_doubtful_noncurrent` | 貸倒引当金(固定) | 固定資産 | 資産 | ≤ 0 | 5 |
| 12 | `__OTHER__jppfs_cor_NoncurrentAssets` | 固定資産その他 | 固定資産 | 資産 | 自由 | 0 |
| 13 | `short_term_loans` | 短期借入金 | 流動負債 | 負債 | ≥ 0 | 6 |
| 14 | `jppfs_cor_IncomeTaxesPayable` | 未払法人税等 | 流動負債 | 負債 | ≥ 0 | 1 |
| 15 | `jppfs_cor_OtherCL` | その他流動負債 | 流動負債 | 負債 | ≥ 0 | 1 |
| 16 | `payables_trade` | 仕入債務 | 流動負債 | 負債 | ≥ 0 | 12 |
| 17 | `accounts_payable_other` | 未払金 | 流動負債 | 負債 | ≥ 0 | 6 |
| 18 | `bonds_current` | 1年内償還予定社債 | 流動負債 | 負債 | ≥ 0 | 1 |
| 19 | `__OTHER__jppfs_cor_CurrentLiabilities` | 流動負債その他 | 流動負債 | 負債 | 自由 | 0 |
| 20 | `__OTHER__jppfs_cor_Liabilities` | 負債その他 | 負債 | 負債 | 自由 | 0 |
| 21 | `long_term_loans` | 長期借入金 | 固定負債 | 負債 | ≥ 0 | 3 |
| 22 | `bonds_noncurrent` | 社債 | 固定負債 | 負債 | ≥ 0 | 2 |
| 23 | `__OTHER__jppfs_cor_NoncurrentLiabilities` | 固定負債その他 | 固定負債 | 負債 | 自由 | 0 |
| 24 | `non_controlling_interests` | 非支配株主持分 | 純資産 | 純資産 | 自由 | 1 |
| 25 | `__OTHER__jppfs_cor_NetAssets` | 純資産その他 | 純資産 | 純資産 | 自由 | 0 |
| 26 | `jppfs_cor_RetainedEarnings` | 利益剰余金 | 株主資本 | 純資産 | 自由 | 1 |
| 27 | `jppfs_cor_CapitalStock` | 資本金 | 株主資本 | 純資産 | ≥ 0 | 1 |
| 28 | `jppfs_cor_CapitalSurplus` | 資本剰余金 | 株主資本 | 純資産 | ≥ 0 | 1 |
| 29 | `jppfs_cor_TreasuryStock` | 自己株式 | 株主資本 | 純資産 | ≤ 0 | 1 |
| 30 | `__OTHER__jppfs_cor_ShareholdersEquity` | 株主資本その他 | 株主資本 | 純資産 | 自由 | 0 |
| 31 | `valuation_difference_securities` | その他有価証券評価差額金 | 評価・換算差額等 | 純資産 | 自由 | 1 |
| 32 | `__OTHER__jppfs_cor_ValuationAndTranslationAdjustments` | 評価・換算差額等その他 | 評価・換算差額等 | 純資産 | 自由 | 0 |

## 小計(基礎科目の和。係数 −1 はない)

| # | 小計 | 日本語 | 側 | 構成する基礎科目(#) |
|---:|---|---|---|---|
| 1 | `jppfs_cor_Assets` | 資産 | 資産 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 |
| 2 | `jppfs_cor_CurrentAssets` | 流動資産 | 資産 | 2, 3, 4, 5, 6, 7 |
| 3 | `jppfs_cor_NoncurrentAssets` | 固定資産 | 資産 | 8, 9, 10, 11, 12 |
| 4 | `jppfs_cor_CurrentLiabilities` | 流動負債 | 負債 | 13, 14, 15, 16, 17, 18, 19 |
| 5 | `jppfs_cor_Liabilities` | 負債 | 負債 | 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 |
| 6 | `jppfs_cor_NoncurrentLiabilities` | 固定負債 | 負債 | 21, 22, 23 |
| 7 | `jppfs_cor_NetAssets` | 純資産 | 純資産 | 24, 25, 26, 27, 28, 29, 30, 31, 32 |
| 8 | `jppfs_cor_ShareholdersEquity` | 株主資本 | 純資産 | 26, 27, 28, 29, 30 |
| 9 | `jppfs_cor_ValuationAndTranslationAdjustments` | 評価・換算差額等 | 純資産 | 31, 32 |

## 総合計

| 区分 | 要素 | 日本語 |
|---|---|---|
| assets | `jppfs_cor_Assets` | 資産 |
| liabilities | `jppfs_cor_Liabilities` | 負債 |
| net_assets | `jppfs_cor_NetAssets` | 純資産 |
| liabilities_and_net_assets | `jppfs_cor_LiabilitiesAndNetAssets` | 負債純資産 |
