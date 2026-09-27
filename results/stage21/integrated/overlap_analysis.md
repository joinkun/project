# ステージ21-C C: 2023 年決算の重複の不一致(統合後)

- 不一致 567 社(重複 3997 社中)
  - 総資産まで変わる: 113
  - 総資産不変、拡張要素 → 標準要素の付け替えの可能性: 53
  - 総資産不変、その他の表示の組替え: 401

総資産まで変わる社の、最大のずれ(総資産比)の分布:
  中央値 0.0031、p90 0.0553、最大 0.6634

## 差分の取り方による違い(不一致の社について、2024 年の Δx)

(a) 同じ書類: fy2024 の当期 − fy2024 の前期(前期は当期の表示に組み替え済み)
(b) 書類をまたぐ: fy2024 の当期 − 2023-24 提出の当期

| 科目 | 違いが出る社 | |Δa − Δb| / 総資産 中央値 | 最大 | 参考: |Δa| / 総資産 中央値 |
|---|---:|---:|---:|---:|
| __OTHER__jppfs_cor_CurrentAssets | 235 | 0.0073 | 0.5714 | 0.0082 |
| jppfs_cor_OtherCL | 227 | 0.0047 | 0.2225 | 0.0062 |
| __OTHER__jppfs_cor_CurrentLiabilities | 219 | 0.0013 | 0.7190 | 0.0094 |
| __OTHER__jppfs_cor_NoncurrentAssets | 177 | 0.0000 | 0.0275 | 0.0000 |
| jppfs_cor_OtherCA | 175 | 0.0018 | 0.1383 | 0.0045 |
| jppfs_cor_InvestmentsAndOtherAssets | 123 | 0.0001 | 0.1343 | 0.0155 |
| receivables_trade | 116 | 0.0028 | 0.5714 | 0.0145 |
| jppfs_cor_IntangibleAssets | 109 | 0.0001 | 0.1664 | 0.0073 |
| __OTHER__jppfs_cor_NoncurrentLiabilities | 92 | 0.0005 | 0.0540 | 0.0053 |
| jppfs_cor_RetainedEarnings | 89 | 0.0003 | 1.0165 | 0.0319 |
| jppfs_cor_PropertyPlantAndEquipment | 87 | 0.0000 | 0.2146 | 0.0163 |
| accounts_payable_other | 75 | 0.0099 | 0.0896 | 0.0102 |
| __OTHER__jppfs_cor_Assets | 62 | 0.0000 | 0.0001 | 0.0000 |
| payables_trade | 57 | 0.0000 | 0.1137 | 0.0093 |
| inventories | 55 | 0.0002 | 0.0384 | 0.0064 |
| __OTHER__jppfs_cor_ShareholdersEquity | 47 | 0.0000 | 0.0011 | 0.0000 |
| __OTHER__jppfs_cor_NetAssets | 45 | 0.0000 | 0.0001 | 0.0000 |
| __OTHER__jppfs_cor_Liabilities | 44 | 0.0000 | 0.0001 | 0.0000 |
| jppfs_cor_IncomeTaxesPayable | 39 | 0.0000 | 0.0160 | 0.0056 |
| __OTHER__jppfs_cor_ValuationAndTranslationAdjustments | 33 | 0.0000 | 0.0052 | 0.0053 |
| non_controlling_interests | 31 | 0.0001 | 0.0346 | 0.0019 |
| jppfs_cor_CashAndDeposits | 29 | 0.0000 | 0.0313 | 0.0373 |
| short_term_loans | 29 | 0.0000 | 0.0340 | 0.0228 |
| jppfs_cor_CapitalSurplus | 29 | 0.0000 | 0.0001 | 0.0003 |
| jppfs_cor_TreasuryStock | 27 | 0.0000 | 0.0001 | 0.0002 |
| allowance_doubtful_current | 24 | 0.0000 | 0.6607 | 0.0001 |
| jppfs_cor_CapitalStock | 24 | 0.0000 | 0.0001 | 0.0000 |
| valuation_difference_securities | 24 | 0.0000 | 0.0021 | 0.0023 |
| allowance_doubtful_noncurrent | 23 | 0.0000 | 0.0663 | 0.0004 |
| long_term_loans | 23 | 0.0000 | 0.0009 | 0.0214 |
| bonds_current | 4 | 0.0136 | 0.0369 | 0.0209 |
| bonds_noncurrent | 1 | 0.0000 | 0.0000 | 0.0025 |

違いが出る社(いずれかの科目): 567 / 567
