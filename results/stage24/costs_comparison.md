# ステージ24 A.4: コスト重みの 3 版(現金預金 = 1.0)

w+ / w- は値の増加・減少のコスト。推定ベースは符号を分けない推定なので w+ = w-。

| 科目 | 等重み | 推定ベース | 定性ベース (w+ / w-) | 推定の根拠 | 定性の根拠 |
|---|---:|---:|---|---|---|
| __OTHER__jppfs_cor_Assets | 1.0 | 1.000(補完) | 1.0 / 1.0 | 推定なし(残余科目) | 根拠なし。等重み |
| inventories | 1.0 | 0.221 | 1.0 / 1.0 | 売上高比・差分 GMM(ステージ23.5)λ=0.663、c=0.508 | 根拠なし。等重み |
| receivables_trade | 1.0 | 0.178 | 1.0 / 1.0 | 売上高比・差分 GMM(ステージ23.5)λ=0.709、c=0.411 | 根拠なし。等重み |
| CashAndDeposits | 1.0 | 1.000 | 1.0 / 1.0 | 総資産比・差分 GMM(ステージ23)λ=0.303、c=2.303 | 根拠なし。等重み |
| OtherCA | 1.0 | 0.219 | 1.0 / 1.0 | 総資産比・差分 GMM(ステージ23)λ=0.665、c=0.504 | 根拠なし。等重み |
| allowance_doubtful_current | 1.0 | 0.327 | 1.0 / 1.0 | 総資産比・差分 GMM(ステージ23)λ=0.570、c=0.753 | 根拠なし。等重み |
| __OTHER__jppfs_cor_CurrentAssets | 1.0 | 1.000(補完) | 1.0 / 1.0 | 推定なし(残余科目) | 根拠なし。等重み |
| PropertyPlantAndEquipment | 1.0 | 1.000(補完) | 1.0 / 2.0 | 総資産比・差分 GMM(ステージ23)λ=0.001 | 減少(売却)= 2.0。Shleifer & Vishny (1992, JF 47(4)) 要旨: 困窮企業が資産を売る際は同業者も困窮しており、最善の用途の価値を下回る価格での売却になる(要旨で確認) |
| IntangibleAssets | 1.0 | 1.000(補完) | 1.0 / 1.0 | 総資産比・差分 GMM(ステージ23)λ=-0.171 | 根拠なし。等重み |
| InvestmentsAndOtherAssets | 1.0 | 0.515 | 1.0 / 1.0 | 総資産比・差分 GMM(ステージ23)λ=0.457、c=1.187 | 根拠なし。等重み |
| allowance_doubtful_noncurrent | 1.0 | 4.219 | 1.0 / 1.0 | 総資産比・差分 GMM(ステージ23)λ=0.093、c=9.717 | 根拠なし。等重み |
| __OTHER__jppfs_cor_NoncurrentAssets | 1.0 | 1.000(補完) | 1.0 / 1.0 | 推定なし(残余科目) | 根拠なし。等重み |
| short_term_loans | 1.0 | 0.207 | 1.0 / 1.0 | 総資産比・差分 GMM(ステージ23)λ=0.677、c=0.477 | 根拠なし。等重み |
| IncomeTaxesPayable | 1.0 | 0.073 | 1.0 / 1.0 | 総資産比・差分 GMM(ステージ23)λ=0.856、c=0.169 | 根拠なし。等重み |
| OtherCL | 1.0 | 0.224 | 1.0 / 1.0 | 総資産比・差分 GMM(ステージ23)λ=0.659、c=0.517 | 根拠なし。等重み |
| payables_trade | 1.0 | 0.295 | 1.0 / 1.0 | 売上高比・差分 GMM(ステージ23.5)λ=0.596、c=0.679 | 根拠なし。等重み |
| accounts_payable_other | 1.0 | 0.275 | 1.0 / 1.0 | 総資産比・差分 GMM(ステージ23)λ=0.612、c=0.633 | 根拠なし。等重み |
| bonds_current | 1.0 | 1.000(補完) | 1.0 / 2.0 | 総資産比・差分 GMM(ステージ23)λ=1.088 | 減少(返済・再交渉)= 2.0、借入金は 1.0。Gilson, John & Lang (1990, JFE 27) 要旨・本文: 銀行借入の割合が高く貸し手が少ない企業ほど私的な債務再編に成功しやすい(本文で確認) |
| __OTHER__jppfs_cor_CurrentLiabilities | 1.0 | 1.000(補完) | 1.0 / 1.0 | 推定なし(残余科目) | 根拠なし。等重み |
| __OTHER__jppfs_cor_Liabilities | 1.0 | 1.000(補完) | 1.0 / 1.0 | 推定なし(残余科目) | 根拠なし。等重み |
| long_term_loans | 1.0 | 0.735 | 1.0 / 1.0 | 総資産比・差分 GMM(ステージ23)λ=0.371、c=1.694 | 根拠なし。等重み |
| bonds_noncurrent | 1.0 | 1.000(補完) | 1.0 / 2.0 | 総資産比・差分 GMM(ステージ23)λ=-0.028 | 減少(返済・再交渉)= 2.0、借入金は 1.0。Gilson, John & Lang (1990) による(bonds_current と同じ) |
| __OTHER__jppfs_cor_NoncurrentLiabilities | 1.0 | 1.000(補完) | 1.0 / 1.0 | 推定なし(残余科目) | 根拠なし。等重み |
| non_controlling_interests | 1.0 | 4.080 | 1.0 / 1.0 | 総資産比・差分 GMM(ステージ23)λ=0.096、c=9.399 | 根拠なし。等重み |
| __OTHER__jppfs_cor_NetAssets | 1.0 | 1.000(補完) | 1.0 / 1.0 | 推定なし(残余科目) | 根拠なし。等重み |
| RetainedEarnings | 1.0 | 1.268 | 1.0 / 1.0 | 総資産比・差分 GMM(ステージ23)λ=0.255、c=2.921 | 根拠なし。等重み |
| CapitalStock | 1.0 | 1.595 | 1.0 / 1.0 | 総資産比・差分 GMM(ステージ23)λ=0.214、c=3.674 | 根拠なし。等重み |
| CapitalSurplus | 1.0 | 0.318 | 1.0 / 1.0 | 総資産比・差分 GMM(ステージ23)λ=0.577、c=0.733 | 減少 = 1.0(最も低い段階のまま)。布袋 (2024): 無償減資は実行頻度が高い(ユーザー確認済み、本ステージでは原典未確認)。既定値と同じため数値は変わらない |
| TreasuryStock | 1.0 | 0.962 | 1.0 / 4.0 | 総資産比・差分 GMM(ステージ23)λ=0.311、c=2.217 | 自己株式の取得(負の値がさらに減る方向)= 4.0。会社法 461 条 1 項: 自己株式の取得等により交付する金銭等の帳簿価額の総額は分配可能額を超えてはならない(e-Gov 法令 API で条文を確認)。分配可能額が 0 の企業(債務超過など)では取得できない。制約は条件付きだが、重みは全社に一律で課す |
| __OTHER__jppfs_cor_ShareholdersEquity | 1.0 | 1.000(補完) | 1.0 / 1.0 | 推定なし(残余科目) | 根拠なし。等重み |
| valuation_difference_securities | 1.0 | 0.037 | 1.0 / 1.0 | 総資産比・差分 GMM(ステージ23)λ=0.921、c=0.086 | 根拠なし。等重み |
| __OTHER__jppfs_cor_ValuationAndTranslationAdjustments | 1.0 | 1.000(補完) | 1.0 / 1.0 | 推定なし(残余科目) | 根拠なし。等重み |

## 推定ベースで等重み(1.0)に補完した科目: 13 / 32

- __OTHER__jppfs_cor_Assets: 推定なし(残余科目)
- __OTHER__jppfs_cor_CurrentAssets: 推定なし(残余科目)
- PropertyPlantAndEquipment: c = 805.7 > 100(総資産比・差分 GMM(ステージ23)λ=0.001)
- IntangibleAssets: λ ∉ (0, 1](総資産比・差分 GMM(ステージ23)λ=-0.171)
- __OTHER__jppfs_cor_NoncurrentAssets: 推定なし(残余科目)
- bonds_current: λ ∉ (0, 1](総資産比・差分 GMM(ステージ23)λ=1.088)
- __OTHER__jppfs_cor_CurrentLiabilities: 推定なし(残余科目)
- __OTHER__jppfs_cor_Liabilities: 推定なし(残余科目)
- bonds_noncurrent: λ ∉ (0, 1](総資産比・差分 GMM(ステージ23)λ=-0.028)
- __OTHER__jppfs_cor_NoncurrentLiabilities: 推定なし(残余科目)
- __OTHER__jppfs_cor_NetAssets: 推定なし(残余科目)
- __OTHER__jppfs_cor_ShareholdersEquity: 推定なし(残余科目)
- __OTHER__jppfs_cor_ValuationAndTranslationAdjustments: 推定なし(残余科目)
