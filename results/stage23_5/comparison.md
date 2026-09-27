# ステージ23.5 B.3: 分母と手法を揃えた再推定

パネル A のうち、すべての年で売上高・売上原価が取れた社(同じ部分標本)。目標は業種中央値。
括弧内は標準誤差(企業クラスタ)。F は第 1 段階の F、Jp は Hansen 検定の p 値。

## 符号を分けない

| 科目 | 分母 | 手法 | λ (se) | F | Jp | 文献値 | 一致するか |
|---|---|---|---|---:|---:|---|---|
| receivables_trade | 総資産 | fe | 0.812 (0.025) | - | - | 0.70 | × |
| receivables_trade | 総資産 | diff_gmm | 0.453 (0.060) | 53 | 0.001 | 0.70 | × |
| receivables_trade | 総資産 | sys_gmm | 0.091 (0.018) | 234 | 0.000 | 0.70 | × |
| receivables_trade | 売上高 | fe | 0.908 (0.030) | - | - | 0.70 | × |
| receivables_trade | 売上高 | diff_gmm | 0.709 (0.048) | 44 | 0.087 | 0.70 | ○(95% CI が含む) |
| receivables_trade | 売上高 | sys_gmm | 0.318 (0.028) | 243 | 0.000 | 0.70 | × |
| payables_trade | 総資産 | fe | 0.902 (0.038) | - | - | (数値を確認できる文献なし) | - |
| payables_trade | 総資産 | diff_gmm | 0.551 (0.048) | 95 | 0.000 | (数値を確認できる文献なし) | - |
| payables_trade | 総資産 | sys_gmm | 0.359 (0.123) | 180 | 0.000 | (数値を確認できる文献なし) | - |
| payables_trade | 売上高 | fe | 1.055 (0.034) | - | - | (数値を確認できる文献なし) | - |
| payables_trade | 売上高 | diff_gmm | 0.596 (0.058) | 86 | 0.054 | (数値を確認できる文献なし) | - |
| payables_trade | 売上高 | sys_gmm | 0.433 (0.628) | 331 | 0.000 | (数値を確認できる文献なし) | - |
| payables_trade | 売上原価 | fe | 0.998 (0.043) | - | - | (数値を確認できる文献なし) | - |
| payables_trade | 売上原価 | diff_gmm | 0.597 (0.055) | 68 | 0.025 | (数値を確認できる文献なし) | - |
| payables_trade | 売上原価 | sys_gmm | 0.229 (0.034) | 331 | 0.000 | (数値を確認できる文献なし) | - |
| inventories | 総資産 | fe | 0.895 (0.032) | - | - | (数値を確認できる文献なし) | - |
| inventories | 総資産 | diff_gmm | 0.577 (0.082) | 37 | 0.000 | (数値を確認できる文献なし) | - |
| inventories | 総資産 | sys_gmm | 0.168 (0.019) | 208 | 0.000 | (数値を確認できる文献なし) | - |
| inventories | 売上高 | fe | 1.007 (0.054) | - | - | (数値を確認できる文献なし) | - |
| inventories | 売上高 | diff_gmm | 0.663 (0.142) | 16 | 0.021 | (数値を確認できる文献なし) | - |
| inventories | 売上高 | sys_gmm | 0.336 (0.073) | 107 | 0.041 | (数値を確認できる文献なし) | - |
| inventories | 売上原価 | fe | 1.049 (0.055) | - | - | (数値を確認できる文献なし) | - |
| inventories | 売上原価 | diff_gmm | 0.731 (0.156) | 15 | 0.034 | (数値を確認できる文献なし) | - |
| inventories | 売上原価 | sys_gmm | 0.515 (0.080) | 107 | 0.274 | (数値を確認できる文献なし) | - |
| CashAndDeposits | 総資産 | fe | 0.906 (0.020) | - | - | (数値を確認できる文献なし) | - |
| CashAndDeposits | 総資産 | diff_gmm | 0.333 (0.058) | 84 | 0.081 | (数値を確認できる文献なし) | - |
| CashAndDeposits | 総資産 | sys_gmm | 0.318 (0.038) | 203 | 0.012 | (数値を確認できる文献なし) | - |
| CashAndDeposits | 売上高 | fe | 0.870 (0.041) | - | - | (数値を確認できる文献なし) | - |
| CashAndDeposits | 売上高 | diff_gmm | 0.454 (0.060) | 43 | 0.227 | (数値を確認できる文献なし) | - |
| CashAndDeposits | 売上高 | sys_gmm | 0.305 (0.041) | 61 | 0.011 | (数値を確認できる文献なし) | - |

## 運転資本 4 科目の順位(差分 GMM、λ の大きい順)

- すべて総資産比: inventories(0.58) > payables_trade(0.55) > receivables_trade(0.45) > CashAndDeposits(0.33)
- すべて売上高比: receivables_trade(0.71) > inventories(0.66) > payables_trade(0.60) > CashAndDeposits(0.45)

## 符号別(差分 GMM)

| 科目 | 分母 | λ+ (se) | n+ | λ− (se) | n− |
|---|---|---|---:|---|---:|
| receivables_trade | 総資産 | 0.400 (0.085) | 2946 | 0.402 (0.106) | 2873 |
| receivables_trade | 売上高 | 0.739 (0.058) | 2770 | 0.666 (0.089) | 2904 |
| payables_trade | 総資産 | 0.412 (0.056) | 2969 | 0.564 (0.073) | 2977 |
| payables_trade | 売上高 | 0.371 (0.074) | 2827 | 0.660 (0.093) | 2987 |
| payables_trade | 売上原価 | 0.420 (0.074) | 2747 | 0.614 (0.082) | 2942 |
| inventories | 総資産 | 0.202 (0.089) | 3049 | 0.574 (0.089) | 2781 |
| inventories | 売上高 | 0.139 (0.174) | 3018 | 0.600 (0.194) | 2837 |
| inventories | 売上原価 | 0.143 (0.212) | 3004 | 0.544 (0.196) | 2910 |
| CashAndDeposits | 総資産 | 0.380 (0.090) | 2724 | 0.474 (0.093) | 2988 |
| CashAndDeposits | 売上高 | 0.405 (0.068) | 2735 | 0.467 (0.074) | 2978 |

## システム GMM(パネル A 全社、総資産比、全 23 科目)

- Hansen 検定が 5% で棄却: 14 / 23
- 推定不能・標準誤差が非有限: 1

| 科目 | λ (se) | F | Jp | status |
|---|---|---:|---:|---|
| inventories | 0.184 (0.018) | 1424 | 0.000 | ok |
| receivables_trade | 0.077 (0.017) | 302 | 0.000 | ok |
| CashAndDeposits | 0.290 (0.032) | 235 | 0.010 | ok |
| OtherCA | 0.513 (0.028) | 521 | 0.000 | ok |
| allowance_doubtful_current | 0.415 (0.062) | 88 | 0.103 | ok |
| PropertyPlantAndEquipment | -0.050 (0.016) | 72 | 0.002 | ok |
| IntangibleAssets | 0.292 (0.118) | 104 | 0.000 | ok |
| InvestmentsAndOtherAssets | 0.181 (0.037) | 138 | 0.034 | ok |
| allowance_doubtful_noncurrent | 0.138 (0.055) | 34 | 0.764 | ok |
| short_term_loans | 0.455 (0.041) | 303 | 0.000 | ok |
| IncomeTaxesPayable | 0.883 (0.026) | 435 | 0.009 | ok |
| OtherCL | 0.584 (0.027) | 315 | 0.000 | ok |
| payables_trade | 0.472 (0.044) | 188 | 0.000 | ok |
| accounts_payable_other | 0.521 (0.037) | 131 | 0.165 | ok |
| bonds_current | 1.023 (0.045) | 326 | 0.106 | ok |
| long_term_loans | 0.251 (0.044) | 71 | 0.000 | ok |
| bonds_noncurrent | 0.185 (0.064) | 31 | 0.057 | ok |
| non_controlling_interests | 0.017 (nan) | 65 | 0.901 | nonfinite_se |
| RetainedEarnings | 0.080 (0.183) | 65 | 0.001 | ok |
| CapitalStock | 0.160 (0.024) | 56 | 0.545 | ok |
| CapitalSurplus | 0.016 (0.027) | 104 | 0.155 | ok |
| TreasuryStock | 0.209 (0.037) | 166 | 0.273 | ok |
| valuation_difference_securities | 0.725 (0.032) | 148 | 0.000 | ok |
