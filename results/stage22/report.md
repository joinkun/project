# ステージ22: 部分調整モデルによる調整速度 λ の推定

括弧内は標準誤差(企業クラスタ頑健、GMM は不均一分散頑健)。
印: 範囲外 = λ ∉ (0, 1]、弱IV = 第 1 段階の F < 10、J棄却 = Hansen の過剰識別検定 p < 0.05。
AR(2) 検定は、水準が 3 期(差分式が 1 本)しかないため計算できない。

## 主表(パネル A、符号を分けない)

| 科目 | 業種中央値・FE | 業種中央値・GMM | 企業特性・FE | 企業特性・GMM |
|---|---|---|---|---|
| inventories | 1.017 (0.035) 範囲外 | 0.721 (0.060) J棄却 | 1.055 (0.038) 範囲外 | -1.466 (1.827) 範囲外、弱IV |
| receivables_trade | 0.835 (0.060) | 0.257 (0.106) | 0.837 (0.063) | 0.170 (0.134) |
| CashAndDeposits | 1.126 (0.025) 範囲外 | 0.251 (0.081) | 1.170 (0.027) 範囲外 | 0.249 (0.121) J棄却 |
| OtherCA | 1.149 (0.030) 範囲外 | 0.670 (0.056) | 1.164 (0.031) 範囲外 | 0.751 (0.057) |
| allowance_doubtful_current | 1.190 (0.084) 範囲外 | -0.113 (0.418) 範囲外、弱IV | 1.190 (0.083) 範囲外 | -0.118 (0.346) 範囲外、弱IV |
| PropertyPlantAndEquipment | 0.859 (0.052) | 0.198 (0.095) J棄却 | 0.887 (0.054) | -0.285 (0.213) 範囲外 |
| IntangibleAssets | 0.979 (0.074) | -0.478 (0.363) 範囲外 | 0.990 (0.074) | -0.130 (0.231) 範囲外、弱IV |
| InvestmentsAndOtherAssets | 0.822 (0.088) | 0.411 (0.103) | 0.823 (0.089) | 0.687 (0.167) 弱IV、J棄却 |
| allowance_doubtful_noncurrent | 1.030 (0.147) 範囲外 | -0.682 (0.780) 範囲外、弱IV | 1.032 (0.147) 範囲外 | 0.078 (0.397) 弱IV |
| short_term_loans | 1.196 (0.036) 範囲外 | 0.459 (0.113) | 1.256 (0.040) 範囲外 | 0.696 (0.095) J棄却 |
| IncomeTaxesPayable | 1.377 (0.023) 範囲外 | 0.927 (0.040) | 1.403 (0.024) 範囲外 | 1.019 (0.040) 範囲外 |
| OtherCL | 0.898 (0.038) | 0.356 (0.070) J棄却 | 0.925 (0.041) | 0.588 (0.091) |
| payables_trade | 1.140 (0.038) 範囲外 | 0.423 (0.101) | 1.159 (0.040) 範囲外 | 0.074 (0.186) J棄却 |
| accounts_payable_other | 1.145 (0.056) 範囲外 | 0.590 (0.116) | 1.154 (0.058) 範囲外 | 0.700 (0.116) |
| bonds_current | 1.442 (0.049) 範囲外 | 1.034 (0.069) 範囲外 | - rank_deficient | - rank_deficient |
| long_term_loans | 1.098 (0.034) 範囲外 | 0.428 (0.096) | 1.164 (0.039) 範囲外 | 0.634 (0.092) J棄却 |
| bonds_noncurrent | 1.139 (0.072) 範囲外 | 0.201 (0.265) | 1.142 (0.072) 範囲外 | 0.274 (0.203) 弱IV |
| non_controlling_interests | 1.037 (0.065) 範囲外 | 0.335 (0.268) 弱IV | 1.051 (0.065) 範囲外 | 0.369 (0.297) 弱IV |
| RetainedEarnings | 1.091 (0.077) 範囲外 | 0.179 (0.198) | 1.113 (0.075) 範囲外 | 0.878 (0.163) 弱IV、J棄却 |
| CapitalStock | 1.091 (0.044) 範囲外 | 0.415 (0.147) | 1.099 (0.042) 範囲外 | 0.536 (0.169) 弱IV |
| CapitalSurplus | 1.103 (0.083) 範囲外 | 0.587 (0.274) 弱IV | 1.106 (0.067) 範囲外 | 0.477 (0.157) |
| TreasuryStock | 1.092 (0.035) 範囲外 | 0.153 (0.263) | 1.080 (0.035) 範囲外 | 0.368 (0.145) |
| valuation_difference_securities | 1.124 (0.054) 範囲外 | 1.058 (0.069) 範囲外、J棄却 | 1.090 (0.067) 範囲外 | 1.342 (0.206) 範囲外、弱IV、J棄却 |

## 主表(パネル B、符号を分けない)

| 科目 | 業種中央値・FE | 業種中央値・GMM | 企業特性・FE | 企業特性・GMM |
|---|---|---|---|---|
| inventories | 1.006 (0.030) 範囲外 | 0.749 (0.071) J棄却 | 1.055 (0.034) 範囲外 | 1.454 (0.363) 範囲外、弱IV、J棄却 |
| receivables_trade | 1.168 (0.034) 範囲外 | 0.479 (0.120) | 1.178 (0.035) 範囲外 | 0.359 (0.216) 弱IV |
| CashAndDeposits | 1.122 (0.027) 範囲外 | 0.287 (0.097) | 1.165 (0.029) 範囲外 | 0.215 (0.144) J棄却 |
| OtherCA | 1.276 (0.028) 範囲外 | 0.721 (0.061) J棄却 | 1.299 (0.028) 範囲外 | 0.835 (0.062) |
| allowance_doubtful_current | 1.097 (0.058) 範囲外 | 0.109 (0.332) 弱IV | 1.101 (0.058) 範囲外 | 0.058 (0.346) 弱IV |
| PropertyPlantAndEquipment | 0.925 (0.038) | 0.132 (0.139) J棄却 | 0.959 (0.042) | -0.192 (0.217) 範囲外 |
| IntangibleAssets | 1.069 (0.051) 範囲外 | -0.227 (0.352) 範囲外 | 1.079 (0.051) 範囲外 | -0.101 (0.269) 範囲外 |
| InvestmentsAndOtherAssets | 1.120 (0.045) 範囲外 | 0.213 (0.127) | 1.138 (0.047) 範囲外 | 0.797 (0.256) 弱IV、J棄却 |
| allowance_doubtful_noncurrent | 1.264 (0.112) 範囲外 | -0.503 (0.694) 範囲外、弱IV | 1.264 (0.113) 範囲外 | -0.028 (0.465) 範囲外、弱IV |
| short_term_loans | 1.205 (0.038) 範囲外 | 0.481 (0.124) | 1.266 (0.041) 範囲外 | 0.669 (0.112) J棄却 |
| IncomeTaxesPayable | 1.372 (0.024) 範囲外 | 0.928 (0.042) | 1.412 (0.024) 範囲外 | 1.050 (0.040) 範囲外 |
| OtherCL | 1.199 (0.025) 範囲外 | 0.382 (0.081) J棄却 | 1.264 (0.025) 範囲外 | 0.962 (0.117) |
| payables_trade | 1.198 (0.039) 範囲外 | 0.428 (0.096) J棄却 | 1.234 (0.040) 範囲外 | -0.279 (0.253) 範囲外、弱IV |
| accounts_payable_other | 1.320 (0.044) 範囲外 | 0.811 (0.112) J棄却 | 1.338 (0.045) 範囲外 | 0.962 (0.100) |
| bonds_current | 1.474 (0.048) 範囲外 | 1.020 (0.076) 範囲外 | - rank_deficient | - rank_deficient |
| long_term_loans | 1.070 (0.033) 範囲外 | 0.369 (0.088) | 1.146 (0.036) 範囲外 | 0.658 (0.094) J棄却 |
| bonds_noncurrent | 1.129 (0.081) 範囲外 | 0.315 (0.259) | 1.132 (0.081) 範囲外 | 0.463 (0.179) 弱IV |
| non_controlling_interests | 0.962 (0.035) | -0.977 (1.296) 範囲外、弱IV | 0.968 (0.036) | 0.255 (0.423) 弱IV |
| RetainedEarnings | 0.983 (0.055) | 0.069 (0.171) | 1.000 (0.052) 範囲外 | 0.757 (0.152) 弱IV、J棄却 |
| CapitalStock | 1.073 (0.051) 範囲外 | 0.401 (0.154) | 1.092 (0.048) 範囲外 | 0.680 (0.134) 弱IV、J棄却 |
| CapitalSurplus | 1.064 (0.056) 範囲外 | 0.382 (0.254) 弱IV | 1.088 (0.058) 範囲外 | 0.585 (0.146) J棄却 |
| TreasuryStock | 1.097 (0.041) 範囲外 | 0.246 (0.272) 弱IV、J棄却 | 1.080 (0.041) 範囲外 | 0.384 (0.160) 弱IV |
| valuation_difference_securities | 1.180 (0.061) 範囲外 | 1.116 (0.074) 範囲外、J棄却 | 1.143 (0.075) 範囲外 | 1.201 (0.240) 範囲外、弱IV、J棄却 |

## 符号別(パネル A)

positive = 目標より下(増やす方向)、negative = 目標より上(減らす方向)。n は推定に使った観測数(FE は 2 年とも同じ群の社の観測、GMM は 2 年とも同じ符号の社数)。

| 科目 | 目標 | 手法 | λ+ (se) | n+ | λ− (se) | n− |
|---|---|---|---|---:|---|---:|
| inventories | industry | fe | 0.707 (0.155)  | 3522 | 1.062 (0.046) 範囲外 | 3382 |
| inventories | industry | diff_gmm | 0.382 (0.092)  | 1761 | 0.818 (0.093) J棄却 | 1691 |
| inventories | firm | fe | 0.654 (0.135)  | 1838 | 0.941 (0.147)  | 1630 |
| inventories | firm | diff_gmm | 0.013 (0.286) 弱IV、J棄却 | 919 | -1.670 (0.772) 範囲外、弱IV | 815 |
| receivables_trade | industry | fe | 0.946 (0.062)  | 3516 | 1.158 (0.049) 範囲外 | 3470 |
| receivables_trade | industry | diff_gmm | 0.285 (0.122) J棄却 | 1758 | 0.399 (0.155)  | 1735 |
| receivables_trade | firm | fe | 0.776 (0.047)  | 2460 | 0.950 (0.054)  | 2200 |
| receivables_trade | firm | diff_gmm | -0.849 (0.348) 範囲外、弱IV | 1230 | -0.230 (0.238) 範囲外、弱IV | 1100 |
| CashAndDeposits | industry | fe | 0.973 (0.051)  | 3320 | 1.156 (0.037) 範囲外 | 3486 |
| CashAndDeposits | industry | diff_gmm | 0.282 (0.121) J棄却 | 1660 | 0.336 (0.132)  | 1743 |
| CashAndDeposits | firm | fe | 0.647 (0.057)  | 1874 | 0.876 (0.080)  | 2290 |
| CashAndDeposits | firm | diff_gmm | -0.311 (0.290) 範囲外、弱IV、J棄却 | 937 | -1.694 (0.504) 範囲外、弱IV | 1145 |
| OtherCA | industry | fe | 1.012 (0.061) 範囲外 | 3224 | 1.248 (0.041) 範囲外 | 3196 |
| OtherCA | industry | diff_gmm | 0.801 (0.135) J棄却 | 1612 | 0.840 (0.073) J棄却 | 1598 |
| OtherCA | firm | fe | 0.548 (0.089)  | 2034 | 0.672 (0.045)  | 1536 |
| OtherCA | firm | diff_gmm | -1.225 (0.828) 範囲外、弱IV | 1017 | -0.292 (0.202) 範囲外 | 768 |
| allowance_doubtful_current | industry | fe | 1.165 (0.101) 範囲外 | 3388 | 1.740 (0.551) 範囲外 | 3358 |
| allowance_doubtful_current | industry | diff_gmm | -0.521 (0.644) 範囲外、弱IV | 1694 | 0.737 (0.413)  | 1679 |
| allowance_doubtful_current | firm | fe | 0.722 (0.342)  | 1234 | 0.448 (0.193)  | 1460 |
| allowance_doubtful_current | firm | diff_gmm | -2.326 (1.078) 範囲外、弱IV | 617 | -0.418 (0.295) 範囲外、弱IV、J棄却 | 730 |
| PropertyPlantAndEquipment | industry | fe | 0.777 (0.046)  | 3560 | 0.798 (0.089)  | 3646 |
| PropertyPlantAndEquipment | industry | diff_gmm | 0.068 (0.119)  | 1780 | 0.113 (0.120) J棄却 | 1823 |
| PropertyPlantAndEquipment | firm | fe | 0.949 (0.092)  | 1860 | 0.619 (0.150)  | 3334 |
| PropertyPlantAndEquipment | firm | diff_gmm | 0.426 (0.243) 弱IV、J棄却 | 930 | -0.664 (0.282) 範囲外 | 1667 |
| IntangibleAssets | industry | fe | 1.063 (0.299) 範囲外 | 3450 | 1.049 (0.047) 範囲外 | 3372 |
| IntangibleAssets | industry | diff_gmm | 1.511 (0.461) 範囲外 | 1725 | -0.721 (0.566) 範囲外、弱IV | 1686 |
| IntangibleAssets | firm | fe | 0.813 (0.104)  | 1978 | 0.831 (0.055)  | 2296 |
| IntangibleAssets | firm | diff_gmm | -1.186 (0.455) 範囲外、弱IV | 989 | -0.129 (0.209) 範囲外 | 1148 |
| InvestmentsAndOtherAssets | industry | fe | 0.973 (0.057)  | 3612 | 0.886 (0.124)  | 3360 |
| InvestmentsAndOtherAssets | industry | diff_gmm | 0.004 (0.112) J棄却 | 1806 | 0.635 (0.200)  | 1680 |
| InvestmentsAndOtherAssets | firm | fe | 0.772 (0.065)  | 3334 | 0.893 (0.072)  | 1884 |
| InvestmentsAndOtherAssets | firm | diff_gmm | -0.164 (0.215) 範囲外、J棄却 | 1667 | -0.518 (0.423) 範囲外、弱IV | 942 |
| allowance_doubtful_noncurrent | industry | fe | 0.861 (0.159)  | 3554 | 0.119 (0.644)  | 2270 |
| allowance_doubtful_noncurrent | industry | diff_gmm | -1.597 (1.275) 範囲外、弱IV | 1777 | 0.937 (1.590)  | 1135 |
| allowance_doubtful_noncurrent | firm | fe | 0.573 (0.204)  | 1696 | 0.704 (0.071)  | 934 |
| allowance_doubtful_noncurrent | firm | diff_gmm | -3.023 (1.252) 範囲外、弱IV | 848 | -1.897 (1.068) 範囲外、弱IV | 467 |
| short_term_loans | industry | fe | 0.653 (0.086)  | 3334 | 1.177 (0.038) 範囲外 | 3406 |
| short_term_loans | industry | diff_gmm | 0.171 (0.079) J棄却 | 1667 | 0.477 (0.126)  | 1703 |
| short_term_loans | firm | fe | 0.350 (0.123)  | 1498 | 0.649 (0.099)  | 1914 |
| short_term_loans | firm | diff_gmm | -0.709 (0.374) 範囲外、弱IV | 749 | -3.012 (0.796) 範囲外、弱IV | 957 |
| IncomeTaxesPayable | industry | fe | 1.364 (0.061) 範囲外 | 2612 | 1.338 (0.043) 範囲外 | 2658 |
| IncomeTaxesPayable | industry | diff_gmm | 0.821 (0.106) J棄却 | 1306 | 0.782 (0.085)  | 1329 |
| IncomeTaxesPayable | firm | fe | 1.210 (0.161) 範囲外 | 1598 | 0.911 (0.122)  | 1246 |
| IncomeTaxesPayable | firm | diff_gmm | 1.273 (0.335) 範囲外、弱IV、J棄却 | 799 | -1.518 (0.471) 範囲外、弱IV、J棄却 | 623 |
| OtherCL | industry | fe | 0.995 (0.052)  | 3490 | 1.014 (0.043) 範囲外 | 3340 |
| OtherCL | industry | diff_gmm | 0.555 (0.082) J棄却 | 1745 | 0.526 (0.091) J棄却 | 1670 |
| OtherCL | firm | fe | 0.705 (0.070)  | 2526 | 0.889 (0.044)  | 1516 |
| OtherCL | firm | diff_gmm | -0.013 (0.307) 範囲外、弱IV | 1263 | -0.139 (0.204) 範囲外 | 758 |
| payables_trade | industry | fe | 0.962 (0.064)  | 3488 | 1.169 (0.053) 範囲外 | 3588 |
| payables_trade | industry | diff_gmm | 0.352 (0.104)  | 1744 | 0.595 (0.131)  | 1794 |
| payables_trade | firm | fe | 0.425 (0.089)  | 1458 | 0.774 (0.054)  | 2376 |
| payables_trade | firm | diff_gmm | -1.310 (0.663) 範囲外、弱IV、J棄却 | 729 | -0.470 (0.197) 範囲外、J棄却 | 1188 |
| accounts_payable_other | industry | fe | 0.481 (0.175)  | 2502 | 1.253 (0.047) 範囲外 | 3294 |
| accounts_payable_other | industry | diff_gmm | 0.187 (0.114) J棄却 | 1251 | 0.765 (0.128) J棄却 | 1647 |
| accounts_payable_other | firm | fe | 0.440 (0.205)  | 1388 | 0.666 (0.073)  | 976 |
| accounts_payable_other | firm | diff_gmm | -1.320 (0.753) 範囲外、弱IV | 694 | -0.314 (0.289) 範囲外、弱IV | 488 |
| bonds_current | industry | fe | - insufficient_obs | 0 | 1.324 (0.086) 範囲外 | 756 |
| bonds_current | industry | diff_gmm | - insufficient_obs | 0 | 0.848 (0.142)  | 378 |
| bonds_current | firm | fe | - rank_deficient | 7638 | - rank_deficient | 7638 |
| bonds_current | firm | diff_gmm | - rank_deficient | 7638 | - rank_deficient | 7638 |
| long_term_loans | industry | fe | 0.721 (0.069)  | 3390 | 1.034 (0.034) 範囲外 | 3508 |
| long_term_loans | industry | diff_gmm | -0.018 (0.159) 範囲外、J棄却 | 1695 | 0.342 (0.103)  | 1754 |
| long_term_loans | firm | fe | 1.013 (0.119) 範囲外 | 1136 | 0.473 (0.067)  | 2484 |
| long_term_loans | firm | diff_gmm | 1.453 (0.565) 範囲外、弱IV、J棄却 | 568 | -0.427 (0.192) 範囲外 | 1242 |
| bonds_noncurrent | industry | fe | 0.612 (0.184)  | 32 | 1.043 (0.093) 範囲外 | 1314 |
| bonds_noncurrent | industry | diff_gmm | - insufficient_obs | 16 | 0.127 (0.364)  | 657 |
| bonds_noncurrent | firm | fe | 0.461 (0.154)  | 298 | 0.376 (0.134)  | 776 |
| bonds_noncurrent | firm | diff_gmm | -0.766 (1.105) 範囲外、弱IV | 149 | -1.421 (0.510) 範囲外、弱IV | 388 |
| non_controlling_interests | industry | fe | -0.015 (0.027) 範囲外 | 1196 | 1.022 (0.062) 範囲外 | 2938 |
| non_controlling_interests | industry | diff_gmm | 0.041 (0.056)  | 598 | 0.141 (0.400) 弱IV | 1469 |
| non_controlling_interests | firm | fe | 0.778 (0.165)  | 1254 | 0.690 (0.093)  | 798 |
| non_controlling_interests | firm | diff_gmm | 0.122 (0.264) 弱IV | 627 | -1.128 (0.767) 範囲外、弱IV | 399 |
| RetainedEarnings | industry | fe | 1.119 (0.086) 範囲外 | 3704 | 0.816 (0.049)  | 3532 |
| RetainedEarnings | industry | diff_gmm | 0.349 (0.246)  | 1852 | -0.132 (0.155) 範囲外 | 1766 |
| RetainedEarnings | firm | fe | 0.800 (0.052)  | 3456 | 0.317 (0.113)  | 1932 |
| RetainedEarnings | firm | diff_gmm | -1.190 (0.674) 範囲外、弱IV | 1728 | -0.616 (0.253) 範囲外 | 966 |
| CapitalStock | industry | fe | 0.351 (0.079)  | 3326 | 1.143 (0.095) 範囲外 | 3822 |
| CapitalStock | industry | diff_gmm | 0.065 (0.068)  | 1663 | -0.264 (0.489) 範囲外、弱IV | 1911 |
| CapitalStock | firm | fe | 0.120 (0.168)  | 676 | 1.323 (0.278) 範囲外 | 4200 |
| CapitalStock | firm | diff_gmm | -0.179 (0.214) 範囲外、弱IV、J棄却 | 338 | -3.134 (1.195) 範囲外、弱IV | 2100 |
| CapitalSurplus | industry | fe | 0.574 (0.190)  | 3424 | 1.129 (0.109) 範囲外 | 3782 |
| CapitalSurplus | industry | diff_gmm | 0.137 (0.137)  | 1712 | 0.358 (0.475) 弱IV | 1891 |
| CapitalSurplus | firm | fe | 0.777 (0.238)  | 948 | 0.850 (0.148)  | 3736 |
| CapitalSurplus | firm | diff_gmm | -0.497 (0.225) 範囲外 | 474 | -0.997 (0.601) 範囲外、弱IV | 1868 |
| TreasuryStock | industry | fe | 1.113 (0.060) 範囲外 | 3366 | 0.969 (0.136)  | 3656 |
| TreasuryStock | industry | diff_gmm | -0.128 (0.403) 範囲外、弱IV | 1683 | 0.564 (0.185)  | 1828 |
| TreasuryStock | firm | fe | 0.697 (0.135)  | 2010 | 0.450 (0.098)  | 1522 |
| TreasuryStock | firm | diff_gmm | -2.089 (0.742) 範囲外、弱IV | 1005 | -1.649 (0.603) 範囲外、弱IV | 761 |
| valuation_difference_securities | industry | fe | 0.421 (0.035)  | 3336 | 1.275 (0.066) 範囲外 | 3046 |
| valuation_difference_securities | industry | diff_gmm | 0.226 (0.061) J棄却 | 1668 | 1.548 (0.124) 範囲外、J棄却 | 1523 |
| valuation_difference_securities | firm | fe | 0.602 (0.112)  | 3032 | 0.517 (0.230)  | 382 |
| valuation_difference_securities | firm | diff_gmm | -1.132 (0.443) 範囲外、J棄却 | 1516 | -0.843 (0.623) 範囲外、弱IV | 191 |

観測数が 100 未満の群: 4 件(bonds_current/industry/fe/positive=0、bonds_current/industry/diff_gmm/positive=0、bonds_noncurrent/industry/fe/positive=32、bonds_noncurrent/industry/diff_gmm/positive=16)

## 重みへの変換 c = (1 − λ) / λ と破綻した科目(パネル A、符号を分けない)

| 科目 | 業種・GMM の c | 企業・GMM の c | 破綻(範囲外・推定不能)の組 |
|---|---:|---:|---|
| inventories | 0.39 | - | industry/fe(λ=1.02)、firm/fe(λ=1.06)、firm/diff_gmm(λ=-1.47) |
| receivables_trade | 2.89 | 4.87 |  |
| CashAndDeposits | 2.98 | 3.02 | industry/fe(λ=1.13)、firm/fe(λ=1.17) |
| OtherCA | 0.49 | 0.33 | industry/fe(λ=1.15)、firm/fe(λ=1.16) |
| allowance_doubtful_current | - | - | industry/fe(λ=1.19)、industry/diff_gmm(λ=-0.11)、firm/fe(λ=1.19)、firm/diff_gmm(λ=-0.12) |
| PropertyPlantAndEquipment | 4.06 | - | firm/diff_gmm(λ=-0.29) |
| IntangibleAssets | - | - | industry/diff_gmm(λ=-0.48)、firm/diff_gmm(λ=-0.13) |
| InvestmentsAndOtherAssets | 1.43 | 0.46 |  |
| allowance_doubtful_noncurrent | - | 11.79 | industry/fe(λ=1.03)、industry/diff_gmm(λ=-0.68)、firm/fe(λ=1.03) |
| short_term_loans | 1.18 | 0.44 | industry/fe(λ=1.20)、firm/fe(λ=1.26) |
| IncomeTaxesPayable | 0.08 | - | industry/fe(λ=1.38)、firm/fe(λ=1.40)、firm/diff_gmm(λ=1.02) |
| OtherCL | 1.81 | 0.70 |  |
| payables_trade | 1.36 | 12.52 | industry/fe(λ=1.14)、firm/fe(λ=1.16) |
| accounts_payable_other | 0.70 | 0.43 | industry/fe(λ=1.14)、firm/fe(λ=1.15) |
| bonds_current | - | - | industry/fe(λ=1.44)、industry/diff_gmm(λ=1.03)、firm/fe(rank_deficient)、firm/diff_gmm(rank_deficient) |
| long_term_loans | 1.34 | 0.58 | industry/fe(λ=1.10)、firm/fe(λ=1.16) |
| bonds_noncurrent | 3.98 | 2.65 | industry/fe(λ=1.14)、firm/fe(λ=1.14) |
| non_controlling_interests | 1.99 | 1.71 | industry/fe(λ=1.04)、firm/fe(λ=1.05) |
| RetainedEarnings | 4.58 | 0.14 | industry/fe(λ=1.09)、firm/fe(λ=1.11) |
| CapitalStock | 1.41 | 0.87 | industry/fe(λ=1.09)、firm/fe(λ=1.10) |
| CapitalSurplus | 0.71 | 1.10 | industry/fe(λ=1.10)、firm/fe(λ=1.11) |
| TreasuryStock | 5.52 | 1.72 | industry/fe(λ=1.09)、firm/fe(λ=1.08) |
| valuation_difference_securities | - | - | industry/fe(λ=1.12)、industry/diff_gmm(λ=1.06)、firm/fe(λ=1.09)、firm/diff_gmm(λ=1.34) |

λ ∉ (0, 1] の組: FE 36 / 46、GMM 12 / 46(推定不能を含む)

## 文献値との照合(パネル A)

| 科目 | 方向 | 文献値 | 出典 | 業種・FE | 業種・GMM | 企業・FE | 企業・GMM |
|---|---|---|---|---|---|---|---|
| short_term_loans | both | 0.27〜0.28 | 岡本(2013)、日本上場企業 | 1.196 (0.036) | 0.459 (0.113) ✓CI | 1.256 (0.040) | 0.696 (0.095) |
| CapitalStock | positive | 0.33 | 岡本(2013) 増資 | 0.351 (0.079) ✓CI | 0.065 (0.068) | 0.120 (0.168) ✓CI | -0.179 (0.214) |
| CapitalStock | negative | 0.037 | 岡本(2013) 減資 | 1.143 (0.095) | -0.264 (0.489) ✓CI | 1.323 (0.278) | -3.134 (1.195) |
| receivables_trade | both | 0.65〜0.7 | Luo(2021)、Abuhommous(2017) | 0.835 (0.060) | 0.257 (0.106) | 0.837 (0.063) | 0.170 (0.134) |
| payables_trade | both | 0.65〜0.7 | Luo(2021)、Abuhommous(2017) | 1.140 (0.038) | 0.423 (0.101) | 1.159 (0.040) | 0.074 (0.186) |

✓CI = 95% 信頼区間が文献値(の範囲)と重なる。

運転資本 4 科目の順位(λ の大きい順、符号を分けない、パネル A):

- industry・fe: payables_trade(1.14) > CashAndDeposits(1.13) > inventories(1.02) > receivables_trade(0.83)
- industry・diff_gmm: inventories(0.72) > payables_trade(0.42) > receivables_trade(0.26) > CashAndDeposits(0.25)
- firm・fe: CashAndDeposits(1.17) > payables_trade(1.16) > inventories(1.06) > receivables_trade(0.84)
- firm・diff_gmm: CashAndDeposits(0.25) > receivables_trade(0.17) > payables_trade(0.07) > inventories(-1.47)

文献(Seidu 2022): 現金が最速、棚卸資産が最遅。

## パネル A(不一致を含む)と B(除く)の比較(符号を分けない)

| 目標 | 手法 | |λ_A − λ_B| の中央値 | 最大 | 最大の科目 | 差 > 2×se_A の科目数 |
|---|---|---:|---:|---|---:|
| firm | diff_gmm | 0.112 | 2.920 | inventories | 2 / 22 |
| firm | fe | 0.074 | 0.341 | receivables_trade | 5 / 22 |
| industry | diff_gmm | 0.066 | 1.312 | non_controlling_interests | 2 / 23 |
| industry | fe | 0.058 | 0.334 | receivables_trade | 5 / 23 |
