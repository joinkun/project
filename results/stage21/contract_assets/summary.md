# ステージ21-B C.3: 契約資産ほか、グループ科目の取りこぼし

## (2) 排他性スコア(fy2024 当期、非金融 4156 社)

| A | B | A の社数 | B の社数 | 併存 | スコア |
|---|---|---:|---:|---:|---:|
| ContractAssets | AccountsReceivableTrade | 437 | 2228 | 388 | 0.888 |
| ContractAssets | NotesAndAccountsReceivableTrade | 437 | 607 | 22 | 0.050 |
| ContractAssets | NotesAndAccountsReceivableTradeAndContractAssets | 437 | 645 | 0 | 0.000 |
| ContractAssets | NotesReceivableTrade | 437 | 1013 | 273 | 0.625 |
| AccountsReceivableTradeAndContractAssets | AccountsReceivableTrade | 58 | 2228 | 0 | 0.000 |
| AccountsReceivableTradeAndContractAssets | NotesAndAccountsReceivableTrade | 58 | 607 | 0 | 0.000 |
| AccountsReceivableTradeAndContractAssets | NotesAndAccountsReceivableTradeAndContractAssets | 58 | 645 | 0 | 0.000 |
| AccountsReceivableTradeAndContractAssets | NotesReceivableTrade | 58 | 1013 | 3 | 0.052 |
| ElectronicallyRecordedMonetaryClaimsOperatingCA | AccountsReceivableTrade | 1037 | 2228 | 547 | 0.527 |
| ElectronicallyRecordedMonetaryClaimsOperatingCA | NotesAndAccountsReceivableTrade | 1037 | 607 | 196 | 0.323 |
| ElectronicallyRecordedMonetaryClaimsOperatingCA | NotesAndAccountsReceivableTradeAndContractAssets | 1037 | 645 | 209 | 0.324 |
| ElectronicallyRecordedMonetaryClaimsOperatingCA | NotesReceivableTrade | 1037 | 1013 | 493 | 0.487 |
| ContractAssets | AccountsReceivableTradeAndContractAssets | 437 | 58 | 0 | 0.000 |
| ContractAssets | ElectronicallyRecordedMonetaryClaimsOperatingCA | 437 | 1037 | 168 | 0.384 |
| AccountsReceivableTradeAndContractAssets | ElectronicallyRecordedMonetaryClaimsOperatingCA | 58 | 1037 | 5 | 0.086 |

ContractAssets を持つが、現行の売上債権 members を 1 つも持たない社: 16

## (4) 金額規模(fy2024 当期、候補を持つ非金融の社)

| 候補 | 社数 | 候補 / __OTHER__CurrentAssets 中央値 | 候補 / 総資産 中央値 | 候補 / 総資産 p90 |
|---|---:|---:|---:|---:|
| contract_assets | 495 | 0.361 | 0.0188 | 0.1795 |
| electronic_claims | 1037 | 0.892 | 0.0352 | 0.0987 |
| construction_receivables | 152 | 0.783 | 0.3011 | 0.4877 |
| operating_receivables | 104 | 0.576 | 0.0216 | 0.1548 |

## 表示の切り替え(同じ社で、要素の有無が期の間で変わる件数)

期: 2024 = fy2024 当期、2023a = fy2024 前期、2023b = 2023-24 当期、2022 = 2023-24 前期。

| 要素 | 比較 | 両期にいる社 | 有→無 | 無→有 |
|---|---|---:|---:|---:|
| AccountsReceivableTrade | 2022→2023b | 3994 | 17 | 37 |
| AccountsReceivableTrade | 2023a→2024 | 4083 | 4 | 9 |
| AccountsReceivableTrade | 2023b→2023a | 3997 | 23 | 19 |
| NotesAndAccountsReceivableTrade | 2022→2023b | 3994 | 77 | 1 |
| NotesAndAccountsReceivableTrade | 2023a→2024 | 4083 | 3 | 1 |
| NotesAndAccountsReceivableTrade | 2023b→2023a | 3997 | 22 | 6 |
| NotesAndAccountsReceivableTradeAndContractAssets | 2022→2023b | 3994 | 0 | 45 |
| NotesAndAccountsReceivableTradeAndContractAssets | 2023a→2024 | 4083 | 0 | 2 |
| NotesAndAccountsReceivableTradeAndContractAssets | 2023b→2023a | 3997 | 19 | 17 |
| NotesReceivableTrade | 2022→2023b | 3994 | 40 | 39 |
| NotesReceivableTrade | 2023a→2024 | 4083 | 58 | 22 |
| NotesReceivableTrade | 2023b→2023a | 3997 | 7 | 8 |
| ContractAssets | 2022→2023b | 3994 | 16 | 26 |
| ContractAssets | 2023a→2024 | 4083 | 13 | 26 |
| ContractAssets | 2023b→2023a | 3997 | 2 | 8 |
| AccountsReceivableTradeAndContractAssets | 2022→2023b | 3994 | 0 | 0 |
| AccountsReceivableTradeAndContractAssets | 2023a→2024 | 4083 | 0 | 0 |
| AccountsReceivableTradeAndContractAssets | 2023b→2023a | 3997 | 0 | 50 |
| ElectronicallyRecordedMonetaryClaimsOperatingCA | 2022→2023b | 3994 | 12 | 19 |
| ElectronicallyRecordedMonetaryClaimsOperatingCA | 2023a→2024 | 4083 | 17 | 18 |
| ElectronicallyRecordedMonetaryClaimsOperatingCA | 2023b→2023a | 3997 | 2 | 39 |

## (1) 取りこぼし候補(標準要素、5 社以上): 60 要素 → leak_candidates.csv
拡張要素でキーワードに当たるもの: 475 要素(424 社)
