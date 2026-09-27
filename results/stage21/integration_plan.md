# canonical の統合計画(ステージ21-C)

__OTHER__(残余科目)に落ちていた標準要素のうち、既存グループのキーワードに該当し、fy2024 提出の非金融企業で 5 社以上が使う 60 要素について、統合先を判定した。統合 42 要素、統合しない 18 要素。状態ベクトルの次元 D = 32 は変えない(既存グループの members を増やすだけ)。

社数・総資産比は fy2024 提出(当期)の非金融企業での値。総資産比は、その要素を持つ社での中央値。

| 要素(jppfs_cor_) | 社数 | 総資産比 | 統合先 | 理由 |
|---|---:|---:|---|---|
| `ElectronicallyRecordedMonetaryClaimsOperatingCA` | 1037 | 0.035 | 売上債権(`receivables_trade`) | 営業取引から生じた電子記録債権。受取手形の代替手段で、売上債権と同じ概念 |
| `ContractAssets` | 437 | 0.013 | 売上債権(`receivables_trade`) | 収益認識基準の契約資産。合算表示の要素(…TradeAndContractAssets)と排他で、売上債権を細かく分けた表示の一部 |
| `NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndOtherCNS` | 99 | 0.368 | 売上債権(`receivables_trade`) | 受取手形・完成工事未収入金等。建設業の売上債権 |
| `OperatingAccountsReceivableCA` | 95 | 0.014 | 売上債権(`receivables_trade`) | 営業未収入金。本業の対価に係る債権 |
| `AccountsReceivableTradeAndContractAssets` | 58 | 0.174 | 売上債権(`receivables_trade`) | 売掛金と契約資産の合算表示。現行の売上債権 members とすべて排他な別表記(2023-12-01 版タクソノミで新設) |
| `AccountsReceivableFromCompletedConstructionContractsCNS` | 53 | 0.094 | 売上債権(`receivables_trade`) | 完成工事未収入金。建設業の売掛金 |
| `AccountsReceivableInstallment` | 18 | 0.456 | 売上債権(`receivables_trade`) | 割賦売掛金。割賦販売による売上債権 |
| `AccountsReceivableInstallmentSalesCALEA` | 14 | 0.037 | 売上債権(`receivables_trade`) | リース業の割賦債権。割賦販売の対価で、本業の営業債権 |
| `AccountsReceivableLeaseCALEA` | 13 | 0.008 | 売上債権(`receivables_trade`) | リース業のリース料の未収。本業の営業債権 |
| `AccountsReceivableCARWY` | 12 | 0.030 | 売上債権(`receivables_trade`) | 鉄道業の未収運賃。本業の営業債権 |
| `NotesAndOperatingAccountsReceivableCA` | 9 | 0.099 | 売上債権(`receivables_trade`) | 受取手形及び営業未収入金 |
| `AccountsReceivableTradeNet` | 6 | 0.029 | 売上債権(`receivables_trade`) | 売掛金の純額表示。売掛金の別表記 |
| `OperatingAccountsReceivableOfHighwayBusinessCAHWY` | 5 | 0.062 | 売上債権(`receivables_trade`) | 高速道路事業の営業未収入金。本業の営業債権 |
| `ElectronicallyRecordedObligationsOperatingCL` | 905 | 0.045 | 仕入債務(`payables_trade`) | 営業取引から生じた電子記録債務。支払手形の代替手段で、仕入債務と同じ概念 |
| `NotesPayableAccountsPayableForConstructionContractsAndOtherCNS` | 105 | 0.149 | 仕入債務(`payables_trade`) | 支払手形・工事未払金等。建設業の仕入債務 |
| `AccountsPayableForConstructionContractsCNS` | 81 | 0.048 | 仕入債務(`payables_trade`) | 工事未払金。建設業の買掛金 |
| `OperatingAccountsPayable` | 75 | 0.034 | 仕入債務(`payables_trade`) | 営業未払金。本業の仕入・役務に係る債務 |
| `NotesAndOperatingAccountsPayableTrade` | 31 | 0.051 | 仕入債務(`payables_trade`) | 支払手形及び営業未払金 |
| `NotesPayableAccountsPayableForConstructionContractsCNS` | 15 | 0.097 | 仕入債務(`payables_trade`) | 支払手形・工事未払金。建設業の仕入債務 |
| `AccountsPayableOperatingSpecific` | 13 | 0.034 | 仕入債務(`payables_trade`) | 業種別様式の営業未払金。本業に係る債務 |
| `AccountsPayableConsignment` | 7 | 0.026 | 仕入債務(`payables_trade`) | 委託販売に係る未払金。商品仕入の対価 |
| `OperatingAccountsPayableHighwayBusinessCLHWY` | 5 | 0.113 | 仕入債務(`payables_trade`) | 高速道路事業の営業未払金。本業の営業債務 |
| `CostsOnUncompletedConstructionContractsCNS` | 239 | 0.006 | 棚卸資産(`inventories`) | 未成工事支出金。建設業の仕掛品 |
| `RealEstateForSale` | 199 | 0.098 | 棚卸資産(`inventories`) | 販売用不動産。不動産業の商品 |
| `RealEstateForSaleInProcess` | 111 | 0.176 | 棚卸資産(`inventories`) | 仕掛販売用不動産。不動産業の仕掛品 |
| `RawMaterialsAndSuppliesCNS` | 59 | 0.004 | 棚卸資産(`inventories`) | 建設業の材料貯蔵品 |
| `OtherInventories` | 49 | 0.003 | 棚卸資産(`inventories`) | その他の棚卸資産 |
| `RealEstateForSaleCNS` | 48 | 0.037 | 棚卸資産(`inventories`) | 建設業の販売用不動産。販売目的の棚卸資産 |
| `CostsOnUncompletedConstructionContractsAndOtherCNS` | 22 | 0.022 | 棚卸資産(`inventories`) | 未成工事支出金等。建設業の仕掛品 |
| `SemiFinishedGoods` | 15 | 0.025 | 棚卸資産(`inventories`) | 半製品 |
| `NotesPayableFacilities` | 115 | 0.002 | 未払金(`accounts_payable_other`) | 設備関係支払手形。営業外の未払で、未払金と同質 |
| `AccountsPayableFacilities` | 41 | 0.007 | 未払金(`accounts_payable_other`) | 設備関係未払金。未払金と同質 |
| `ElectronicallyRecordedObligationsFacilitiesCL` | 37 | 0.008 | 未払金(`accounts_payable_other`) | 設備投資に係る電子記録債務。営業外の未払で、未払金と同質 |
| `ElectronicallyRecordedObligationsNonOperatingCL` | 31 | 0.004 | 未払金(`accounts_payable_other`) | 営業外の電子記録債務。未払金と同質 |
| `AccountsPayableOtherAndAccruedExpenses` | 25 | 0.046 | 未払金(`accounts_payable_other`) | 未払金及び未払費用の合算表示。未払金の別表記として扱う(未払費用の単独表示は __OTHER__ に残るため、合算表示の社は未払費用の分だけ未払金が大きくなる) |
| `CommercialPapersLiabilities` | 103 | 0.030 | 短期借入金(`short_term_loans`) | コマーシャル・ペーパー。短期の資金調達手段で、短期借入金と同じ機能 |
| `ShortTermLoansPayableToSubsidiariesAndAffiliates` | 37 | 0.081 | 短期借入金(`short_term_loans`) | 関係会社短期借入金。借入先が関係会社であるだけの短期借入金 |
| `CurrentPortionOfLongTermLoansPayableToSubsidiariesAndAffiliates` | 12 | 0.016 | 短期借入金(`short_term_loans`) | 1 年内返済予定の関係会社長期借入金。short_term_loans は 1 年内返済予定の長期借入金を含む |
| `ShortTermBondsPayable` | 8 | 0.010 | 短期借入金(`short_term_loans`) | 短期社債。電子 CP の法的形式で、CP と同じく短期の資金調達 |
| `LongTermLoansPayableToSubsidiariesAndAffiliates` | 27 | 0.094 | 長期借入金(`long_term_loans`) | 関係会社長期借入金。借入先が関係会社であるだけの長期借入金 |
| `LongTermLoansPayableToRoadBuildingNCLHWY` | 5 | 0.108 | 長期借入金(`long_term_loans`) | 高速道路事業の道路建設関係長期借入金 |
| `RoadBuildingBondsPayableNCLHWY` | 5 | 0.527 | 社債(固定)(`bonds_noncurrent`) | 高速道路事業の道路建設関係社債。普通社債と同じ負債 |
| `AccountsReceivableOther` | 754 | 0.014 | 統合しない | 未収入金。営業外の債権で、売上債権ではない |
| `LongTermAccountsPayableOther` | 425 | 0.004 | 統合しない | 長期未払金。固定負債の営業外の未払で、対応するグループが無い |
| `IncomeTaxesReceivable` | 226 | 0.002 | 統合しない | 未収法人税等。税金の還付請求で、対応するグループが無い |
| `AdvancesReceivedOnUncompletedConstructionContractsCNS` | 149 | 0.025 | 統合しない | 未成工事受入金。顧客からの前受で契約負債に相当し、仕入債務ではない(契約負債 ContractLiabilities も __OTHER__ に残している) |
| `BondIssuanceCostDA` | 147 | 0.000 | 統合しない | 社債発行費。繰延資産で、社債ではない |
| `ConvertibleBondTypeBondsWithSubscriptionRightsToShares` | 51 | 0.075 | 統合しない | 転換社債型新株予約権付社債。資本性を持ち、株式への転換は再建の資本政策そのものなので普通社債と混ぜない(58 社、ステージ21-C A.1) |
| `ReserveForFinancialProductsTransactionLiabilitiesReservesUnderTheSpecialLawsSEC` | 26 | 0.001 | 統合しない | 証券業の金融商品取引責任準備金。キーワードの誤検出(棚卸資産ではない) |
| `BorrowingsOnMarginTransactionsCLSEC` | 21 | 0.012 | 統合しない | 証券業の信用取引借入金。金融業の固有科目 |
| `CashCollateralPledgedForSecuritiesBorrowingOnMarginTransactionsCASEC` | 20 | 0.003 | 統合しない | 証券業の信用取引借証券担保金。金融業の固有科目 |
| `AccountsReceivableOperatingLoansCALEA` | 12 | 0.177 | 統合しない | リース業の営業貸付金。貸付金(金融資産)で、売上債権ではない |
| `AccountsReceivableOtherLoansToCustomersCALEA` | 9 | 0.042 | 統合しない | リース業のその他の営業貸付債権。貸付金で、売上債権ではない |
| `AccountsReceivableOtherFromSubsidiariesAndAffiliates` | 8 | 0.006 | 統合しない | 関係会社未収入金。営業外の債権 |
| `LongTermAccountsPayableInstallmentPurchase` | 8 | 0.008 | 統合しない | 長期割賦未払金。設備の割賦購入で、長期未払金と同じ扱い |
| `AccountsReceivableOperatingLoansCASPF` | 7 | 0.536 | 統合しない | 特定金融業の営業貸付金。貸付金で、売上債権ではない |
| `BondsWithSubscriptionRightsToSharesNCL` | 6 | 0.098 | 統合しない | 新株予約権付社債。同上の理由で普通社債と混ぜない |
| `LongTermAccountsPayableToJapanRailwayConstructionTransportAndTechnologyAgencyNCLRWY` | 5 | 0.032 | 統合しない | 鉄道・運輸機構長期未払金。鉄道施設の譲渡代金で、長期未払金と同じ扱い |
| `BorrowingsOnGensakiTransactionCLSEC` | 5 | 0.122 | 統合しない | 証券業の現先取引借入金。金融業の固有科目 |
| `TradingProductsCASEC` | 5 | 0.006 | 統合しない | 証券業のトレーディング商品。金融業の固有科目 |

## 対象外としたもの

- 会社独自の拡張要素(キーワード該当 475 要素、424 社): 要素名は各社が付けるため、要素名による写像では拾えない。金額規模はステージ21-C A.2 に記録した。
- 5 社未満の標準要素: 判定の対象外(__OTHER__ に残る)。
