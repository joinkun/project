# 引き継ぎ文書(ステージ1〜43、実装完了)

2026-09-28 時点の状態。**ステージ31 で最終設定を確定し、実装を完了とした。** 以降は論文執筆が中心になる。
ステージ20 以降の各ステージの詳細は `results/stage21`〜`stage31` の `report.md` にある。
この文書は、別のマシン・別の Claude Code セッションで作業を再開するためのもの。

## 1. 研究の目的と主張

EDINET の有価証券報告書(XBRL)から貸借対照表を集計し、継続企業の前提に関する注記(GC 注記)がある企業(class 2)について、
健全な状態へ移る「再建経路」を勘定科目レベルで生成する。

主張は 2 つ。

1. 会計恒等式(貸借一致・集計関係)を生成過程の全体で厳密に満たす。
2. **ただし恒等式を満たすだけでは、実務的に妥当な貸借対照表にはならない。**
   第 1 層(恒等式・符号)だけでは 68 社中 63 社が 13 項目チェックのいずれかに違反し、30 社が致命的な違反
   (資産合計・小計・総負債が負など)を起こす。第 2 層の制約を加えると違反は 0 になる(ステージ17)。
   統合後のデータ・乱数種の修正後に取り直すと 67/68 社(ステージ27)。

## 2. 環境とデータの移行

- Python 3.11+、`uv sync` で環境構築。`.env` に `EDINET_API_KEY`(再取得する場合のみ必要)。
- **`data/` は `.gitignore` 対象(約 3.3 GB)。git では移らないので `rsync` で送る。**
  ```bash
  rsync -avz --progress 元マシン:~/project/data/ ~/project/data/
  ```
  主な中身:
  - `data/raw/documents/*.zip`: EDINET の XBRL(再取得すると時間がかかる)
  - `data/interim/values/aggregated_core.csv`: 32 科目での集計(ステージ16 以降)
  - `data/interim/labels/labels.csv`: 3 クラスラベル
  - `data/interim/industry_12/folds.npz`: 5-fold 分割(全ステージ共通。**作り直さないこと**)
  - `data/interim/stage13`〜`stage18`: 各ステージの輸送計画・生成結果
  - `data/interim/canonical_v30/`: 貸倒引当金追加前(30 科目)の canonical と集計の退避
  - ステージ21〜23 で取得した 2022・2023 年度の書類: `data/raw/documents_2022_2023`、`documents_2023_2024`
    (一覧は `data/raw/index/document_list_*.csv`)。推定用パネルは `data/interim/stage23/panel/`
  - 除外リスト: `data/interim/stage28/excluded.csv`(ゴルフ場・金融。`--exclude` で各スクリプトに渡す)
  - 年度ごとの GC ラベル: `data/interim/stage30/labels_*.csv`、`firm_years.csv`
  - 最終設定の生成: `data/interim/stage31/r3_cap076`(R1 は `stage31/r1` → `stage28/r1`)
- 論文に使う表・BS 表はコミット対象の `results/` にある(最終設定は `results/stage31`)。

## 3. 作業ルール(これまでの指示で確立したもの)

- **プロセス分離を必ず守る**(macOS の libomp 二重ロードでクラッシュするため。ステージ11-A)。
  1. POT のプロセス: `precompute_*`。torch/sklearn の間接 import も無効化している
  2. torch/cvxpy のプロセス: `run_generation_*`、`run_projection_*`
  3. lightgbm/sklearn/scipy/matplotlib のプロセス: `run_metrics_*`、`run_paired_*`、`run_diagnose_*`、`run_bs_check_*`
  - **`KMP_DUPLICATE_LIB_OK` は使わない。**
- 乱数シードは 0/1/2 の 3 seed。分類器は seed=42。速度場は必ず `seeded_velocity_field`(種を設定してから
  構築)で作る(ステージ26 で見つけた再現性のバグ。`results/stage27/silent_failures.md` の 8 件目)。
- `uv run mypy --strict src scripts tests` と `uv run ruff check .` をクリーンに保つ
  (既存の `pipelines/p03_build_dataset.py` の E501 1 件のみ例外)。
- 最適化の呼び出しはすべて収束を検査する。未収束は例外(`SolverNotConvergedError`、`InfeasibleProjectionError`)。
- **15 項目チェックの違反 0 を必須条件とする**(ステージ40 で 14、41 で 15 を追加。最終設定では 165 行 = 55 社 × 3 seed、class 1 は 393 行 = 131 社 × 3 seed。ステージ19 までは 204 行 = 68 社 × 3 seed)。
- 評価は企業単位(最終設定では n=55)に集約してから、ペア Wilcoxon 検定とブートストラップ 95% CI(2,000 回)で行う。
  204 行を独立標本として扱わない(擬似反復)。
- 報告は日本語。**結論が変わることを恐れず、数字を有利に見せない。**「指標が良いこと」と「出力が妥当であること」は別で、BS 表を必ず目で読む。
- 結果ファイルは `results/stageNN/` に置く。

## 4. 確定した最終設定(ステージ31、ステージ41・42 で不活性制約を追加)

| 要素 | 設定 |
|---|---|
| データ | 非金融(`is_financial`)のうち、**金融 4 業種とゴルフ場を除外**(ステージ28)。class 0 = 3,768 書類、class 2 = 57 社。評価は外れ値 2 社を除く **55 社** |
| 状態ベクトル | canonical 32 科目(`conf/accounts/core.yaml`)、総資産で正規化(tilde 空間) |
| 輸送(対応先) | **等重み最近傍**(`conf/accounts/costs_equal.yaml`、非操作科目の重み 0 = `nearest_na0`)。目標は絞り込みなし(baseline) |
| 生成 | Flow Matching(K=13、5-fold、50 epoch)+ **PCFM**(毎ステップ射影) |
| 射影の重み | **W1** |
| 利益剰余金 | **R3 + 上限**: 固定せず、RE(1) − RE(0) ≤ **γ·A0、γ = 0.757**(`--freeze no_re --re-cap 0.7565…`)。その他有価証券評価差額金・非支配株主持分は固定 |
| 第 2 層の制約 | 小計の非負、純資産の非悪化、残余科目の下限、控除後純額の非負、**α=0.5**、**β=0.3** |
| 不活性制約 | **始点で 0 の 4 科目(売上債権、固定資産その他、自己株式、評価・換算差額等その他)を経路上で固定**(規則 M、ステージ41・42。`INACTIVE_ACCOUNTS`、`--inactive selected`)。0 → 非0 だけを禁じ、非0 → 0 は禁じない |
| 時間軸 | t∈[0,1] は 5 年(中小企業活性化協議会の「5 年以内の債務超過解消」。**原典は未確認**、ステージ31) |
| 主指標 | 純資産の改善、債務超過の解消(改善)。\|総資産変化\|、sparsity(実行可能性) |
| 副指標 | validity(**勘定振替にも反応する限界あり**、ステージ29)、proximity。参考: 識別 AUC |
| 感度分析 | R1(利益剰余金を固定) |

主な数値(`results/stage42/final.md`、最終設定): **class 2(55 社)** 違反 0、純資産の改善 0.238、債務超過の解消 8/12 社、
\|総資産変化\| 0.045、sparsity 13、validity LightGBM 0.998、proximity 0.548、符号付き順位相関 +0.325。
**class 1(131 社)** 違反 0、純資産の改善 0.189、解消 3/6、\|総資産変化\| 0.047、sparsity 12、proximity 0.311、順位相関 −0.133。
不活性制約なしの従来(ステージ31): class 2 で純資産の改善 0.233、\|総資産変化\| 0.044(R1: 0.000・0.115)。
感度分析 **R1 + 不活性制約**(ステージ43、`results/stage43/r1.md`): class 2 で純資産の改善 0.000、解消 2/12、\|総資産変化\| 0.115、
validity 0.751、順位相関 +0.499。R3 − R1(どちらも不活性制約あり)は純資産の改善 +0.186 [+0.157, +0.212]。
最終数値の表(95% 区間つき、中間報告 第 4 版用)は `results/stage43/final_table.md`。
23/57 社が上限 γ に達する。

**主設定の根拠(ステージ30・31)**: 2022 年度に GC 注記があり 2024 年度に外れた企業(再建群 15 社)の 87% で
利益剰余金が増え、注記が残った企業(30 社)では 20%。増資はむしろ非再建群に多い。R1 は純資産の改善を増資に
限るので再建を表現できない。**ただし最終設定は、再建群から作った基準 2(RE が増える社の割合、Wilson 95%
62.1〜96.3%)を満たさない(生成は 100%)。** 採用は基準の合格ではなく、この判断による。

制約の一覧(層・数式・根拠・拘束社数)は `results/stage17/constraints.md`。

## 5. 15 項目のチェック(`src/fsgen/accounting/plausibility.py`)

1 資産合計が負 / 2 小計が負 / 3 現金がほぼ 0 / 4 主要資産が全滅 / 5 負債が 2 倍超 / 6 純資産が悪化 /
7 健全→債務超過 / 8 資本金が半減以下 / 9 総負債が負 / 10 残余科目が負 / 11 現金 > 総資産 /
12 基礎科目の符号違反 / 13 控除後の純額が負 / 14 残余科目が下限未満(ステージ40)/ 15 不活性科目が動いた(ステージ41)

許容はすべて総資産比の絶対量。丸めとソルバ精度の許容は `_ROUND_TOL = 1e-3`。
チェック 5 は終点の負債 > 総資産(0) の 1% の場合に限る(ステージ19)。
`evaluate_checks`(1〜8)と `evaluate_extra_checks`(9〜15)に分かれている。
15 は `INACTIVE_ACCOUNTS`(規則 M の 4 科目)のうち始点で |x|/A < 1e-6 の科目が 1e-6 を超えて動いたら違反。
**不活性制約なしの従来の最終設定は 15 に掛かる**(class 2 128 / 165 行、class 1 315 / 393 行。規則 S の 17 科目で数えると
class 2 162 / 165 行)。自己株式を持たない苦境企業の 80〜95% に自己株式が生じていた(実データ 78 組中 0)。
符号制約では検出できない(自己株式は負で、0 から負は符号として正しい)。論文に書く(ステージ42)。
14 は canonical に依存しない: 状態ベクトルの全残余(`RESIDUAL_SIGN_FREE` = 評価・換算差額等の残余を除く)が
min(−1e-3 × 総資産(始点), 始点の値) を下回れば違反。10 は固定の 8 個(core では 14 と同じ科目)。
下限の制約(`residual_nonneg_accounts`)も同じ規則で、新しい残余は既定で下限の対象になる。

## 6. ステージの経緯(要点)

| ステージ | 内容と決定 |
|---|---|
| 1〜4 | EDINET 取得、XBRL 解析、canonical の構築(出現率 90% の閾値 + `always_include`) |
| 4.5〜4.6 | 残余科目 `__OTHER__*` を差額定義に変更。恒等式が構成的に厳密になる |
| 5〜7 | 制約行列 C、低次元基底、BS 表示、符号の確定(残余科目は自由符号) |
| 8〜9 | OT で対応先、3 分類器で validity。`costs.yaml`(文献ベースのコスト重み) |
| 10〜10.6 | FM / PCFM / CAFM。PCFM の改善は重み設計の差で、射影の不具合ではない |
| 11-A | OpenMP のクラッシュを 3 プロセス分離で解決 |
| 12 | 業種制約を試作(当時の基準は識別 AUC。不採用) |
| 13 | **旧 Sinkhorn は収束していなかった**。対数領域・収束検査を必須化。balanced OT が資産の崩壊の原因。最近傍と片側 OT が安全 |
| 14 | BS の会計的妥当性。チェック導入と第 2 層の制約(S1〜S5)。残余科目の抜け穴を発見 |
| 15 | 等重み vs コスト重みの最近傍。主指標がコスト重み proximity だったため自己参照的だった |
| 16 | **貸倒引当金 2 科目を canonical に追加**(30→32 科目)。控除後純額の制約、β を導入。主指標を \|総資産変化\| と validity に変更し、コスト重み最近傍が勝つ |
| 17 | **LogReg に標準化が欠けていたバグを修正**。業種制約の再評価(不採用)。制約を 2 層に整理(`constraint_layers`) |
| 18 | 業種 × 規模制約の 6 設計を比較。全設計で必須条件または主指標が悪化(不採用)。「不明」業種の始点には業種制約を課さない |
| 19 | 違反 2 社の原因は流れ場の外挿(資本構成の外れ値)。チェック 5・7 の修正。R1/R3 の比較と記述案 |
| 21〜23.5 | 2022・2023 年度の書類を取得して推定用パネル(4 期)を構築。部分調整モデルでコスト重みの推定を試みた |
| 24〜25 | コスト重みの 3 版(等重み・推定ベース・定性ベース)を生成・比較。回復指標(純資産の改善、債務超過の解消)を導入 |
| 26 | 自己資本比率による目標の絞り込み(不採用、baseline を維持)。**生成の乱数種の順序のバグを発見**(fold 0 が再現しない) |
| 27 | 種のバグを全生成スクリプトで修正し、主要な結論を再確認(CAFM_warm の劣位は有意でない、convex_qp が validity で上)。R1/R2/R3 と RE の上限(γ=0.5)。静かな失敗 8 件の一覧 |
| 28 | ゴルフ場・金融を機械的に除外(class 2 で 11 社)。修正後に非決定性が残らないことを確認(3 回の実行が完全一致)。CAFM を効果量で再評価 |
| 29 | 欠損填補の振替と利益を分ける R4(2 分岐の和集合への射影)。GC 企業全体から作った基準では全版が不合格 |
| 30 | GC 注記の遷移から再建群(15 社)を特定し、基準の出所を修正。全版が不合格(R3 + 上限は基準 2 のみ) |
| 31 | 基準に標本誤差を反映、γ = 0.757 を生成前に決定。**主設定を R3 + 上限に変更して確定**、R1 は感度分析 |
| 32 | 重みの最終試行(停止規則を事前に宣言)。施策の頻度(AGS 1994 の WP で確認できた 3 施策・6 科目。Ofek 1993 は入手できず不使用)、持続性(N&P 2001 の ATO・FLEV を再現)、自前の非対称性。**いずれも等重みに勝てず、重みの探索を終了** |
| 33 | 科目別の持続性の整理(記述統計)。ゼロの同順位・残余科目の注意、苦境企業では 21 科目で持続性が有意に低い |
| 34 | class 1 を始点とした検証(違反 0、実際の回復企業と大きさは近いが RE が増える割合は 100% 対 86%)。class 1 の回復率 43%。分類器が class 1 を学習に含めていたことが判明(validity への影響なし) |
| 36 | 重みの 5 版を class 1(131 社)で再比較。class 1 では 4 版とも純資産の改善でわずかに上回ったが class 2 で再現せず(2 版は逆向き)。**等重みで確定**。どの重みでも生成の科目の向きは実データと一致しない(符号付き順位相関 約 −0.1) |
| 38 | 遷移を T1(class 2 → 1)・T2(class 1 → 0)に分割、重みを射影・速度場にも作用させた(持続性ベース)。T2 は向きの負の相関を解消(−0.11 → +0.04)したが違反 5 社、連結は違反多数。**段階分割も作用点の拡大も不採用**。数値条件(最適性・勾配)は問題なし。ステージ37 は未実施 |
| 39 | 有形固定資産・投資その他の資産を開いた細分化(D = 38)と OT の再評価(`results/stage39/report.md`)。細分化は class 2 で \|総資産変化\| が悪化し**不採用**。不均衡 OT は純資産の改善が大きいが \|総資産変化\| が悪化し違反も出て**不採用**。持続性の重みは細分化の空間の最近傍では両 class で勝つが、細分化が不採用なので最終設定は**変えない**。細分化の生成は有形固定資産の残余を負にして土地などを作る(34〜54% の行) |
| 40 | 【canonical の最終試行】残余に下限を課した細分化(チェック 14 を追加)。14 項目の違反 0 だが、class 2 で \|総資産変化\| が悪化し(+0.014・+0.018)class 1 でのみ勝つ。**停止規則により細分化を不採用で確定、canonical の探索を終了**。下限を課しても土地の出現(46 / 99)は減らない(`results/stage40/report.md`) |
| 41 | 不活性制約(始点で 0 の科目を固定)。実データで苦境企業の自己株式の新規計上は 0 / 78。17 科目に課すと、自己株式 80〜95% → 0%、評価・換算差額等その他 85〜98% → 0% など科目別には実データに近づき、主指標の悪化なし、違反 0。ただし事前に決めた指標(企業ごとの出現数の乖離)は下がらず、事前の指標は不適切と判断し(科目別の割合で判定)、**規則 M(4 科目)を採用**(`results/stage41/report.md`)。ステージ40 の出現の集計の誤り(符号が自由な科目の負を数えていなかった)を訂正 |
| 42 | 確定した最終設定(不活性制約 規則 M を含む)で全指標を再計算(`results/stage42/`)。15 項目の違反 0(class 2・1)、主指標は従来とほぼ同じ(純資産の改善 +0.002)。中間報告 第 4 版用 |
| 43 | R1 を不活性制約つきで再計算(R3 − R1 の純資産の改善 +0.186、不活性制約は比較に影響しない)。class 1 の向きの不一致を診断(生成の class 1 は生成の class 2 とほぼ同じ向き、順位相関 +0.937。短期借入金・未払金などを減らすが実際の class 1 は変えない)。最終数値を 95% 区間つきで確定。**実装を完了** |

## 7. 却下した設計と理由

| 設計 | 理由 |
|---|---|
| balanced OT | 対応先が約 850 社に分散し、総資産が崩壊する。制約だけでは直せない(ステージ13・14) |
| 不均衡 OT | 外れ値の始点を落とす(τ = 0.1 で 57 社中 8 社、質量 0.18)。純資産の改善は上がるが \|総資産変化\| が両 class で悪化し違反も出る(ステージ39、D = 32・38 とも) |
| CAFM | PCFM に対する優位がない(除外後 57 社で、4 比較中 3 つは効果量が実質ゼロ、残る 1 つは小さい劣位。ステージ28)。「4 比較すべて有意に劣る」は現データでは成り立たない |
| W3(コスト重みの射影) | 毎ステップ射影で再生成しても、総資産の維持で W1 に劣る(ステージ16) |
| コスト重み最近傍(旧 `costs.yaml`) | ステージ16 では採用したが、重みの根拠を見直し(ステージ24〜26)、最終設定は等重み最近傍 |
| 根拠のある重み(5 通り) | 文献値・部分調整モデル・制度的根拠・施策の頻度・持続性のいずれも等重みに勝てない(ステージ24〜26・32)。停止規則により探索を終了 |
| R1(利益剰余金の固定) | 再建を表現できない(ステージ30・31)。感度分析として残す |
| R2(RE の近い目標を選んで固定) | 対応先の多くが RE の大きく負の企業になる(ステージ27) |
| R4(振替と利益の分離) | γ=0 では振替だけで割合を合わせ、純資産が改善しない。γ>0 では R3 + 上限とほぼ同じ(ステージ29〜31) |
| 目標の絞り込み(自己資本比率) | 改善がわずかで \|総資産変化\| が悪化(ステージ26) |
| α=1.0(減資禁止) | 主指標は α=0.5 と同じで、拘束される社が 3→42 社に増える。感度分析として残す |
| 総資産の変化幅 [0.5, 2.0](旧 S4) | チェック違反を減らさない |
| 業種ハード/ソフト制約 | ステージ17・18。必須条件を満たさないか、主指標が有意に悪化する |
| 科目の細分化(D = 38、ステージ39・40) | 残余に下限を課しても(ステージ40)、class 2 で \|総資産変化\| が有意に悪化し(等重み +0.014、持続性 +0.018)、class 1 でしか勝たない。停止規則により不採用で確定し、canonical の探索は終了(ステージ21 の統合、39・40 の細分化の 3 回) |
| 規模制約 (0.5, 2.0) | **採否は保留**。外れ値 2 社を除けば違反 0 で、対応先の総資産比 3 倍超が 17→0 社になるが、\|総資産変化\| が +0.020 悪化する(`results/stage19/tradeoff_size.csv`) |

## 8. 既知の限界

- **外れ値 2 社**(オンコセラピー・サイエンス S100TSS9、桃太郎源 S100TWFQ): 資本剰余金と欠損が総資産の 16〜30 倍
  (class 0 の最大は 4.33)。流れ場が外挿で破綻し、射影後の出力は設定・seed によって不安定になる。
  現行の設定で違反が出ないのは偶然で、総資産は 2〜4.5 倍に膨らんでいる。
- **チェック 8** は、資本金が総資産の 1% 未満の 7 社では許容 1e-3 が半減の幅を上回るため、実質的に働かない
  (制約 α は QP で課しているので、出力自体は守られる)。
- 業種マスタに未登録の企業がある(class 2 に 1 社、class 0 に 8 社)。
- 最近傍は業種を見ないため、対応先の 27/68 社が現金比 0.5 超(創薬ベンチャーなど)。小山カントリー倶楽部は IT 企業に対応づけられ、「借入して現金を抱える」出力になる。
- LogReg の validity は、標準化後も LightGBM・RF より 1 桁近く低い(線形分類器の限界)。絶対値は解釈に使わず、向きの一致だけを根拠にする。
- 識別 AUC は約 0.999 で、始点自体も 0.993。再建案の現実味の指標としては不適切(ステージ13)。
- K=13。canonical 変更で 99% 実効次元が 14 になったが、K=14 でも主指標はほぼ同じ(ステージ17)。
- 恒等式残差 r3 は 145 社で 1e6 円(decimals=−6 の丸め由来、canonical 変更前から存在)。
- **最終設定は全社で利益剰余金を増やす**(再建群は 87%、Wilson 区間の上端 96.3%、Fisher p = 0.04)。生成は構成上
  健全企業へ向かう経路で、再建に失敗する経路を含まない。
- 再建群は 15 社(2 年の観測)で、基準の区間は広い。γ は少数の企業に依存する(全社の p90 では 2.39)。
- **validity は純資産を動かさない勘定振替にも反応する**(R4 γ=0 で純資産の改善 0 のまま +0.011、ステージ29)。
- ゴルフ場の判定規則は class 0 側で 4 社を余分に含み、1 社を漏らす(ステージ28)。
- 「5 年以内の債務超過解消」の原典(中小企業活性化協議会実施基本要領)は未確認。
- 生成の科目別の動きは、大きさは実データと一致する(順位相関 +0.80)が、向きは一致しない(符号付き −0.1)。
  短期借入金を減らし有形固定資産を増やすが、実際の class 1 は逆(ステージ34・36)。重みでは変わらない。
- validity の分類器は、class 1 も「class 2 でない」側に含めて学習している(各スクリプトの `load_real`)。
  class 1 を除いても最終設定の validity は 0.998 で変わらない(ステージ34)。
- 回復企業(class 2 → 注記なし 15 社と class 1 → class 0 の 57 社)で利益剰余金が増えたのは 86%
  (Wilson 95% 76〜92%)。最終設定の生成(100%)はこれを上回る(ステージ34)。

- core の基礎科目は計算リンクの木の葉の和で、報告された小計のファクトと、木が不規則な書類で食い違う(差はその小計の残余に入る)。
  32 科目で総資産比 1e-3 を超える差は 4,156 書類中 16 件(最大 0.45、S100TMWI。ステージ39)。
- 生成は固定資産を持たない企業にも土地などを作る(細分化で見えた。D = 38 の class 2 で土地が 0 から正になる行が 47 / 99、実データは 1 年で 1.9%。ステージ39)。

- 生成は始点で 0 の科目を作る(自己株式 0 → 負が 79〜95%、実データ 0〜7%。class 1 の有形固定資産 0 → 正が 81%)。
  細分化では土地・建物・機械を同じ向きに動かし、実データの「土地を保持して設備を減らす」を再現しない(ステージ40)。

- **少額からの膨張(今は対処しない)**: 不活性制約は始点が 0 の企業にしか効かない。自己株式などは少額の保有からも
  膨らむ(始点が総資産比 1e-6〜1e-3 の行の 67〜90% が 1e-3 超へ。実データの 1 年は 12%)。対処には倍率制限が要るが、
  制約を増やし続けないため保留(ステージ41・42)。

- **class 1 の向きの不一致**: 生成の class 1 は class 2 の再建の向き(有利子負債・その他の負債の圧縮)をそのまま当てる
  (生成 class 1 対 生成 class 2 の順位相関 +0.937)。実際の class 1 は回復しても短期借入金を変えず、向きは動いた 22 科目中 5 科目しか
  一致しない(ステージ43)。
- **class 2 では R1 のほうが実データとの向きが近い**(順位相関 +0.499 対 +0.325)。R3 は減資・振替で利益剰余金を増やし、
  再建群は増資する(払込資本が増えた社: 生成 16%、再建群 60%)。R1 は純資産を改善しない(ステージ43)。

## 9. 今後の課題(実装しない。論文に書く)

- 取引ベクトル辞書(再建策を取引の組み合わせとして表す)
- 業種別のコスト重み
- 外れ値(資本構成が学習データの範囲外の企業)の自動検出
- 再建群の拡大(2021 年度以前の書類。ディスク容量の制約で未実施)と、再建に失敗する経路を含む生成
- K の最適化、規模制約の採否

**ステージ43 で実装を区切った。** 論文の「今後の課題」として書くもの:

- 少額からの膨張(倍率制限)
- 段階的な輸送(ステージ38 で試し不採用)
- 重みの測定(長期パネルによる持続性の推定)
- 外れ値検出の自動化
- class 1 の向きの不一致(class 1 を含めた速度場の学習)

## 10. 主要スクリプトと make ターゲット

| make | 内容 | 主なスクリプト |
|---|---|---|
| `make test` / `lint` / `typecheck` | テスト・ruff・mypy | — |
| `make fetch` / `parse` / `build-dataset` | EDINET 取得〜データセット構築 | `pipelines/p01〜p03` |
| — | canonical 集計・ラベル | `scripts/inspect_values.py`、`scripts/build_labels.py`、`scripts/build_canonical.py` |
| `make evaluate-stage13` | 収束版 OT・輸送設計 | `precompute_stage13_transport.py`(`--industry` 対応)、`run_generation_stage13.py`、`run_metrics_stage13.py` |
| `make evaluate-stage14` | 追加制約の段階別の再射影 | `run_projection_stage14.py`、`run_metrics_stage14.py`、`run_bs_check_stage14.py`(BS 表 10 社 + チェック) |
| `make evaluate-stage16` | **最終設定の生成**(32 科目、全設定) | `run_generation_stage16.py`(FM を 1 回学習し、全設定を毎ステップ射影。`--configs`、`--freeze no_re`、`--layers-from`、`--k`)、`run_metrics_stage16.py`(`--dir`、`--auc`)、`run_paired_stage16.py` |
| `make evaluate-stage17` | 業種制約の再評価 | 同上 + `precompute_stage13_transport.py --industry hard/soft` |
| `make evaluate-stage18` | 業種 × 規模の 6 設計 | `precompute_stage18_transport.py`(フォールバック 4 段階)、`run_paired_stage18.py` |
| `make evaluate-stage19` | R3 生成、規模制約の trade-off | `run_tradeoff_stage19.py` |
| (make なし) | ステージ39 細分化・OT | 細分化用ルート `data/stage39_root`(`conf/` は core_expanded、`data/raw` などはリンク)。`build_core_expanded_stage39.py`、`persistence_stage39.py`、`dims_stage39.py`、`gen_stage39.sh`、`metrics_stage39.py`(各ルートで実行)、`compare_stage39.py`、`bs_stage39.py`。再現手順は `results/stage39/report.md` |
| (make なし) | ステージ40 下限あり細分化 | `residuals_stage40.py`(残余の分布)、`emergence_stage40.py`(始点 0 → 正)、`gen_stage39.sh ... data/interim/stage40`、`metrics_stage39.py --gen-dir data/interim/stage40/gen`、`compare_stage40.py` |
| (make なし) | ステージ41 不活性制約 | `inactive_rates_stage41.py`(実データの出現率)、`inactive_select_stage41.py`(選定)、`run_generation_stage16.py`・`run_generation_stage34.py` の `--inactive`、`compare_stage41.py` |
| (make なし) | ステージ42・43 最終数値 | `evaluate_stage42.py`、`evaluate_stage43.py`(R1 の再計算・向きの診断・95% 区間つきの表)、`render_bs_stage29.py --versions final,r1_inactive`、`render_actual_bs_stage31.py`。R1 の生成は `R1=1 bash scripts/gen_stage39.sh ...` |

中核のコード:
- `src/fsgen/accounting/projection_qp.py`: 射影 QP(`extra_constraints`、strict な収束検査、OSQP→CLARABEL/ECOS へのフォールバック)
- `src/fsgen/accounting/plausibility.py`: 第 2 層の制約(`PlausibilityConfig`、`ConstraintLayers`)と 15 項目チェック
- `src/fsgen/evaluation/classifiers.py`: validity 用の 3 分類器(LogReg は `StandardScaler` 込み)
- `src/fsgen/models/transport.py`、`cost.py`、`industry.py`: 輸送・コスト・業種

最終設定(ステージ42。ステージ31 の設定 + 不活性制約)の再現:

```bash
uv run python scripts/precompute_stage13_transport.py --designs nearest_na0 --costs conf/accounts/costs_equal.yaml \
    --exclude data/interim/stage28/excluded.csv --out-dir data/interim/stage28/plans_equal
for s in 0 1 2; do
  bash scripts/gen_stage39.sh "$PWD" c2 data/interim/stage28/plans_equal conf/accounts/costs_equal.yaml \
      final nearest_na0 $s data/interim/stage42 selected   # class 2(run_generation_stage16.py --inactive selected)
  bash scripts/gen_stage39.sh "$PWD" c1 data/interim/stage28/plans_equal conf/accounts/costs_equal.yaml \
      final nearest_na0 $s data/interim/stage42 selected   # class 1(run_generation_stage34.py --inactive selected)
done   # 比較用の従来設定は setting=prev、最後の引数 none
uv run python scripts/metrics_stage39.py --gen-dir data/interim/stage42/gen --out data/interim/stage42/metrics
uv run python scripts/evaluate_stage42.py
uv run python scripts/bs_stage39.py --gen-dir data/interim/stage42/gen/c2 --settings final,prev --out results/stage42/bs.md
```

`gen_stage39.sh` の中身は `run_generation_stage16.py --configs W1_S5_a05_b03_re0.756519 --re-cap 0.7565192502545388
--freeze no_re --inactive selected`(class 2)と `run_generation_stage34.py --inactive selected`(class 1)。
ステージ31 の評価スクリプト(`run_stage25.py`、`evaluate_stage31.py`、`render_bs_stage29.py`)は不活性制約なしの出力用。

除外リストは `extract_business_stage28.py` → `special_industry_stage28.py`、基準は `criteria_stage31.py`
(入力はステージ30 の `transitions_stage30.py` → `recovery_stage30.py`)。

## 11. 会話記録

元マシンのセッション記録(約 40 MB):
`~/.claude/projects/-Users-yuto-project/0fd36790-775d-43da-9b0a-2b53c5f03396.jsonl`
細部(各ステージの全数値、ユーザーの元の指示文)が必要なときだけ参照する。
