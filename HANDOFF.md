# 引き継ぎ文書(ステージ1〜19)

2026-09-26 時点の状態。ステージ19で実装は一区切り。以降は論文執筆と、下記「今後の課題」の検討が中心になる。
この文書は、別のマシン・別の Claude Code セッションで作業を再開するためのもの。

## 1. 研究の目的と主張

EDINET の有価証券報告書(XBRL)から貸借対照表を集計し、継続企業の前提に関する注記(GC 注記)がある企業(class 2)について、
健全な状態へ移る「再建経路」を勘定科目レベルで生成する。

主張は 2 つ。

1. 会計恒等式(貸借一致・集計関係)を生成過程の全体で厳密に満たす。
2. **ただし恒等式を満たすだけでは、実務的に妥当な貸借対照表にはならない。**
   第 1 層(恒等式・符号)だけでは 68 社中 63 社が 13 項目チェックのいずれかに違反し、30 社が致命的な違反
   (資産合計・小計・総負債が負など)を起こす。第 2 層の制約を加えると違反は 0 になる(ステージ17)。

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
- 論文に使う表・BS 表はコミット対象の `results/` にある(`results/stage17`〜`stage19`)。

## 3. 作業ルール(これまでの指示で確立したもの)

- **プロセス分離を必ず守る**(macOS の libomp 二重ロードでクラッシュするため。ステージ11-A)。
  1. POT のプロセス: `precompute_*`。torch/sklearn の間接 import も無効化している
  2. torch/cvxpy のプロセス: `run_generation_*`、`run_projection_*`
  3. lightgbm/sklearn/scipy/matplotlib のプロセス: `run_metrics_*`、`run_paired_*`、`run_diagnose_*`、`run_bs_check_*`
  - **`KMP_DUPLICATE_LIB_OK` は使わない。**
- 乱数シードは 0/1/2 の 3 seed。分類器は seed=42。
- `uv run mypy --strict src scripts tests` と `uv run ruff check .` をクリーンに保つ
  (既存の `pipelines/p03_build_dataset.py` の E501 1 件のみ例外)。
- 最適化の呼び出しはすべて収束を検査する。未収束は例外(`SolverNotConvergedError`、`InfeasibleProjectionError`)。
- **13 項目チェックの違反 0 を必須条件とする**(204 行 = 68 社 × 3 seed)。
- 評価は企業単位(n=68)に集約してから、ペア Wilcoxon 検定とブートストラップ 95% CI(2,000 回)で行う。
  204 行を独立標本として扱わない(擬似反復)。
- 報告は日本語。**結論が変わることを恐れず、数字を有利に見せない。**「指標が良いこと」と「出力が妥当であること」は別で、BS 表を必ず目で読む。
- 結果ファイルは `results/stageNN/` に置く。

## 4. 確定した最終設定

| 要素 | 設定 |
|---|---|
| データ | 非金融 4,156 社。class 0 = 3,951 社(健全)、class 2 = 68 社(GC 注記あり)。class 1(重要事象のみ、137 社)は不使用 |
| 状態ベクトル | canonical 32 科目(`conf/accounts/core.yaml`)、総資産で正規化(tilde 空間) |
| 輸送(対応先) | **コスト重み最近傍**(`costs.yaml` の非対称重み、非操作科目の重み 0 = `nearest_na0`)。業種・規模の制約なし |
| 生成 | Flow Matching(K=13 の低次元部分空間、5-fold、50 epoch)+ **PCFM**(毎ステップ射影) |
| 射影の重み | **W1**(`1/(\|x\|+1e-3)`) |
| 非操作科目 | **R1**: 利益剰余金・その他有価証券評価差額金・非支配株主持分を等式で固定 |
| 第 2 層の制約 | 小計の非負、純資産の非悪化、残余科目の下限(総資産比 −1e-3)、控除後純額の非負、**α=0.5**(資本金・資本剰余金)、**β=0.3**(有形固定資産・棚卸資産・売上債権) |
| 時間軸 | t∈[0,1] は 5 年(中小企業活性化協議会の「5 年以内の債務超過解消」) |
| 主指標 | \|総資産変化\|(小さいほど良い)と validity(LightGBM が主、RF・LogReg は向きの確認) |
| 副指標 | 等重み proximity、コスト重み proximity。参考: sparsity、識別 AUC |

主な数値(最終設定、204 行): 違反 0、\|総資産変化\| 0.065、validity LightGBM 0.898 / RF 0.800 / LogReg 0.141。
R3(利益剰余金を固定しない)を併記する: 違反 0、ΔRE/資産の中央値 +0.27(年率 5.4%)、上位 9% は年率 30% 超。

制約の一覧(層・数式・根拠・拘束社数)は `results/stage17/constraints.md`。

## 5. 13 項目のチェック(`src/fsgen/accounting/plausibility.py`)

1 資産合計が負 / 2 小計が負 / 3 現金がほぼ 0 / 4 主要資産が全滅 / 5 負債が 2 倍超 / 6 純資産が悪化 /
7 健全→債務超過 / 8 資本金が半減以下 / 9 総負債が負 / 10 残余科目が負 / 11 現金 > 総資産 /
12 基礎科目の符号違反 / 13 控除後の純額が負

許容はすべて総資産比の絶対量。丸めとソルバ精度の許容は `_ROUND_TOL = 1e-3`。
チェック 5 は終点の負債 > 総資産(0) の 1% の場合に限る(ステージ19)。
`evaluate_checks`(1〜8)と `evaluate_extra_checks`(9〜13)に分かれている。

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

## 7. 却下した設計と理由

| 設計 | 理由 |
|---|---|
| balanced OT | 対応先が約 850 社に分散し、総資産が崩壊する。制約だけでは直せない(ステージ13・14) |
| 不均衡 OT | 外れ値の始点を落とす。最近傍に明確には勝たない |
| CAFM | validity で PCFM に劣る(4 比較すべて有意)。採用しないので第 2 層の制約も組み込んでいない |
| W3(コスト重みの射影) | 毎ステップ射影で再生成しても、総資産の維持で W1 に劣る(ステージ16) |
| 等重み最近傍 | \|総資産変化\| と validity でコスト重み最近傍に劣る(ステージ16) |
| α=1.0(減資禁止) | 主指標は α=0.5 と同じで、拘束される社が 3→42 社に増える。感度分析として残す |
| 総資産の変化幅 [0.5, 2.0](旧 S4) | チェック違反を減らさない |
| 業種ハード/ソフト制約 | ステージ17・18。必須条件を満たさないか、主指標が有意に悪化する |
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

## 9. 今後の課題(実装しない。論文に書く)

- 取引ベクトル辞書(再建策を取引の組み合わせとして表す)
- K の最適化
- 利益剰余金の成長率の上限(R1 と R3 の中間)
- 学習範囲を外れた始点の検出と、生成対象からの除外(外れ値 2 社)
- 規模制約の採否(外れ値の扱いと一緒に決める)

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

中核のコード:
- `src/fsgen/accounting/projection_qp.py`: 射影 QP(`extra_constraints`、strict な収束検査、OSQP→CLARABEL/ECOS へのフォールバック)
- `src/fsgen/accounting/plausibility.py`: 第 2 層の制約(`PlausibilityConfig`、`ConstraintLayers`)と 13 項目チェック
- `src/fsgen/evaluation/classifiers.py`: validity 用の 3 分類器(LogReg は `StandardScaler` 込み)
- `src/fsgen/models/transport.py`、`cost.py`、`industry.py`: 輸送・コスト・業種

最終設定の BS 表は `run_bs_check_stage14.py --gen16 nearest_na0:W1_S5_a05_b03 --gen16-dir data/interim/stage16 ...`
で出す(`results/stage17/bs_final/` が同じもの)。

## 11. 会話記録

元マシンのセッション記録(約 40 MB):
`~/.claude/projects/-Users-yuto-project/0fd36790-775d-43da-9b0a-2b53c5f03396.jsonl`
細部(各ステージの全数値、ユーザーの元の指示文)が必要なときだけ参照する。
