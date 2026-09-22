# fsgen

倒産リスクの高い企業の貸借対照表を、健全な状態へ移行させる「経路」を
勘定科目レベルで生成する研究のためのプロジェクト。

**本研究の核心的主張は「会計恒等式(貸借一致・集計関係)を生成過程の全体で
厳密に満たす」ことである。** そのため制約(`fsgen.accounting`)まわりのコードは
性能より正確性を優先し、テストで担保する。

データソースは EDINET(金融庁の開示システム)の XBRL。対象は日本の
上場企業約1,000社の有価証券報告書。

## セットアップ

```bash
uv sync
cp .env.example .env   # EDINET_API_KEY を設定
```

`uv` が無い場合:

```bash
pip install -e ".[dev]"
```

## ディレクトリ構成

```
conf/                  Hydra 設定(config.yaml, data/, accounts/)
src/fsgen/
  acquisition/          層1: EDINET からの取得
  parsing/              層2: XBRL(_cal.xml 等) → 生テーブル
  accounting/           層3: 会計ドメインロジック(制約行列・射影)
  datasets/             データセット組み立て・分割
  models/               経路生成モデル
  evaluation/           制約充足などの評価指標
  viz/                  可視化
pipelines/              実行入口(ロジックは置かず薄く保つ)
tests/                  テスト(fixtures/ に実 XBRL を数社分固定)
notebooks/              探索専用。ロジックは書かない
data/                   .gitignore 対象(raw/interim/processed)
```

## パイプラインの実行

```bash
make fetch                                    # 層1: 取得
make parse                                    # 層2: パース
make build-dataset                            # 層3: データセット構築(level2)
uv run python pipelines/p03_build_dataset.py accounts=level3   # level3 に切替
```

## テスト

```bash
make test
make lint
make typecheck
```

## 設計上の注意点(ドメイン知識)

- 計算関係(親子・合計)は `_cal.xml` の `calculationArc` から取得する。
  `_pre.xml` は表示順のみで weight を持たないため使わない。
- 貸借対照表の計算リンクは資産側(`Assets`)と負債純資産側
  (`LiabilitiesAndNetAssets`)の2つの根に分かれており、両者を結ぶ
  「資産合計 = 負債純資産合計」というアークは XBRL 上に存在しない。
  この1本は `fsgen.accounting.constraints` で明示的に追加している。
- 勘定科目の符号は「非負・非正・自由」の3分類。減価償却累計額・
  貸倒引当金・自己株式は値そのものが負で報告される。判定ルールは
  `conf/config.yaml` の `sign_rules` で上書き可能。
- 金額は円単位の整数のまま保持する。科目ごとの丸めは合計と内訳の
  整合性を壊すため、正規化はデータセット構築の最終段階で一度だけ行う。
- 会社独自の拡張要素(`jpcrp...` 等)は `_cal.xml` の親をたどって
  標準タクソノミの「その他」科目に合算する。
- 階層をどこまで展開するかは `conf/accounts/level{2,3}.yaml` で切替える。

## テスト用 fixtures の配置手順

`tests/fixtures/` には EDINET から取得した実際の XBRL 提出書類一式を、
業種の異なる数社分、以下のように配置する(API 非依存でテストするため):

```
tests/fixtures/
  <企業名や証券コードなど識別子>/
    XBRL/PublicDoc/
      jpcrp030000-asr-...-cal.xml     # 計算リンク(必須)
      jpcrp030000-asr-...-pre.xml     # 表示リンク(参考用)
      jpcrp030000-asr-....xbrl        # インスタンス文書
      ...
```

EDINET の書類取得API(`/documents/{docID}`, type=1 で ZIP 一式)で
取得した ZIP を展開し、そのままのディレクトリ構造で
`tests/fixtures/<識別子>/` 配下にコピーする。個人情報や巨大なファイルを
含まないよう、必要なファイル(`*_cal.xml`, `*.xbrl`、GC注記調査に
必要な本文 htm など)だけを残して問題ない。最低2〜3社、可能なら
業種の異なる会社(例: 製造業と小売業)を含めること。
