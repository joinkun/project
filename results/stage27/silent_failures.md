# エラーを出さずに誤った値を返した事象(静かな失敗)の一覧

論文の実装知見として記録する。各項目の根拠は、現在のコード(対処の実装とコメント)、`HANDOFF.md` §6、
`src/fsgen/errors.py` の記述、本セッションでの確認である。第 1〜7 件の発見時の細かな数値は、
元マシンの会話記録(`HANDOFF.md` §11)にあり、このマシンでは再確認していない。

| # | ステージ | 事象 | 症状 | 発見の経緯 | 影響範囲 | 対処(現在のコード) |
|---|---|---|---|---|---|---|
| 1 | 7 | 重み付き射影 QP の誤った解 | ソルバが status = optimal を返しながら、実際には最適でない解(コメントによれば数百億円規模のずれ)を返す。重みの比が大きく、目的関数のヘッセ行列の条件数が悪化するため | 射影後の経路の値を確認して判明(`projection_qp.py` の docstring) | 重み付き射影を使う生成すべて | 白色化(変数変換)で条件数を改善。`strict=True` で status が optimal 以外なら例外。OSQP → CLARABEL/ECOS へのフォールバック(`src/fsgen/accounting/projection_qp.py`) |
| 2 | 8.5 | validity の符号反転 | 健全度を表すはずの validity が、class 2 の確率をそのまま使うなど向きが逆になっていた(errors.py の記述) | 詳細は会話記録 | validity を使う評価 | 分類器の出力を `predict_proba_class2` に統一し、validity = 1 − P(class 2) と明示(`src/fsgen/evaluation/classifiers.py`) |
| 3 | 10.5 | 微分可能 QP 層の勾配破綻 | CAFM の学習で、射影層の解析勾配が数値勾配と一致しないまま学習が進む | 勾配の照合で判明 | CAFM の学習 | 学習前に解析勾配を数値勾配と照合し、破綻していれば例外(`check_layer_gradient`、`scripts/run_generation_stage13.py`) |
| 4 | 13 | SCS の前進解の精度不足 | 微分可能射影(diffcp/SCS)の既定精度 eps=1e-4 では、前進解に相対 2.8e-3 の制約違反が残る | 学習中の実測(`differentiable_projection.py` のコメント) | CAFM の学習 | eps=1e-6・max_iters=20000 に変更し、前進解の実行可能性を検査、違反は例外。失敗率が 1% を超えたら例外(`src/fsgen/models/differentiable_projection.py`、`flow_matching.py`) |
| 5 | 12〜13 | Sinkhorn の未収束 | エントロピー正則化 OT が収束判定を満たさないまま計画を返していた。発見はステージ13(`HANDOFF.md` §6) | 収束判定の値を確認して判明 | ステージ12 以前の輸送計画と、それを使った業種制約の判定 | 対数領域で解き、停止基準 1e-9 の収束を検査、未収束は `SolverNotConvergedError`。計画の健全性(NaN、質量 0 の行、周辺分布)を検査(`src/fsgen/models/transport.py`) |
| 6 | 13 | POT の不均衡 OT の破綻 | 不均衡 OT が一部の始点の質量を 0 にし(外れ値を切り捨て)、argmax が任意の対応先を返す | 計画の行和を確認して判明 | 不均衡 OT の設計 | 質量 0 の行は `allow_dropped_sources=True` を明示しない限り例外(`transport.py` の `check_plan_health`) |
| 7 | 17 | LogReg の標準化の欠落 | validity 用のロジスティック回帰が、特徴量を標準化せずに学習していた(ほかの箇所の LogReg は標準化していた) | validity の診断で判明 | LogReg の validity(ステージ16 以前) | `StandardScaler` をパイプラインに組み込み(`classifiers.py`) |
| 8 | 26 | 乱数種の設定順序 | 生成スクリプトが、速度場のネットワークを初期化した**後で** `train_fm` の中で初めて `torch.manual_seed` を呼んでいた。fold 0(68 社中 14 社)の初期値だけが実行ごとに変わり、fold 1〜4 は前の fold の学習後の乱数状態を引き継いで再現されていた | 同じ設定での 2 回の生成が一致しないことに、ステージ26 で気づいた(ステージ24 の結果との差が fold 0 の 14 行に限られていた) | 本研究の生成スクリプトすべて(ステージ10〜26)。seed を固定しても fold 0 は再現されず、手法・設定の比較に偶然の差が混ざっていた | ネットワークの構築を `seeded_velocity_field`(種を設定してから作る)に統一(`src/fsgen/models/flow_matching.py`、全生成スクリプト)。テスト(`tests/models/test_flow_matching_seed.py`)。主要な結論の再確認はステージ27 A |

## 共通する教訓

- ソルバや学習が「成功した」と報告すること(status = optimal、例外が出ない、損失が有限)は、出力が正しい
  ことを意味しない。出力そのもの(制約の充足、勾配の一致、周辺分布、再実行での一致)を検査する。
- 8 件目は数値計算ではなく再現性の失敗で、同じ入力で 2 回実行して結果を突き合わせるまで気づかなかった。
  生成を比較に使う前に、同じ設定の 2 回の実行が一致することを確かめる。
