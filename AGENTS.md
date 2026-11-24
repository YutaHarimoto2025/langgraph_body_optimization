# AGENTS.md

## プロジェクト

2D の質点・バネ身体（mass-spring creature）の形態と筋肉制御を最適化する実験プロジェクト。最適化ループを LangGraph で構造化し、LLM を設計者・戦略立案者として使うことが目標だが、first goal は LangGraph・LLM なしの純粋 GA による周期制御最適化の確立にある。

目的は最強の人工生命 simulator ではなく、**数値最適化・LLM 判断・失敗処理・戦略変更をどこまできれいに構造化・観測できるか**の検証にある。背景・構想は  docs/open/0912_langgraph_artificial_body_optimization_idea.md  を参照。

## 参照

実装仕様の正本は docs/agents/ に分割してある。変更内容に応じて次を読む。

- 全体の目次と参照マップ: @docs/agents/index.md
- first goal の範囲・ロードマップ: @docs/agents/overview.md
- 身体・辺・遺伝子を変更する場合: @docs/agents/sim/creature.md
- 積分法・接触・摩擦・3D 対応・計算構成: @docs/agents/sim/physics.md
- 評価式・無効判定・エネルギー: @docs/agents/fitness.md
- GA の演算子・パラメータ・島モデル: @docs/agents/optimizer/ga.md
- 記録・再現メタデータ・viewer・動画: @docs/agents/recording.md
- コマンド・テスト・report/review/HTML の書式: @docs/agents/operations.md
- 人間向けの仕様と進捗: docs/humans/spec.html

## 必須事項

- 実装後は「静的テスト → 短い GA run による動的テスト → docs 正本化」まで行う。詳細は @docs/agents/operations.md。
- GA の世代ループ内に CPU 同期（int()・float()・np.asarray・print）を入れない。同期は run の最後の 1 回だけ。
- 仕様が複雑になったら docs/agents を内容別に分割し、このファイルには参照リンクだけを置く。
