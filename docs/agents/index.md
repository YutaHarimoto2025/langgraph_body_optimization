# bodyopt 実装正本 (index)

このディレクトリは実装の詳細な正本である。人間向けの要約は docs/humans/spec.html に置く。実装と正本が乖離したら、どちらを直すかを利用者に確認する。docs/open/ は履歴・構想の参考資料であり、正本ではない。

## 変更内容ごとの参照先

| 変更するもの | 読む文書 |
|---|---|
| 身体の形・辺・遺伝子 | @docs/agents/sim/creature.md |
| 積分法・接触・摩擦・3D 対応・計算構成 | @docs/agents/sim/physics.md |
| 評価式・無効判定・エネルギー | @docs/agents/fitness.md |
| GA の演算子・パラメータ・島モデル | @docs/agents/optimizer/ga.md |
| 記録レベル・再現メタデータ・viewer | @docs/agents/recording.md |
| コマンド・テスト・report/review の書式・運用 | @docs/agents/operations.md |
| first goal の範囲・ロードマップ | @docs/agents/overview.md |

## 文書一覧

- overview.md: 目的・first goal・拡張ロードマップ
- sim/creature.md: 2D 質点・バネ身体と遺伝子
- sim/physics.md: JAX シミュレータ
- fitness.md: 評価関数とエネルギー記録
- optimizer/ga.md: 固定 GA と拡張
- recording.md: 記録・再現・再生
- operations.md: 実行・検証・報告の運用手順
